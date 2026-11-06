import torch

from ad_vla.dataset.data_types import E2EDataBatch


def compute_avg_jerk(traj: torch.Tensor) -> float:
    """
    Computes traj jerk for comfort penalty.

    Args:
        traj (torch.Tensor): tensor shaped [T, 2] where T is number of timesteps.

    Returns:
        float: trajectory jerk.
    """
    d_1 = traj[1:] - traj[:-1]
    d_2 = d_1[1:] - d_1[:-1]
    d_3 = d_2[1:] - d_2[:-1]
    return d_3.norm(dim=-1).mean(dim=-1).item()


def compute_tortuosity(traj: torch.Tensor) -> float:
    """
    Computes traj tortuosity for comfort penalty.

    Args:
        traj (torch.Tensor): tensor shaped [T, 2] where T is number of timesteps.

    Returns:
        float: trajectory tortuosity.
    """
    diff = (traj[1:] - traj[:-1]).norm(dim=-1).sum()
    total_displ = (traj[-1] - traj[0]).norm(dim=-1)
    return (diff / total_displ).item()


def get_thresholds(speed: torch.Tensor) -> tuple[float, float]:
    """
    Calculates lateral and longitudinal thresholds based on initial speed.
    """
    thresh_lat, thresh_lon = 1.8, 7.2

    if speed < 1.4:
        scale = 0.5
    elif speed < 11:
        scale = 0.5 * (1 + ((speed - 1.4) / (11 - 1.4)))
    else:
        scale = 1.0

    return scale * thresh_lat, scale * thresh_lon


def compute_similarity(
    fut_traj: torch.Tensor,
    pred_traj: torch.Tensor,
    special_trajs: dict[str, torch.Tensor],
) -> tuple[str, torch.Tensor, float]:
    """
    Returns reference trajectory with highest similarity to prediction.
    Returns no reference trajectory if similarity is at most 0 between
    prediction and special trajectories.
    """
    # Get initial speed from expert and get thresholds
    init_speed = fut_traj[0].norm(dim=-1) / 0.2

    thresh_lat, thresh_lon = get_thresholds(init_speed.item())

    # Initialize special trajectories and their scores
    reference_trajs = {
        "expert": fut_traj,
        **special_trajs,
    }

    mms_ref_dict = {
        "wrong_speed": 7.0,
        "expert": 10.0,
        "neglect_instruction": 4.0,
        "off_road": 1.0,
        "crash": 0.0,
    }

    # Find reference trajectory match
    max_sim = -1.0
    selected_traj_type, selected_traj = None, None
    for ref_traj_type, ref_traj in reference_trajs.items():
        ref_traj = ref_traj.to(pred_traj.device)

        final_displ = ref_traj[-1] - ref_traj[-2]
        final_heading = torch.atan2(final_displ[1], final_displ[0])

        # At last time step of each reference trajectory,
        # at that coordinate frame, get prediction's displacement errors,
        # use those and thresholds to calculate similarity
        dx, dy = pred_traj[-1] - ref_traj[-1]
        dx_rot = dx * torch.cos(final_heading) + dy * torch.sin(final_heading)
        dy_rot = -dx * torch.sin(final_heading) + dy * torch.cos(final_heading)
        dx_rot = dx_rot.abs().item()
        dy_rot = dy_rot.abs().item()

        if dx_rot <= thresh_lon and dy_rot <= thresh_lat:
            sim = 1.0
        else:
            sim_lat = max(0, 1.0 - ((dy_rot - thresh_lat) / thresh_lat))
            sim_lon = max(0, 1.0 - ((dx_rot - thresh_lon) / thresh_lon))
            sim = min(sim_lat, sim_lon)

        # Assign new match if similarity increased
        if max_sim < sim:
            max_sim = sim
            selected_traj_type = ref_traj_type
            selected_traj = ref_traj
        elif max_sim == sim:
            # If similarity stays the same but now matches to higher MMS, update match
            if mms_ref_dict.get(selected_traj_type, -1.0) < mms_ref_dict.get(
                ref_traj_type, -1.0
            ):
                selected_traj_type = ref_traj_type
                selected_traj = ref_traj

    # If no match, return None
    if max_sim <= 0:
        return None, None, 0.0

    return selected_traj_type, selected_traj, max_sim


def compute_mms(
    fut_traj: torch.Tensor,
    pred_traj: torch.Tensor,
    special_trajs: dict[str, torch.Tensor] | None = None,
) -> float:
    if special_trajs is None:
        return -1.0

    # Get initial velocities for case 1 later
    init_ref_speed = (fut_traj[0] / 0.2).cpu()
    init_pred_speed = (pred_traj[0] / 0.2).cpu()

    # Get reference trajectory match
    ref_traj_type, ref_traj, sim = compute_similarity(
        fut_traj, pred_traj, special_trajs
    )

    comfort_penalty = 0.0

    # If there is a match, compute comfort_penalty
    if ref_traj is not None:
        ref_jerk = compute_avg_jerk(ref_traj)
        pred_jerk = compute_avg_jerk(pred_traj)
        if pred_jerk > 1.44 * ref_jerk:
            comfort_penalty += 1.0

        ref_tort = compute_tortuosity(ref_traj)
        pred_tort = compute_tortuosity(pred_traj)
        if pred_tort > 1.06 * ref_tort:
            comfort_penalty += 1.0

    # Get reference score of match, update with comfort penalty if > 1.0
    mms_ref = {
        "wrong_speed": 7.0,
        "expert": 10.0,
        "neglect_instruction": 4.0,
        "off_road": 1.0,
        "crash": 0.0,
    }.get(ref_traj_type, 0)

    if mms_ref > 1.0:
        mms_ref -= comfort_penalty

    # Case 1: initial speed is inconsistent with actual initial speed
    if (init_ref_speed * init_pred_speed).sum().item() <= 0.5 * (
        init_ref_speed.norm().item() ** 2
    ):
        return 0.0
    # Case 2: predictions near off-roads and crashes
    elif (mms_ref == 0 or mms_ref == 1.0) and sim >= 0.4:
        return mms_ref
    # Case 3:
    elif sim * mms_ref >= 3.5 - comfort_penalty:
        return sim * mms_ref
    else:
        return 3.5 - comfort_penalty


def compute_batch_mms(batch: E2EDataBatch, pred_trajs: torch.Tensor) -> torch.Tensor:
    mms_list = []
    for i, sample in enumerate(batch.samples):
        mms_list.append(
            compute_mms(sample.fut_traj, pred_trajs[i], sample.special_trajs)
        )
    return torch.tensor(mms_list).to(pred_trajs.device)
