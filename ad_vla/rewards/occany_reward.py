import torch
import os

from ad_vla.dataset.data_types import E2EDataBatch, TrajectorySampling
from ad_vla.utils.traj_utils import resample_tensor


class OccAnyBEVReward:
    def __init__(
        self,
        cached_bevs_path: str,
        bev_weight: float = 0.5,
    ):
        self.cached_bevs_path = cached_bevs_path
        self.bev_weight = bev_weight

    def __call__(self, pred_trajs: torch.Tensor, batch: E2EDataBatch) -> torch.Tensor:
        """
        Computes BEV reward for predicted trajectories.

        Args:
            pred_trajs (torch.Tensor): Predicted trajectories shaped [B, G, T, 2].
            batch (E2EDataBatch): Data batch with ground truth trajectories.

        Returns:
            torch.Tensor: Rewards shaped [B, G]

        B = Batch size, G = Group size (number of predictions per sample), T = Trajectory length
        """
        gt_trajs = torch.stack([sample.fut_traj for sample in batch.samples]).to(
            pred_trajs.device
        )
        gt_traj_sampling = batch.samples[0].fut_traj_sampling

        target_sampling = TrajectorySampling(
            time_horizon=gt_traj_sampling.time_horizon, interval_length=0.1
        )

        if target_sampling != gt_traj_sampling:
            gt_trajs = resample_tensor(gt_trajs, gt_traj_sampling, target_sampling)

        # Get L2 reward
        l2_reward = -(pred_trajs - gt_trajs[:, None]).norm(dim=-1).mean(dim=-1)

        # Get coordinate positions in BEV grid
        quant_pred_trajs = torch.cat(
            [
                (2 * pred_trajs[..., 0:1]).int(),
                (50 - 2 * pred_trajs[..., 1:2]).int(),
            ],
            dim=-1,
        )

        # Load cached BEVs predicted from OccAny
        # (OccAny inference takes too long and requires too much memory for on-the-fly teacher output)
        bev_occ_map = []
        for sample in batch.samples:
            bev_path = os.path.join(
                self.cached_bevs_path, sample.metadata["scenario_id"] + ".pt"
            )
            if os.path.isfile(bev_path):
                cur_bev = torch.load(bev_path)
                bev_occ_map.append(cur_bev)
            else:
                print("Skipped BEV reward for sample", sample.metadata["scenario_id"])
                bev_occ_map.append(torch.zeros(200, 200).int())
        bev_occ_map = torch.stack(bev_occ_map).to(pred_trajs.device)

        # Save mask with valid positions
        bev_valid_mask = (quant_pred_trajs < 100) & (quant_pred_trajs >= 0)
        quant_pred_trajs[~bev_valid_mask] = 0

        batch_size = bev_occ_map.shape[0]

        batch_idx = torch.arange(batch_size, device=bev_occ_map.device)[:, None, None]

        # Take values of BEV along trajectory
        bev_values = bev_occ_map[:, 100:, 50:150][
            batch_idx,
            quant_pred_trajs[..., 0],
            quant_pred_trajs[..., 1],
        ]

        # BEV class convention:
        # 0: Background
        # 1: Static object
        # 2: Dynamic object
        # 3: Non-drivable area
        # 4: Drivable area
        # Reward calculation happens at each trajectory timestep depending on class
        col_mask = torch.logical_or(bev_values == 1, bev_values == 3)
        drivable_mask = bev_values == 4
        dyn_col_mask = bev_values == 2
        bg_mask = bev_values == 0

        bev_reward = (
            col_mask * (-10) + drivable_mask * (5) + dyn_col_mask * (-2) + bg_mask * (0)
        )

        # Aggregate trajectory reward using valid mask
        bev_reward = (bev_reward * bev_valid_mask.all(dim=-1)).sum(dim=-1) / (
            bev_valid_mask.all(dim=-1).sum(dim=-1) + 1e-6
        )

        reward = bev_reward * self.bev_weight + l2_reward

        return reward
