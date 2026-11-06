"""Collision reward from ``AgentInfo`` tracks, for dataset samples without a simulator context.

``agent_ctx`` resamples the tracks to a plan's time grid in the format of ``SimPDMReward``'s reward context
(``agent_corners``, ``agent_valid``, ``agent_times_s``). A lost track is ignored at that time, never a free road.
"""

from __future__ import annotations

import torch

from ad_vla.dataset.data_types import AgentInfo, E2EDataBatch
from ad_vla.rewards.sim_pdm_reward import box_points, boxes_overlap, headings

COLLISION_TYPES = ("vehicle", "person", "cyclist")


def agent_ctx(
    info: AgentInfo, num_poses: int, dt: float, types: tuple[str, ...] = COLLISION_TYPES
) -> dict:
    """``agent_corners`` [T, A, 4, 2], ``agent_valid`` [T, A], ``agent_times_s`` [T] on the grid dt, 2 dt, ..."""
    times = torch.arange(1, num_poses + 1, dtype=torch.float32) * dt
    if info.boxes is None or info.fut_boxes is None or len(info.boxes) == 0:
        return {
            "agent_corners": torch.zeros(num_poses, 0, 4, 2),
            "agent_valid": torch.zeros(num_poses, 0, dtype=torch.bool),
            "agent_times_s": times,
        }
    keep = torch.tensor(
        [t in types for t in (info.coarse_types or [])], dtype=torch.bool
    )
    boxes = torch.cat(
        [info.boxes[keep, None], info.fut_boxes[keep]], dim=1
    ).float()  # [A, K + 1, 7]; step 0 is now
    mask = torch.cat(
        [torch.ones(int(keep.sum()), 1, dtype=torch.bool), info.fut_mask[keep].bool()],
        dim=1,
    )
    step = times / float(info.fut_interval_s)
    low = step.floor().long().clamp(max=boxes.shape[1] - 1)
    high = (low + 1).clamp(max=boxes.shape[1] - 1)
    frac = step - low.float()
    on_step = (
        frac < 1e-6
    )  # no interpolation here, so the next step's validity doesn't matter
    w = frac[None, :, None]
    a, b = boxes[:, low], boxes[:, high]
    dyaw = torch.atan2(
        torch.sin(b[..., 6] - a[..., 6]), torch.cos(b[..., 6] - a[..., 6])
    )
    center = a[..., :2] * (1 - w) + b[..., :2] * w
    yaw = a[..., 6] + dyaw * w[..., 0]
    # box_points takes one size for all boxes, so the per-agent corners are built here, in its order.
    half_l, half_w = a[..., 3, None] / 2, a[..., 4, None] / 2
    sx, sy = a.new_tensor([1, 1, -1, -1]), a.new_tensor([1, -1, -1, 1])
    cos, sin = torch.cos(yaw)[..., None], torch.sin(yaw)[..., None]
    dx, dy = sx * half_l, sy * half_w
    corners = torch.stack(
        [
            center[..., 0, None] + dx * cos - dy * sin,
            center[..., 1, None] + dx * sin + dy * cos,
        ],
        dim=-1,
    )  # [A, T, 4, 2]
    # Past the last step, or with a missing end point, the agent is unknown.
    beyond = step > boxes.shape[1] - 1
    valid = mask[:, low] & (mask[:, high] | on_step[None]) & ~beyond[None]
    return {
        "agent_corners": corners.permute(1, 0, 2, 3),
        "agent_valid": valid.permute(1, 0),
        "agent_times_s": times,
    }


class AgentCollisionReward:
    """Rewards [B, G]: 1 without an at-fault collision, 0 with one. Same at-fault rule as ``SimPDMReward``.

    Samples without annotations score 1 for every plan, so they give no signal, and ``last_coverage`` reports
    the share of samples that had them.
    """

    def __init__(
        self,
        num_poses: int = 20,
        interval_length: float = 0.25,
        ego_length: float = 4.9,
        ego_width: float = 2.0,
        rig_to_center: float = 1.4,
    ) -> None:
        self.num_poses, self.dt = num_poses, interval_length
        self.ego_length, self.ego_width, self.rig_to_center = (
            ego_length,
            ego_width,
            rig_to_center,
        )
        self.last_coverage = float("nan")

    def score(self, plans: torch.Tensor, info: AgentInfo | None) -> torch.Tensor:
        """Reward [G] for plans [G, T, 2] in the current ego frame."""
        out = torch.ones(plans.shape[0], device=plans.device, dtype=plans.dtype)
        if info is None or info.boxes is None:
            return out
        ctx = agent_ctx(info, plans.shape[1], self.dt)
        corners = ctx["agent_corners"].to(plans.device).float()
        if corners.shape[1] == 0:
            return out
        valid = ctx["agent_valid"].to(plans.device)
        positions = torch.cat([plans.new_zeros(plans.shape[0], 1, 2), plans], dim=1)
        yaw = headings(positions)
        cos, sin = torch.cos(yaw), torch.sin(yaw)
        centers = plans + torch.stack([cos, sin], dim=-1) * self.rig_to_center
        moving = (positions[:, 1:] - positions[:, :-1]).norm(dim=-1) > 0.05
        ego = box_points(centers, yaw, self.ego_length, self.ego_width, mids=False)
        hit = boxes_overlap(ego[:, :, None], corners[None])  # [G, T, A]
        rel = corners.mean(dim=2)[None] - centers[:, :, None]
        rel_x = rel[..., 0] * cos[..., None] + rel[..., 1] * sin[..., None]
        at_fault = (rel_x > -self.ego_length / 4.0) & moving[..., None]
        return torch.where(
            (hit & valid[None] & at_fault).any(dim=(1, 2)), out * 0.0, out
        )

    def __call__(self, pred_trajs: torch.Tensor, batch: E2EDataBatch) -> torch.Tensor:
        infos = [getattr(sample, "agent_info", None) for sample in batch.samples]
        self.last_coverage = sum(
            i is not None and i.boxes is not None for i in infos
        ) / max(1, len(infos))
        return torch.stack(
            [self.score(pred_trajs[b].float(), infos[b]) for b in range(len(infos))]
        ).to(pred_trajs.dtype)
