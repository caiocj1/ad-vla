import torch

from ad_vla.dataset.data_types import E2EDataBatch, TrajectorySampling
from ad_vla.utils.traj_utils import resample_tensor


class NormL2Reward:
    def __init__(self):
        pass

    def __call__(self, pred_trajs: torch.Tensor, batch: E2EDataBatch) -> torch.Tensor:
        """
        Computes the L2 norm reward for predicted trajectories.

        Args:
            pred_trajs (torch.Tensor): Predicted trajectories shaped [B, G, T, 2].
            batch (E2EDataBatch): Data batch with ground truth trajectories.

        Returns:
            torch.Tensor: Rewards shaped [B, G].

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

        reward = -(gt_trajs[:, None] - pred_trajs).norm(dim=-1).mean(-1)
        return reward
