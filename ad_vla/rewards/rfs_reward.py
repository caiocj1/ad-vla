import torch
import numpy as np

from ad_vla.dataset.data_types import E2EDataBatch
from ad_vla.dataset.waymo.waymo_utils import prepare_for_rfs, get_rater_feedback_score


class RFSReward:
    def __init__(self):
        pass

    def __call__(self, pred_trajs: torch.Tensor, batch: E2EDataBatch) -> torch.Tensor:
        """
        Computes the Rater Feedback Score (RFS) reward for predicted trajectories.

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

        gt_dict = {
            "gt_trajs": gt_trajs.to(pred_trajs.device),
        }
        gt_dict.update(prepare_for_rfs(batch, pred_trajs.device))

        inference_probs = (
            np.ones((pred_trajs.shape[0], pred_trajs.shape[1])) / pred_trajs.shape[1]
        )

        rfs = get_rater_feedback_score(
            inference_trajectories=pred_trajs.cpu().numpy(),
            inference_probs=inference_probs,
            rater_specified_trajectories=gt_dict["pref_trajs"].cpu().numpy(),
            rater_feedback_labels=gt_dict["pref_scores"].cpu().numpy(),
            init_speed=gt_dict["init_speed"].cpu().numpy(),
        )["rater_feedback_score_per_inference"]  # [B, rollouts_per_gpu]

        reward = torch.from_numpy(rfs).to(pred_trajs.device)
        return reward
