import torch
from torchmetrics import Metric
import pandas as pd
import numpy as np

from ad_vla.dataset.waymo.waymo_utils import get_rater_feedback_score


class RFS(Metric):
    def __init__(self, **kwargs):
        super().__init__(**kwargs)
        self.add_state("preds", default=[], dist_reduce_fx="cat")
        self.add_state("gts", default=[], dist_reduce_fx="cat")
        self.add_state("pref_trajs", default=[], dist_reduce_fx="cat")
        self.add_state("pref_scores", default=[], dist_reduce_fx="cat")
        self.add_state("init_speeds", default=[], dist_reduce_fx="cat")
        self.add_state("scenario_clusters", default=[], dist_reduce_fx="cat")

    def update(self, pred_dict, gt_dict):
        if "pref_trajs" in gt_dict:
            pred = pred_dict["pred_trajs"]
            gt = gt_dict["gt_trajs"]
            pref_trajs = gt_dict["pref_trajs"]
            pref_scores = gt_dict["pref_scores"]
            init_speeds = gt_dict["init_speed"]
            scenario_clusters = gt_dict["scenario_clusters"]
            self.preds.append(pred.float())
            self.gts.append(gt.float())
            self.pref_trajs.append(pref_trajs.float())
            self.pref_scores.append(pref_scores.float())
            self.init_speeds.append(init_speeds.float())
            self.scenario_clusters.append(scenario_clusters)

    def compute(self):
        if isinstance(self.pref_trajs, torch.Tensor) and self.pref_trajs.numel() > 0:
            rfs = get_rater_feedback_score(
                inference_trajectories=self.preds[:, None].cpu().numpy(),
                inference_probs=np.ones((self.preds.shape[0], 1)),
                rater_specified_trajectories=self.pref_trajs.cpu().numpy(),
                rater_feedback_labels=self.pref_scores.cpu().numpy(),
                init_speed=self.init_speeds.cpu().numpy(),
            )["rater_feedback_score"]

            avg_rfs = (
                pd.DataFrame(
                    data={
                        "scenario_clusters": self.scenario_clusters.cpu().numpy(),
                        "rfs": rfs,
                    }
                )
                .groupby("scenario_clusters")
                .mean()
                .rfs.mean()
            )

            return avg_rfs
        return -1.0

    @staticmethod
    def _compute_metric(
        pred_traj: torch.Tensor,
        pref_trajs: torch.Tensor,
        pref_scores: torch.Tensor,
        init_speed: torch.Tensor,
    ) -> torch.Tensor:
        """
        Computes RFS for a batch of predictions.
        Assumes only one predicted trajectory per scene in the batch.

        Args:
            pred_traj (torch.Tensor): Predicted trajectory for each scene. Shape [B, T, 2]
            pref_trajs (torch.Tensor): Preference trajectories. Shape [B, 3, T, 2]
            pref_scores (torch.Tensor): Score label for each preference trajectory. Shape [B, 3]
            init_speed (torch.Tensor): Initial speed of ego-vehicle for RFS calculation. Shape [B]
        """
        rfs = get_rater_feedback_score(
            inference_trajectories=pred_traj[:, None].cpu().numpy(),
            inference_probs=np.ones((pred_traj.shape[0], 1)),
            rater_specified_trajectories=pref_trajs.cpu().numpy(),
            rater_feedback_labels=pref_scores.cpu().numpy(),
            init_speed=init_speed.cpu().numpy(),
        )["rater_feedback_score"]

        return rfs
