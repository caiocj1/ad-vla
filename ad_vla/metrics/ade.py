import torch
from torchmetrics import Metric


class ADE(Metric):
    def __init__(self, **kwargs):
        super().__init__(**kwargs)

        self.add_state("metric", default=torch.tensor(0.0), dist_reduce_fx="sum")
        self.add_state("total", default=torch.tensor(0), dist_reduce_fx="sum")

    def update(self, pred_dict, gt_dict):
        pred = pred_dict["pred_trajs"]
        gt = gt_dict["gt_trajs"].float()

        ade = ADE._compute_metric(pred, gt)

        self.metric += ade.sum()
        self.total += ade.shape[0]

    def compute(self):
        return self.metric / self.total

    @staticmethod
    def _compute_metric(pred_traj: torch.Tensor, gt_traj: torch.Tensor) -> torch.Tensor:
        return (pred_traj - gt_traj).norm(dim=-1).mean(dim=-1)
