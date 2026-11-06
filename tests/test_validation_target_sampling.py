from types import SimpleNamespace

import torch
from omegaconf import OmegaConf

import ad_vla.train_stages.base_module as base_module
from ad_vla.dataset.data_types import TrajectorySampling
from ad_vla.train_stages.base_module import BaseTrainingModule


def _module(target_sampling):
    module = BaseTrainingModule.__new__(BaseTrainingModule)
    torch.nn.Module.__init__(module)
    module.val_target_sampling = BaseTrainingModule._coerce_val_target_sampling(
        target_sampling
    )
    module.val_generate_cfg = None
    return module


def _batch(fut_traj, sampling):
    sample = SimpleNamespace(
        fut_traj=fut_traj,
        fut_traj_sampling=sampling,
    )
    return SimpleNamespace(samples=[sample])


def test_null_validation_target_preserves_native_sampling(monkeypatch):
    source_sampling = TrajectorySampling(num_poses=50, interval_length=0.1)
    fut_traj = torch.arange(100, dtype=torch.float32).reshape(50, 2)

    def fail_if_resampled(*_args, **_kwargs):
        raise AssertionError("null target sampling must not invoke resample_tensor")

    monkeypatch.setattr(base_module, "resample_tensor", fail_if_resampled)
    gt_trajs, target_sampling = _module(None)._prepare_validation_trajectories(
        _batch(fut_traj, source_sampling)
    )

    assert target_sampling is source_sampling
    assert torch.equal(gt_trajs, fut_traj.unsqueeze(0))


def test_null_validation_target_preserves_missing_native_sampling(monkeypatch):
    fut_traj = torch.arange(100, dtype=torch.float32).reshape(50, 2)

    def fail_if_resampled(*_args, **_kwargs):
        raise AssertionError("null target sampling must not invoke resample_tensor")

    monkeypatch.setattr(base_module, "resample_tensor", fail_if_resampled)
    gt_trajs, target_sampling = _module(None)._prepare_validation_trajectories(
        _batch(fut_traj, None)
    )

    assert target_sampling is None
    assert torch.equal(gt_trajs, fut_traj.unsqueeze(0))


def test_validation_target_resamples_ground_truth_to_requested_grid():
    source_sampling = TrajectorySampling(num_poses=50, interval_length=0.1)
    target_cfg = OmegaConf.create({"num_poses": 25, "interval_length": 0.2})
    fut_traj = torch.arange(100, dtype=torch.float32).reshape(50, 2)

    gt_trajs, target_sampling = _module(target_cfg)._prepare_validation_trajectories(
        _batch(fut_traj, source_sampling)
    )

    assert target_sampling == TrajectorySampling(num_poses=25, interval_length=0.2)
    assert torch.equal(gt_trajs, fut_traj[1::2].unsqueeze(0))


def test_validation_with_null_target_uses_native_grid_end_to_end():
    source_sampling = TrajectorySampling(num_poses=50, interval_length=0.1)
    fut_traj = torch.arange(100, dtype=torch.float32).reshape(50, 2)

    class Model(torch.nn.Module):
        def __init__(self):
            super().__init__()
            self.anchor = torch.nn.Parameter(torch.zeros(()))
            self.received_target_sampling = None

        def predict_from_processed_inputs(self, **kwargs):
            self.received_target_sampling = kwargs["target_sampling"]
            return torch.zeros(1, 50, 2), {}

    class Metrics:
        def update(self, pred_dict, gt_dict):
            self.pred_dict = pred_dict
            self.gt_dict = gt_dict

    module = _module(None)
    module.model = Model()
    module.aux_head = None
    module.val_metrics = Metrics()
    module.val_log_every_n_steps = 100
    batch = _batch(fut_traj, source_sampling)
    batch.model_inputs = {}

    module.validation_step(batch, batch_idx=1)

    assert module.model.received_target_sampling is source_sampling
    assert module.val_metrics.pred_dict["pred_trajs"].shape == (1, 50, 2)
    assert torch.equal(
        module.val_metrics.gt_dict["gt_trajs"],
        fut_traj.unsqueeze(0),
    )


def test_validation_uses_requested_grid_for_prediction_and_metrics():
    source_sampling = TrajectorySampling(num_poses=50, interval_length=0.1)
    target_sampling = TrajectorySampling(num_poses=25, interval_length=0.2)
    fut_traj = torch.arange(100, dtype=torch.float32).reshape(50, 2)

    class Model(torch.nn.Module):
        def __init__(self):
            super().__init__()
            self.anchor = torch.nn.Parameter(torch.zeros(()))
            self.received_target_sampling = None

        def predict_from_processed_inputs(self, **kwargs):
            self.received_target_sampling = kwargs["target_sampling"]
            return torch.zeros(1, 25, 2), {}

    class Metrics:
        def update(self, pred_dict, gt_dict):
            self.pred_dict = pred_dict
            self.gt_dict = gt_dict

    module = _module(target_sampling)
    module.model = Model()
    module.aux_head = None
    module.val_metrics = Metrics()
    module.val_log_every_n_steps = 100
    batch = _batch(fut_traj, source_sampling)
    batch.model_inputs = {}

    module.validation_step(batch, batch_idx=1)

    assert module.model.received_target_sampling is target_sampling
    assert module.val_metrics.pred_dict["pred_trajs"].shape == (1, 25, 2)
    assert torch.equal(
        module.val_metrics.gt_dict["gt_trajs"],
        fut_traj[1::2].unsqueeze(0),
    )
