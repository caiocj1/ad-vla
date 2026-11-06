"""GRPO probe (fixed held-out states, deterministic plans) and the state-list input (CPU)."""

from types import SimpleNamespace

import pytest
import torch

from ad_vla.dataset.alpasim.alpasim_state_dataset import AlpaSimStateDataset
from ad_vla.rewards.sim_pdm_reward import SimPDMReward
from ad_vla.train_stages.grpo_module import GRPOModule
from tests.test_sim_pdm_reward import _ctx, _static_agent, _straight


class _PlanModel:
    """Returns the same straight 8 m/s plan for every state; tracks train / eval mode."""

    def __init__(self):
        self.training = True

    def eval(self):
        self.training = False

    def train(self):
        self.training = True

    def predict_from_processed_inputs(
        self, num_traj_samples, generate_cfg, target_sampling, **_
    ):
        assert num_traj_samples == 1 and generate_cfg == {"sampler": "ode"}
        assert target_sampling.num_poses == 20
        return torch.stack([_straight()] * 2), {}


class _Logger:
    def __init__(self):
        self.calls = []

    def log_metrics(self, metrics, step):
        self.calls.append((step, metrics))


def _probe_module():
    states = [
        SimpleNamespace(
            metadata={"rl_ctx": _ctx(_static_agent(25.0, 0.0)), "probe_group": "fail"}
        ),
        SimpleNamespace(metadata={"rl_ctx": _ctx(), "probe_group": "control"}),
    ]
    batch = SimpleNamespace(samples=states, model_inputs={})
    logger = _Logger()
    module = SimpleNamespace(
        probe_cfg={"every_n_batches": 250},
        _probe_step=None,
        global_step=31,
        model=_PlanModel(),
        _probe_dataloader=lambda: [batch],
        transfer_batch_to_device=lambda b, device, idx: b,
        device="cpu",
        val_generate_cfg={"sampler": "ode"},
        reward_fn=SimPDMReward(),
        probe_extra_rewards={},
        _viz_probe_states=lambda: None,
        trainer=SimpleNamespace(is_global_zero=True),
        loggers=[logger],
    )
    return module, logger


def test_probe_reports_all_states_and_each_group_once_per_step():
    module, logger = _probe_module()
    metrics = GRPOModule.run_probe(module)
    assert metrics["probe/num_states"] == 2.0
    assert metrics["probe/collision_rate"] == pytest.approx(0.5)
    assert (
        metrics["probe_fail/collision_rate"] == 1.0
        and metrics["probe_fail/reward"] == 0.0
    )
    assert metrics["probe_control/collision_rate"] == 0.0
    assert metrics["probe_control/reward"] == pytest.approx(1.0, abs=1e-4)
    assert not any(k.endswith("/ep") for k in metrics)
    assert logger.calls == [(31, metrics)]
    assert module.model.training  # back in training mode
    # Same optimizer step (e.g. epoch end right after a periodic probe): not repeated.
    assert GRPOModule.run_probe(module) == {} and len(logger.calls) == 1
    module.global_step = 62
    assert GRPOModule.run_probe(module) and len(logger.calls) == 2


def test_state_list_with_groups(tmp_path):
    listing = tmp_path / "probe.tsv"
    listing.write_text(
        "# probe states\n/a/s1.npz\tfail\n/a/s2.npz\n\n/a/s3.npz\tcontrol\n"
    )
    ds = AlpaSimStateDataset(state_list=str(listing))
    assert ds.files == ["/a/s1.npz", "/a/s2.npz", "/a/s3.npz"]
    assert ds.groups == {"/a/s1.npz": "fail", "/a/s3.npz": "control"}
    with pytest.raises(ValueError):
        AlpaSimStateDataset(state_dirs="/a", state_list=str(listing))


def test_probe_logs_extra_rewards_on_the_same_plans():
    module, _ = _probe_module()
    module.reward_fn = SimPDMReward(mode="tiered")
    module.probe_extra_rewards = {"v9b": SimPDMReward()}
    metrics = GRPOModule.run_probe(module)
    assert (
        metrics["probe_fail/reward_v9b"] == 0.0 and metrics["probe_fail/reward"] < 0.0
    )
    assert metrics["probe_control/reward_v9b"] == pytest.approx(1.0, abs=1e-4)


def test_probe_and_viz_require_a_state_reward():
    from ad_vla.rewards.norm_l2_reward import NormL2Reward

    for kwargs in (
        {"probe_cfg": {"every_n_batches": 10}, "viz_cfg": None},
        {"probe_cfg": None, "viz_cfg": {}},
    ):
        module = SimpleNamespace(reward_fn=NormL2Reward(), probe_extra_rewards={})
        with pytest.raises(ValueError, match="score"):
            GRPOModule._check_state_reward(
                module, kwargs["probe_cfg"], kwargs["viz_cfg"]
            )
    module = SimpleNamespace(
        reward_fn=SimPDMReward(), probe_extra_rewards={"old": NormL2Reward()}
    )
    with pytest.raises(ValueError, match="extra_reward_fns.old"):
        GRPOModule._check_state_reward(module, {"every_n_batches": 10}, None)
    GRPOModule._check_state_reward(
        SimpleNamespace(reward_fn=SimPDMReward(), probe_extra_rewards={}), {}, {}
    )
