"""FlowMatchingVLA multi-sample prediction and past-trajectory state tokens (CPU, no backbone)."""

import pytest
import torch
import torch.nn as nn

from ad_vla.dataset.data_types import TrajectorySampling
from ad_vla.models.diffusion.flow_matching import FlowMatching
from ad_vla.models.layers.kv_prefix_action_expert import (
    KVPrefixActionExpert,
    PastTrajEncoder,
)
from ad_vla.models.vlas.fm_vla import (
    EVAL_DEFAULTS,
    FlowMatchingVLA,
    FlowMatchingVLAConfig,
)

NUM_POSES = 5
PAST_POSES = 4
HIDDEN = 32
KV_HEADS, HEADS, HEAD_DIM = 2, 4, 8
PROMPT_LEN = 7


def _past_traj_cfg(mode: str = "per_pose") -> dict:
    return {
        "num_poses": PAST_POSES,
        "interval_length": 0.5,
        "features": ("xy", "vel"),
        "mode": mode,
        "normalization": 20.0,
    }


def _expert(past_traj_cfg: dict | None = None) -> KVPrefixActionExpert:
    return KVPrefixActionExpert(
        num_poses=NUM_POSES,
        action_dim=2,
        hidden_size=HIDDEN,
        intermediate_size=64,
        num_layers=2,
        num_heads=HEADS,
        num_kv_heads=KV_HEADS,
        head_dim=HEAD_DIM,
        time_embed_dim=16,
        rms_norm_eps=1e-6,
        past_traj_cfg=past_traj_cfg,
    )


def _ctx(batch_size: int, expert: KVPrefixActionExpert) -> dict:
    n_tokens = expert.num_state_tokens + NUM_POSES
    rotary = HEAD_DIM // 2
    gen = torch.Generator().manual_seed(0)
    kv_shape = (batch_size, KV_HEADS, PROMPT_LEN, HEAD_DIM)
    return {
        "prompt_kv": [
            (torch.randn(kv_shape, generator=gen), torch.randn(kv_shape, generator=gen))
            for _ in expert.blocks
        ],
        "attn_mask": torch.ones(
            batch_size, 1, 1, PROMPT_LEN + n_tokens, dtype=torch.bool
        ),
        "cos": torch.ones(batch_size, n_tokens, rotary),
        "sin": torch.zeros(batch_size, n_tokens, rotary),
    }


def _past_traj(batch_size: int) -> torch.Tensor:
    # Constant 10 m/s along +x, present at the origin.
    x = torch.arange(-PAST_POSES + 1, 1, dtype=torch.float32) * 5.0
    return (
        torch.stack((x, torch.zeros_like(x)), dim=-1).expand(batch_size, -1, -1).clone()
    )


@pytest.mark.parametrize("mode, num_tokens", [("per_pose", PAST_POSES), ("pooled", 1)])
def test_past_traj_encoder_shapes_and_null_tokens(mode: str, num_tokens: int):
    encoder = PastTrajEncoder(hidden_size=HIDDEN, **_past_traj_cfg(mode))
    valid = torch.tensor([True, False, True])
    tokens = encoder(_past_traj(3), valid)

    assert encoder.num_tokens == num_tokens
    assert tokens.shape == (3, num_tokens, HIDDEN)
    torch.testing.assert_close(tokens[1], encoder.null_tokens)
    assert not torch.allclose(tokens[0], encoder.null_tokens)

    # The null tokens stay in the graph when every row is valid (no unused DDP params).
    encoder(_past_traj(2), torch.ones(2, dtype=torch.bool)).sum().backward()
    assert encoder.null_tokens.grad is not None


def test_past_traj_encoder_velocity_features():
    encoder = PastTrajEncoder(hidden_size=HIDDEN, **_past_traj_cfg())
    feats = encoder._features(_past_traj(1))
    # [x, y, vx, vy] / normalization; 5 m per 0.5 s step = 10 m/s everywhere.
    torch.testing.assert_close(feats[0, :, 2], torch.full((PAST_POSES,), 10.0 / 20.0))
    torch.testing.assert_close(feats[0, :, 3], torch.zeros(PAST_POSES))


def test_expert_without_past_traj_keeps_original_parameters():
    assert not any("past_traj" in name for name, _ in _expert().named_parameters())


def test_expert_with_state_tokens_outputs_action_shape():
    expert = _expert(_past_traj_cfg())
    ctx = _ctx(2, expert)
    state = expert.past_traj_encoder(_past_traj(2), torch.ones(2, dtype=torch.bool))
    out = expert(
        torch.randn(2, NUM_POSES, 2), torch.rand(2, 1, 1), state_tokens=state, **ctx
    )
    assert out.shape == (2, NUM_POSES, 2)

    with pytest.raises(ValueError, match="state_tokens"):
        expert(torch.randn(2, NUM_POSES, 2), torch.rand(2, 1, 1), **ctx)


def test_flow_matching_zero_init_is_deterministic_and_generator_reproducible():
    fm = FlowMatching(input_dims=(NUM_POSES, 2), num_inference_steps=4)

    def step_fn(x, t):
        return -x + t

    zero_a = fm.sample(batch_size=3, step_fn=step_fn, device="cpu", init="zero")
    zero_b = fm.sample(batch_size=3, step_fn=step_fn, device="cpu", init="zero")
    torch.testing.assert_close(zero_a, zero_b)

    rand_a = fm.sample(3, step_fn, "cpu", generator=torch.Generator().manual_seed(0))
    rand_b = fm.sample(3, step_fn, "cpu", generator=torch.Generator().manual_seed(0))
    torch.testing.assert_close(rand_a, rand_b)

    with pytest.raises(ValueError, match="init"):
        fm.sample(3, step_fn, "cpu", init="ones")


def test_aggregate_samples_mean_and_medoid():
    trajs = torch.zeros(1, 3, NUM_POSES, 2)
    trajs[0, 0] = 1.0
    trajs[0, 1] = 0.2
    trajs[0, 2] = -0.5
    mean = FlowMatchingVLA._aggregate_samples(trajs, "mean")
    torch.testing.assert_close(mean[0], torch.full((NUM_POSES, 2), 0.7 / 3))
    medoid = FlowMatchingVLA._aggregate_samples(trajs, "medoid")
    torch.testing.assert_close(medoid[0], trajs[0, 1])


class _StubFMVLA(FlowMatchingVLA):
    """FlowMatchingVLA with the backbone replaced by a fixed random prompt context."""

    def __init__(self, diff_cfg: dict, past_traj_cfg: dict | None = None):
        nn.Module.__init__(self)
        self.config = FlowMatchingVLAConfig.__new__(FlowMatchingVLAConfig)
        self.config.diff_cfg = {**EVAL_DEFAULTS, **diff_cfg}
        self.config.action_stats = None
        self.pred_traj_sampling = TrajectorySampling(
            num_poses=NUM_POSES, interval_length=1.0
        )
        self.traj_normalization = 20.0
        self.action_representation = "trajs"
        self.action_expert = _expert(past_traj_cfg)
        self.diffusion = FlowMatching(
            input_dims=(NUM_POSES, 2),
            num_inference_steps=diff_cfg.get("num_inference_steps", 3),
        )
        self.encoded_batch_sizes = []

    @property
    def device(self):
        return torch.device("cpu")

    def _encode_prompt(self, model_inputs):
        batch_size = model_inputs["input_ids"].shape[0]
        self.encoded_batch_sizes.append(batch_size)
        ctx = _ctx(batch_size, self.action_expert)
        if self.action_expert.past_traj_encoder is not None:
            ctx["state_tokens"] = self.action_expert.past_traj_encoder(
                model_inputs["past_traj"], model_inputs["past_traj_valid"]
            )
        return ctx


def _inputs(batch_size: int) -> dict:
    return {
        "input_ids": torch.ones(batch_size, PROMPT_LEN, dtype=torch.long),
        "attention_mask": torch.ones(batch_size, PROMPT_LEN, dtype=torch.long),
        "past_traj": _past_traj(batch_size),
        "past_traj_valid": torch.ones(batch_size, dtype=torch.bool),
    }


@pytest.mark.parametrize("past_traj_cfg", [None, _past_traj_cfg()])
def test_predict_aggregates_eval_samples_per_output(past_traj_cfg):
    model = _StubFMVLA({"num_eval_samples": 4}, past_traj_cfg).eval()
    pred, extra = model.predict_from_processed_inputs(**_inputs(2), num_traj_samples=3)

    assert pred.shape == (2 * 3, NUM_POSES, 2)
    assert extra["single_sample_trajs"].shape == pred.shape
    # The backbone runs once per scene; samples only repeat the expert.
    assert model.encoded_batch_sizes == [2]


def test_predict_single_sample_has_no_extra_and_generate_cfg_overrides():
    model = _StubFMVLA({"num_eval_samples": 1}).eval()
    pred, extra = model.predict_from_processed_inputs(**_inputs(2))
    assert pred.shape == (2, NUM_POSES, 2)
    assert extra == {}

    a, _ = model.predict_from_processed_inputs(
        **_inputs(2), generate_cfg={"eval_init": "zero"}
    )
    b, _ = model.predict_from_processed_inputs(
        **_inputs(2), generate_cfg={"eval_init": "zero"}
    )
    torch.testing.assert_close(a, b)

    with pytest.raises(ValueError, match="Unknown FlowMatchingVLA generate_cfg"):
        model.predict_from_processed_inputs(
            **_inputs(2), generate_cfg={"temperature": 1.0}
        )
    with pytest.raises(ValueError, match="deterministic"):
        model.predict_from_processed_inputs(
            **_inputs(2), generate_cfg={"eval_init": "zero", "num_eval_samples": 2}
        )


def test_cuda_graph_setting_falls_back_to_eager_off_gpu():
    model = _StubFMVLA({"num_eval_samples": 2}).eval()
    torch.manual_seed(0)
    eager, _ = model.predict_from_processed_inputs(**_inputs(2))
    torch.manual_seed(0)
    graph, _ = model.predict_from_processed_inputs(
        **_inputs(2), generate_cfg={"cuda_graph": True}
    )
    torch.testing.assert_close(graph, eager, rtol=0, atol=0)
    with pytest.raises(ValueError, match="cuda_graph must be a bool"):
        model.predict_from_processed_inputs(
            **_inputs(2), generate_cfg={"cuda_graph": 1}
        )


def test_config_fills_eval_defaults_for_old_checkpoints():
    # Saved configs from before the eval options have only the training keys.
    config = FlowMatchingVLAConfig(
        vlm_name_or_path="unused",
        diff_cfg={"num_inference_steps": 10, "num_repeat_steps": 8},
    )
    assert config.diff_cfg["num_eval_samples"] == 1
    assert config.diff_cfg["num_repeat_steps"] == 8
    assert config.expert_cfg["past_traj_tokens"] is False


def test_past_traj_encoder_missing_token_replaces_padded_poses():
    encoder = PastTrajEncoder(
        hidden_size=HIDDEN, missing_token=True, **_past_traj_cfg()
    )
    past = _past_traj(2)
    past[1, :2] = past[1, 2]  # Row 1: two oldest poses are left padding.
    pose_valid = torch.tensor([[True] * PAST_POSES, [False, False, True, True]])
    tokens = encoder(past, torch.ones(2, dtype=torch.bool), pose_valid)

    for i in range(2):
        torch.testing.assert_close(
            tokens[1, i], encoder.missing_token + encoder.pos_embed[i]
        )
    assert not torch.allclose(
        tokens[0, 0], encoder.missing_token + encoder.pos_embed[0]
    )
    # Real poses do not depend on the padding values.
    past[1, :2] = 123.0
    torch.testing.assert_close(
        encoder(past, torch.ones(2, dtype=torch.bool), pose_valid)[1, 2:], tokens[1, 2:]
    )
    # Dropped rows still get the null tokens; the missing token stays in the graph.
    dropped = encoder(past, torch.tensor([True, False]), pose_valid)
    torch.testing.assert_close(dropped[1], encoder.null_tokens)
    dropped.sum().backward()
    assert encoder.missing_token.grad is not None

    with pytest.raises(ValueError, match="per-pose validity"):
        encoder(past, torch.ones(2, dtype=torch.bool))
    with pytest.raises(ValueError, match="per_pose"):
        PastTrajEncoder(
            hidden_size=HIDDEN, missing_token=True, **_past_traj_cfg("pooled")
        )


def test_past_traj_encoder_velocity_ignores_padding():
    encoder = PastTrajEncoder(
        hidden_size=HIDDEN, missing_token=True, **_past_traj_cfg()
    )
    past = _past_traj(1)
    past[0, 0] = past[0, 1]  # Oldest pose is padding (a copy of the oldest real one).
    pose_valid = torch.tensor([[False, True, True, True]])
    feats = encoder._features(past, pose_valid)
    # The oldest real pose takes the next step's velocity, not the zero step to its padding.
    torch.testing.assert_close(
        feats[0, 1:, 2], torch.full((PAST_POSES - 1,), 10.0 / 20.0)
    )


def test_config_without_prompt_grid_or_missing_token_keeps_legacy_behaviour():
    config = FlowMatchingVLAConfig(vlm_name_or_path="unused")
    assert config.prompt_past_traj_sampling is None
    assert config.expert_cfg["past_traj_missing_token"] is False
