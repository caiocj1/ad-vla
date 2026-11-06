"""Flow-GRPO pieces: marginal-preserving SDE sampler, step log-probs and the FM-VLA RL hooks (CPU)."""

import math

import pytest
import torch
import torch.nn as nn

from ad_vla.models.diffusion.flow_matching import FlowMatching
from tests.test_fm_vla_inference import (
    HEAD_DIM,
    HEADS,
    KV_HEADS,
    NUM_POSES,
    PROMPT_LEN,
    _StubFMVLA,
    _ctx,
    _inputs,
    _past_traj_cfg,
)

DIMS = (NUM_POSES, 2)


def _gaussian_field(mean: float, std: float):
    """Exact flow-matching velocity E[x - eps | x_t] for data x ~ N(mean, std^2 I)."""

    def step_fn(x, t):
        var_t = t**2 * std**2 + (1 - t) ** 2
        centred = x - t * mean
        x_hat = mean + t * std**2 / var_t * centred
        eps_hat = (1 - t) / var_t * centred
        return x_hat - eps_hat

    return step_fn


def test_sde_step_mean_without_noise_is_euler():
    x, v = torch.randn(3, *DIMS), torch.randn(3, *DIMS)
    torch.testing.assert_close(
        FlowMatching.sde_step_mean(x, v, 0.3, 0.1, 0.0), x + 0.1 * v
    )


def test_sde_schedule_first_step_uses_next_grid_point():
    timesteps, stds = FlowMatching.sde_schedule(4, 0.7, torch.device("cpu"))
    assert timesteps.tolist() == pytest.approx([0.0, 0.25, 0.5, 0.75, 1.0])
    expected = [0.7 * math.sqrt((1 - t) / t) * 0.5 for t in (0.25, 0.25, 0.5, 0.75)]
    assert stds.tolist() == pytest.approx(expected, rel=1e-6)


def test_sde_keeps_the_ode_marginal_on_gaussian_data():
    # Data N(1.5, 0.5^2): the SDE sample must have the data distribution, as the ODE's does.
    fm = FlowMatching(input_dims=DIMS, num_inference_steps=100)
    torch.manual_seed(0)
    out = fm.sample_sde(
        4000,
        _gaussian_field(1.5, 0.5),
        torch.device("cpu"),
        num_steps=100,
        noise_level=0.7,
    )
    sample = out["chain"][:, -1]
    assert sample.mean().item() == pytest.approx(1.5, abs=0.02)
    assert sample.std().item() == pytest.approx(0.5, abs=0.02)
    # Without the drift correction (plain noise injection) the spread is visibly wrong.
    x = out["chain"][:, 0]
    step_fn = _gaussian_field(1.5, 0.5)
    for k in range(100):
        t, std = float(out["timesteps"][k]), float(out["stds"][k])
        x = x + 0.01 * step_fn(x, t) + std * torch.randn_like(x)
    assert x.std().item() > 0.6
    assert out["chain"].shape == (4000, 101, *DIMS)
    assert out["log_probs"].shape == (4000, 100)


def test_sde_approaches_the_ode_as_noise_vanishes():
    fm = FlowMatching(input_dims=DIMS, num_inference_steps=8)
    step_fn = _gaussian_field(0.5, 1.2)
    ode = fm.sample(
        16, step_fn, torch.device("cpu"), generator=torch.Generator().manual_seed(3)
    )
    sde = fm.sample_sde(
        16,
        step_fn,
        torch.device("cpu"),
        num_steps=8,
        noise_level=1e-5,
        generator=torch.Generator().manual_seed(3),
    )
    # Same initial noise (first draw of the generator); the SDE perturbation is O(noise_level).
    torch.testing.assert_close(
        sde["chain"][:, 0],
        torch.randn(16, *DIMS, generator=torch.Generator().manual_seed(3)),
    )
    torch.testing.assert_close(sde["chain"][:, -1], ode, atol=1e-3, rtol=0)


def test_recomputed_log_probs_match_sampling_and_have_gradients():
    fm = FlowMatching(input_dims=DIMS, num_inference_steps=6)
    net = nn.Sequential(
        nn.Linear(2 * NUM_POSES + 1, 32), nn.Tanh(), nn.Linear(32, 2 * NUM_POSES)
    )

    def step_fn(x, t):
        flat = torch.cat([x.flatten(1), t.reshape(x.shape[0], -1)[:, :1]], dim=1)
        return net(flat).view_as(x)

    torch.manual_seed(1)
    out = fm.sample_sde(8, step_fn, torch.device("cpu"), num_steps=6, noise_level=0.7)
    means = fm.chain_step_means(out["chain"], out["timesteps"], out["stds"], step_fn)
    log_probs = torch.stack(
        [
            fm.sde_step_log_prob(
                out["chain"][:, k + 1], means[:, k], float(out["stds"][k])
            )
            for k in range(6)
        ],
        dim=1,
    )
    torch.testing.assert_close(log_probs, out["log_probs"], atol=1e-5, rtol=1e-5)
    log_probs.sum().backward()
    grads = torch.cat([p.grad.flatten() for p in net.parameters()])
    assert torch.isfinite(grads).all() and grads.abs().sum() > 0

    subset = fm.chain_step_means(
        out["chain"], out["timesteps"], out["stds"], step_fn, [1, 4]
    )
    torch.testing.assert_close(subset, means[:, [1, 4]].detach())


def test_policy_gradient_on_chain_log_probs_moves_samples_toward_reward():
    # Learnable offset on an exact Gaussian field; reward prefers large y at the last pose.
    fm = FlowMatching(input_dims=DIMS, num_inference_steps=6)
    base = _gaussian_field(0.0, 1.0)
    offset = nn.Parameter(torch.zeros(*DIMS))
    opt = torch.optim.Adam([offset], lr=0.05)

    def step_fn(x, t):
        return base(x, t) + offset

    def mean_reward(n=512):
        with torch.no_grad():
            return fm.sample(n, step_fn, torch.device("cpu"))[:, -1, 1].mean().item()

    torch.manual_seed(0)
    before = mean_reward()
    for _ in range(40):
        out = fm.sample_sde(
            64, step_fn, torch.device("cpu"), num_steps=6, noise_level=0.7
        )
        reward = out["chain"][:, -1, -1, 1]
        adv = (reward - reward.mean()) / (reward.std() + 1e-6)
        means = fm.chain_step_means(
            out["chain"], out["timesteps"], out["stds"], step_fn
        )
        log_probs = torch.stack(
            [
                fm.sde_step_log_prob(
                    out["chain"][:, k + 1], means[:, k], float(out["stds"][k])
                )
                for k in range(6)
            ],
            dim=1,
        )
        loss = -(torch.exp(log_probs - out["log_probs"]) * adv[:, None]).mean()
        opt.zero_grad()
        loss.backward()
        opt.step()
    assert mean_reward() > before + 0.5


class _RLStubFMVLA(_StubFMVLA):
    """Stub FM-VLA whose fixed random prompt context comes from ``_encode_backbone``."""

    def __init__(self, diff_cfg: dict, past_traj_cfg: dict | None = None):
        super().__init__(diff_cfg, past_traj_cfg)
        self.insulate_backbone = True

    def _encode_backbone(self, model_inputs):
        batch_size = model_inputs["input_ids"].shape[0]
        self.encoded_batch_sizes.append(batch_size)
        return _ctx(batch_size, self.action_expert)


@pytest.mark.parametrize("past_traj_cfg", [None, _past_traj_cfg()])
def test_fm_vla_sde_rollouts_and_policy_terms(past_traj_cfg):
    model = _RLStubFMVLA({"num_inference_steps": 4}, past_traj_cfg)
    reference = model.grpo_reference()
    assert all(not p.requires_grad for p in reference.parameters())
    generate_cfg = {"sampler": "sde", "sde_steps": 5, "noise_level": 0.7}

    model.eval()
    with torch.no_grad():
        pred, extra = model.predict_from_processed_inputs(
            **_inputs(2), num_traj_samples=3, generate_cfg=generate_cfg
        )
    assert pred.shape == (6, NUM_POSES, 2)
    assert model.encoded_batch_sizes == [
        2
    ]  # backbone once per scene, shared by the group
    rl = extra["rl"]
    assert rl["chain"].shape == (6, 6, NUM_POSES, 2) and rl["repeats"] == 3
    # Trajectories are the decoded last chain state ("trajs": x * traj_normalization).
    torch.testing.assert_close(pred, rl["chain"][:, -1] * model.traj_normalization)

    model.train()
    terms = model.grpo_policy_terms(extra, reference=reference)
    # Same weights as at sampling: ratio 1 and zero KL to an identical reference.
    torch.testing.assert_close(
        terms["log_probs"], terms["old_log_probs"], atol=1e-4, rtol=1e-4
    )
    torch.testing.assert_close(
        terms["kl"], torch.zeros_like(terms["kl"]), atol=1e-8, rtol=0
    )
    terms["log_probs"].mean().backward()
    grads = [p.grad for p in model.action_expert.parameters() if p.grad is not None]
    assert grads and all(torch.isfinite(g).all() for g in grads)
    assert sum(g.abs().sum() for g in grads) > 0

    with torch.no_grad():
        for p in model.action_expert.parameters():
            p.add_(0.01 * torch.randn_like(p))
    terms = model.grpo_policy_terms(extra, reference=reference, step_fraction=0.4)
    assert terms["log_probs"].shape == (6, 2)
    assert (terms["kl"] > 0).all()


def test_fm_vla_rejects_inconsistent_sde_settings():
    model = _RLStubFMVLA({"num_inference_steps": 4})
    with pytest.raises(ValueError, match="num_eval_samples"):
        model.predict_from_processed_inputs(
            **_inputs(1), generate_cfg={"sampler": "sde", "num_eval_samples": 2}
        )
    with pytest.raises(ValueError, match="sampler must be"):
        model.predict_from_processed_inputs(
            **_inputs(1), generate_cfg={"sampler": "ddim"}
        )
    model.insulate_backbone = False
    with pytest.raises(ValueError, match="insulate_backbone"):
        model.predict_from_processed_inputs(
            **_inputs(1), generate_cfg={"sampler": "sde"}
        )


def test_fm_vla_ode_path_unchanged_by_rl_keys():
    model = _RLStubFMVLA({"num_inference_steps": 4}).eval()
    torch.manual_seed(0)
    a, extra = model.predict_from_processed_inputs(**_inputs(2))
    torch.manual_seed(0)
    b, _ = model.predict_from_processed_inputs(
        **_inputs(2), generate_cfg={"sampler": "ode"}
    )
    torch.testing.assert_close(a, b)
    assert "rl" not in extra


@pytest.mark.parametrize("past_traj_cfg", [None, _past_traj_cfg()])
def test_shared_prompt_kv_matches_repeated_kv(past_traj_cfg):
    # 2 scenes x 3 samples; scene 1 masks its last 3 prompt tokens. Sharing one prompt KV per scene
    # (RL groups) must give the repeated-KV outputs and gradients.
    model = _RLStubFMVLA({"num_inference_steps": 4}, past_traj_cfg)
    expert = model.action_expert
    inputs = _inputs(2)
    base = _ctx(2, expert)
    base["attn_mask"][1, ..., PROMPT_LEN - 3 : PROMPT_LEN] = False
    torch.manual_seed(1)
    x, t = torch.randn(6, NUM_POSES, 2), torch.rand(6, 1, 1)
    outs = {}
    for share in (False, True):
        expert.zero_grad()
        ctx = model._expert_context(
            base, inputs
        )  # fresh graph per backward (state tokens carry grads)
        out = model._predict_vector_field(
            x, t, model._repeat_context(ctx, 3, share_prompt_kv=share)
        )
        out.square().sum().backward()
        outs[share] = (
            out.detach(),
            [p.grad.clone() for p in expert.parameters() if p.grad is not None],
        )
    torch.testing.assert_close(outs[True][0], outs[False][0], atol=1e-5, rtol=1e-5)
    for a, b in zip(outs[True][1], outs[False][1]):
        torch.testing.assert_close(a, b, atol=1e-5, rtol=1e-5)
    # The masked prompt tokens matter: unmasking them changes scene 1 only.
    base2 = {**base, "attn_mask": torch.ones_like(base["attn_mask"])}
    out2 = model._predict_vector_field(
        x,
        t,
        model._repeat_context(
            model._expert_context(base2, inputs), 3, share_prompt_kv=True
        ),
    ).detach()
    torch.testing.assert_close(out2[:3], outs[True][0][:3], atol=1e-5, rtol=1e-5)
    assert (out2[3:] - outs[True][0][3:]).abs().max() > 1e-4


def test_shared_prompt_kv_rejects_mismatched_groups():
    from ad_vla.models.layers.kv_prefix_action_expert import _shared_prompt_attention

    q = torch.randn(5, HEADS, 4, HEAD_DIM)
    kv = torch.randn(5, KV_HEADS, 4, HEAD_DIM)
    prompt = torch.randn(2, KV_HEADS, PROMPT_LEN, HEAD_DIM)
    with pytest.raises(ValueError, match="multiple"):
        _shared_prompt_attention(
            q,
            kv,
            kv,
            prompt,
            prompt,
            torch.ones(5, 1, 1, PROMPT_LEN + 4, dtype=torch.bool),
        )
