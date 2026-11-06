"""Small flow-matching action expert conditioned on a VLM's per-layer prompt KV.

Each block pairs with one backbone full-attention layer: action-token queries attend
bidirectionally to that layer's cached prompt keys/values (already k-normed and rotated
by the backbone) together with the action tokens' own keys/values. Head geometry
(``head_dim``, KV heads, rotary tables) therefore follows the backbone, while the
residual width and MLP are independent. Time conditioning uses adaLN-Zero.

Shape notation:

* ``B``: batch size.
* ``N``: number of action tokens (waypoints).
* ``S``: padded prompt length.
* ``D``: expert hidden width.
"""

import math

import torch
import torch.nn as nn
import torch.nn.functional as F
from alpamayo_r1.models.action_in_proj import PerWaypointActionInProjV2


def _modulate(
    x: torch.Tensor, shift: torch.Tensor, scale: torch.Tensor
) -> torch.Tensor:
    """adaLN modulation: ``x * (1 + scale) + shift``."""
    return x * (1 + scale) + shift


def _rotate_half(x: torch.Tensor) -> torch.Tensor:
    """RoPE helper: ``[x1, x2] -> [-x2, x1]`` on the last dimension."""
    x1, x2 = x.chunk(2, dim=-1)
    return torch.cat((-x2, x1), dim=-1)


def apply_rotary(x: torch.Tensor, cos: torch.Tensor, sin: torch.Tensor) -> torch.Tensor:
    """Rotate the leading `cos.shape[-1]` dims of x [B, heads, L, head_dim], as the backbone does.

    Partial rotary (Qwen3.5 rotates 64 of 256 dims) leaves the remaining dims untouched.
    """
    cos, sin = cos.unsqueeze(1), sin.unsqueeze(1)
    rotary_dim = cos.shape[-1]
    x_rot, x_pass = x[..., :rotary_dim], x[..., rotary_dim:]
    x_rot = x_rot * cos + _rotate_half(x_rot) * sin
    return torch.cat((x_rot, x_pass), dim=-1)


class TimestepEmbedding(nn.Module):
    """Sinusoidal flow-matching time embedding followed by an MLP."""

    def __init__(self, freq_dim: int, hidden_size: int):
        super().__init__()
        self.freq_dim = freq_dim
        self.mlp = nn.Sequential(
            nn.Linear(freq_dim, hidden_size),
            nn.SiLU(),
            nn.Linear(hidden_size, hidden_size),
        )

    def forward(self, t: torch.Tensor) -> torch.Tensor:
        """Embed times ``t`` [B] in [0, 1] (scaled by 1000 before the sinusoids) -> [B, D]."""
        half = self.freq_dim // 2
        freqs = torch.exp(
            -math.log(10_000.0)
            * torch.arange(half, device=t.device, dtype=torch.float32)
            / half
        )
        args = 1000.0 * t.float()[:, None] * freqs[None]
        return self.mlp(torch.cat((args.cos(), args.sin()), dim=-1))


PAST_TRAJ_FEATURES = ("xy", "vel")
PAST_TRAJ_MODES = ("per_pose", "pooled")


class PastTrajEncoder(nn.Module):
    """Embeds the ego past trajectory as state tokens prepended to the action tokens.

    Per pose, features are the ego-frame position and/or its finite-difference velocity.
    ``per_pose`` gives one token per past pose (shared MLP + learned position embedding);
    ``pooled`` flattens the whole history into a single token. Rows whose past trajectory
    was dropped (prompt input dropout) or is unavailable are replaced by learned null tokens.
    With ``missing_token`` (per_pose only), poses older than the available history get a
    learned missing-pose embedding (plus their position embedding) instead of their features.
    """

    def __init__(
        self,
        num_poses: int,
        interval_length: float,
        features: tuple[str, ...],
        mode: str,
        normalization: float,
        hidden_size: int,
        missing_token: bool = False,
    ):
        super().__init__()
        if not features or any(f not in PAST_TRAJ_FEATURES for f in features):
            raise ValueError(
                f"past_traj_features must be a non-empty subset of {PAST_TRAJ_FEATURES}."
            )
        if mode not in PAST_TRAJ_MODES:
            raise ValueError(
                f"past_traj_mode must be one of {PAST_TRAJ_MODES}, got {mode!r}."
            )
        if "vel" in features and num_poses < 2:
            raise ValueError("Velocity features need at least two past poses.")
        if missing_token and mode != "per_pose":
            raise ValueError("past_traj_missing_token needs past_traj_mode=per_pose.")
        self.features = tuple(features)
        self.mode = mode
        self.interval_length = interval_length
        self.normalization = normalization
        self.num_tokens = num_poses if mode == "per_pose" else 1

        feat_dim = 2 * len(self.features)
        in_dim = feat_dim if mode == "per_pose" else feat_dim * num_poses
        self.mlp = nn.Sequential(
            nn.Linear(in_dim, hidden_size),
            nn.SiLU(),
            nn.Linear(hidden_size, hidden_size),
        )
        # Position (per_pose only) + a type embedding marking state vs. action tokens.
        self.pos_embed = nn.Parameter(torch.randn(self.num_tokens, hidden_size) * 0.02)
        self.null_tokens = nn.Parameter(
            torch.randn(self.num_tokens, hidden_size) * 0.02
        )
        self.missing_token = (
            nn.Parameter(torch.randn(hidden_size) * 0.02) if missing_token else None
        )

    def _features(
        self, past_traj: torch.Tensor, pose_valid: torch.Tensor | None = None
    ) -> torch.Tensor:
        """[B, P, 2] ego-frame XY (present last) -> normalized per-pose features [B, P, F]."""
        feats = []
        if "xy" in self.features:
            feats.append(past_traj)
        if "vel" in self.features:
            vel = torch.diff(past_traj, dim=1) / self.interval_length
            # The oldest pose has no predecessor; reuse the next step's velocity.
            backward = torch.cat((vel[:, :1], vel), dim=1)
            if pose_valid is not None:
                # Same for the oldest available pose, whose predecessor is padding.
                forward = torch.cat((vel, vel[:, -1:]), dim=1)
                prev_valid = torch.cat((pose_valid[:, :1], pose_valid[:, :-1]), dim=1)
                backward = torch.where(prev_valid[..., None], backward, forward)
            feats.append(backward)
        return torch.cat(feats, dim=-1) / self.normalization

    def forward(
        self,
        past_traj: torch.Tensor,
        valid: torch.Tensor,
        pose_valid: torch.Tensor | None = None,
    ) -> torch.Tensor:
        """
        Args:
            past_traj: Ego-frame past XY [B, P, 2], oldest first, present last.
            valid: Bool [B]; False rows get the learned null tokens.
            pose_valid: Bool [B, P]; False poses (older than the available history) get the
                missing-pose embedding. Used only with missing_token.

        Returns:
            State tokens [B, num_tokens, D].
        """
        if self.missing_token is None:
            pose_valid = None
        elif pose_valid is None:
            raise ValueError(
                "past_traj_missing_token needs per-pose validity (past_traj_pose_valid)."
            )
        feats = self._features(past_traj.float(), pose_valid)
        if self.mode == "pooled":
            feats = feats.flatten(1)[:, None]
        tokens = self.mlp(feats)
        if pose_valid is not None:
            tokens = torch.where(
                pose_valid[..., None], tokens, self.missing_token.expand_as(tokens)
            )
        tokens = tokens + self.pos_embed
        # torch.where keeps null_tokens in the graph even when every row is valid, so DDP
        # never sees it as an unused parameter.
        return torch.where(
            valid[:, None, None], tokens, self.null_tokens.expand_as(tokens)
        )


CAMERA_GEOMETRY_MODES = ("ray_kv",)


class CameraGeometryKV(nn.Module):
    """Adds camera geometry to the image tokens' prompt keys/values (camera_geometry="ray_kv").

    Per image token, the raw geometry (ego-frame viewing ray, camera centre, normalized image
    coordinates, frame index; see ad_vla/utils/camera_geometry.py) is normalized, expanded
    with Fourier features and embedded by a shared MLP. Per expert block, zero-initialized
    linears map the embedding to a key offset on the non-rotary dims (the rotary dims keep the
    backbone's positional encoding) and a value offset on all dims. Offsets are applied only at
    image tokens, once per prompt, so the denoising loop is unchanged; at initialization the
    keys/values, and therefore the expert's outputs, are exactly the backbone's.
    """

    def __init__(
        self,
        num_blocks: int,
        num_kv_heads: int,
        head_dim: int,
        rotary_dim: int,
        num_features: int,
        hidden_size: int = 256,
        fourier_bands: int = 4,
        translation_norm_m: float = 10.0,
    ):
        super().__init__()
        if not 0 <= rotary_dim < head_dim:
            raise ValueError(
                f"rotary_dim={rotary_dim} must be in [0, head_dim={head_dim})."
            )
        self.num_kv_heads = num_kv_heads
        self.head_dim = head_dim
        self.rotary_dim = rotary_dim
        self.num_features = num_features
        self.translation_norm_m = translation_norm_m
        # A plain attribute, not a buffer: from_pretrained re-creates non-persistent buffers it
        # does not load as uninitialized memory, which would corrupt warm starts.
        self.fourier_bands = fourier_bands
        in_dim = num_features * (1 + 2 * fourier_bands)
        self.mlp = nn.Sequential(
            nn.Linear(in_dim, hidden_size),
            nn.SiLU(),
            nn.Linear(hidden_size, hidden_size),
            nn.SiLU(),
        )
        self.k_out = nn.ModuleList(
            nn.Linear(hidden_size, num_kv_heads * (head_dim - rotary_dim))
            for _ in range(num_blocks)
        )
        self.v_out = nn.ModuleList(
            nn.Linear(hidden_size, num_kv_heads * head_dim) for _ in range(num_blocks)
        )
        for layer in (*self.k_out, *self.v_out):
            # FlowMatchingVLA._init_weights keeps these at zero when HF (re)initializes them,
            # e.g. as missing keys when warm-starting from a checkpoint without geometry.
            layer._zero_init = True
        self.init_zero()

    @torch.no_grad()
    def init_zero(self) -> None:
        """Zero the key / value offset layers, so the geometry starts as a no-op."""
        for layer in (*self.k_out, *self.v_out):
            nn.init.zeros_(layer.weight)
            nn.init.zeros_(layer.bias)

    def _features(self, geom: torch.Tensor) -> torch.Tensor:
        """Raw geometry [B, S, F] -> normalized features with Fourier terms [B, S, F * (1 + 2 bands)]."""
        geom = geom.float()
        # Camera centre (features 3:6) in units of translation_norm_m.
        scale = torch.ones(self.num_features, device=geom.device)
        scale[3:6] = 1.0 / self.translation_norm_m
        geom = geom * scale
        freqs = math.pi * 2.0 ** torch.arange(
            self.fourier_bands, device=geom.device, dtype=torch.float32
        )
        args = geom[..., None] * freqs
        return torch.cat((geom, args.sin().flatten(-2), args.cos().flatten(-2)), dim=-1)

    def forward(
        self,
        prompt_kv: list[tuple[torch.Tensor, torch.Tensor]],
        geom: torch.Tensor,
        mask: torch.Tensor,
    ) -> list[tuple[torch.Tensor, torch.Tensor]]:
        """
        Args:
            prompt_kv: Per-block backbone (keys, values), each [B, kv_heads, S, head_dim].
            geom: Raw per-token geometry [B, S, F]; ignored where mask is False.
            mask: Bool [B, S], True at image tokens with geometry.

        Returns:
            Per-block FP32 (keys, values) with the geometry offsets added at image tokens.
        """
        if len(prompt_kv) != len(self.k_out):
            raise ValueError(
                f"Expected KV for {len(self.k_out)} blocks, got {len(prompt_kv)}."
            )
        batch_size, seq_len = mask.shape
        h = self.mlp(self._features(geom))
        token_mask = mask[:, None, :, None].float()
        out = []
        for (k, v), k_out, v_out in zip(prompt_kv, self.k_out, self.v_out):
            dk = (
                k_out(h)
                .view(batch_size, seq_len, self.num_kv_heads, -1)
                .transpose(1, 2)
                * token_mask
            )
            dv = (
                v_out(h)
                .view(batch_size, seq_len, self.num_kv_heads, -1)
                .transpose(1, 2)
                * token_mask
            )
            k = k.float()
            k = torch.cat(
                (k[..., : self.rotary_dim], k[..., self.rotary_dim :] + dk), dim=-1
            )
            out.append((k, v.float() + dv))
        return out


def _shared_prompt_attention(
    q: torch.Tensor,
    k: torch.Tensor,
    v: torch.Tensor,
    prompt_k: torch.Tensor,
    prompt_v: torch.Tensor,
    attn_mask: torch.Tensor,
) -> torch.Tensor:
    """Joint attention when R = B / P contiguous samples share one prompt KV (RL groups).

    Same result as repeating ``prompt_k`` / ``prompt_v`` R times and attending per sample, without
    materializing the R copies (or the R concatenated KVs autograd keeps per block and step): the
    R samples' expert tokens of a prompt become one query sequence [P, heads, R * T, head_dim]
    over [prompt KV, all R samples' own KV], with a block-diagonal mask so each sample still only
    sees the prompt and its own tokens. Equal up to floating-point summation order.

    Args:
        q: Expert queries [B, heads, T, head_dim]; k, v: expert keys / values [B, kv_heads, T, head_dim].
        prompt_k, prompt_v: Shared prompt keys / values [P, kv_heads, S, head_dim].
        attn_mask: The per-sample mask [B, 1, 1, S + T] (identical within a group).

    Returns:
        Attention output [B, heads, T, head_dim].
    """
    batch_size, num_heads, seq_len, head_dim = q.shape
    num_prompts, prompt_len = prompt_k.shape[0], prompt_k.shape[2]
    if batch_size % num_prompts:
        raise ValueError(
            f"Batch {batch_size} is not a multiple of the {num_prompts} shared prompts."
        )
    repeats = batch_size // num_prompts

    def fold(x: torch.Tensor) -> torch.Tensor:  # [P * R, h, T, d] -> [P, h, R * T, d]
        h = x.shape[1]
        return (
            x.view(num_prompts, repeats, h, seq_len, -1)
            .transpose(1, 2)
            .reshape(num_prompts, h, repeats * seq_len, -1)
        )

    keys = torch.cat((prompt_k.to(k.dtype), fold(k)), dim=2)
    values = torch.cat((prompt_v.to(v.dtype), fold(v)), dim=2)
    prompt_mask = attn_mask[::repeats, :, :, :prompt_len]  # [P, 1, 1, S]
    own_valid = attn_mask[:, 0, 0, prompt_len:].reshape(num_prompts, repeats * seq_len)
    sample = torch.arange(repeats, device=q.device).repeat_interleave(seq_len)
    own_mask = (sample[:, None] == sample[None, :])[None] & own_valid[
        :, None, :
    ]  # [P, R * T, R * T]
    mask = torch.cat(
        (prompt_mask.expand(-1, -1, repeats * seq_len, -1), own_mask[:, None]), dim=-1
    )
    out = F.scaled_dot_product_attention(
        fold(q), keys, values, attn_mask=mask, enable_gqa=True
    )
    return (
        out.view(num_prompts, num_heads, repeats, seq_len, head_dim)
        .transpose(1, 2)
        .reshape(batch_size, num_heads, seq_len, head_dim)
    )


class KVPrefixExpertBlock(nn.Module):
    """adaLN-Zero transformer block whose attention reads backbone KV as a prefix.

    Action-token queries attend jointly (bidirectionally) to one backbone layer's prompt
    keys/values and to the action tokens' own keys/values. Head geometry therefore matches
    the backbone's attention; only the residual width and MLP are free.
    """

    def __init__(
        self,
        hidden_size: int,
        intermediate_size: int,
        num_heads: int,
        num_kv_heads: int,
        head_dim: int,
        eps: float,
    ):
        super().__init__()
        self.num_heads = num_heads
        self.num_kv_heads = num_kv_heads
        self.head_dim = head_dim

        self.norm_attn = nn.RMSNorm(hidden_size, eps=eps, elementwise_affine=False)
        self.q_proj = nn.Linear(hidden_size, num_heads * head_dim, bias=False)
        self.k_proj = nn.Linear(hidden_size, num_kv_heads * head_dim, bias=False)
        self.v_proj = nn.Linear(hidden_size, num_kv_heads * head_dim, bias=False)
        self.q_norm = nn.RMSNorm(head_dim, eps=eps)
        self.k_norm = nn.RMSNorm(head_dim, eps=eps)
        self.o_proj = nn.Linear(num_heads * head_dim, hidden_size, bias=False)

        self.norm_mlp = nn.RMSNorm(hidden_size, eps=eps, elementwise_affine=False)
        self.gate_proj = nn.Linear(hidden_size, intermediate_size, bias=False)
        self.up_proj = nn.Linear(hidden_size, intermediate_size, bias=False)
        self.down_proj = nn.Linear(intermediate_size, hidden_size, bias=False)

        self.ada = nn.Sequential(nn.SiLU(), nn.Linear(hidden_size, 6 * hidden_size))

    def forward(
        self,
        h: torch.Tensor,
        cond: torch.Tensor,
        prompt_k: torch.Tensor,
        prompt_v: torch.Tensor,
        attn_mask: torch.Tensor,
        cos: torch.Tensor,
        sin: torch.Tensor,
    ) -> torch.Tensor:
        """One block: adaLN self-attention of the expert tokens over [prompt KV, own KV], then
        an adaLN SwiGLU MLP, both gated (residual).

        Args:
            h: Expert tokens [B, T, D].
            cond: Time conditioning [B, D].
            prompt_k, prompt_v: This block's backbone keys / values [B, kv_heads, S, head_dim]
                (already normalized and rotated at the prompt positions). May instead hold one
                prompt per group of B / P contiguous samples ([P, kv_heads, S, head_dim], RL
                groups): see ``_shared_prompt_attention``.
            attn_mask: Boolean mask [B, 1, 1, S + T], True where attention is allowed.
            cos, sin: Rotary tables of the expert tokens [B, T, rotary_dim].
        """
        batch_size, seq_len, _ = h.shape
        shift_a, scale_a, gate_a, shift_m, scale_m, gate_m = self.ada(cond)[
            :, None
        ].chunk(6, dim=-1)

        x = _modulate(self.norm_attn(h), shift_a, scale_a)
        q = self.q_norm(
            self.q_proj(x).view(batch_size, seq_len, self.num_heads, self.head_dim)
        )
        k = self.k_norm(
            self.k_proj(x).view(batch_size, seq_len, self.num_kv_heads, self.head_dim)
        )
        v = self.v_proj(x).view(batch_size, seq_len, self.num_kv_heads, self.head_dim)
        q = apply_rotary(q.transpose(1, 2), cos, sin)
        k = apply_rotary(k.transpose(1, 2), cos, sin)
        v = v.transpose(1, 2)

        # Backbone keys already carry k_norm and RoPE at their own positions.
        if prompt_k.shape[0] == batch_size:
            k = torch.cat((prompt_k.to(k.dtype), k), dim=2)
            v = torch.cat((prompt_v.to(v.dtype), v), dim=2)
            out = F.scaled_dot_product_attention(
                q, k, v, attn_mask=attn_mask, enable_gqa=True
            )
        else:
            out = _shared_prompt_attention(q, k, v, prompt_k, prompt_v, attn_mask)
        out = out.transpose(1, 2).reshape(batch_size, seq_len, -1)
        h = h + gate_a * self.o_proj(out)

        x = _modulate(self.norm_mlp(h), shift_m, scale_m)
        h = h + gate_m * self.down_proj(F.silu(self.gate_proj(x)) * self.up_proj(x))
        return h


class KVPrefixActionExpert(nn.Module):
    """Small action expert: one KVPrefixExpertBlock per selected backbone full-attention layer.

    With ``past_traj_cfg``, state tokens from a PastTrajEncoder are prepended to the action
    tokens; they share the blocks, attend to the prompt KV and to each other, and are
    discarded at the output. With ``camera_geometry_cfg``, ``camera_geometry`` is a
    CameraGeometryKV that the caller applies to the prompt KV once per prompt.
    """

    def __init__(
        self,
        num_poses: int,
        action_dim: int,
        hidden_size: int,
        intermediate_size: int,
        num_layers: int,
        num_heads: int,
        num_kv_heads: int,
        head_dim: int,
        time_embed_dim: int,
        rms_norm_eps: float,
        past_traj_cfg: dict | None = None,
        camera_geometry_cfg: dict | None = None,
    ):
        """
        Args:
            num_poses: Predicted waypoints (action tokens).
            action_dim: Channels per waypoint (2).
            hidden_size, intermediate_size: Expert width and MLP width.
            num_layers: Blocks, one per backbone KV layer read.
            num_heads, num_kv_heads, head_dim: Attention shape; KV heads and head_dim must
                match the backbone's, whose keys / values are reused as a prefix.
            time_embed_dim: Sinusoidal time-embedding width.
            rms_norm_eps: RMSNorm epsilon.
            past_traj_cfg: PastTrajEncoder kwargs (without hidden_size); None: no state tokens.
            camera_geometry_cfg: CameraGeometryKV kwargs (without num_blocks / heads / head_dim);
                None: no camera geometry.
        """
        super().__init__()
        self.past_traj_encoder = (
            PastTrajEncoder(hidden_size=hidden_size, **past_traj_cfg)
            if past_traj_cfg
            else None
        )
        self.action_in_proj = PerWaypointActionInProjV2(
            in_dims=(num_poses, action_dim),
            out_dim=hidden_size,
        )
        self.time_embed = TimestepEmbedding(time_embed_dim, hidden_size)
        self.blocks = nn.ModuleList(
            KVPrefixExpertBlock(
                hidden_size=hidden_size,
                intermediate_size=intermediate_size,
                num_heads=num_heads,
                num_kv_heads=num_kv_heads,
                head_dim=head_dim,
                eps=rms_norm_eps,
            )
            for _ in range(num_layers)
        )
        self.norm_out = nn.RMSNorm(
            hidden_size, eps=rms_norm_eps, elementwise_affine=False
        )
        self.ada_out = nn.Sequential(nn.SiLU(), nn.Linear(hidden_size, 2 * hidden_size))
        self.action_out_proj = nn.Linear(hidden_size, action_dim)
        self.camera_geometry = None
        if camera_geometry_cfg:
            # Built last and on a forked RNG, so enabling it leaves the other modules' random
            # initialization unchanged (same seed => same weights as without geometry).
            with torch.random.fork_rng(devices=[]):
                self.camera_geometry = CameraGeometryKV(
                    num_blocks=num_layers,
                    num_kv_heads=num_kv_heads,
                    head_dim=head_dim,
                    **camera_geometry_cfg,
                )

    @torch.no_grad()
    def init_adaln_zero(self) -> None:
        """Zero the modulation and output layers so every block starts as the identity."""
        for block in self.blocks:
            nn.init.zeros_(block.ada[-1].weight)
            nn.init.zeros_(block.ada[-1].bias)
        nn.init.zeros_(self.ada_out[-1].weight)
        nn.init.zeros_(self.ada_out[-1].bias)
        nn.init.zeros_(self.action_out_proj.weight)
        nn.init.zeros_(self.action_out_proj.bias)

    @property
    def num_state_tokens(self) -> int:
        """Number of past-trajectory state tokens prepended to the action tokens."""
        return (
            0 if self.past_traj_encoder is None else self.past_traj_encoder.num_tokens
        )

    def forward(
        self,
        x: torch.Tensor,
        t: torch.Tensor,
        prompt_kv: list[tuple[torch.Tensor, torch.Tensor]],
        attn_mask: torch.Tensor,
        cos: torch.Tensor,
        sin: torch.Tensor,
        state_tokens: torch.Tensor | None = None,
    ) -> torch.Tensor:
        """
        Args:
            x: Noisy actions [B, num_poses, action_dim].
            t: Flow-matching time [B, 1, 1].
            prompt_kv: Per-block backbone (keys, values), each [B, kv_heads, S, head_dim].
            attn_mask: Boolean mask [B, 1, 1, S + T], True where attention is allowed,
                where T = num_state_tokens + num_poses.
            cos, sin: Rotary tables for the state then action tokens [B, T, rotary_dim].
            state_tokens: Output of past_traj_encoder [B, num_state_tokens, D]; computed once
                per prompt by the caller since it does not depend on x or t.
        """
        h = self.action_in_proj(x.float(), t.float())
        if self.past_traj_encoder is not None:
            if state_tokens is None:
                raise ValueError(
                    "This expert was built with past-trajectory tokens; pass state_tokens."
                )
            h = torch.cat((state_tokens.to(h.dtype), h), dim=1)
        cond = self.time_embed(t.reshape(t.shape[0], -1)[:, 0])
        for block, (prompt_k, prompt_v) in zip(self.blocks, prompt_kv):
            h = block(h, cond, prompt_k, prompt_v, attn_mask, cos, sin)
        h = h[:, self.num_state_tokens :]
        shift, scale = self.ada_out(cond)[:, None].chunk(2, dim=-1)
        return self.action_out_proj(_modulate(self.norm_out(h), shift, scale))
