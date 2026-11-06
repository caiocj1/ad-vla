"""Diffusion-transformer action expert with camera-aware cross-attention.

Each block first performs bidirectional self-attention over the recurrent
``[state, action]`` sequence, then cross-attends to one fixed VLM memory, and
finally applies a SwiGLU feed-forward network. Even blocks read visual tokens
and odd blocks read language tokens, matching the Qwen-RobotManip action expert.

Visual cross-attention uses a compact Camera Positional Encoding (CaPE). The
first half of every attention head is transformed by camera-to-ego poses and
the second half receives temporal RoPE. With query pose ``P_q`` and key pose
``P_k``, the transformed dot product depends on ``P_q^-1 P_k``; consequently a
change to the arbitrary global coordinate origin cancels out.

Shape notation used throughout this module:

* ``B``: batch size.
* ``D``: expert hidden width.
* ``H``: number of attention heads.
* ``S``: number of recurrent state/action tokens.
* ``C``: number of fixed context tokens.

All pose tensors are homogeneous camera-to-ego transforms shaped ``[..., 4, 4]``.
Masks are boolean with ``True`` meaning that a token is valid and attendable.
"""

import math

import torch
import torch.nn as nn
import torch.nn.functional as F
from torch.utils.checkpoint import checkpoint


def _rotate_half(x: torch.Tensor) -> torch.Tensor:
    """Rotate the two feature halves for a rotary-position update.

    Args:
        x: Tensor with an even final dimension. Leading dimensions are arbitrary.

    Returns:
        Tensor shaped like ``x`` containing ``[-x_second, x_first]``. Combining
        this value with sine and cosine terms rotates each RoPE feature pair.
    """
    x1, x2 = x.chunk(2, dim=-1)
    return torch.cat((-x2, x1), dim=-1)


class RotaryEmbedding(nn.Module):
    """Parameter-free rotary position embedding with runtime position IDs.

    RoPE rotates query/key features by angles derived from token positions. Dot
    products then depend on relative position differences instead of a learned
    absolute-position table.

    Args:
        dim: Number of per-head channels to rotate. It must be positive and even.
        base: Frequency base controlling how quickly the rotary bands change.
            ``10_000`` is the standard transformer default.
    """

    def __init__(self, dim: int, base: float = 10_000.0):
        super().__init__()
        if dim <= 0 or dim % 2:
            raise ValueError(f"RoPE dimension must be positive and even, got {dim}.")
        inv_freq = 1.0 / (base ** (torch.arange(0, dim, 2, dtype=torch.float32) / dim))
        self.register_buffer("inv_freq", inv_freq, persistent=True)

    def forward(
        self,
        x: torch.Tensor,
        position_ids: torch.Tensor,
    ) -> torch.Tensor:
        """Apply rotary positions to an attention query or key tensor.

        Args:
            x: Attention features shaped ``[B, H, length, dim]``.
            position_ids: Integer or integer-like positions shaped
                ``[B, length]``. Negative positions are supported.

        Returns:
            Rotated features with the same shape and dtype as ``x``.
        """
        if position_ids.shape != (x.shape[0], x.shape[2]):
            raise ValueError(
                "position_ids must have shape [batch, length], got "
                f"{position_ids.shape} for attention tensor {x.shape}."
            )
        frequencies = position_ids.float().unsqueeze(-1) * self.inv_freq.float()
        angles = torch.cat((frequencies, frequencies), dim=-1).unsqueeze(1)
        return x * angles.cos().to(x.dtype) + _rotate_half(x) * angles.sin().to(x.dtype)


class SelfAttention(nn.Module):
    """Multi-head attention over the recurrent state-and-action sequence.

    Normalization of the residual stream happens in ``ActionExpertBlock``. This
    module performs Q/K/V projection, optional per-head Q/K normalization, RoPE,
    scaled dot-product attention, and output projection.

    Args:
        dim: Input and output width ``D``.
        num_heads: Number of attention heads ``H``. ``dim`` must divide evenly.
        attention_dropout: Dropout probability applied to attention weights.
        rope_base: Frequency base for state/action rotary positions.
        qk_norm: Apply RMSNorm independently to each query and key head.
        norm_eps: Numerical epsilon used by per-head RMSNorm.
    """

    def __init__(
        self,
        dim: int,
        num_heads: int,
        attention_dropout: float,
        rope_base: float,
        qk_norm: bool,
        norm_eps: float,
    ):
        super().__init__()
        if dim % num_heads:
            raise ValueError(f"dim={dim} must be divisible by num_heads={num_heads}.")
        self.num_heads = num_heads
        self.head_dim = dim // num_heads
        self.attention_dropout = attention_dropout

        self.q_proj = nn.Linear(dim, dim, bias=False)
        self.k_proj = nn.Linear(dim, dim, bias=False)
        self.v_proj = nn.Linear(dim, dim, bias=False)
        self.out_proj = nn.Linear(dim, dim, bias=False)
        self.q_norm = (
            nn.RMSNorm(self.head_dim, eps=norm_eps) if qk_norm else nn.Identity()
        )
        self.k_norm = (
            nn.RMSNorm(self.head_dim, eps=norm_eps) if qk_norm else nn.Identity()
        )
        self.rope = RotaryEmbedding(self.head_dim, base=rope_base)

    def _split_heads(self, x: torch.Tensor) -> torch.Tensor:
        """Convert ``[B, length, D]`` into ``[B, H, length, D/H]``."""
        batch, length, _ = x.shape
        return x.view(batch, length, self.num_heads, self.head_dim).transpose(1, 2)

    def forward(
        self,
        hidden_states: torch.Tensor,
        token_mask: torch.Tensor,
        position_ids: torch.Tensor,
        causal: bool,
    ) -> torch.Tensor:
        """Mix information among all valid state/action tokens.

        Args:
            hidden_states: Normalized recurrent sequence shaped ``[B, S, D]``.
                The state token(s) precede noisy continuous action tokens.
            token_mask: Valid-token mask shaped ``[B, S]``. Invalid positions
                cannot be used as keys or values.
            position_ids: RoPE positions shaped ``[B, S]``. The expert assigns
                negative positions to state tokens and non-negative positions to
                action tokens.
            causal: If ``True``, token ``i`` cannot attend to later tokens. Flow
                matching normally sets this to ``False`` so the entire noisy
                action trajectory is denoised jointly.

        Returns:
            Attention update shaped ``[B, S, D]``. The caller adds the residual.
        """
        q = self.rope(
            self.q_norm(self._split_heads(self.q_proj(hidden_states))), position_ids
        )
        k = self.rope(
            self.k_norm(self._split_heads(self.k_proj(hidden_states))), position_ids
        )
        v = self._split_heads(self.v_proj(hidden_states))

        attention_mask = token_mask[:, None, None, :]
        if causal:
            length = hidden_states.shape[1]
            causal_mask = torch.ones(
                length, length, dtype=torch.bool, device=hidden_states.device
            ).tril()
            attention_mask = attention_mask & causal_mask[None, None]

        output = F.scaled_dot_product_attention(
            q,
            k,
            v,
            attn_mask=attention_mask,
            dropout_p=self.attention_dropout if self.training else 0.0,
        )
        output = output.transpose(1, 2).reshape_as(hidden_states)
        return self.out_proj(output)


class CrossAttention(nn.Module):
    """Cross-attention, optionally using pose-aware CaPE and temporal RoPE.

    CaPE follows the relative-pose construction from EscherNet: query features
    are transformed by ``P_q^-T`` and key features by ``P_k``. Qwen-RobotManip
    additionally applies the pose transform to values and converts the attended
    result back through ``P_q^-1``; this implementation retains that extension.

    ``cape_dim=0`` selects ordinary cross-attention. When CaPE is active, the
    first ``cape_dim`` channels of every head carry geometry and the remaining
    channels carry temporal RoPE.

    Args:
        dim: Input and output width ``D`` for queries and context.
        num_heads: Number of attention heads ``H``.
        attention_dropout: Dropout probability applied to attention weights.
        rope_base: Frequency base for the temporal part of visual attention.
        qk_norm: Apply per-head RMSNorm to projected queries and keys.
        norm_eps: Numerical epsilon for Q/K RMSNorm.
        cape_dim: Number of channels per head transformed by camera poses. It
            must be divisible by four because each group is multiplied by a
            homogeneous 4x4 pose matrix. Zero disables CaPE.
        cape_translation_scale: Factor applied to the translation column of all
            poses before encoding. Rotations are not rescaled.
    """

    def __init__(
        self,
        dim: int,
        num_heads: int,
        attention_dropout: float,
        rope_base: float,
        qk_norm: bool,
        norm_eps: float,
        cape_dim: int,
        cape_translation_scale: float,
    ):
        super().__init__()
        if dim % num_heads:
            raise ValueError(f"dim={dim} must be divisible by num_heads={num_heads}.")
        self.num_heads = num_heads
        self.head_dim = dim // num_heads
        self.attention_dropout = attention_dropout
        self.cape_dim = cape_dim
        self.cape_translation_scale = cape_translation_scale

        if cape_dim:
            rope_dim = self.head_dim - cape_dim
            if cape_dim % 4:
                raise ValueError(
                    f"CaPE dimension must be divisible by four, got {cape_dim}."
                )
            if rope_dim <= 0 or rope_dim % 2:
                raise ValueError(
                    "The non-CaPE part of a visual attention head must be positive "
                    f"and even, got head_dim={self.head_dim}, cape_dim={cape_dim}."
                )
            self.temporal_rope = RotaryEmbedding(rope_dim, base=rope_base)
        else:
            self.temporal_rope = None

        self.q_proj = nn.Linear(dim, dim, bias=False)
        self.k_proj = nn.Linear(dim, dim, bias=False)
        self.v_proj = nn.Linear(dim, dim, bias=False)
        self.out_proj = nn.Linear(dim, dim, bias=False)
        self.q_norm = (
            nn.RMSNorm(self.head_dim, eps=norm_eps) if qk_norm else nn.Identity()
        )
        self.k_norm = (
            nn.RMSNorm(self.head_dim, eps=norm_eps) if qk_norm else nn.Identity()
        )

    def _split_heads(self, x: torch.Tensor) -> torch.Tensor:
        """Convert ``[B, length, D]`` into ``[B, H, length, D/H]``."""
        batch, length, _ = x.shape
        return x.view(batch, length, self.num_heads, self.head_dim).transpose(1, 2)

    @staticmethod
    def _validate_poses(
        poses: torch.Tensor,
        batch_size: int,
        length: int,
        name: str,
    ) -> None:
        """Check that one homogeneous transform is supplied per token.

        Args:
            poses: Candidate pose tensor.
            batch_size: Expected batch size ``B``.
            length: Expected query or context length.
            name: Argument name included in validation errors.
        """
        if poses.shape != (batch_size, length, 4, 4):
            raise ValueError(
                f"{name} must have shape {(batch_size, length, 4, 4)}, "
                f"got {poses.shape}."
            )

    def _scaled_poses(self, poses: torch.Tensor) -> torch.Tensor:
        """Copy poses to fp32 and rescale only their translation components.

        Args:
            poses: Camera-to-ego transforms shaped ``[B, length, 4, 4]``.

        Returns:
            An fp32 copy with ``[..., :3, 3]`` multiplied by the configured
            translation scale. The input tensor is never modified in place.
        """
        poses = poses.float().clone()
        poses[..., :3, 3] *= self.cape_translation_scale
        return poses

    def _apply_pose(
        self,
        x: torch.Tensor,
        matrices: torch.Tensor,
    ) -> torch.Tensor:
        """Apply one 4x4 matrix to every four-channel CaPE feature group.

        Args:
            x: Per-head features shaped ``[B, H, length, head_dim]``.
            matrices: Token-aligned transforms shaped ``[B, length, 4, 4]``.

        Returns:
            Tensor shaped like ``x``. Its first ``cape_dim`` channels have been
            transformed; all remaining channels are copied unchanged.
        """
        cape, remainder = x[..., : self.cape_dim], x[..., self.cape_dim :]
        groups = cape.reshape(*cape.shape[:-1], self.cape_dim // 4, 4)
        transformed = torch.einsum(
            "blij,bhlrj->bhlri", matrices.to(x.dtype), groups
        ).flatten(-2)
        return torch.cat((transformed, remainder), dim=-1)

    def forward(
        self,
        query_states: torch.Tensor,
        context_states: torch.Tensor,
        context_mask: torch.Tensor,
        *,
        query_time_ids: torch.Tensor | None = None,
        context_time_ids: torch.Tensor | None = None,
        query_poses: torch.Tensor | None = None,
        context_poses: torch.Tensor | None = None,
    ) -> torch.Tensor:
        """Let recurrent state/action queries read one fixed VLM memory.

        Args:
            query_states: Normalized state/action states shaped ``[B, S, D]``.
            context_states: Normalized visual or language memory shaped
                ``[B, C, D]``.
            context_mask: Valid context tokens shaped ``[B, C]``.
            query_time_ids: Temporal positions for state/action queries shaped
                ``[B, S]``. Required only when CaPE is active.
            context_time_ids: Temporal positions for visual tokens shaped
                ``[B, C]``. Required only when CaPE is active.
            query_poses: Reference-camera poses shaped ``[B, S, 4, 4]``. In the
                driving model every row is the latest FRONT camera pose.
            context_poses: Source-camera pose for each visual token, shaped
                ``[B, C, 4, 4]``.

        Returns:
            Cross-attention update shaped ``[B, S, D]``. The caller adds it to
            the recurrent sequence through a residual connection.

        Note:
            Language blocks construct this module with ``cape_dim=0`` and do not
            pass any time IDs or camera poses.
        """
        batch_size, query_length, _ = query_states.shape
        context_length = context_states.shape[1]

        q = self.q_norm(self._split_heads(self.q_proj(query_states)))
        k = self.k_norm(self._split_heads(self.k_proj(context_states)))
        v = self._split_heads(self.v_proj(context_states))

        if self.cape_dim:
            if query_poses is None or context_poses is None:
                raise ValueError(
                    "Visual cross-attention requires query and context poses."
                )
            if query_time_ids is None or context_time_ids is None:
                raise ValueError(
                    "Visual cross-attention requires temporal position IDs."
                )
            self._validate_poses(query_poses, batch_size, query_length, "query_poses")
            self._validate_poses(
                context_poses, batch_size, context_length, "context_poses"
            )

            query_poses = self._scaled_poses(query_poses)
            context_poses = self._scaled_poses(context_poses)
            query_inverse = torch.linalg.inv(query_poses)
            context_matrices = context_poses

            q = self._apply_pose(q, query_inverse.transpose(-1, -2))
            k = self._apply_pose(k, context_matrices)
            v = self._apply_pose(v, context_matrices)

            q_cape, q_temporal = q.split(
                (self.cape_dim, self.head_dim - self.cape_dim), dim=-1
            )
            k_cape, k_temporal = k.split(
                (self.cape_dim, self.head_dim - self.cape_dim), dim=-1
            )
            q = torch.cat(
                (q_cape, self.temporal_rope(q_temporal, query_time_ids)), dim=-1
            )
            k = torch.cat(
                (k_cape, self.temporal_rope(k_temporal, context_time_ids)), dim=-1
            )

        output = F.scaled_dot_product_attention(
            q,
            k,
            v,
            attn_mask=context_mask[:, None, None, :],
            dropout_p=self.attention_dropout if self.training else 0.0,
        )
        if self.cape_dim:
            # Values are in the shared ego frame; return them to each query frame.
            output = self._apply_pose(output, query_inverse)
        output = output.transpose(1, 2).reshape(batch_size, query_length, -1)
        return self.out_proj(output)


class SwiGLU(nn.Module):
    """SiLU-gated feed-forward network used after both attention sublayers.

    Args:
        dim: Residual-stream input and output width ``D``.
        hidden_dim: Intermediate width of the gate and value branches.
        dropout: Dropout probability applied after projecting back to ``D``.
    """

    def __init__(self, dim: int, hidden_dim: int, dropout: float):
        super().__init__()
        self.gate_proj = nn.Linear(dim, hidden_dim, bias=False)
        self.up_proj = nn.Linear(dim, hidden_dim, bias=False)
        self.down_proj = nn.Linear(hidden_dim, dim, bias=False)
        self.dropout = nn.Dropout(dropout)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        """Apply ``down(SiLU(gate(x)) * up(x))`` without adding a residual.

        Args:
            x: Hidden states with arbitrary leading dimensions and final width
                ``dim``.

        Returns:
            Feed-forward update with the same shape as ``x``.
        """
        return self.dropout(self.down_proj(F.silu(self.gate_proj(x)) * self.up_proj(x)))


class ActionExpertBlock(nn.Module):
    """One pre-norm DiT block with three residual branches.

    Processing order is always state/action self-attention, one modality-specific
    cross-attention, and SwiGLU. ``visual_cross_attention`` determines whether
    this block reads visual memory or language memory; ``ActionExpert`` alternates
    that flag by layer index.

    Args:
        dim: Recurrent and projected-context width ``D``.
        num_heads: Number of heads for self- and cross-attention.
        ffn_dim: SwiGLU intermediate width.
        dropout: Dropout on attention residuals and the SwiGLU output.
        attention_dropout: Dropout on attention probability matrices.
        rope_base: Frequency base used by rotary embeddings.
        qk_norm: Enable per-head query/key RMSNorm.
        norm_eps: Epsilon for every RMSNorm in the block.
        visual_cross_attention: If ``True``, the block reads visual tokens and
            may use CaPE. Otherwise it reads language tokens with ordinary
            cross-attention.
        cape_dim: Per-head geometry width for a visual block. Ignored by a
            language block.
        cape_translation_scale: Translation scaling used by visual CaPE.
    """

    def __init__(
        self,
        dim: int,
        num_heads: int,
        ffn_dim: int,
        dropout: float,
        attention_dropout: float,
        rope_base: float,
        qk_norm: bool,
        norm_eps: float,
        visual_cross_attention: bool,
        cape_dim: int,
        cape_translation_scale: float,
    ):
        super().__init__()
        self.visual_cross_attention = visual_cross_attention
        self.self_norm = nn.RMSNorm(dim, eps=norm_eps)
        self.self_attention = SelfAttention(
            dim,
            num_heads,
            attention_dropout,
            rope_base,
            qk_norm,
            norm_eps,
        )
        self.cross_norm = nn.RMSNorm(dim, eps=norm_eps)
        self.context_norm = nn.RMSNorm(dim, eps=norm_eps)
        self.cross_attention = CrossAttention(
            dim,
            num_heads,
            attention_dropout,
            rope_base,
            qk_norm,
            norm_eps,
            cape_dim=cape_dim if visual_cross_attention else 0,
            cape_translation_scale=cape_translation_scale,
        )
        self.ffn_norm = nn.RMSNorm(dim, eps=norm_eps)
        self.ffn = SwiGLU(dim, ffn_dim, dropout)
        self.residual_dropout = nn.Dropout(dropout)

    def forward(
        self,
        hidden_states: torch.Tensor,
        token_mask: torch.Tensor,
        self_position_ids: torch.Tensor,
        context_states: torch.Tensor,
        context_mask: torch.Tensor,
        causal: bool,
        *,
        query_time_ids: torch.Tensor | None = None,
        context_time_ids: torch.Tensor | None = None,
        query_poses: torch.Tensor | None = None,
        context_poses: torch.Tensor | None = None,
    ) -> torch.Tensor:
        """Run the three residual updates for one expert layer.

        Args:
            hidden_states: Recurrent ``[state, action]`` sequence ``[B, S, D]``.
            token_mask: Valid recurrent tokens shaped ``[B, S]``.
            self_position_ids: RoPE positions for recurrent tokens ``[B, S]``.
            context_states: The already projected visual or language memory,
                shaped ``[B, C, D]``.
            context_mask: Valid memory tokens shaped ``[B, C]``.
            causal: Whether the self-attention branch masks future tokens.
            query_time_ids: CaPE temporal positions for recurrent queries.
            context_time_ids: CaPE temporal positions for visual keys.
            query_poses: Reference-camera pose for each recurrent query.
            context_poses: Source-camera pose for each visual memory token.

        Returns:
            Updated recurrent sequence ``[B, S, D]``. Masked positions are
            explicitly zeroed so they cannot carry information between layers.
        """
        hidden_states = hidden_states + self.residual_dropout(
            self.self_attention(
                self.self_norm(hidden_states),
                token_mask,
                self_position_ids,
                causal,
            )
        )
        hidden_states = hidden_states + self.residual_dropout(
            self.cross_attention(
                self.cross_norm(hidden_states),
                self.context_norm(context_states),
                context_mask,
                query_time_ids=query_time_ids,
                context_time_ids=context_time_ids,
                query_poses=query_poses,
                context_poses=context_poses,
            )
        )
        hidden_states = hidden_states + self.ffn(self.ffn_norm(hidden_states))
        return hidden_states.masked_fill(~token_mask.unsqueeze(-1), 0.0)


class ActionExpert(nn.Module):
    """Qwen-RobotManip-style DiT expert simplified for 2D driving actions.

    The proprioceptive state tokens are prepended once and remain part of the
    recurrent expert sequence. Visual and language VLM states are immutable
    memories extracted from the backbone's final layer. Block zero cross-attends
    to vision, block one to language, and that order repeats.

    For the 768-wide, 12-head default, ``cape_dim=32`` assigns half of each
    64-dimensional visual attention head to camera pose and the other half to
    temporal RoPE, as in Qwen-RobotManip.

    Setting ``use_cape=False`` produces the geometry-free ablation: visual
    blocks use ordinary cross-attention, the learned intrinsics projection is
    not instantiated, and camera/time metadata is ignored. Self-attention RoPE
    over the state/action sequence is retained in both variants.

    Args:
        dim: State/action token width.
        depth: Number of transformer blocks. Even blocks read vision and odd
            blocks read language.
        num_heads: Number of heads in both attention sublayers.
        language_dim: Width of final-layer backbone language states.
        visual_dim: Width of final-layer backbone visual states.
        ffn_dim: SwiGLU intermediate width.
        dropout: Residual and feed-forward dropout probability.
        attention_dropout: Attention-probability dropout probability.
        rope_base: Rotary-embedding frequency base.
        qk_norm: Whether to RMS-normalize each query/key head.
        norm_eps: Epsilon for RMS normalization.
        causal: Whether state/action self-attention is causal. Diffusion models
            normally use the default bidirectional behavior.
        use_cape: Enable extrinsic CaPE, visual temporal RoPE, and intrinsics.
        cape_dim: Per-head channels assigned to homogeneous pose transforms.
        cape_translation_scale: Scale applied to pose translations before CaPE.
            Meter-valued driving extrinsics normally use ``1.0``.
    """

    def __init__(
        self,
        dim: int,
        depth: int,
        num_heads: int,
        language_dim: int | None = None,
        visual_dim: int | None = None,
        ffn_dim: int | None = None,
        dropout: float = 0.0,
        attention_dropout: float = 0.0,
        rope_base: float = 10_000.0,
        qk_norm: bool = True,
        norm_eps: float = 1e-6,
        causal: bool = False,
        use_cape: bool = True,
        cape_dim: int = 32,
        cape_translation_scale: float = 1.0,
    ):
        super().__init__()
        if dim <= 0 or depth <= 0 or num_heads <= 0:
            raise ValueError("dim, depth, and num_heads must all be positive.")
        if dim % num_heads:
            raise ValueError(f"dim={dim} must be divisible by num_heads={num_heads}.")
        if not 0.0 <= dropout < 1.0:
            raise ValueError(f"dropout must be in [0, 1), got {dropout}.")
        if not 0.0 <= attention_dropout < 1.0:
            raise ValueError(
                f"attention_dropout must be in [0, 1), got {attention_dropout}."
            )
        if cape_translation_scale <= 0.0:
            raise ValueError("cape_translation_scale must be positive.")

        language_dim = language_dim or dim
        visual_dim = visual_dim or dim
        ffn_dim = ffn_dim or 64 * math.ceil((8 * dim / 3) / 64)
        self.dim = dim
        self.depth = depth
        self.language_dim = language_dim
        self.visual_dim = visual_dim
        self.causal = causal
        self.use_cape = use_cape

        self.language_projection = self._projection(language_dim, dim)
        self.visual_projection = self._projection(visual_dim, dim)
        # The baseline has neither intrinsics computation nor an unused module.
        self.intrinsics_projection = nn.Linear(2, dim, bias=False) if use_cape else None
        self.layers = nn.ModuleList(
            [
                ActionExpertBlock(
                    dim=dim,
                    num_heads=num_heads,
                    ffn_dim=ffn_dim,
                    dropout=dropout,
                    attention_dropout=attention_dropout,
                    rope_base=rope_base,
                    qk_norm=qk_norm,
                    norm_eps=norm_eps,
                    visual_cross_attention=layer_index % 2 == 0,
                    cape_dim=cape_dim if use_cape else 0,
                    cape_translation_scale=cape_translation_scale,
                )
                for layer_index in range(depth)
            ]
        )
        self.final_norm = nn.RMSNorm(dim, eps=norm_eps)
        self.reset_parameters()

    @staticmethod
    def _projection(input_dim: int, output_dim: int) -> nn.Module:
        """Create a bias-free context adapter, or identity when widths match.

        Args:
            input_dim: Width produced by the VLM backbone.
            output_dim: Action-expert width ``D``.

        Returns:
            ``nn.Identity`` for matching widths, otherwise ``nn.Linear``.
        """
        if input_dim <= 0:
            raise ValueError(f"Context dimensions must be positive, got {input_dim}.")
        if input_dim == output_dim:
            return nn.Identity()
        return nn.Linear(input_dim, output_dim, bias=False)

    def reset_parameters(self) -> None:
        """Initialize linears and depth-scale all residual branch exits.

        Input matrices use a truncated normal with standard deviation ``0.02``.
        Self-attention, cross-attention, and feed-forward output matrices are
        additionally scaled by ``1/sqrt(3 * depth)`` because each block has three
        residual branches.
        """
        for module in self.modules():
            if isinstance(module, nn.Linear):
                nn.init.trunc_normal_(module.weight, std=0.02)
                if module.bias is not None:
                    nn.init.zeros_(module.bias)

        residual_std = 0.02 / math.sqrt(3 * self.depth)
        for layer in self.layers:
            for projection in (
                layer.self_attention.out_proj,
                layer.cross_attention.out_proj,
                layer.ffn.down_proj,
            ):
                nn.init.trunc_normal_(projection.weight, std=residual_std)

    @staticmethod
    def _mask(
        mask: torch.Tensor | None,
        batch_size: int,
        length: int,
        device: torch.device,
        name: str,
    ) -> torch.Tensor:
        """Validate a token mask or create an all-valid mask.

        Args:
            mask: Optional tensor shaped ``[B, length]``. Nonzero values become
                ``True``.
            batch_size: Expected batch size ``B``.
            length: Expected token count.
            device: Device on which the returned boolean mask must live.
            name: Argument name used in validation errors.

        Returns:
            Boolean mask shaped ``[B, length]``.
        """
        if mask is None:
            return torch.ones(batch_size, length, dtype=torch.bool, device=device)
        if mask.shape != (batch_size, length):
            raise ValueError(
                f"{name} must have shape {(batch_size, length)}, got {mask.shape}."
            )
        return mask.to(device=device, dtype=torch.bool)

    @staticmethod
    def _positions(
        position_ids: torch.Tensor | None,
        batch_size: int,
        length: int,
        device: torch.device,
        name: str,
        start: int = 0,
    ) -> torch.Tensor:
        """Validate, broadcast, or construct a batch of position IDs.

        Args:
            position_ids: Optional ``[length]`` or ``[B, length]`` positions.
            batch_size: Expected batch size ``B``.
            length: Number of positions required.
            device: Target device.
            name: Argument name used in validation errors.
            start: First position when constructing defaults.

        Returns:
            Long tensor shaped ``[B, length]``.
        """
        if position_ids is None:
            position_ids = torch.arange(start, start + length, device=device)
        else:
            position_ids = position_ids.to(device=device, dtype=torch.long)
        if position_ids.shape == (length,):
            return position_ids.unsqueeze(0).expand(batch_size, -1)
        if position_ids.shape != (batch_size, length):
            raise ValueError(
                f"{name} must have shape {(length,)} or {(batch_size, length)}, "
                f"got {position_ids.shape}."
            )
        return position_ids

    @staticmethod
    def _validate_context(
        context: torch.Tensor,
        batch_size: int,
        feature_dim: int,
        name: str,
    ) -> None:
        """Validate one final-layer VLM memory.

        Args:
            context: Language or visual states shaped ``[B, C, context_dim]``.
            batch_size: Expected batch size ``B``.
            feature_dim: Expected final width.
            name: Argument name used in validation errors.
        """
        if context.ndim != 3 or context.shape[0] != batch_size:
            raise ValueError(f"{name} must have shape [batch, length, dim].")
        if context.shape[-1] != feature_dim:
            raise ValueError(
                f"{name} feature dimension is {context.shape[-1]}, "
                f"expected {feature_dim}."
            )

    @staticmethod
    def _identity_poses(
        batch_size: int,
        length: int,
        device: torch.device,
    ) -> torch.Tensor:
        """Construct broadcasted identity camera poses.

        Args:
            batch_size: Number of batch elements ``B``.
            length: Number of token-aligned poses per element.
            device: Target device.

        Returns:
            Float32 identity transforms shaped ``[B, length, 4, 4]``.
        """
        return (
            torch.eye(4, dtype=torch.float32, device=device)
            .view(1, 1, 4, 4)
            .expand(batch_size, length, -1, -1)
        )

    def forward(
        self,
        action_tokens: torch.Tensor,
        state_tokens: torch.Tensor,
        language_states: torch.Tensor,
        visual_states: torch.Tensor,
        *,
        action_mask: torch.Tensor | None = None,
        state_mask: torch.Tensor | None = None,
        language_mask: torch.Tensor | None = None,
        visual_mask: torch.Tensor | None = None,
        action_position_ids: torch.Tensor | None = None,
        visual_time_ids: torch.Tensor | None = None,
        visual_camera_poses: torch.Tensor | None = None,
        reference_camera_pose: torch.Tensor | None = None,
        visual_patch_coords: torch.Tensor | None = None,
    ) -> torch.Tensor:
        """Refine actions with alternating visual/language grounding.

        Camera tensors are required only when ``use_cape=True``. They are
        deliberately ignored by the geometry-free variant, making the switch a
        controlled ablation of camera-aware positional information.

        This method does not care whether cameras were concatenated before Qwen
        preprocessing. The collator converts concatenated and separate-camera
        layouts into the same token-aligned visual metadata interface.

        Args:
            action_tokens: Embeddings of noisy continuous actions and the
                denoising timestep, shaped ``[B, A, D]``.
            state_tokens: Encoded proprioceptive state shaped ``[B, R, D]``. The
                driving model currently uses ``R=1`` and encodes the flattened
                resampled past XY trajectory in that token.
            language_states: Prompt-only final-layer VLM states shaped
                ``[B, L, language_dim]``. Assistant/discrete-action states must
                already have been removed.
            visual_states: Final-layer VLM image-token states shaped
                ``[B, V, visual_dim]``.
            action_mask: Optional valid-action mask ``[B, A]``.
            state_mask: Optional valid-state mask ``[B, R]``.
            language_mask: Optional valid-language mask ``[B, L]``.
            visual_mask: Optional valid-visual mask ``[B, V]``.
            action_position_ids: Optional action RoPE positions ``[A]`` or
                ``[B, A]``. Defaults to ``0, ..., A-1``.
            visual_time_ids: Image-frame time for every visual token ``[B, V]``.
                The newest frame is zero and earlier frames are negative.
            visual_camera_poses: Camera-to-ego source pose for each visual token,
                shaped ``[B, V, 4, 4]``. In a three-camera concatenated image,
                tokens still receive LEFT, FRONT, or RIGHT poses individually.
            reference_camera_pose: Camera-to-ego pose used by state/action
                queries. It may be ``[B, 4, 4]`` or token-specific
                ``[B, R+A, 4, 4]``; the driving collator always supplies the
                latest FRONT pose in the first form.
            visual_patch_coords: Intrinsics-normalized ``(x, y)`` ray coordinate
                for every visual token, shaped ``[B, V, 2]``.

        Returns:
            Final-normalized action states shaped ``[batch, actions, dim]``.
            Recurrent state-token outputs are intentionally discarded.
        """
        if action_tokens.ndim != 3 or action_tokens.shape[-1] != self.dim:
            raise ValueError(
                f"action_tokens must have shape [batch, length, {self.dim}]."
            )
        if state_tokens.ndim != 3 or state_tokens.shape[-1] != self.dim:
            raise ValueError(
                f"state_tokens must have shape [batch, length, {self.dim}]."
            )
        batch_size, action_length, _ = action_tokens.shape
        state_length = state_tokens.shape[1]
        if action_length == 0 or state_length == 0:
            raise ValueError("The expert requires at least one state and action token.")
        if state_tokens.shape[0] != batch_size:
            raise ValueError("State and action token batch sizes must match.")

        self._validate_context(
            language_states, batch_size, self.language_dim, "language_states"
        )
        self._validate_context(
            visual_states, batch_size, self.visual_dim, "visual_states"
        )
        language_length = language_states.shape[1]
        visual_length = visual_states.shape[1]
        if language_length == 0 or visual_length == 0:
            raise ValueError("Both language and visual memories must contain tokens.")

        device = action_tokens.device
        state_mask = self._mask(
            state_mask, batch_size, state_length, device, "state_mask"
        )
        action_mask = self._mask(
            action_mask, batch_size, action_length, device, "action_mask"
        )
        language_mask = self._mask(
            language_mask, batch_size, language_length, device, "language_mask"
        )
        visual_mask = self._mask(
            visual_mask, batch_size, visual_length, device, "visual_mask"
        )
        if not language_mask.any(dim=-1).all() or not visual_mask.any(dim=-1).all():
            raise ValueError(
                "Every batch item needs at least one language and visual token."
            )

        action_position_ids = self._positions(
            action_position_ids,
            batch_size,
            action_length,
            device,
            "action_position_ids",
        )
        state_position_ids = self._positions(
            None,
            batch_size,
            state_length,
            device,
            "state_position_ids",
            start=-state_length,
        )
        self_position_ids = torch.cat((state_position_ids, action_position_ids), dim=1)
        token_mask = torch.cat((state_mask, action_mask), dim=1)

        visual_states = self.visual_projection(visual_states)
        language_states = self.language_projection(language_states)

        if self.use_cape:
            # State is at the present; future action positions begin at one.
            visual_time_ids = self._positions(
                visual_time_ids,
                batch_size,
                visual_length,
                device,
                "visual_time_ids",
            )
            query_time_ids = torch.cat(
                (
                    torch.zeros(
                        batch_size, state_length, dtype=torch.long, device=device
                    ),
                    action_position_ids + 1,
                ),
                dim=1,
            )
            if visual_patch_coords is None:
                visual_patch_coords = visual_states.new_zeros(
                    batch_size, visual_length, 2
                )
            if visual_patch_coords.shape != (batch_size, visual_length, 2):
                raise ValueError(
                    "visual_patch_coords must have shape "
                    f"{(batch_size, visual_length, 2)}, got "
                    f"{visual_patch_coords.shape}."
                )
            # K^-1 patch centers provide per-token field-of-view awareness.
            visual_states = visual_states + self.intrinsics_projection(
                visual_patch_coords.to(visual_states.dtype)
            )

            if visual_camera_poses is None:
                visual_camera_poses = self._identity_poses(
                    batch_size, visual_length, device
                )
            else:
                visual_camera_poses = visual_camera_poses.to(device=device)
            if visual_camera_poses.shape != (batch_size, visual_length, 4, 4):
                raise ValueError(
                    "visual_camera_poses must have shape "
                    f"{(batch_size, visual_length, 4, 4)}, got "
                    f"{visual_camera_poses.shape}."
                )

            sequence_length = state_length + action_length
            if reference_camera_pose is None:
                reference_camera_pose = self._identity_poses(batch_size, 1, device)[
                    :, 0
                ]
            else:
                reference_camera_pose = reference_camera_pose.to(device=device)
            if reference_camera_pose.shape == (batch_size, 4, 4):
                # FRONT is shared by the state query and all future actions.
                query_poses = reference_camera_pose[:, None].expand(
                    -1, sequence_length, -1, -1
                )
            elif reference_camera_pose.shape == (
                batch_size,
                sequence_length,
                4,
                4,
            ):
                query_poses = reference_camera_pose
            else:
                raise ValueError(
                    "reference_camera_pose must have shape [batch, 4, 4] or "
                    f"[batch, {sequence_length}, 4, 4], got "
                    f"{reference_camera_pose.shape}."
                )
        else:
            query_time_ids = None
            query_poses = None

        hidden_states = torch.cat((state_tokens, action_tokens), dim=1)
        for layer in self.layers:
            if layer.visual_cross_attention and self.use_cape:
                context_states = visual_states
                context_mask = visual_mask
                cross_kwargs = {
                    "query_time_ids": query_time_ids,
                    "context_time_ids": visual_time_ids,
                    "query_poses": query_poses,
                    "context_poses": visual_camera_poses,
                }
            elif layer.visual_cross_attention:
                context_states = visual_states
                context_mask = visual_mask
                cross_kwargs = {}
            else:
                context_states = language_states
                context_mask = language_mask
                cross_kwargs = {}

            layer_args = (
                hidden_states,
                token_mask,
                self_position_ids,
                context_states,
                context_mask,
                self.causal,
            )
            if self.training and torch.is_grad_enabled():
                # Context K/V and SwiGLU activations dominate expert memory,
                # especially after flow-matching repeats. Non-reentrant
                # checkpointing recomputes them in backward without changing
                # the layer's forward result or parameter/state-dict layout.
                hidden_states = checkpoint(
                    layer,
                    *layer_args,
                    use_reentrant=False,
                    **cross_kwargs,
                )
            else:
                hidden_states = layer(*layer_args, **cross_kwargs)

        action_output = self.final_norm(hidden_states[:, state_length:])
        return action_output.masked_fill(~action_mask.unsqueeze(-1), 0.0)
