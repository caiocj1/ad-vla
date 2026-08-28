import torch
import torch.nn as nn
import math


class BEVCrossAttentionLayer(nn.Module):
    """Cross-attend learned BEV queries to image-token features.

    ``img_hidden_states`` is expected to have shape
    ``[batch, num_images, num_tokens, img_hidden_dim]``. The image and token
    dimensions are flattened into the key/value sequence, then learned BEV
    query embeddings attend to that sequence. Optional ``extra_hidden_states``
    can provide additional key/value tokens, such as projected calibration
    tokens, with the same final hidden dimension as the image features.

    The learned query grid uses the x/z dimensions from ``grid_size`` and
    returns only the forward half in z and center half in x, producing
    ``[batch, grid_z / 2, grid_x / 2, 1]``. When positional embeddings are
    enabled, ``bev_bounds`` gives the metric extent ``(x_min, x_max, y_min,
    y_max, z_min, z_max)`` used to normalize query positions for that cropped
    output region.
    """

    def __init__(
        self,
        grid_size: tuple[int, int, int] = (200, 8, 200),
        hidden_dim: int = 128,
        img_hidden_dim: int = 2048,
        add_pos_embedding: bool = False,
        add_residual: bool = False,
        add_ffn: bool = False,
        num_layers: int = 1,
        num_attention_heads: int = 1,
        ffn_hidden_dim: int | None = None,
        bev_bounds: tuple[float, float, float, float, float, float] = (
            -50,
            50,
            -5,
            5,
            -50,
            50,
        ),
    ):
        super().__init__()

        if num_layers < 1:
            raise ValueError("num_layers must be at least 1.")
        if num_attention_heads < 1:
            raise ValueError("num_attention_heads must be at least 1.")
        if hidden_dim % num_attention_heads != 0:
            raise ValueError(
                "hidden_dim must be divisible by num_attention_heads, "
                f"got hidden_dim={hidden_dim}, num_attention_heads={num_attention_heads}."
            )

        self.hidden_dim = hidden_dim
        self.img_hidden_dim = img_hidden_dim
        self.add_pos_embedding = add_pos_embedding
        self.add_residual = add_residual
        self.add_ffn = add_ffn
        self.num_layers = num_layers
        self.num_attention_heads = num_attention_heads
        self.head_dim = hidden_dim // num_attention_heads

        self.z, y, self.x = grid_size
        self.query = nn.Parameter(
            torch.randn(self.z // 2, self.x // 2, hidden_dim) * 0.02
        )

        if self.add_pos_embedding:
            self.register_buffer(
                "bev_pos_coords",
                self._build_bev_pos_coords(bev_bounds),
                persistent=False,
            )
            self.bev_pos_embed = nn.Sequential(
                nn.Linear(2, hidden_dim),
                nn.GELU(),
                nn.Linear(hidden_dim, hidden_dim),
            )

        self.to_q = nn.Linear(hidden_dim, hidden_dim)

        self.to_bev_proj = nn.Sequential(
            nn.Linear(img_hidden_dim, hidden_dim),
            nn.GELU(),
            nn.Linear(hidden_dim, hidden_dim),
        )
        self.to_bev_norm = nn.BatchNorm1d(img_hidden_dim)

        self.to_k = nn.Linear(hidden_dim, hidden_dim)
        self.to_v = nn.Linear(hidden_dim, hidden_dim)

        if self.add_ffn:
            self.ffn = self._build_ffn(hidden_dim, ffn_hidden_dim)

        self.layers = nn.ModuleList()
        for _ in range(num_layers - 1):
            layer = nn.ModuleDict(
                {
                    "to_q": nn.Linear(hidden_dim, hidden_dim),
                    "to_k": nn.Linear(hidden_dim, hidden_dim),
                    "to_v": nn.Linear(hidden_dim, hidden_dim),
                }
            )
            if self.add_ffn:
                layer["ffn"] = self._build_ffn(hidden_dim, ffn_hidden_dim)
            self.layers.append(layer)

        self.to_out = nn.Linear(hidden_dim, 1)

    def _build_bev_pos_coords(
        self,
        bev_bounds: tuple[float, float, float, float, float, float],
    ) -> torch.Tensor:
        x_min, x_max, _, _, z_min, z_max = bev_bounds
        z_step = (z_max - z_min) / self.z
        x_step = (x_max - x_min) / self.x

        z_start = self.z // 2
        z_end = self.z
        x_start = (self.x - self.x // 2) // 2
        x_end = x_start + self.x // 2

        z_indices = torch.arange(z_start, z_end, dtype=torch.float32)
        x_indices = torch.arange(x_start, x_end, dtype=torch.float32)
        z_meters = z_min + (z_indices + 0.5) * z_step
        x_meters = x_min + (x_indices + 0.5) * x_step

        z_edge_min = z_min + z_start * z_step
        z_edge_max = z_min + z_end * z_step
        x_edge_min = x_min + x_start * x_step
        x_edge_max = x_min + x_end * x_step

        z_pos = 2 * (z_meters - z_edge_min) / (z_edge_max - z_edge_min) - 1
        x_pos = 2 * (x_meters - x_edge_min) / (x_edge_max - x_edge_min) - 1
        z_pos, x_pos = torch.meshgrid(z_pos, x_pos, indexing="ij")
        return torch.stack([z_pos, x_pos], dim=-1).reshape(1, -1, 2)

    def _build_ffn(
        self,
        hidden_dim: int,
        ffn_hidden_dim: int | None,
    ) -> nn.Sequential:
        ffn_hidden_dim = ffn_hidden_dim or 4 * hidden_dim
        return nn.Sequential(
            nn.LayerNorm(hidden_dim),
            nn.Linear(hidden_dim, ffn_hidden_dim),
            nn.GELU(),
            nn.Linear(ffn_hidden_dim, hidden_dim),
        )

    def _get_query_states(self, bs: int) -> torch.Tensor:
        q = self.query.reshape(1, -1, self.hidden_dim).expand(bs, -1, -1)
        if self.add_pos_embedding:
            pos = self.bev_pos_embed(
                self.bev_pos_coords.to(device=q.device, dtype=q.dtype)
            )
            q = q + pos
        return q

    def _cross_attention(
        self,
        q_states: torch.Tensor,
        kv: torch.Tensor,
        to_q: nn.Linear,
        to_k: nn.Linear,
        to_v: nn.Linear,
    ) -> torch.Tensor:
        q = to_q(q_states)
        k = to_k(kv)
        v = to_v(kv)

        if self.num_attention_heads == 1:
            scores = torch.matmul(q, k.transpose(-1, -2)) / math.sqrt(q.shape[-1])
            attn = torch.softmax(scores, dim=-1)
            return torch.matmul(attn, v)

        bs, num_queries, _ = q.shape
        num_kv = k.shape[1]

        q = q.reshape(bs, num_queries, self.num_attention_heads, self.head_dim)
        k = k.reshape(bs, num_kv, self.num_attention_heads, self.head_dim)
        v = v.reshape(bs, num_kv, self.num_attention_heads, self.head_dim)

        q = q.transpose(1, 2)
        k = k.transpose(1, 2)
        v = v.transpose(1, 2)

        scores = torch.matmul(q, k.transpose(-1, -2)) / math.sqrt(self.head_dim)
        attn = torch.softmax(scores, dim=-1)
        out = torch.matmul(attn, v)
        out = out.transpose(1, 2).reshape(bs, num_queries, self.hidden_dim)
        return out

    def _apply_attention_block(
        self,
        q_states: torch.Tensor,
        kv: torch.Tensor,
        to_q: nn.Linear,
        to_k: nn.Linear,
        to_v: nn.Linear,
        ffn: nn.Module | None = None,
    ) -> torch.Tensor:
        attn_out = self._cross_attention(q_states, kv, to_q, to_k, to_v)
        if self.add_residual:
            q_states = q_states + attn_out
        else:
            q_states = attn_out

        if ffn is not None:
            ffn_out = ffn(q_states)
            if self.add_residual:
                q_states = q_states + ffn_out
            else:
                q_states = ffn_out
        return q_states

    def forward(
        self,
        img_hidden_states: torch.Tensor,
        extra_hidden_states: torch.Tensor | None = None,
    ) -> torch.Tensor:
        bs, num_imgs, num_tokens, _ = img_hidden_states.shape

        kv = img_hidden_states.reshape(bs, -1, self.img_hidden_dim)
        kv = self.to_bev_norm(kv.permute(0, 2, 1)).permute(0, 2, 1)
        kv = self.to_bev_proj(kv)

        if extra_hidden_states is not None:
            extra_kv = extra_hidden_states.reshape(bs, -1, self.img_hidden_dim)
            extra_kv = self.to_bev_proj(extra_kv)
            kv = torch.cat([kv, extra_kv], dim=1)

        out = self._get_query_states(bs)
        out = self._apply_attention_block(
            out,
            kv,
            self.to_q,
            self.to_k,
            self.to_v,
            self.ffn if self.add_ffn else None,
        )
        for layer in self.layers:
            out = self._apply_attention_block(
                out,
                kv,
                layer["to_q"],
                layer["to_k"],
                layer["to_v"],
                layer["ffn"] if self.add_ffn else None,
            )

        out = self.to_out(out)
        out = out.reshape(bs, self.z // 2, self.x // 2, -1)
        return out
