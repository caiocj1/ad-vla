"""CUDA-graph inference for FlowMatchingVLA on Qwen3.5-style backbones.

At small batch sizes FM-VLA inference is dominated by CPU launch overhead: thousands of small
kernels in the backbone prefill (the Gated-DeltaNet layers especially) and in every Euler step
of the expert. This runner records the whole inference for one input shape (vision tower,
LLM prefill, expert context and the Euler loop) as a single CUDA graph and replays it.

Graphs need fixed shapes, but prompt lengths vary by a few dozen tokens with the trajectory
text. Prompts are therefore right-padded to one fixed length: backbone attention and DeltaNet
are causal, so tokens appended after the prompt leave the real tokens' KV bit-identical, and
the expert masks the padded keys. M-RoPE positions are computed on the CPU (the backbone's
own `get_rope_index`, which syncs the GPU), so nothing in the graph depends on data values.

One graph is kept per (batch size, samples per scene, steps, image grid). Inputs are copied
into the graph's static buffers before each replay. Unsupported inputs return None so the
caller falls back to the eager path.
"""

import logging
import math
from typing import Any

import torch
from transformers.vision_utils import (
    get_vision_attention_seqlens,
    get_vision_interpolation_indices_and_weights,
    get_vision_position_ids,
)

LOGGER = logging.getLogger(__name__)

# Right-padding granularity and headroom when the padded length is chosen from the data.
_LEN_ROUND = 64
_LEN_MARGIN = 64
_WARMUP_RUNS = 2


def _round_up(n: int, multiple: int) -> int:
    """Smallest multiple of ``multiple`` that is >= ``n``."""
    return int(math.ceil(n / multiple) * multiple)


class FMVLACudaGraphRunner:
    """Records and replays FlowMatchingVLA inference as CUDA graphs (see the module docstring).

    Args:
        model: The FlowMatchingVLA to run; the runner reads its submodules, it owns no weights.
        prompt_len: Fixed padded prompt length; None picks it from the first batch
            (rounded up, plus headroom). A longer prompt later re-records at a larger length.
    """

    def __init__(self, model, prompt_len: int | None = None):
        self.model = model
        hf = model._backbone()
        self.backbone = (
            hf.model
        )  # Qwen3_5Model-like: .visual, .language_model, .get_rope_index
        self.visual = self.backbone.visual
        self.lm = self.backbone.language_model
        self.expert = model.action_expert
        self.expert_dtype = next(self.expert.parameters()).dtype
        self.pad_id = model.processor.tokenizer.pad_token_id
        self.n_expert_tokens = (
            self.expert.num_state_tokens + model.pred_traj_sampling.num_poses
        )
        self.geometry = self.expert.camera_geometry
        self.prompt_len = prompt_len
        self._graphs: dict[tuple, dict[str, Any]] = {}
        self._failed: set[tuple] = set()
        self._vision_meta: dict[tuple, dict[str, Any]] = {}
        self._warned: set[str] = set()

    def _warn_once(self, key: str, message: str) -> None:
        """Log ``message`` the first time ``key`` is seen."""
        if key not in self._warned:
            self._warned.add(key)
            LOGGER.warning(message)

    # ------------------------------------------------------------------ CPU preparation
    def _prepare(self, model_inputs: dict[str, Any]) -> dict[str, torch.Tensor] | None:
        """Left-padded collator inputs -> right-padded CPU tensors and per-row positions."""
        required = (
            "input_ids",
            "attention_mask",
            "mm_token_type_ids",
            "pixel_values",
            "image_grid_thw",
        )
        missing = [k for k in required if k not in model_inputs]
        if missing:
            self._warn_once(
                "missing",
                f"CUDA-graph inference needs {missing}; using the eager path.",
            )
            return None
        ids = model_inputs["input_ids"].cpu()
        mask = model_inputs["attention_mask"].cpu().bool()
        mm = model_inputs["mm_token_type_ids"].cpu()
        if (mm == 2).any():
            self._warn_once(
                "video",
                "CUDA-graph inference does not support video tokens; using the eager path.",
            )
            return None
        if self.geometry is not None and "visual_geom" not in model_inputs:
            self._warn_once(
                "geometry",
                "CUDA-graph inference needs visual_geom; using the eager path.",
            )
            return None
        lengths = mask.sum(dim=1)
        longest = int(lengths.max())
        if self.prompt_len is None or longest > self.prompt_len:
            new_len = _round_up(longest + _LEN_MARGIN, _LEN_ROUND)
            if self.prompt_len is not None:
                LOGGER.warning(
                    "Prompt of %d tokens exceeds the CUDA-graph length %d; re-recording at %d.",
                    longest,
                    self.prompt_len,
                    new_len,
                )
            self.prompt_len = new_len
            self._graphs.clear()
            self._failed.clear()
            torch.cuda.empty_cache()

        batch_size, length = ids.shape[0], self.prompt_len
        ids_r = torch.full((batch_size, length), self.pad_id, dtype=ids.dtype)
        mm_r = torch.zeros((batch_size, length), dtype=mm.dtype)
        valid_r = torch.zeros((batch_size, length), dtype=torch.long)
        for b in range(batch_size):
            n = int(lengths[b])
            ids_r[b, :n] = ids[b, mask[b]]
            mm_r[b, :n] = mm[b, mask[b]]
            valid_r[b, :n] = 1
        geometry = {}
        if self.geometry is not None:
            # Right-padded like the tokens; the padding carries no geometry.
            geom = model_inputs["visual_geom"].cpu()
            geom_mask = model_inputs["visual_geom_mask"].cpu().bool()
            geometry["visual_geom"] = torch.zeros(
                (batch_size, length, geom.shape[-1]), dtype=geom.dtype
            )
            geometry["visual_geom_mask"] = torch.zeros(
                (batch_size, length), dtype=torch.bool
            )
            for b in range(batch_size):
                n = int(lengths[b])
                geometry["visual_geom"][b, :n] = geom[b, mask[b]]
                geometry["visual_geom_mask"][b, :n] = geom_mask[b, mask[b]]
        grid = model_inputs["image_grid_thw"].cpu()
        # Same call the backbone makes inside its forward; pad positions stay 0 (never attended).
        position_ids, rope_deltas = self.backbone.get_rope_index(
            ids_r, mm_r, image_grid_thw=grid, attention_mask=valid_r
        )
        return {
            "input_ids": ids_r,
            "position_ids": position_ids,
            # Flat (row-major) positions of the image tokens: the order the vision features fill.
            "image_index": (mm_r.reshape(-1) == 1).nonzero()[:, 0],
            "lengths": lengths.view(-1, 1),
            "next_pos": lengths.view(-1, 1) + rope_deltas.view(-1, 1).to(lengths.dtype),
            "pixel_values": model_inputs["pixel_values"],
            "grid": grid,
            "past_traj": model_inputs.get("past_traj"),
            "past_traj_valid": model_inputs.get("past_traj_valid"),
            "past_traj_pose_valid": model_inputs.get("past_traj_pose_valid"),
            **geometry,
        }

    def _vision_kwargs(self, grid: torch.Tensor) -> dict[str, Any]:
        """Grid-dependent vision metadata, computed once per camera grid (it syncs the GPU)."""
        key = tuple(map(tuple, grid.tolist()))
        if key not in self._vision_meta:
            visual = self.visual
            grid_gpu = grid.to(self.model.device)
            meta = {"grid_thw": grid_gpu}
            meta["interp_indices"], meta["interp_weights"] = (
                get_vision_interpolation_indices_and_weights(
                    grid_gpu,
                    num_grid_per_side=visual.num_grid_per_side,
                    mode=visual.interpolation_mode,
                    align_corners=visual.interpolation_align_corners,
                    spatial_merge_size=visual.config.spatial_merge_size,
                )
            )
            meta["position_ids"] = get_vision_position_ids(
                grid_gpu, visual.spatial_merge_size
            )
            meta["cu_seqlens"], meta["max_seqlen"] = get_vision_attention_seqlens(
                grid_gpu, visual.config
            )
            self._vision_meta[key] = meta
        return self._vision_meta[key]

    # ------------------------------------------------------------------ graph
    def _build(
        self, prep: dict[str, Any], repeats: int, num_steps: int
    ) -> dict[str, Any]:
        """Allocate static buffers for this input shape, warm up and record the graph.

        Returns the graph entry: static input / output buffers and the recorded graph.
        """
        device = self.model.device
        batch_size, length = prep["input_ids"].shape
        static = {
            "input_ids": prep["input_ids"].to(device),
            "position_ids": prep["position_ids"].to(device),
            "image_index": prep["image_index"].to(device),
            "lengths": prep["lengths"].to(device),
            "next_pos": prep["next_pos"].to(device),
            "pixel_values": prep["pixel_values"].to(device).clone(),
            "x0": torch.zeros(
                batch_size * repeats, *self.model.diffusion.input_dims, device=device
            ),
        }
        has_state = self.expert.past_traj_encoder is not None
        if has_state:
            static["past_traj"] = prep["past_traj"].to(device).clone()
            static["past_traj_valid"] = prep["past_traj_valid"].to(device).clone()
            pose_valid = prep["past_traj_pose_valid"]
            if pose_valid is None:
                pose_valid = torch.ones(prep["past_traj"].shape[:2], dtype=torch.bool)
            static["past_traj_pose_valid"] = pose_valid.to(device).clone()
        if self.geometry is not None:
            static["visual_geom"] = prep["visual_geom"].to(device).clone()
            static["visual_geom_mask"] = prep["visual_geom_mask"].to(device).clone()
        vision_kwargs = self._vision_kwargs(prep["grid"])
        n_tok = self.n_expert_tokens
        key_index = torch.arange(length + n_tok, device=device)
        token_offsets = torch.arange(n_tok, device=device)

        def run():
            vision = self.visual(
                static["pixel_values"].type(self.visual.dtype), **dict(vision_kwargs)
            )
            embeds = self.lm.embed_tokens(static["input_ids"])
            flat = embeds.view(-1, embeds.shape[-1])
            flat[static["image_index"]] = vision.pooler_output.to(flat.dtype)
            # No attention mask: the padding is at the end and the backbone is causal.
            out = self.lm(
                inputs_embeds=embeds,
                position_ids=static["position_ids"],
                attention_mask=None,
                use_cache=True,
            )
            cache = out.past_key_values
            # The expert computes in its own dtype; cast the read-only prompt KV once, not per step.
            prompt_kv = [
                (
                    cache.layers[i].keys.to(self.expert_dtype),
                    cache.layers[i].values.to(self.expert_dtype),
                )
                for i in self.model.kv_layers
            ]
            if self.geometry is not None:
                prompt_kv = self.geometry(
                    prompt_kv, static["visual_geom"], static["visual_geom_mask"]
                )
            # Real prompt keys and the expert's own tokens are visible; the right padding is not.
            attn_mask = (key_index < static["lengths"]) | (key_index >= length)
            position_ids = (static["next_pos"] + token_offsets)[None].expand(3, -1, -1)
            cos, sin = self.lm.rotary_emb(
                torch.empty(0, device=device, dtype=torch.float32), position_ids
            )
            ctx = {
                "prompt_kv": prompt_kv,
                "attn_mask": attn_mask[:, None, None, :],
                "cos": cos,
                "sin": sin,
            }
            if has_state:
                ctx["state_tokens"] = self.expert.past_traj_encoder(
                    static["past_traj"],
                    static["past_traj_valid"].bool(),
                    static["past_traj_pose_valid"].bool(),
                )
            ctx = self.model._repeat_context(ctx, repeats)
            return self.model.diffusion._euler(
                x=static["x0"],
                step_fn=lambda x, t: self.model._predict_vector_field(x, t, ctx),
                num_inference_steps=num_steps,
            )

        stream = torch.cuda.Stream()
        stream.wait_stream(torch.cuda.current_stream())
        with torch.cuda.stream(stream):
            for _ in range(_WARMUP_RUNS):
                run()
        torch.cuda.current_stream().wait_stream(stream)
        graph = torch.cuda.CUDAGraph()
        # thread_local: dataloader pin-memory threads may touch CUDA during capture.
        with torch.cuda.graph(graph, capture_error_mode="thread_local"):
            output = run()
        torch.cuda.synchronize()
        # The graph replays raw addresses: keep everything it reads alive with it.
        return {
            "graph": graph,
            "static": static,
            "output": output,
            "run": run,
            "vision_kwargs": vision_kwargs,
        }

    # ------------------------------------------------------------------ public
    def sample(
        self,
        model_inputs: dict[str, Any],
        repeats: int,
        num_steps: int,
        init: str = "random",
    ) -> torch.Tensor | None:
        """Diffusion-space samples [B * repeats, T, D], or None to use the eager path."""
        if self.model.device.type != "cuda":
            return None
        with torch.inference_mode(False), torch.no_grad():
            prep = self._prepare(model_inputs)
            if prep is None:
                return None
            batch_size = prep["input_ids"].shape[0]
            key = (
                batch_size,
                repeats,
                num_steps,
                self.prompt_len,
                tuple(prep["pixel_values"].shape),
                tuple(map(tuple, prep["grid"].tolist())),
                int(prep["image_index"].numel()),
            )
            if key in self._failed:
                return None
            entry = self._graphs.get(key)
            if entry is None:
                try:
                    entry = self._build(prep, repeats, num_steps)
                except Exception as error:  # noqa: BLE001 - any capture failure falls back
                    self._failed.add(key)
                    self._warn_once(
                        f"capture-{key}",
                        f"CUDA-graph capture failed ({error!r}); using the eager path.",
                    )
                    torch.cuda.synchronize()
                    return None
                self._graphs[key] = entry
            static = entry["static"]
            for name in (
                "input_ids",
                "position_ids",
                "image_index",
                "lengths",
                "next_pos",
                "pixel_values",
            ):
                static[name].copy_(prep[name], non_blocking=True)
            if "past_traj" in static:
                static["past_traj"].copy_(prep["past_traj"], non_blocking=True)
                static["past_traj_valid"].copy_(
                    prep["past_traj_valid"], non_blocking=True
                )
                if prep["past_traj_pose_valid"] is not None:
                    static["past_traj_pose_valid"].copy_(
                        prep["past_traj_pose_valid"], non_blocking=True
                    )
            if "visual_geom" in static:
                static["visual_geom"].copy_(prep["visual_geom"], non_blocking=True)
                static["visual_geom_mask"].copy_(
                    prep["visual_geom_mask"], non_blocking=True
                )
            # Same RNG draw as the eager sampler, so both paths start from the same noise.
            static["x0"].copy_(
                self.model.diffusion.initial_noise(
                    batch_size * repeats, self.model.device, init=init
                )
            )
            entry["graph"].replay()
            return entry["output"].clone()
