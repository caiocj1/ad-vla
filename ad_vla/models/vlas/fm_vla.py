"""FlowMatchingVLA: a VLM backbone plus a flow-matching action expert that plans trajectories.

The backbone (e.g. a trained TextTrajVLM's Qwen3.5) encodes the driving prompt (camera images,
past trajectory text, intent) once. A small transformer expert (KVPrefixActionExpert) attends
to the prompt's key/value cache of selected backbone layers, plus optional past-trajectory state
tokens and camera geometry, and predicts the flow-matching velocity of the future trajectory.
Training regresses that velocity (``FlowMatching``); inference integrates it with a few Euler
steps from noise (or from the noise mean with ``eval_init: zero``), optionally replayed as CUDA
graphs (fm_vla_cuda_graph.py). With ``insulate_backbone`` the backbone stays frozen.
"""

import copy
import random
from typing import Any

import torch
import torch.nn.functional as F
from hydra.utils import get_class
from transformers import AutoProcessor
from transformers import initialization as hf_init

from ad_vla.dataset.data_types import E2EDataBatch, TrajectorySampling, E2EDataSample
from ad_vla.models.base_vla import BaseVLA, BaseVLAConfig
from ad_vla.models.base_traj_planner import BaseTrajPlanner
from ad_vla.utils.prompts import PromptBuilder, ALL_INPUTS
from ad_vla.utils.traj_utils import resample_tensor
from ad_vla.utils.action_utils import (
    ACTION_REPRESENTATIONS,
    decode_actions,
    encode_actions,
)
from ad_vla.models.diffusion.flow_matching import FlowMatching
from ad_vla.models.layers.kv_prefix_action_expert import (
    CAMERA_GEOMETRY_MODES,
    KVPrefixActionExpert,
)
from ad_vla.utils.camera_geometry import NUM_GEOMETRY_FEATURES, token_geometry
from ad_vla.models.vlas.text_traj_vlm import pad_past_traj_text


SPECIAL_TOKENS_KEYS = ["traj_future_start", "traj_future_end"]
SPECIAL_TOKENS = {k: "<|" + k + "|>" for k in SPECIAL_TOKENS_KEYS}

DEFAULT_EXPERT_CFG = {
    "hidden_size": 1024,
    "intermediate_size": 4096,
    # Query heads; None uses the backbone's. Must be a multiple of the backbone's KV heads.
    "num_heads": None,
    # Backbone layers whose KV the expert reads, one expert block per entry. None selects the
    # backbone's full-attention layers, evenly subsampled down to at most `num_layers`.
    "kv_layers": None,
    "num_layers": 6,
    # Detach backbone KV so the diffusion loss does not update the VLM.
    "insulate_backbone": True,
    "time_embed_dim": 256,
    "rms_norm_eps": 1e-6,
    # Prepend ego past-trajectory state tokens to the action tokens (see PastTrajEncoder).
    # The history grid is past_traj_sampling (the prompt's is prompt_past_traj_sampling).
    "past_traj_tokens": False,
    "past_traj_features": ["xy", "vel"],  # subset of xy | vel, per past pose
    "past_traj_mode": "per_pose",  # per_pose: one token per pose | pooled: one token
    # Learned embedding for past poses older than the available history (per_pose only).
    # False keeps the legacy left-padding with the oldest pose.
    "past_traj_missing_token": False,
    # Not listed here, so configs and checkpoints without them stay unchanged:
    # "camera_geometry": None | "ray_kv". ray_kv gives the expert each image token's camera
    # geometry (ego-frame viewing ray, camera centre, normalized image coordinates, frame
    # index) as zero-initialized offsets to the image tokens' prompt keys/values (see
    # CameraGeometryKV); None (the default) is the geometry-free expert.
    # "camera_geometry_cfg": overrides of DEFAULT_CAMERA_GEOMETRY_CFG.
}

DEFAULT_CAMERA_GEOMETRY_CFG = {
    "hidden_size": 256,  # width of the shared geometry MLP
    "fourier_bands": 4,  # sin/cos at pi * 2^k, k < fourier_bands, of each normalized feature
    "translation_norm_m": 10.0,  # camera centres are divided by this
    # Raise when a camera has no calibration; False leaves its tokens without geometry.
    "require_calibration": True,
}

# Inference-only settings; read from diff_cfg, overridable per call through generate_cfg.
EVAL_DEFAULTS = {
    # Trajectories sampled per output and aggregated (in XY) into one.
    "num_eval_samples": 1,
    "eval_aggregation": "mean",  # mean | medoid (the sample closest to the mean)
    "eval_init": "random",  # random: N(0, I) start | zero: deterministic start at the noise mean
    # Replay inference as CUDA graphs (see fm_vla_cuda_graph.py). Same outputs as the eager
    # path; unsupported inputs fall back to it.
    "cuda_graph": False,
    # Fixed prompt length the graphs right-pad to. None picks it from the first batch
    # (+64 headroom); a longer prompt later re-records the graphs at a larger length.
    "cuda_graph_prompt_len": None,
    # "ode": deterministic Euler steps (evaluation / deployment). "sde": Flow-GRPO's
    # marginal-preserving SDE from random noise (RL rollouts); the denoising chain and its
    # per-step log-probs are returned in extra["rl"] (see grpo_policy_terms).
    "sampler": "ode",
    "noise_level": 0.7,  # SDE noise scale a in sigma_t = a * sqrt((1 - t) / t)
    "sde_steps": None,  # SDE denoising steps; None uses num_inference_steps
}
EVAL_AGGREGATIONS = ("mean", "medoid")
SAMPLERS = ("ode", "sde")

# Planner inputs the expert itself encodes (camera geometry, past-trajectory state tokens).
_EXPERT_CONTEXT_KEYS = {
    "past_traj",
    "past_traj_valid",
    "past_traj_pose_valid",
    "visual_geom",
    "visual_geom_mask",
}

# Keys the collator adds for the planner that the backbone forward must not receive.
_NON_VLM_KEYS = {
    "init_vel",
    "labels",
    "past_traj",
    "past_traj_valid",
    "past_traj_pose_valid",
    "visual_geom",
    "visual_geom_mask",
}


class FlowMatchingVLAConfig(BaseVLAConfig):
    """Configuration of FlowMatchingVLA (saved with the checkpoint).

    Args:
        past_traj_sampling: Past grid of the expert's state tokens (and of the prompt unless
            ``prompt_past_traj_sampling`` is set), as TrajectorySampling kwargs.
        pred_traj_sampling: Predicted future grid.
        prompt_past_traj_sampling: Past grid written in the prompt; None uses
            ``past_traj_sampling``.
        num_inference_steps: Legacy field; the sampler reads ``diff_cfg.num_inference_steps``.
        traj_normalization: XY scale of ``trajs`` targets (and of the past state tokens).
        diff_cfg: FlowMatching kwargs (``num_inference_steps``, ``num_repeat_steps``) plus the
            inference defaults of ``EVAL_DEFAULTS``.
        action_representation: ``trajs`` | ``acc_curv`` | ``polar`` (see action_utils.py).
        action_stats: Optional per-waypoint normalization of the diffusion target
            (scripts/utils/compute_action_stats.py); required for ``polar``.
        expert_cfg: Action expert settings, merged over ``DEFAULT_EXPERT_CFG``.
        backbone_ckpt: Optional trained wrapper checkpoint whose ``.vlm`` is the backbone.
        backbone_class_path: Class of ``backbone_ckpt``.
        chat_template_kwargs: Extra ``apply_chat_template`` kwargs.
        **kwargs: BaseVLAConfig fields (backbone, prompt, cameras, ...).
    """

    model_type = "fm_vla"

    def __init__(
        self,
        past_traj_sampling: dict | None = None,
        pred_traj_sampling: dict | None = None,
        prompt_past_traj_sampling: dict | None = None,
        num_inference_steps: int = 10,
        traj_normalization: float = 20.0,
        diff_cfg: dict | None = None,
        action_representation: str = "trajs",
        action_stats: dict | None = None,
        expert_cfg: dict | None = None,
        backbone_ckpt: str | None = None,
        backbone_class_path: str = "ad_vla.models.vlas.text_traj_vlm.TextTrajVLM",
        chat_template_kwargs: dict | None = None,
        **kwargs: Any,
    ):
        super().__init__(**kwargs)
        self.past_traj_sampling = past_traj_sampling or {
            "num_poses": 4,
            "interval_length": 0.5,
        }
        self.pred_traj_sampling = pred_traj_sampling or {
            "num_poses": 25,
            "interval_length": 0.2,
        }
        # Past grid shown in the prompt, e.g. the one a trained backbone_ckpt saw; None uses
        # past_traj_sampling (also the expert's state-token grid). With pad_missing_past_traj,
        # prompt poses older than the available history are written as "[X.X, X.X]".
        self.prompt_past_traj_sampling = prompt_past_traj_sampling
        self.num_inference_steps = num_inference_steps
        self.traj_normalization = traj_normalization
        self.diff_cfg = {
            **EVAL_DEFAULTS,
            **(diff_cfg or {"num_inference_steps": 10, "num_repeat_steps": 8}),
        }
        # trajs: XY waypoints; acc_curv: [acceleration, curvature]; polar: per-step
        # [delta_heading, step_distance] (see ad_vla/utils/action_utils.py).
        assert action_representation in ACTION_REPRESENTATIONS
        self.action_representation = action_representation
        # Per-(waypoint, channel) normalization of the diffusion target, from
        # scripts/utils/compute_action_stats.py: {"mean": [[..]] * num_poses, "std": [[..]] * num_poses, ...}.
        # None keeps the legacy scaling (trajs / traj_normalization; acc_curv unnormalized).
        # Required for polar.
        if action_stats is not None and not isinstance(action_stats, dict):
            from omegaconf import OmegaConf

            action_stats = OmegaConf.to_container(action_stats, resolve=True)
        self.action_stats = action_stats
        self.expert_cfg = {**DEFAULT_EXPERT_CFG, **(expert_cfg or {})}
        # Optional trained ad_vla wrapper (e.g. a TextTrajVLM hf_ckpt/) whose `.vlm` becomes
        # the backbone when building from pretrained submodules.
        self.backbone_ckpt = backbone_ckpt
        self.backbone_class_path = backbone_class_path
        # Extra kwargs for the processor's chat template, e.g. {"enable_thinking": False} for
        # backbones whose default generation prompt opens a reasoning block (Cosmos3-Edge).
        if chat_template_kwargs is not None and not isinstance(
            chat_template_kwargs, dict
        ):
            from omegaconf import OmegaConf

            chat_template_kwargs = OmegaConf.to_container(
                chat_template_kwargs, resolve=True
            )
        self.chat_template_kwargs = chat_template_kwargs or {}

    def _build_processor(self) -> AutoProcessor:
        """Build the processor with trajectory tokens."""
        processor = AutoProcessor.from_pretrained(self.vlm_name_or_path)
        tokenizer = processor.tokenizer

        special_tokens = list(SPECIAL_TOKENS.values())
        tokenizer.add_tokens(special_tokens, special_tokens=True)

        return processor


def initial_speed(sample: E2EDataSample) -> torch.Tensor:
    """Current ego speed from the last past-trajectory step (the pose before the origin).

    A single-pose history (e.g. the first nuScenes sample of a scene with require_full_past
    off) has no step: speed 0, as the AlpaSim driver uses at the start of a rollout.
    """
    past_traj = sample.agent_input["past_traj"]
    if past_traj.shape[0] < 2:
        return past_traj.new_zeros(())
    return (past_traj[-2] * sample.metadata["sampling_freq"]).norm()


def _select_kv_layers(text_config, max_layers: int) -> list[int]:
    """Full-attention layers of the backbone, evenly subsampled (keeping the last) to max_layers."""
    num_layers = text_config.num_hidden_layers
    layer_types = (
        getattr(text_config, "layer_types", None) or ["full_attention"] * num_layers
    )
    full = [
        i for i, layer_type in enumerate(layer_types) if layer_type == "full_attention"
    ]
    if not full:
        raise ValueError(
            "Backbone has no full-attention layers to condition the action expert on."
        )
    if len(full) <= max_layers:
        return full
    return [
        full[round((i + 1) * len(full) / max_layers) - 1] for i in range(max_layers)
    ]


class FlowMatchingVLA(BaseVLA, BaseTrajPlanner):
    """VLM backbone + KV-prefix flow-matching action expert (see the module docstring).

    ``compute_loss`` trains the expert with the flow-matching loss on ground-truth futures;
    ``predict_from_processed_inputs`` samples trajectories; ``get_collator`` builds the
    matching ``FlowMatchingVLACollator``.
    """

    config_class = FlowMatchingVLAConfig
    base_model_prefix = "vlm"
    # Small and numerically sensitive (Fourier features, adaLN); keep in FP32 even when the
    # checkpoint is loaded in bf16.
    _keep_in_fp32_modules_strict = ["action_expert"]

    def __init__(self, config, pretrained_modules=None):
        """Build the expert and sampler around the backbone.

        Args:
            config: FlowMatchingVLAConfig; ``expert_cfg.kv_layers`` is resolved and written
                back so the saved config rebuilds the same expert.
            pretrained_modules: Optional prebuilt submodules, e.g. ``{"vlm": backbone}``.
        """
        BaseVLA.__init__(self, config=config, pretrained_modules=pretrained_modules)
        BaseTrajPlanner.__init__(
            self,
            past_traj_sampling=config.past_traj_sampling,
            pred_traj_sampling=config.pred_traj_sampling,
        )
        prompt_past = getattr(config, "prompt_past_traj_sampling", None)
        self.prompt_past_traj_sampling = (
            TrajectorySampling(**prompt_past)
            if prompt_past
            else self.past_traj_sampling
        )

        text_config = self.vlm.config.get_text_config()
        expert_cfg = {**DEFAULT_EXPERT_CFG, **config.expert_cfg}
        if expert_cfg["kv_layers"] is None:
            expert_cfg["kv_layers"] = _select_kv_layers(
                text_config, expert_cfg["num_layers"]
            )
        self.camera_geometry_mode = expert_cfg.get("camera_geometry")
        if self.camera_geometry_mode is not None:
            if self.camera_geometry_mode not in CAMERA_GEOMETRY_MODES:
                raise ValueError(
                    f"expert_cfg.camera_geometry must be None or one of {CAMERA_GEOMETRY_MODES}, "
                    f"got {self.camera_geometry_mode!r}."
                )
            geometry_cfg = dict(expert_cfg.get("camera_geometry_cfg") or {})
            unknown = set(geometry_cfg) - set(DEFAULT_CAMERA_GEOMETRY_CFG)
            if unknown:
                raise ValueError(
                    f"Unknown expert_cfg.camera_geometry_cfg keys {sorted(unknown)}."
                )
            expert_cfg["camera_geometry_cfg"] = {
                **DEFAULT_CAMERA_GEOMETRY_CFG,
                **geometry_cfg,
            }
        # Persist the resolved layers so the saved config rebuilds the same expert.
        config.expert_cfg = expert_cfg
        self.kv_layers = list(expert_cfg["kv_layers"])
        self.insulate_backbone = expert_cfg["insulate_backbone"]
        if self.insulate_backbone:
            # The diffusion loss never reaches the backbone, so freeze it: no autograd graph through
            # the VLM and no unused DDP parameters. LoRA added later by the train stage still
            # creates trainable adapters (useful only with another loss, e.g. an aux head).
            self.vlm.requires_grad_(False)

        num_kv_heads = text_config.num_key_value_heads
        num_heads = expert_cfg["num_heads"] or text_config.num_attention_heads
        if num_heads % num_kv_heads != 0:
            raise ValueError(
                f"expert num_heads={num_heads} must be a multiple of backbone KV heads={num_kv_heads}."
            )
        head_dim = getattr(text_config, "head_dim", None) or (
            text_config.hidden_size // text_config.num_attention_heads
        )
        past_traj_cfg = None
        if expert_cfg["past_traj_tokens"]:
            past_traj_cfg = {
                "num_poses": self.past_traj_sampling.num_poses,
                "interval_length": self.past_traj_sampling.interval_length,
                "features": tuple(expert_cfg["past_traj_features"]),
                "mode": expert_cfg["past_traj_mode"],
                "normalization": config.traj_normalization,
                "missing_token": expert_cfg["past_traj_missing_token"],
            }
        camera_geometry_cfg = None
        if self.camera_geometry_mode is not None:
            geometry_cfg = expert_cfg["camera_geometry_cfg"]
            camera_geometry_cfg = {
                # Key offsets skip the backbone's rotary dims (Qwen3.5: 64 of 256).
                "rotary_dim": 2
                * self._backbone().model.language_model.rotary_emb.inv_freq.shape[-1],
                "num_features": NUM_GEOMETRY_FEATURES,
                "hidden_size": geometry_cfg["hidden_size"],
                "fourier_bands": geometry_cfg["fourier_bands"],
                "translation_norm_m": geometry_cfg["translation_norm_m"],
            }
        self.action_expert = KVPrefixActionExpert(
            num_poses=self.pred_traj_sampling.num_poses,
            action_dim=2,
            hidden_size=expert_cfg["hidden_size"],
            intermediate_size=expert_cfg["intermediate_size"],
            num_layers=len(self.kv_layers),
            num_heads=num_heads,
            num_kv_heads=num_kv_heads,
            head_dim=head_dim,
            time_embed_dim=expert_cfg["time_embed_dim"],
            rms_norm_eps=expert_cfg["rms_norm_eps"],
            past_traj_cfg=past_traj_cfg,
            camera_geometry_cfg=camera_geometry_cfg,
        )

        self.diffusion = FlowMatching(
            input_dims=(self.pred_traj_sampling.num_poses, 2),
            **{k: v for k, v in config.diff_cfg.items() if k not in EVAL_DEFAULTS},
        )
        self._check_eval_settings(self._eval_settings(None))

        self.traj_normalization = config.traj_normalization

        self.action_representation = config.action_representation
        self._check_action_stats(config.action_stats)

        self.post_init()
        # post_init applies the generic HF init; restore FP32 and adaLN-Zero afterwards.
        self.action_expert.float()
        self.action_expert.init_adaln_zero()
        if self.action_expert.camera_geometry is not None:
            self.action_expert.camera_geometry.init_zero()

    @torch.no_grad()
    def _init_weights(self, module):
        """HF weight init, keeping layers flagged ``_zero_init`` at zero."""
        super()._init_weights(module)
        # Zero-initialized layers (CameraGeometryKV's offsets) must stay zero when HF
        # initializes them, e.g. as missing keys when a checkpoint without camera geometry is
        # loaded into a model with it; hf_init skips weights that were loaded.
        if getattr(module, "_zero_init", False):
            hf_init.zeros_(module.weight)
            if module.bias is not None:
                hf_init.zeros_(module.bias)

    @classmethod
    def from_pretrained_submodules(cls, config):
        """Build from the raw HF backbone, or from a trained wrapper's `.vlm` if backbone_ckpt is set."""
        if not config.backbone_ckpt:
            return super().from_pretrained_submodules(config)

        backbone_wrapper = get_class(config.backbone_class_path).from_pretrained(
            config.backbone_ckpt,
            dtype=torch.bfloat16,
            attn_implementation="flash_attention_2",
        )
        # The processor is rebuilt from vlm_name_or_path, so it must be the one the backbone
        # was trained with.
        if backbone_wrapper.config.vlm_name_or_path != config.vlm_name_or_path:
            raise ValueError(
                f"backbone_ckpt was trained on {backbone_wrapper.config.vlm_name_or_path}, "
                f"but vlm_name_or_path is {config.vlm_name_or_path}."
            )
        return cls(config, pretrained_modules={"vlm": backbone_wrapper.vlm})

    def _backbone(self):
        """The HF VLM, unwrapped from PEFT if needed."""
        return (
            self.vlm.get_base_model()
            if hasattr(self.vlm, "get_base_model")
            else self.vlm
        )

    @staticmethod
    def _vlm_inputs(model_inputs: dict[str, Any]) -> dict[str, Any]:
        """Collator outputs minus the planner-only keys the backbone must not receive."""
        return {k: v for k, v in model_inputs.items() if k not in _NON_VLM_KEYS}

    def forward(self, **model_inputs):
        """Plain backbone forward on the collator's inputs (used e.g. by auxiliary heads)."""
        return self.vlm(**self._vlm_inputs(model_inputs))

    def _encode_prompt(self, model_inputs: dict[str, Any]) -> dict[str, Any]:
        """Run the backbone once and collect everything the expert needs.

        Returns read-only per-block prompt KV, the joint attention mask, the rotary tables
        for the expert tokens (positions continuing after each prompt: state tokens, then
        action tokens), and the past-trajectory state tokens when enabled.
        """
        return self._expert_context(self._encode_backbone(model_inputs), model_inputs)

    def _encode_backbone(self, model_inputs: dict[str, Any]) -> dict[str, Any]:
        """Backbone part of the expert context: prompt KV of the selected layers (detached with
        ``insulate_backbone``), the joint attention mask and the rotary tables of the expert
        tokens. Holds no expert parameters, so RL keeps it fixed across policy updates."""
        # Only the cache is used; skip full-vocabulary logits over the whole prompt.
        vlm_inputs = {
            **self._vlm_inputs(model_inputs),
            "use_cache": True,
            "logits_to_keep": 1,
        }
        vlm_outputs = self.vlm(**vlm_inputs)
        cache = vlm_outputs.past_key_values
        prompt_kv = [
            (cache.layers[i].keys, cache.layers[i].values) for i in self.kv_layers
        ]
        if self.insulate_backbone:
            prompt_kv = [(k.detach(), v.detach()) for k, v in prompt_kv]

        prompt_mask = model_inputs["attention_mask"].to(self.device).bool()
        batch_size = prompt_mask.shape[0]
        n_tokens = (
            self.action_expert.num_state_tokens + self.pred_traj_sampling.num_poses
        )
        attn_mask = torch.cat(
            [prompt_mask, prompt_mask.new_ones(batch_size, n_tokens)], dim=-1
        )[:, None, None, :]

        # Next text position after each (left-padded) prompt is rope_delta + #valid tokens.
        rope_deltas = self._backbone().model.rope_deltas
        next_pos = prompt_mask.sum(dim=-1, keepdim=True)
        if rope_deltas is not None:
            next_pos = next_pos + rope_deltas.to(self.device)
        position_ids = next_pos + torch.arange(n_tokens, device=self.device)
        position_ids = position_ids[None].expand(3, -1, -1)
        rotary_emb = self._backbone().model.language_model.rotary_emb
        cos, sin = rotary_emb(
            torch.empty(0, device=self.device, dtype=torch.float32), position_ids
        )
        return {
            "prompt_kv": prompt_kv,
            "attn_mask": attn_mask,
            "cos": cos,
            "sin": sin,
        }

    def _expert_context(
        self,
        base: dict[str, Any],
        model_inputs: dict[str, Any],
        expert: KVPrefixActionExpert | None = None,
    ) -> dict[str, Any]:
        """Add the expert's own prompt-side computations to ``_encode_backbone``'s output:
        camera-geometry offsets on the prompt KV and the past-trajectory state tokens.

        These depend on expert parameters, so RL recomputes them (with gradients) for every
        log-probability; ``expert`` selects another copy, e.g. the frozen RL reference.
        """
        expert = self.action_expert if expert is None else expert
        prompt_kv = base["prompt_kv"]
        geometry = expert.camera_geometry
        if geometry is not None:
            if "visual_geom" not in model_inputs:
                raise ValueError(
                    "This model uses camera geometry; its collator must provide visual_geom "
                    "(FlowMatchingVLACollator(camera_geometry=True))."
                )
            if base["cos"].shape[-1] != geometry.rotary_dim:
                raise ValueError(
                    f"Backbone rotary dim {base['cos'].shape[-1]} != camera geometry's "
                    f"{geometry.rotary_dim}."
                )
            prompt_kv = geometry(
                prompt_kv,
                model_inputs["visual_geom"].to(self.device),
                model_inputs["visual_geom_mask"].to(self.device).bool(),
            )

        ctx = {**base, "prompt_kv": prompt_kv}
        encoder = expert.past_traj_encoder
        if encoder is not None:
            # Independent of the noisy actions and time: encode once per prompt.
            pose_valid = model_inputs.get("past_traj_pose_valid")
            ctx["state_tokens"] = encoder(
                model_inputs["past_traj"].to(self.device),
                model_inputs["past_traj_valid"].to(self.device).bool(),
                None if pose_valid is None else pose_valid.to(self.device).bool(),
            )
        return ctx

    @staticmethod
    def _repeat_context(
        ctx: dict[str, Any], repeats: int, share_prompt_kv: bool = False
    ) -> dict[str, Any]:
        """Repeat each scene's expert context ``repeats`` times along the batch (contiguously).

        ``share_prompt_kv``: keep one prompt KV per scene; the expert blocks then attend to it from
        all ``repeats`` samples without copies (``_shared_prompt_attention``; same result up to
        floating-point summation order). Used by the RL path, where 16 samples share a prompt.
        """
        if repeats == 1:
            return ctx
        return {
            "prompt_kv": ctx["prompt_kv"]
            if share_prompt_kv
            else [
                (
                    k.repeat_interleave(repeats, dim=0),
                    v.repeat_interleave(repeats, dim=0),
                )
                for k, v in ctx["prompt_kv"]
            ],
            **{
                key: value.repeat_interleave(repeats, dim=0)
                for key, value in ctx.items()
                if key != "prompt_kv"
            },
        }

    def _predict_vector_field(
        self,
        x: torch.Tensor,
        t: torch.Tensor,
        ctx: dict[str, Any],
        expert: KVPrefixActionExpert | None = None,
    ) -> torch.Tensor:
        """Expert velocity for noisy actions ``x`` at time ``t`` given the encoded prompt ``ctx``
        (``expert``: another copy of the expert, e.g. the frozen RL reference)."""
        return (self.action_expert if expert is None else expert)(x, t, **ctx)

    def _check_action_stats(self, stats: dict | None) -> None:
        """Validate ``action_stats`` against the representation and prediction grid."""
        if stats is None:
            if self.action_representation == "polar":
                raise ValueError(
                    "polar actions need action_stats (scripts/utils/compute_action_stats.py)."
                )
            return
        expected = [self.pred_traj_sampling.num_poses, 2]
        for key in ("mean", "std"):
            if list(torch.tensor(stats[key]).shape) != expected:
                raise ValueError(f"action_stats[{key!r}] must have shape {expected}.")
        if min(min(row) for row in stats["std"]) <= 0:
            raise ValueError("action_stats['std'] must be positive.")
        # Stats are only meaningful for the representation and grid they were computed on.
        representation = stats.get("action_representation")
        if representation is not None and representation != self.action_representation:
            raise ValueError(
                f"action_stats were computed for {representation!r}, model uses "
                f"{self.action_representation!r}."
            )
        sampling = stats.get("pred_traj_sampling")
        if (
            sampling is not None
            and TrajectorySampling(**sampling) != self.pred_traj_sampling
        ):
            raise ValueError(
                f"action_stats were computed for {sampling}, model predicts {self.pred_traj_sampling}."
            )

    def _action_stats(self, ref: torch.Tensor) -> tuple[torch.Tensor, torch.Tensor]:
        """``action_stats`` mean and std as tensors on ``ref``'s device and dtype."""
        stats = self.config.action_stats
        mean = torch.tensor(stats["mean"], device=ref.device, dtype=ref.dtype)
        std = torch.tensor(stats["std"], device=ref.device, dtype=ref.dtype)
        return mean, std

    def _normalize_actions(self, actions: torch.Tensor) -> torch.Tensor:
        """Raw actions (see encode_actions) -> diffusion space."""
        if self.config.action_stats is not None:
            mean, std = self._action_stats(actions)
            return (actions - mean) / std
        if self.action_representation == "trajs":
            return actions / self.traj_normalization
        return actions

    def _denormalize_actions(self, x: torch.Tensor) -> torch.Tensor:
        """Diffusion space -> raw actions; inverse of _normalize_actions."""
        if self.config.action_stats is not None:
            mean, std = self._action_stats(x)
            return x * std + mean
        if self.action_representation == "trajs":
            return x * self.traj_normalization
        return x

    def compute_loss(self, batch: E2EDataBatch) -> dict[str, torch.Tensor]:
        """Flow-matching loss of the expert on the batch's ground-truth futures.

        The prompt is encoded once; every scene gets ``num_repeat_steps`` noise / time draws.
        Returns ``{"loss", "diff_loss"}`` (the same value).
        """
        ctx = self._encode_prompt(batch.model_inputs)

        # Resample to prediction frequency/horizon
        # Not batched since datasets may have different GT frequencies
        gt_trajs = []
        for sample in batch.samples:
            gt_traj = resample_tensor(
                sample.fut_traj,
                sample.fut_traj_sampling,
                self.pred_traj_sampling,
            )
            gt_trajs.append(gt_traj)
        gt_trajs = torch.stack(gt_trajs).to(self.device)

        # Prepare diffusion data
        actions = encode_actions(
            gt_trajs,
            self.action_representation,
            dt=self.pred_traj_sampling.interval_length,
            initial_velocity=batch.model_inputs["init_vel"],
        )
        diff_target = self._normalize_actions(actions)
        diff_training_data = self.diffusion.construct_training_data(diff_target)

        # Each scenario is paired with num_repeat_steps noise/time samples; the prompt is
        # encoded once and shared across them.
        ctx = self._repeat_context(ctx, self.diffusion.num_repeat_steps)
        pred = self._predict_vector_field(
            diff_training_data["noisy_x"].float(),
            diff_training_data["timesteps"].float(),
            ctx,
        )

        diff_loss = self.diffusion.compute_loss_from_pred(diff_training_data, pred)

        return {"loss": diff_loss, "diff_loss": diff_loss}

    def _eval_settings(self, generate_cfg: dict | None) -> dict[str, Any]:
        """Inference settings: diff_cfg defaults, overridden by a per-call generate_cfg."""
        settings = {
            **{k: self.config.diff_cfg.get(k, v) for k, v in EVAL_DEFAULTS.items()},
            "num_inference_steps": self.diffusion.num_inference_steps,
        }
        if generate_cfg:
            unknown = set(generate_cfg) - set(settings)
            if unknown:
                raise ValueError(
                    f"Unknown FlowMatchingVLA generate_cfg keys {sorted(unknown)}; "
                    f"expected a subset of {sorted(settings)}."
                )
            settings.update(generate_cfg)
        return settings

    @staticmethod
    def _check_eval_settings(settings: dict[str, Any]) -> None:
        """Reject inconsistent inference settings (see ``EVAL_DEFAULTS``)."""
        if settings["num_eval_samples"] < 1:
            raise ValueError("num_eval_samples must be >= 1.")
        if settings["eval_aggregation"] not in EVAL_AGGREGATIONS:
            raise ValueError(
                f"eval_aggregation must be one of {EVAL_AGGREGATIONS}, "
                f"got {settings['eval_aggregation']!r}."
            )
        if settings["eval_init"] == "zero" and settings["num_eval_samples"] > 1:
            raise ValueError(
                "eval_init='zero' is deterministic; use num_eval_samples=1."
            )
        if not isinstance(settings["cuda_graph"], bool):
            raise ValueError(
                f"cuda_graph must be a bool, got {settings['cuda_graph']!r}."
            )
        if settings["sampler"] not in SAMPLERS:
            raise ValueError(
                f"sampler must be one of {SAMPLERS}, got {settings['sampler']!r}."
            )
        if settings["sampler"] == "sde":
            if settings["num_eval_samples"] != 1:
                raise ValueError(
                    "sampler='sde' returns raw samples; use num_eval_samples=1."
                )
            if settings["noise_level"] <= 0:
                raise ValueError("sampler='sde' needs noise_level > 0.")

    @staticmethod
    def _aggregate_samples(trajs: torch.Tensor, aggregation: str) -> torch.Tensor:
        """[N, K, T, 2] sampled XY trajectories -> [N, T, 2]."""
        mean = trajs.mean(dim=1)
        if aggregation == "mean":
            return mean
        # medoid: the sample with the lowest average displacement from the mean, which
        # stays a feasible trajectory when the samples are multimodal.
        dist = (trajs - mean[:, None]).norm(dim=-1).mean(dim=-1)
        idx = dist.argmin(dim=1)
        return trajs[torch.arange(trajs.shape[0], device=trajs.device), idx]

    def predict_from_processed_inputs(
        self,
        # Model specific inputs
        input_ids: torch.Tensor,
        attention_mask: torch.Tensor,
        init_vel: torch.Tensor | None = None,
        # General inputs
        num_traj_samples: int = 1,
        generate_cfg: dict | None = None,
        target_sampling: TrajectorySampling | None = None,
        **kwargs,
    ):
        """
        Returns [B * num_traj_samples, T, 2] trajectories (each scene's outputs contiguous).
        Each output aggregates num_eval_samples flow samples (see EVAL_DEFAULTS); generate_cfg
        may override those keys and num_inference_steps. With num_eval_samples > 1, extra
        holds "single_sample_trajs": the first raw sample of each output, for comparison.
        """
        settings = self._eval_settings(generate_cfg)
        self._check_eval_settings(settings)

        # The collator already emits the prompt only, left-padded with at least one pad token,
        # so its inputs go to the backbone unchanged.
        model_inputs = {
            "input_ids": input_ids,
            "attention_mask": attention_mask,
            **{k: v for k, v in kwargs.items() if isinstance(v, torch.Tensor)},
        }
        model_inputs = {k: v.to(self.device) for k, v in model_inputs.items()}

        batch_size = input_ids.shape[0]
        num_eval_samples = settings["num_eval_samples"]
        repeats = num_traj_samples * num_eval_samples
        pred = None
        extra = {}
        if settings["sampler"] == "sde":
            pred, extra["rl"] = self._sample_sde(model_inputs, repeats, settings)
        elif settings["cuda_graph"] and self.device.type == "cuda":
            pred = self._cuda_graph_runner(settings["cuda_graph_prompt_len"]).sample(
                model_inputs,
                repeats=repeats,
                num_steps=settings["num_inference_steps"],
                init=settings["eval_init"],
            )
        if pred is None:
            # Every sampling step reads the same prompt KV; nothing is written back.
            ctx = self._repeat_context(self._encode_prompt(model_inputs), repeats)
            pred = self.diffusion.sample(
                batch_size=batch_size * repeats,
                step_fn=lambda x, t: self._predict_vector_field(x, t, ctx),
                device=self.device,
                num_inference_steps=settings["num_inference_steps"],
                init=settings["eval_init"],
                return_all_steps=False,
            )

        # Decode every sample before aggregating: polar/acc_curv are nonlinear in XY.
        actions = self._denormalize_actions(pred)
        trajs = decode_actions(
            actions,
            self.action_representation,
            dt=self.pred_traj_sampling.interval_length,
            initial_velocity=(
                None
                if init_vel is None
                else init_vel.to(self.device).repeat_interleave(repeats)
            ),
        )
        trajs = trajs.view(
            batch_size * num_traj_samples, num_eval_samples, *trajs.shape[1:]
        )
        pred_trajs = self._aggregate_samples(trajs, settings["eval_aggregation"])

        if num_eval_samples > 1:
            extra["single_sample_trajs"] = self._maybe_resample(
                trajs[:, 0], target_sampling
            )
        pred_trajs = self._maybe_resample(pred_trajs, target_sampling)

        return pred_trajs, extra

    # ------------------------------------------------------------------ RL (Flow-GRPO)
    def _sample_sde(
        self, model_inputs: dict[str, Any], repeats: int, settings: dict[str, Any]
    ) -> tuple[torch.Tensor, dict[str, Any]]:
        """SDE rollouts for RL: ``repeats`` samples per scene (contiguous), in diffusion space.

        Returns the samples ``[B * repeats, num_poses, 2]`` and the record ``grpo_policy_terms``
        needs to re-score them: the backbone context (fixed: the backbone is insulated), the
        planner inputs the expert re-encodes, the chain, time grid, step stds and log-probs.
        """
        if not self.insulate_backbone:
            raise ValueError(
                "sampler='sde' (RL on the expert) needs expert_cfg.insulate_backbone=True."
            )
        base = self._encode_backbone(model_inputs)
        ctx = self._repeat_context(
            self._expert_context(base, model_inputs), repeats, share_prompt_kv=True
        )
        batch_size = model_inputs["input_ids"].shape[0]
        out = self.diffusion.sample_sde(
            batch_size=batch_size * repeats,
            step_fn=lambda x, t: self._predict_vector_field(x, t, ctx),
            device=self.device,
            num_steps=settings["sde_steps"] or settings["num_inference_steps"],
            noise_level=settings["noise_level"],
        )
        expert_inputs = {
            k: v for k, v in model_inputs.items() if k in _EXPERT_CONTEXT_KEYS
        }
        record = {
            "base": base,
            "expert_inputs": expert_inputs,
            "repeats": repeats,
            **out,
        }
        return out["chain"][:, -1], record

    def grpo_reference(self) -> torch.nn.Module:
        """Frozen copy of what RL trains (the expert) for the KL term; the backbone is shared."""
        reference = copy.deepcopy(self.action_expert).eval()
        reference.requires_grad_(False)
        return reference

    def grpo_policy_terms(
        self,
        extra: dict[str, Any],
        reference: torch.nn.Module | None = None,
        step_fraction: float = 1.0,
    ) -> dict[str, torch.Tensor]:
        """Per-denoising-step terms of the GRPO loss for rollouts from ``sampler='sde'``.

        Re-runs the current expert (with gradients) on the stored chains. Returns, each
        ``[B * repeats, S]`` for the S trained steps: ``log_probs`` of the sampled transitions,
        their ``old_log_probs`` at sampling time, ``kl`` to ``reference`` (closed form for two
        Gaussians with the same std, averaged over elements like the log-probs; zeros without a
        reference) and ``mask``. ``step_fraction`` < 1 trains a random subset of the steps
        (Flow-GRPO's timestep_fraction); the chain itself always has every step.
        """
        rl = extra["rl"]
        chain, timesteps, stds = rl["chain"], rl["timesteps"], rl["stds"]
        num_steps = len(stds)
        n_train = max(1, round(step_fraction * num_steps))
        steps = sorted(random.sample(range(num_steps), n_train))

        ctx = self._repeat_context(
            self._expert_context(rl["base"], rl["expert_inputs"]),
            rl["repeats"],
            share_prompt_kv=True,
        )
        means = self.diffusion.chain_step_means(
            chain,
            timesteps,
            stds,
            lambda x, t: self._predict_vector_field(x, t, ctx),
            steps,
        )
        log_probs = torch.stack(
            [
                self.diffusion.sde_step_log_prob(
                    chain[:, k + 1], means[:, i], float(stds[k])
                )
                for i, k in enumerate(steps)
            ],
            dim=1,
        )
        kl = torch.zeros_like(log_probs)
        if reference is not None:
            with torch.no_grad():
                ref_ctx = self._repeat_context(
                    self._expert_context(rl["base"], rl["expert_inputs"], reference),
                    rl["repeats"],
                    share_prompt_kv=True,
                )
                ref_means = self.diffusion.chain_step_means(
                    chain,
                    timesteps,
                    stds,
                    lambda x, t: self._predict_vector_field(x, t, ref_ctx, reference),
                    steps,
                )
            step_var = stds[steps].to(means.dtype) ** 2
            kl = ((means - ref_means) ** 2).flatten(2).mean(dim=2) / (2.0 * step_var)
        return {
            "log_probs": log_probs,
            "old_log_probs": rl["log_probs"][:, steps],
            "kl": kl,
            "mask": torch.ones_like(log_probs),
        }

    def _cuda_graph_runner(self, prompt_len: int | None):
        """Lazily built; holds the recorded graphs for this model instance (never saved)."""
        runner = self.__dict__.get("_graph_runner")
        if runner is None:
            from ad_vla.models.vlas.fm_vla_cuda_graph import FMVLACudaGraphRunner

            runner = FMVLACudaGraphRunner(self, prompt_len=prompt_len)
            # Plain attribute, not a submodule: no parameters, not in state_dict.
            self.__dict__["_graph_runner"] = runner
        return runner

    def get_collator(
        self,
        drop_cfg: dict[str, float] | None = None,
        inference_mode: bool = False,
        past_truncation_prob: float = 0.0,
    ):
        """Collator for this model's prompts, expert inputs and (with geometry) camera rays.

        Args:
            drop_cfg: Input dropout probabilities, e.g. ``{past_traj: 0.2, expert_past_traj:
                0.2}`` (training only).
            inference_mode: Validation / inference batches.
            past_truncation_prob: Share of training samples whose past is shortened.
        """
        return FlowMatchingVLACollator(
            self.processor,
            self.prompt_builder,
            self.past_traj_sampling,
            self.pred_traj_sampling,
            self.camera_sequence,
            self.num_past_image_frames,
            drop_cfg=drop_cfg,
            inference_mode=inference_mode,
            concat_cameras=self.concat_cameras,
            chat_template_kwargs=self.config.chat_template_kwargs,
            prompt_past_traj_sampling=self.prompt_past_traj_sampling,
            pad_missing_past_traj=self.pad_missing_past_traj,
            past_truncation_prob=past_truncation_prob,
            camera_geometry=self.camera_geometry_mode is not None,
            require_calibration=(
                self.camera_geometry_mode is not None
                and self.config.expert_cfg["camera_geometry_cfg"]["require_calibration"]
            ),
        )


class FlowMatchingVLACollator:
    """Turn ``E2EDataSample``s into FlowMatchingVLA inputs.

    Builds the prompt only (no assistant turn: the future supervises the expert, not text),
    left-padded with at least one pad token, plus the expert-only inputs: ``init_vel``,
    ``past_traj`` [B, P, 2] on the expert grid with ``past_traj_valid`` [B] (prompt / expert
    dropout) and ``past_traj_pose_valid`` [B, P] (poses older than the available history), and
    with camera geometry ``visual_geom`` / ``visual_geom_mask`` aligned to the image tokens.

    Args:
        processor: Backbone processor (tokenizer + image processor).
        prompt_builder: Builds the chat message from a sample.
        past_traj_sampling: Expert state-token grid.
        pred_traj_sampling: Model output grid.
        camera_sequence: Cameras in prompt order.
        num_past_image_frames: Image frames per camera.
        drop_cfg: Input dropout probabilities (``expert_past_traj`` drops the expert's past
            independently of the prompt's).
        inference_mode: Validation / inference (no truncation; labels all ignored).
        concat_cameras: Cameras tiled into one image per frame.
        chat_template_kwargs: Extra ``apply_chat_template`` kwargs.
        prompt_past_traj_sampling: Past grid in the prompt (None: ``past_traj_sampling``).
        pad_missing_past_traj: Write prompt poses older than the history as ``[X.X, X.X]``.
        past_truncation_prob: Probability of shortening a training sample's past.
        camera_geometry: Also emit per-image-token camera geometry.
        require_calibration: Raise when a camera has no calibration (geometry only).
    """

    def __init__(
        self,
        processor: AutoProcessor,
        prompt_builder: PromptBuilder,
        past_traj_sampling: TrajectorySampling,
        pred_traj_sampling: TrajectorySampling,
        camera_sequence: list[str],
        num_past_image_frames: int,
        drop_cfg: dict[str, float] | None = None,
        inference_mode: bool = False,
        concat_cameras: bool = True,
        chat_template_kwargs: dict | None = None,
        prompt_past_traj_sampling: TrajectorySampling | None = None,
        pad_missing_past_traj: bool = False,
        past_truncation_prob: float = 0.0,
        camera_geometry: bool = False,
        require_calibration: bool = True,
    ):
        if not 0.0 <= past_truncation_prob <= 1.0:
            raise ValueError("past_truncation_prob must be between zero and one.")
        self.processor = processor
        self.prompt_builder = prompt_builder
        # Expert state-token grid; the prompt's defaults to the same grid.
        self.past_traj_sampling = past_traj_sampling
        self.prompt_past_traj_sampling = prompt_past_traj_sampling or past_traj_sampling
        self.pad_missing_past_traj = pad_missing_past_traj
        self.pred_traj_sampling = pred_traj_sampling
        self.camera_sequence = camera_sequence
        self.num_past_image_frames = num_past_image_frames
        self.drop_cfg = drop_cfg
        self.inference_mode = inference_mode
        self.concat_cameras = concat_cameras
        self.chat_template_kwargs = chat_template_kwargs or {}
        # Training augmentation (see _maybe_truncate_past); never applied at inference.
        self.past_truncation_prob = 0.0 if inference_mode else past_truncation_prob
        # Token-aligned camera geometry for the expert (see ad_vla/utils/camera_geometry.py).
        self.camera_geometry = camera_geometry
        self.require_calibration = require_calibration

    def _maybe_truncate_past(self, sample: E2EDataSample) -> E2EDataSample:
        """With probability past_truncation_prob, keep only the latest k of the n raw past
        poses (k uniform in 1..n-1), as at the start of a scene or a closed-loop rollout.

        The prompt and the expert then both see the shorter history (missing-pose text
        slots and embeddings). Returns a copy; the dataset sample is not modified.
        """
        past_traj = sample.agent_input["past_traj"]
        num_poses = past_traj.shape[0]
        if num_poses < 2 or random.random() >= self.past_truncation_prob:
            return sample
        keep = random.randint(1, num_poses - 1)
        agent_input = {**sample.agent_input, "past_traj": past_traj[-keep:]}
        return sample.model_copy(update={"agent_input": agent_input})

    @staticmethod
    def _resample_past_traj(
        sample: E2EDataSample, target: TrajectorySampling
    ) -> torch.Tensor:
        """Resample the past trajectory to a target grid (only the available target poses)."""
        past_traj = sample.agent_input["past_traj"]
        dataset_sampling = TrajectorySampling(
            num_poses=past_traj.shape[0],
            interval_length=1.0 / sample.metadata["sampling_freq"],
        )
        return resample_tensor(
            past_traj,
            dataset_sampling,
            target,
            is_past=True,
        )

    def _get_fut_traj_for_prompt(self, sample: E2EDataSample) -> torch.Tensor:
        """Resample future trajectory to model's output grid for prompt embedding (ephemeral)."""
        if sample.fut_traj is None:
            return None
        if sample.fut_traj_sampling != self.pred_traj_sampling:
            return resample_tensor(
                sample.fut_traj, sample.fut_traj_sampling, self.pred_traj_sampling
            )
        return sample.fut_traj

    def _pad_past_traj(
        self, past_traj: torch.Tensor
    ) -> tuple[torch.Tensor, torch.Tensor]:
        """Left-pad a history shorter than past_traj_sampling with its oldest pose.

        resample_tensor does not extrapolate before the oldest observation, so datasets with
        a shorter history return fewer poses; the expert's state tokens need a fixed length.
        Returns the padded XY [P, 2] and which poses are real [P] (False for padding).
        """
        num_poses = self.past_traj_sampling.num_poses
        missing = num_poses - past_traj.shape[0]
        pose_valid = torch.arange(num_poses) >= missing
        if missing <= 0:
            return past_traj[-num_poses:], pose_valid
        return torch.cat(
            (past_traj[:1].expand(missing, -1), past_traj), dim=0
        ), pose_valid

    def _get_sample_message(
        self, sample: E2EDataSample
    ) -> tuple[list[dict[str, Any]], torch.Tensor, torch.Tensor, bool]:
        """Returns the prompt message, the fixed-length past XY [P, 2] for the expert's state
        tokens with its per-pose validity [P], and whether the expert sees the past (by default
        whether the prompt shows it; drawn independently when drop_cfg has expert_past_traj)."""
        if self.drop_cfg is not None:
            active_inputs = self.prompt_builder.sample_active_inputs(self.drop_cfg)
        else:
            active_inputs = ALL_INPUTS

        past_traj_for_prompt = self._resample_past_traj(
            sample, self.prompt_past_traj_sampling
        )

        message = self.prompt_builder.build_message(
            cameras=sample.cameras,
            past_traj=past_traj_for_prompt,
            intent=sample.intent,
            active_inputs=active_inputs,
            camera_sequence=self.camera_sequence,
            num_past_image_frames=self.num_past_image_frames,
            reasoning_trace=None,
            fut_traj=None,  # Future trajectory supervises action expert only.
            inference_mode=True,
            concat_cameras=self.concat_cameras,
        )

        if message[-1]["role"] == "assistant" and not message[-1]["content"]:
            message = message[:-1]

        num_missing = (
            self.prompt_past_traj_sampling.num_poses - past_traj_for_prompt.shape[0]
        )
        if (
            self.pad_missing_past_traj
            and num_missing > 0
            and "past_traj" in active_inputs
        ):
            pad_past_traj_text(message, num_missing)

        if self.prompt_past_traj_sampling == self.past_traj_sampling:
            past_traj_for_expert = past_traj_for_prompt
        else:
            past_traj_for_expert = self._resample_past_traj(
                sample, self.past_traj_sampling
            )
        past_traj, pose_valid = self._pad_past_traj(past_traj_for_expert[:, :2].float())
        expert_past_valid = "past_traj" in active_inputs
        if self.drop_cfg is not None and "expert_past_traj" in self.drop_cfg:
            # Independent expert dropout: drop_cfg["past_traj"] then drops the prompt's past only,
            # and the expert's state tokens become null tokens with probability expert_past_traj.
            expert_past_valid = random.random() >= self.drop_cfg["expert_past_traj"]
        return message, past_traj, pose_valid, expert_past_valid

    def _ensure_padding_token(
        self, inputs: dict[str, torch.Tensor]
    ) -> dict[str, torch.Tensor]:
        """Left-pad every prompt by one pad token when the batch has no padding at all, so
        each batch carries at least one pad position (as ``predict_from_processed_inputs``
        assumes); every per-token tensor is padded alike."""
        if not inputs["attention_mask"].bool().all():
            return inputs

        pad_id = getattr(self.processor.tokenizer, "pad_token_id", None)
        if pad_id is None:
            return inputs

        # Pad every per-token tensor (e.g. mm_token_type_ids) so lengths stay aligned.
        token_shape = inputs["input_ids"].shape
        for key, value in inputs.items():
            if isinstance(value, torch.Tensor) and value.shape == token_shape:
                inputs[key] = F.pad(
                    value, (1, 0), value=pad_id if key == "input_ids" else 0
                )
        return inputs

    def _token_geometry(
        self, batch: list[E2EDataSample], prompt_inputs: dict[str, torch.Tensor]
    ) -> tuple[torch.Tensor, torch.Tensor]:
        """Per-image-token camera geometry [B, L, F] and its mask [B, L] for the prompts."""
        image_token_id = getattr(self.processor, "image_token_id", None)
        if image_token_id is None:
            image_token_id = self.processor.tokenizer.convert_tokens_to_ids(
                "<|image_pad|>"
            )
        return token_geometry(
            [sample.cameras for sample in batch],
            camera_sequence=self.camera_sequence,
            num_past_image_frames=self.num_past_image_frames,
            concat_cameras=self.concat_cameras,
            input_ids=prompt_inputs["input_ids"],
            image_grid_thw=prompt_inputs["image_grid_thw"],
            image_token_id=image_token_id,
            merge_size=int(getattr(self.processor.image_processor, "merge_size", 2)),
            require_calibration=self.require_calibration,
        )

    def __call__(self, batch: list[E2EDataSample]) -> E2EDataBatch:
        """Collate samples into an ``E2EDataBatch`` (see the class docstring for the keys)."""
        if self.past_truncation_prob > 0.0:
            batch = [self._maybe_truncate_past(sample) for sample in batch]
        prompt_messages, past_trajs, past_pose_valid, past_traj_valid = zip(
            *(self._get_sample_message(sample) for sample in batch)
        )

        prompt_inputs = self.processor.apply_chat_template(
            list(prompt_messages),
            tokenize=True,
            add_generation_prompt=True,
            continue_final_message=False,
            return_dict=True,
            return_tensors="pt",
            padding=True,
            **self.chat_template_kwargs,
        )
        prompt_inputs = self._ensure_padding_token(prompt_inputs)
        prompt_inputs["use_cache"] = True

        if self.inference_mode:
            prompt_inputs["labels"] = torch.full_like(prompt_inputs["input_ids"], -100)

        prompt_inputs["init_vel"] = torch.stack(
            [initial_speed(sample) for sample in batch]
        )
        # Expert-only state: the same history as the prompt, masked when input dropout
        # removed it from the prompt so the dropout regularizes both paths.
        prompt_inputs["past_traj"] = torch.stack(past_trajs)
        prompt_inputs["past_traj_valid"] = torch.tensor(
            past_traj_valid, dtype=torch.bool
        )
        # Poses older than the available history (left padding).
        prompt_inputs["past_traj_pose_valid"] = torch.stack(past_pose_valid)
        if self.camera_geometry:
            prompt_inputs["visual_geom"], prompt_inputs["visual_geom_mask"] = (
                self._token_geometry(batch, prompt_inputs)
            )

        out = {}
        out["model_inputs"] = prompt_inputs
        out["samples"] = batch

        return E2EDataBatch(**out)
