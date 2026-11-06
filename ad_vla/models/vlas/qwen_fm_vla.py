"""Qwen VLA with joint text supervision and a continuous action expert.

The Qwen backbone keeps its ordinary teacher-forced trajectory-text objective.
Its final hidden layer is also exposed to a flow-matching DiT expert, but only
prompt-side states are eligible: assistant reasoning and discrete trajectory
tokens are explicitly removed before the expert sees the VLM memories.

The continuous branch follows this data path:

1. The collator resamples and flattens the past ego XY trajectory.
2. Qwen encodes prompt text and images, producing one final hidden-state tensor.
3. Prompt-only states are split into language and visual memories.
4. The action expert denoises a sequence of two-dimensional continuous actions.
5. Flow-matching output is converted back into XY trajectories for evaluation.

Camera geometry is aligned to Qwen token positions by the collator. The action
expert therefore uses exactly the same forward interface for horizontally
concatenated cameras and separately tokenized cameras.
"""

from collections.abc import Callable, Mapping
from typing import Any

import torch
import torch.nn as nn
import torch.nn.functional as F
from transformers.modeling_outputs import ModelOutput

from alpamayo_r1.models.action_in_proj import PerWaypointActionInProjV2

from ad_vla.dataset.data_types import (
    CameraData,
    E2EDataBatch,
    E2EDataSample,
    QuestionAnswerPair,
    TrajectorySampling,
)
from ad_vla.dataset.camera_config import qa_cameras
from ad_vla.models.base_traj_planner import BaseTrajPlanner
from ad_vla.models.base_vla import BaseVLA, BaseVLAConfig
from ad_vla.models.diffusion.flow_matching import FlowMatching
from ad_vla.models.layers.action_expert import ActionExpert
from ad_vla.models.vlas.text_traj_vlm import TextTrajCollator
from ad_vla.utils.prompts import normalize_qa_prediction
from ad_vla.utils.action_utils import action_to_traj, traj_to_action
from ad_vla.utils.traj_utils import resample_tensor


_EXPERT_INPUT_KEYS = frozenset(
    {
        "init_vel",
        "proprio_state",
        "visual_camera_poses",
        "visual_patch_coords",
        "visual_time_ids",
        "reference_camera_pose",
    }
)


class QwenFlowMatchingVLAConfig(BaseVLAConfig):
    """Configuration for joint discrete-text and continuous-action training.

    Expert architecture lives in the nested ``action_expert_cfg`` mapping. Its
    defaults reproduce the intended ten-block, 768-wide, twelve-head expert
    with a 2048-wide SwiGLU. Keeping the keys identical to ``ActionExpert``'s
    constructor makes initialization compact and checkpoint configuration easy
    to inspect.

    These are ordinary Hugging Face config fields: they serialize with model
    checkpoints and can be overridden from ``conf/model/qwen_fm_vla.yaml``.

    Args:
        past_traj_sampling: Temporal grid used to resample the past ego XY
            trajectory before it becomes the state token. The dictionary uses
            ``num_poses`` and ``interval_length``.
        pred_traj_sampling: Temporal grid of the predicted trajectory/action
            sequence, also described by ``num_poses`` and ``interval_length``.
        traj_normalization: Positive scale used for XY trajectory values and the
            past-trajectory state input.
        diff_cfg: Keyword arguments for ``FlowMatching``, such as the number of
            training repeats and numerical integration steps.
        action_representation: ``"trajs"`` for normalized XY waypoints or
            ``"acc_curv"`` for acceleration/curvature controls.
        ce_loss_weight: Weight of Qwen's teacher-forced text cross entropy.
        qa_loss_weight: Relative weight of QA cross entropy within mixed
            QA/driving batches. ``1.0`` makes each task's influence follow its
            fraction of batch rows rather than its number of target tokens.
        diff_loss_weight: Weight of the continuous flow-matching regression.
        action_expert_cfg: Optional partial mapping passed to ``ActionExpert``.
            Missing entries inherit documented defaults. Supported keys are
            ``dim``, ``depth``, ``num_heads``, ``ffn_dim``, ``dropout``,
            ``attention_dropout``, ``rope_base``, ``qk_norm``, ``norm_eps``,
            ``causal``, ``use_cape``, ``cape_dim``, and
            ``cape_translation_scale``.
        **kwargs: Additional ``BaseVLAConfig``/Hugging Face configuration fields,
            including the backbone path, prompt config, and camera layout.
    """

    model_type = "qwen_fm_vla"

    def __init__(
        self,
        past_traj_sampling: dict | None = None,
        pred_traj_sampling: dict | None = None,
        traj_normalization: float = 20.0,
        diff_cfg: dict | None = None,
        action_representation: str = "trajs",
        ce_loss_weight: float = 1.0,
        qa_loss_weight: float = 1.0,
        diff_loss_weight: float = 1.0,
        action_expert_cfg: dict[str, Any] | None = None,
        **kwargs: Any,
    ):
        super().__init__(**kwargs)

        # Merge partial overrides into one complete, serializable expert config.
        default_action_expert_cfg = {
            "dim": 768,
            "depth": 10,
            "num_heads": 12,
            "ffn_dim": 2048,
            "dropout": 0.0,
            "attention_dropout": 0.0,
            "rope_base": 10_000.0,
            "qk_norm": True,
            "norm_eps": 1e-6,
            "causal": False,
            "use_cape": True,
            "cape_dim": 32,
            "cape_translation_scale": 1.0,
        }
        if action_expert_cfg is not None:
            if not isinstance(action_expert_cfg, Mapping):
                raise TypeError("action_expert_cfg must be a mapping.")
            unknown_keys = set(action_expert_cfg) - set(default_action_expert_cfg)
            if unknown_keys:
                raise ValueError(
                    "Unknown action_expert_cfg keys: " + ", ".join(sorted(unknown_keys))
                )
            default_action_expert_cfg.update(dict(action_expert_cfg))
        action_expert_cfg = default_action_expert_cfg

        if action_representation not in {"trajs", "acc_curv"}:
            raise ValueError(
                "action_representation must be 'trajs' or 'acc_curv', got "
                f"{action_representation!r}."
            )
        if traj_normalization <= 0.0:
            raise ValueError(
                f"traj_normalization must be positive, got {traj_normalization}."
            )
        if ce_loss_weight < 0.0 or qa_loss_weight < 0.0 or diff_loss_weight < 0.0:
            raise ValueError("Loss weights must be non-negative.")
        if ce_loss_weight == 0.0 and diff_loss_weight == 0.0:
            raise ValueError("At least one loss weight must be positive.")
        if action_expert_cfg["dim"] <= 0:
            raise ValueError("action_expert_cfg.dim must be positive.")
        if action_expert_cfg["depth"] <= 0:
            raise ValueError("action_expert_cfg.depth must be positive.")
        if action_expert_cfg["num_heads"] <= 0:
            raise ValueError("action_expert_cfg.num_heads must be positive.")
        if action_expert_cfg["dim"] % action_expert_cfg["num_heads"]:
            raise ValueError("action_expert_cfg.dim must be divisible by num_heads.")
        if action_expert_cfg["ffn_dim"] <= 0:
            raise ValueError("action_expert_cfg.ffn_dim must be positive.")
        if action_expert_cfg["rope_base"] <= 0.0:
            raise ValueError("action_expert_cfg.rope_base must be positive.")
        if action_expert_cfg["norm_eps"] <= 0.0:
            raise ValueError("action_expert_cfg.norm_eps must be positive.")
        cape_dim = action_expert_cfg["cape_dim"]
        if cape_dim < 0 or cape_dim % 4:
            raise ValueError(
                "action_expert_cfg.cape_dim must be non-negative and divisible by four."
            )
        if action_expert_cfg["use_cape"]:
            if cape_dim == 0:
                raise ValueError(
                    "action_expert_cfg.cape_dim must be positive when CaPE is on."
                )
            head_dim = action_expert_cfg["dim"] // action_expert_cfg["num_heads"]
            temporal_dim = head_dim - cape_dim
            if temporal_dim <= 0 or temporal_dim % 2:
                raise ValueError(
                    "With CaPE enabled, head_dim - action_expert_cfg.cape_dim "
                    "must be positive and even."
                )
        if action_expert_cfg["cape_translation_scale"] <= 0.0:
            raise ValueError(
                "action_expert_cfg.cape_translation_scale must be positive."
            )

        self.past_traj_sampling = past_traj_sampling or {
            "num_poses": 4,
            "interval_length": 0.5,
        }
        self.pred_traj_sampling = pred_traj_sampling or {
            "num_poses": 25,
            "interval_length": 0.2,
        }
        self.traj_normalization = traj_normalization
        self.diff_cfg = diff_cfg or {
            "num_inference_steps": 10,
            "num_repeat_steps": 8,
        }
        self.action_representation = action_representation
        self.ce_loss_weight = ce_loss_weight
        self.qa_loss_weight = qa_loss_weight
        self.diff_loss_weight = diff_loss_weight
        self.action_expert_cfg = action_expert_cfg


class QwenFlowMatchingVLA(BaseVLA, BaseTrajPlanner):
    """Qwen trajectory model with a configurable flow-matching DiT expert.

    Final-layer, prompt-only visual and language states form two fixed memories.
    A separate state/action transformer predicts the continuous 2D flow field.
    Camera awareness is controlled by the serialized model config; disabling it
    does not change expert depth, width, or modality alternation.

    Args:
        config: Validated ``QwenFlowMatchingVLAConfig`` containing both backbone
            and continuous-expert settings.
        pretrained_modules: Optional mapping containing a preloaded ``"vlm"``.
            When absent, ``BaseVLA`` constructs the backbone from ``config``.
    """

    config_class = QwenFlowMatchingVLAConfig
    trains_qa = True  # data.train entries with task: qa
    base_model_prefix = "vlm"

    def __init__(self, config, pretrained_modules=None):
        BaseVLA.__init__(self, config=config, pretrained_modules=pretrained_modules)
        BaseTrajPlanner.__init__(
            self,
            past_traj_sampling=config.past_traj_sampling,
            pred_traj_sampling=config.pred_traj_sampling,
        )

        text_config = getattr(self.vlm.config, "text_config", None)
        if text_config is None or not hasattr(text_config, "hidden_size"):
            raise ValueError(
                "QwenFlowMatchingVLA requires a Qwen-style VLM config with "
                "text_config.hidden_size."
            )
        if getattr(self.vlm.config, "image_token_id", None) is None:
            raise ValueError(
                "QwenFlowMatchingVLA requires the backbone config to define "
                "image_token_id."
            )

        backbone_dim = text_config.hidden_size
        # Every expert architecture value comes from the checkpointed config.
        self.action_expert = ActionExpert(
            language_dim=backbone_dim,
            visual_dim=backbone_dim,
            **config.action_expert_cfg,
        )
        action_expert_dim = config.action_expert_cfg["dim"]
        self.state_encoder = nn.Sequential(
            nn.Linear(self.past_traj_sampling.num_poses * 2, action_expert_dim),
            nn.SiLU(),
            nn.Linear(action_expert_dim, action_expert_dim),
        )
        self.action_in_proj = PerWaypointActionInProjV2(
            in_dims=(self.pred_traj_sampling.num_poses, 2),
            out_dim=action_expert_dim,
        )
        self.action_out_proj = nn.Linear(action_expert_dim, 2)
        self.diffusion = FlowMatching(
            input_dims=(self.pred_traj_sampling.num_poses, 2),
            **config.diff_cfg,
        )

        self.traj_normalization = config.traj_normalization
        self.action_representation = config.action_representation
        self.ce_loss_weight = config.ce_loss_weight
        self.qa_loss_weight = config.qa_loss_weight
        self.diff_loss_weight = config.diff_loss_weight
        self.use_cape = config.action_expert_cfg["use_cape"]

        # Fourier features remain fp32; the large expert follows the Qwen dtype.
        expert_dtype = self.vlm.dtype
        self.action_expert.to(dtype=expert_dtype)
        self.state_encoder.to(dtype=expert_dtype)
        self.action_out_proj.to(dtype=expert_dtype)
        self.post_init()

    def forward(
        self,
        input_ids: torch.Tensor,
        attention_mask: torch.Tensor,
        pixel_values: torch.Tensor,
        image_grid_thw: torch.Tensor,
        labels: torch.Tensor | None = None,
        init_vel: torch.Tensor | None = None,
        proprio_state: torch.Tensor | None = None,
        visual_camera_poses: torch.Tensor | None = None,
        visual_patch_coords: torch.Tensor | None = None,
        visual_time_ids: torch.Tensor | None = None,
        reference_camera_pose: torch.Tensor | None = None,
        **kwargs,
    ) -> ModelOutput:
        """Run only the Qwen backbone forward path.

        The expert-only arguments are accepted so a full collator dictionary can
        be passed to this model, but they are deliberately not forwarded to Qwen.
        Joint continuous training is implemented in ``compute_loss``.

        Args:
            input_ids: Token IDs shaped ``[B, sequence]``.
            attention_mask: Valid Qwen tokens shaped ``[B, sequence]``.
            pixel_values: Images after Qwen processor patchification.
            image_grid_thw: Per-image temporal/height/width patch grids.
            labels: Optional next-token targets; ignored positions equal ``-100``.
            init_vel: Expert-only current speed estimate.
            proprio_state: Expert-only flattened past trajectory ``[B, 2P]``.
            visual_camera_poses: Token-aligned camera poses for CaPE.
            visual_patch_coords: Token-aligned normalized image coordinates.
            visual_time_ids: Token-aligned image-frame indices.
            reference_camera_pose: Latest FRONT camera pose for expert queries.
            **kwargs: Additional arguments supported by the Qwen backbone.

        Returns:
            The standard Qwen ``ModelOutput``.
        """
        return self.vlm(
            input_ids=input_ids,
            attention_mask=attention_mask,
            pixel_values=pixel_values,
            image_grid_thw=image_grid_thw,
            labels=labels,
            **kwargs,
        )

    @staticmethod
    def _pack_values(
        values: torch.Tensor,
        token_mask: torch.Tensor,
    ) -> tuple[torch.Tensor, torch.Tensor]:
        """Select tokens and right-align each variable-length batch item.

        Args:
            values: Token-aligned tensor shaped ``[B, sequence, ...]``. Trailing
                dimensions may contain hidden features, poses, or coordinates.
            token_mask: Boolean selection mask shaped ``[B, sequence]``.

        Returns:
            A pair ``(packed_values, packed_mask)``. Both use the longest selected
            length in the batch, with shorter examples padded on the left.
        """
        if values.ndim < 2:
            raise ValueError("values must have batch and sequence dimensions.")
        if token_mask.shape != values.shape[:2]:
            raise ValueError(
                "token_mask must match values' first two dimensions, got "
                f"{token_mask.shape} and {values.shape[:2]}."
            )
        token_mask = token_mask.to(device=values.device, dtype=torch.bool)
        lengths = token_mask.sum(dim=-1)
        max_length = int(lengths.max().item()) if lengths.numel() else 0
        batch_size = values.shape[0]
        packed = values.new_zeros((batch_size, max_length, *values.shape[2:]))
        packed_mask = torch.zeros(
            batch_size, max_length, device=values.device, dtype=torch.bool
        )

        if max_length:
            token_rank = token_mask.long().cumsum(dim=-1) - 1
            destination = max_length - lengths[:, None] + token_rank
            batch_indices = torch.arange(batch_size, device=values.device)[
                :, None
            ].expand_as(token_mask)
            packed[batch_indices[token_mask], destination[token_mask]] = values[
                token_mask
            ]
            packed_mask = (
                torch.arange(max_length, device=values.device)[None]
                >= max_length - lengths[:, None]
            )
        return packed, packed_mask

    def _split_backbone_context(
        self,
        last_hidden_states: torch.Tensor,
        input_ids: torch.Tensor,
        attention_mask: torch.Tensor,
        labels: torch.Tensor | None,
        visual_camera_poses: torch.Tensor,
        visual_patch_coords: torch.Tensor,
        visual_time_ids: torch.Tensor,
    ) -> tuple[
        torch.Tensor,
        torch.Tensor,
        torch.Tensor,
        torch.Tensor,
        torch.Tensor,
        torch.Tensor,
        torch.Tensor,
    ]:
        """Create fixed final-layer language and visual memories for the expert.

        During training, ``labels == -100`` is the prompt boundary. Applying it
        before the text/image split guarantees that continuous action queries can
        never attend to teacher-forced reasoning or discrete action outputs.

        Args:
            last_hidden_states: Qwen final-layer states ``[B, sequence, D_vlm]``.
            input_ids: Token IDs aligned with the hidden states.
            attention_mask: Valid-token mask aligned with the hidden states.
            labels: Optional SFT labels. Positions unequal to ``-100`` are
                assistant targets and are removed from both expert memories.
            visual_camera_poses: Full-sequence poses ``[B, sequence, 4, 4]``.
            visual_patch_coords: Full-sequence ray coordinates
                ``[B, sequence, 2]``.
            visual_time_ids: Full-sequence image times ``[B, sequence]``.

        Returns:
            Seven packed tensors: language states/mask, visual states/mask,
            visual poses, visual coordinates, and visual time IDs. Geometry is
            selected using the exact same image-token mask as visual states.
        """
        expected_shape = last_hidden_states.shape[:2]
        if input_ids.shape != expected_shape or attention_mask.shape != expected_shape:
            raise ValueError("Qwen hidden states, IDs, and masks must be aligned.")
        for name, values in (
            ("visual_camera_poses", visual_camera_poses),
            ("visual_patch_coords", visual_patch_coords),
            ("visual_time_ids", visual_time_ids),
        ):
            if values.shape[:2] != expected_shape:
                raise ValueError(f"{name} must align with the Qwen token sequence.")

        prompt_mask = attention_mask.to(dtype=torch.bool)
        if labels is not None:
            if labels.shape != input_ids.shape:
                raise ValueError("labels must have the same shape as input_ids.")
            prompt_mask = prompt_mask & labels.eq(-100)

        image_tokens = input_ids.eq(self.vlm.config.image_token_id)
        image_mask = prompt_mask & image_tokens
        language_mask = prompt_mask & ~image_tokens

        language_states, language_attention_mask = self._pack_values(
            last_hidden_states, language_mask
        )
        visual_states, visual_attention_mask = self._pack_values(
            last_hidden_states, image_mask
        )
        packed_poses, _ = self._pack_values(visual_camera_poses, image_mask)
        packed_coords, _ = self._pack_values(visual_patch_coords, image_mask)
        packed_times, _ = self._pack_values(visual_time_ids, image_mask)

        # Padded poses must stay invertible even though their attention mask is false.
        if packed_poses.numel():
            identity = torch.eye(
                4, dtype=packed_poses.dtype, device=packed_poses.device
            )
            packed_poses = torch.where(
                visual_attention_mask[..., None, None], packed_poses, identity
            )
        return (
            language_states,
            language_attention_mask,
            visual_states,
            visual_attention_mask,
            packed_poses,
            packed_coords,
            packed_times,
        )

    @staticmethod
    def _repeat(tensor: torch.Tensor, repeat_factor: int) -> torch.Tensor:
        """Repeat each batch item for FM training repeats or sampled trajectories.

        Args:
            tensor: Any tensor whose first dimension is the scene batch.
            repeat_factor: Number of consecutive copies per scene.

        Returns:
            Tensor with first dimension multiplied by ``repeat_factor``.
        """
        if repeat_factor == 1:
            return tensor
        return tensor.repeat_interleave(repeat_factor, dim=0)

    def _predict_vector_field(
        self,
        x: torch.Tensor,
        t: torch.Tensor,
        proprio_state: torch.Tensor,
        language_states: torch.Tensor,
        language_mask: torch.Tensor,
        visual_states: torch.Tensor,
        visual_mask: torch.Tensor,
        visual_camera_poses: torch.Tensor,
        visual_patch_coords: torch.Tensor,
        visual_time_ids: torch.Tensor,
        reference_camera_pose: torch.Tensor,
    ) -> torch.Tensor:
        """Predict one flow-matching vector field evaluation.

        Args:
            x: Noisy continuous actions ``x_t`` shaped ``[B, A, 2]``.
            t: Denoising times in the shape expected by the Fourier action input
                projector, normally one scalar per batch item.
            proprio_state: Flattened, resampled past XY trajectory ``[B, 2P]``.
            language_states: Packed final-layer language memory ``[B, L, D_vlm]``.
            language_mask: Valid language tokens ``[B, L]``.
            visual_states: Packed final-layer visual memory ``[B, V, D_vlm]``.
            visual_mask: Valid visual tokens ``[B, V]``.
            visual_camera_poses: Source pose per visual token ``[B, V, 4, 4]``.
            visual_patch_coords: Normalized ray per visual token ``[B, V, 2]``.
            visual_time_ids: Relative frame index per visual token ``[B, V]``.
            reference_camera_pose: Latest FRONT pose ``[B, 4, 4]``.

        Returns:
            Predicted flow velocity shaped ``[B, A, 2]`` in fp32.
        """
        action_input_dtype = next(self.action_in_proj.parameters()).dtype
        action_tokens = self.action_in_proj(
            x.to(dtype=action_input_dtype),
            t.to(dtype=action_input_dtype),
        )

        expert_dtype = self.action_expert.final_norm.weight.dtype
        normalized_state = (
            proprio_state.to(dtype=expert_dtype) / self.traj_normalization
        )
        state_tokens = self.state_encoder(normalized_state).unsqueeze(1)
        expert_output = self.action_expert(
            action_tokens.to(dtype=expert_dtype),
            state_tokens,
            language_states.to(dtype=expert_dtype),
            visual_states.to(dtype=expert_dtype),
            language_mask=language_mask,
            visual_mask=visual_mask,
            visual_camera_poses=visual_camera_poses,
            reference_camera_pose=reference_camera_pose,
            visual_patch_coords=visual_patch_coords,
            visual_time_ids=visual_time_ids,
        )
        # ODE integration and the FM regression loss remain fp32.
        return self.action_out_proj(expert_output).float()

    def _get_gt_trajectories(self, samples: list[E2EDataSample]) -> torch.Tensor:
        """Resample selected ground-truth trajectories to the model output grid.

        Args:
            samples: Driving samples containing ``fut_traj`` and
                ``fut_traj_sampling``.

        Returns:
            Future XY trajectories shaped ``[B, A, 2]`` on the model device.
        """
        trajectories = []
        for sample in samples:
            if sample.fut_traj is None or sample.fut_traj_sampling is None:
                raise ValueError(
                    "QwenFlowMatchingVLA training requires future trajectories "
                    "and their sampling metadata."
                )
            trajectories.append(
                resample_tensor(
                    sample.fut_traj,
                    sample.fut_traj_sampling,
                    self.pred_traj_sampling,
                )
            )
        return torch.stack(trajectories).to(self.device)

    def _to_diffusion_target(
        self,
        gt_trajs: torch.Tensor,
        initial_velocity: torch.Tensor,
    ) -> torch.Tensor:
        """Convert XY supervision into the configured flow-matching target.

        Args:
            gt_trajs: Ground-truth XY waypoints shaped ``[B, A, 2]``.
            initial_velocity: Present scalar ego speed shaped ``[B]``.

        Returns:
            Either normalized XY waypoints or acceleration/curvature controls,
            always shaped ``[B, A, 2]``.
        """
        if self.action_representation == "trajs":
            return gt_trajs / self.traj_normalization
        return traj_to_action(
            gt_trajs,
            dt=self.pred_traj_sampling.interval_length,
            initial_velocity=initial_velocity,
        )

    @staticmethod
    def _expert_inputs(model_inputs: dict[str, Any]) -> dict[str, torch.Tensor]:
        """Extract and validate collator fields consumed only by the expert.

        Args:
            model_inputs: Complete dictionary produced by this model's collator.

        Returns:
            Dictionary containing precisely ``_EXPERT_INPUT_KEYS``.
        """
        missing = _EXPERT_INPUT_KEYS - model_inputs.keys()
        if missing:
            raise ValueError(
                "The flow-matching collator did not provide: "
                + ", ".join(sorted(missing))
            )
        return {key: model_inputs[key] for key in _EXPERT_INPUT_KEYS}

    @staticmethod
    def _token_ce_for_group(
        logits: torch.Tensor,
        labels: torch.Tensor,
        sample_mask: torch.Tensor,
    ) -> torch.Tensor:
        """Compute causal-LM token loss for one subset of batch rows.

        Args:
            logits: Backbone vocabulary logits for supervised causal positions,
                shaped ``[B, targets, vocab]``.
            labels: Targets aligned with ``logits`` and shaped ``[B, targets]``;
                ignored positions equal ``-100``.
            sample_mask: Boolean row selector shaped ``[B]``.

        Returns:
            Mean next-token loss over supervised tokens in selected rows. An
            empty group returns a differentiable scalar zero.
        """
        sample_mask = sample_mask.to(device=logits.device, dtype=torch.bool)
        if sample_mask.shape != (logits.shape[0],):
            raise ValueError("sample_mask must contain one value per batch row.")
        if labels.shape != logits.shape[:2]:
            raise ValueError("labels must align with the first two logits dimensions.")
        if not sample_mask.any():
            return logits[..., 0].sum() * 0.0

        labels = labels.to(logits.device)
        token_mask = sample_mask[:, None] & labels.ne(-100)
        token_indices = token_mask.reshape(-1).nonzero(as_tuple=True)[0]
        if token_indices.numel() == 0:
            raise ValueError("Selected loss group contains no supervised tokens.")

        # Select only assistant target positions, never complete sequence rows.
        # This keeps the gathered vocabulary logits in the backbone's BF16 dtype
        # and avoids PyTorch's large-tensor CUDA indexing path.
        selected_logits = logits.reshape(-1, logits.shape[-1]).index_select(
            0, token_indices
        )
        selected_labels = labels.reshape(-1).index_select(0, token_indices)
        return F.cross_entropy(selected_logits, selected_labels)

    def compute_loss(self, batch: E2EDataBatch) -> dict[str, torch.Tensor]:
        """Compute text cross entropy and continuous flow-matching regression.

        Qwen runs once on the full teacher-forced conversation. QA rows train
        only that backbone. For driving rows, its final hidden layer supplies
        prompt-only memories to the action expert without being detached, so the
        continuous loss also trains the backbone.

        Args:
            batch: Output of ``QwenFlowMatchingVLACollator`` containing Qwen SFT
                inputs, past trajectory state, initial speed, and camera metadata.

        Returns:
            Dictionary with the weighted total and diagnostic CE, QA CE,
            driving CE, diffusion loss, and QA fraction scalars.
        """
        expert_inputs = self._expert_inputs(batch.model_inputs)
        model_inputs = {
            key: value
            for key, value in batch.model_inputs.items()
            if key not in _EXPERT_INPUT_KEYS
        }
        labels = model_inputs.pop("labels", None)
        if labels is None:
            raise ValueError("Joint training requires text trajectory labels.")

        if batch.qa_mask is None:
            raise ValueError("Joint training requires batch.qa_mask from the collator.")
        qa_mask = batch.qa_mask.to(device=labels.device, dtype=torch.bool)
        if qa_mask.shape != (batch.batch_size,):
            raise ValueError("qa_mask must contain one flag per batch sample.")
        driving_mask = ~qa_mask
        future_mask = torch.tensor(
            [
                sample.fut_traj is not None
                and sample.fut_traj.numel() > 0
                and sample.fut_traj_sampling is not None
                for sample in batch.samples
            ],
            dtype=torch.bool,
            device=labels.device,
        )
        action_mask = driving_mask & future_mask
        qa_fraction = qa_mask.float().mean()
        driving_fraction = driving_mask.float().mean()
        action_fraction = action_mask.float().mean()

        # Batches without action targets do not need the memory-heavy hidden states.
        model_inputs["output_hidden_states"] = bool(action_mask.any())
        model_inputs["use_cache"] = False
        shifted_labels = labels[:, 1:]
        logit_positions = shifted_labels.ne(-100).any(dim=0).nonzero(as_tuple=True)[0]
        if logit_positions.numel() == 0:
            raise ValueError("Joint training batch contains no supervised tokens.")
        # Qwen accepts explicit sequence positions here. Computing the LM head
        # only at positions that predict an assistant target avoids materializing
        # prompt/image logits and, with labels withheld, its full FP32 loss copy.
        model_inputs["logits_to_keep"] = logit_positions
        vlm_outputs = self.vlm(**model_inputs)
        target_labels = shifted_labels.index_select(1, logit_positions)

        # Average the two task losses by row fraction so short QA answers are
        # not drowned out by longer trajectory strings. The same helper handles
        # driving-only batches without changing the token-mean objective.
        if qa_mask.any():
            qa_ce_loss = self._token_ce_for_group(
                vlm_outputs.logits, target_labels, qa_mask
            )
            drive_ce_loss = self._token_ce_for_group(
                vlm_outputs.logits, target_labels, driving_mask
            )
            ce_loss = (
                driving_fraction * drive_ce_loss
                + self.qa_loss_weight * qa_fraction * qa_ce_loss
            )
        else:
            ce_loss = self._token_ce_for_group(
                vlm_outputs.logits, target_labels, driving_mask
            )
            drive_ce_loss = ce_loss
            qa_ce_loss = ce_loss.detach().new_zeros(())

        diff_loss = ce_loss.detach().new_zeros(())
        if action_mask.any():
            if not vlm_outputs.hidden_states:
                raise RuntimeError("The Qwen backbone did not return hidden states.")
            (
                language_states,
                language_mask,
                visual_states,
                visual_mask,
                visual_poses,
                visual_coords,
                visual_times,
            ) = self._split_backbone_context(
                vlm_outputs.hidden_states[-1][action_mask],
                model_inputs["input_ids"][action_mask],
                model_inputs["attention_mask"][action_mask],
                labels[action_mask],
                expert_inputs["visual_camera_poses"][action_mask],
                expert_inputs["visual_patch_coords"][action_mask],
                expert_inputs["visual_time_ids"][action_mask],
            )

            initial_velocity = expert_inputs["init_vel"][action_mask].to(self.device)
            action_samples = [
                sample
                for sample, has_action in zip(
                    batch.samples, action_mask.detach().cpu().tolist()
                )
                if has_action
            ]
            gt_trajs = self._get_gt_trajectories(action_samples)
            diffusion_target = self._to_diffusion_target(gt_trajs, initial_velocity)
            diffusion_data = self.diffusion.construct_training_data(diffusion_target)

            action_batch_size = int(action_mask.sum().item())
            repeated_batch_size = diffusion_data["noisy_x"].shape[0]
            if repeated_batch_size % action_batch_size:
                raise RuntimeError(
                    "Diffusion repeat count is not divisible by action batch size."
                )
            repeat_factor = repeated_batch_size // action_batch_size
            repeated = [
                self._repeat(value, repeat_factor)
                for value in (
                    expert_inputs["proprio_state"][action_mask],
                    language_states,
                    language_mask,
                    visual_states,
                    visual_mask,
                    visual_poses,
                    visual_coords,
                    visual_times,
                    expert_inputs["reference_camera_pose"][action_mask],
                )
            ]

            pred = self._predict_vector_field(
                diffusion_data["noisy_x"],
                diffusion_data["timesteps"],
                *repeated,
            )
            diff_loss = self.diffusion.compute_loss_from_pred(diffusion_data, pred)

        # Rows without valid action targets contribute zero diffusion loss.
        loss = self.ce_loss_weight * ce_loss + (
            self.diff_loss_weight * action_fraction * diff_loss
        )
        return {
            "loss": loss,
            "ce_loss": ce_loss,
            "drive_ce_loss": drive_ce_loss,
            "qa_ce_loss": qa_ce_loss,
            "diff_loss": diff_loss,
            "qa_fraction": qa_fraction,
        }

    @staticmethod
    def _pad_selected_metadata(
        values: torch.Tensor,
        selections: list[torch.Tensor],
        padded_length: int,
        identity_fill: bool = False,
    ) -> torch.Tensor:
        """Apply prompt selection and tokenizer-style left padding to metadata.

        Args:
            values: Full-sequence metadata shaped ``[B, sequence, ...]``.
            selections: One boolean prompt-selection mask per batch item.
            padded_length: Target length chosen by tokenizer padding.
            identity_fill: Fill padding with 4x4 identity matrices instead of
                zeros. Pose padding must be invertible even when masked out.

        Returns:
            Selected, left-padded metadata shaped ``[B, padded_length, ...]``.
        """
        output = values.new_zeros((len(selections), padded_length, *values.shape[2:]))
        if identity_fill:
            output[:] = torch.eye(4, dtype=values.dtype, device=values.device)
        for batch_index, selection in enumerate(selections):
            selected = values[batch_index, selection]
            if selected.shape[0]:
                output[batch_index, -selected.shape[0] :] = selected
        return output

    def predict_from_processed_inputs(
        self,
        input_ids: torch.Tensor,
        pixel_values: torch.Tensor,
        image_grid_thw: torch.Tensor,
        init_vel: torch.Tensor,
        proprio_state: torch.Tensor,
        visual_camera_poses: torch.Tensor,
        visual_patch_coords: torch.Tensor,
        visual_time_ids: torch.Tensor,
        reference_camera_pose: torch.Tensor,
        attention_mask: torch.Tensor | None = None,
        labels: torch.Tensor | None = None,
        num_traj_samples: int = 1,
        target_sampling: TrajectorySampling | None = None,
        **kwargs,
    ) -> tuple[torch.Tensor, dict[str, Any]]:
        """Sample continuous trajectories from already processed model inputs.

        Teacher-forced assistant tokens are removed before Qwen is rerun, keeping
        inference conditioning identical to a real prompt. Metadata is filtered
        and padded by the same masks, then the flow ODE is integrated once for
        each requested trajectory sample.

        Args:
            input_ids: Possibly teacher-forced token IDs ``[B, sequence]``.
            pixel_values: Qwen-processed image patches.
            image_grid_thw: Temporal/height/width patch grid for each image.
            init_vel: Present scalar ego speeds ``[B]``.
            proprio_state: Flattened past XY trajectory ``[B, 2P]``.
            visual_camera_poses: Full-token camera poses
                ``[B, sequence, 4, 4]``.
            visual_patch_coords: Full-token normalized rays
                ``[B, sequence, 2]``.
            visual_time_ids: Full-token relative frame IDs ``[B, sequence]``.
            reference_camera_pose: Latest FRONT pose ``[B, 4, 4]``.
            attention_mask: Optional valid-token mask ``[B, sequence]``.
            labels: Optional SFT labels used to identify prompt tokens.
            num_traj_samples: Number of independently initialized FM samples per
                scene.
            target_sampling: Optional output grid to which trajectories are
                resampled after prediction.
            **kwargs: Extra caller arguments accepted for planner compatibility.

        Returns:
            ``(pred_trajs, extras)`` where trajectories have shape
            ``[B * num_traj_samples, output_steps, 2]`` and ``extras`` is empty.
        """
        if num_traj_samples < 1:
            raise ValueError("num_traj_samples must be at least one.")

        if attention_mask is None:
            valid_mask = input_ids.ne(self.processor.tokenizer.pad_token_id)
        else:
            valid_mask = attention_mask.to(dtype=torch.bool)
        if labels is not None:
            valid_mask = valid_mask & labels.eq(-100)
        selections = [valid_mask[index] for index in range(input_ids.shape[0])]
        prompt_ids = [
            input_ids[index, selection] for index, selection in enumerate(selections)
        ]

        gen_inputs = self.processor.tokenizer.pad(
            {"input_ids": prompt_ids}, padding=True, return_tensors="pt"
        )
        padded_length = gen_inputs["input_ids"].shape[1]
        prompt_poses = self._pad_selected_metadata(
            visual_camera_poses, selections, padded_length, identity_fill=True
        )
        prompt_coords = self._pad_selected_metadata(
            visual_patch_coords, selections, padded_length
        )
        prompt_times = self._pad_selected_metadata(
            visual_time_ids, selections, padded_length
        )

        # Qwen's multimodal RoPE path expects at least one padding position.
        if gen_inputs["attention_mask"].bool().all():
            gen_inputs["input_ids"] = F.pad(
                gen_inputs["input_ids"],
                (1, 0),
                value=self.processor.tokenizer.pad_token_id,
            )
            gen_inputs["attention_mask"] = F.pad(
                gen_inputs["attention_mask"], (1, 0), value=0
            )
            prompt_poses = F.pad(prompt_poses, (0, 0, 0, 0, 1, 0))
            prompt_poses[:, 0] = torch.eye(
                4, dtype=prompt_poses.dtype, device=prompt_poses.device
            )
            prompt_coords = F.pad(prompt_coords, (0, 0, 1, 0))
            prompt_times = F.pad(prompt_times, (1, 0))

        gen_inputs.update(
            {
                "pixel_values": pixel_values,
                "image_grid_thw": image_grid_thw,
                # Qwen3.5 in transformers >= 5.18 builds its multimodal RoPE from these; without
                # them image tokens silently get 1D text positions (see grpo_module.py).
                "mm_token_type_ids": (
                    gen_inputs["input_ids"] == self.vlm.config.image_token_id
                ).to(torch.int),
                "output_hidden_states": True,
                "use_cache": False,
            }
        )
        gen_inputs = {
            key: value.to(self.device) if isinstance(value, torch.Tensor) else value
            for key, value in gen_inputs.items()
        }
        prompt_poses = prompt_poses.to(self.device)
        prompt_coords = prompt_coords.to(self.device)
        prompt_times = prompt_times.to(self.device)

        with torch.no_grad():
            vlm_outputs = self.vlm(**gen_inputs)
        if not vlm_outputs.hidden_states:
            raise RuntimeError("The Qwen backbone did not return hidden states.")
        (
            language_states,
            language_mask,
            visual_states,
            visual_mask,
            visual_poses,
            visual_coords,
            visual_times,
        ) = self._split_backbone_context(
            vlm_outputs.hidden_states[-1],
            gen_inputs["input_ids"],
            gen_inputs["attention_mask"],
            None,
            prompt_poses,
            prompt_coords,
            prompt_times,
        )

        repeated = [
            self._repeat(value, num_traj_samples)
            for value in (
                proprio_state.to(self.device),
                language_states,
                language_mask,
                visual_states,
                visual_mask,
                visual_poses,
                visual_coords,
                visual_times,
                reference_camera_pose.to(self.device),
            )
        ]

        def step_fn(x: torch.Tensor, t: torch.Tensor) -> torch.Tensor:
            return self._predict_vector_field(x, t, *repeated)

        sample_batch_size = input_ids.shape[0] * num_traj_samples
        with torch.no_grad():
            pred = self.diffusion.sample(
                batch_size=sample_batch_size,
                step_fn=step_fn,
                device=self.device,
                return_all_steps=False,
            )

        if self.action_representation == "trajs":
            pred_trajs = pred * self.traj_normalization
        else:
            repeated_velocity = init_vel.to(self.device).repeat_interleave(
                num_traj_samples, dim=0
            )
            pred_trajs = action_to_traj(
                pred,
                dt=self.pred_traj_sampling.interval_length,
                initial_velocity=repeated_velocity,
            )
        return self._maybe_resample(pred_trajs, target_sampling), {}

    @torch.inference_mode()
    def predict_qa_answers(
        self,
        qa_batch: list[tuple[E2EDataSample, QuestionAnswerPair]],
        max_new_tokens: int = 16,
    ) -> list[str | None]:
        """Greedily answer one deterministic batch of NuScenesQA questions.

        This is intentionally a backbone-only path: QA validation does not
        construct continuous-action inputs or execute the action expert. The
        prompt is built by the same helper used for QA supervision, with only
        the teacher-forced assistant turn removed.

        The tokenizer and nested Qwen config can disagree about the end token
        after a wrapped model is restored. Passing both known IDs explicitly
        makes generation stop on ``<|im_end|>`` as it did during supervision.
        """
        if max_new_tokens < 1:
            raise ValueError("max_new_tokens must be at least one.")
        if not qa_batch:
            return []

        self.processor.tokenizer.padding_side = "left"
        messages = [
            self.prompt_builder.build_qa_message(
                sample, qa_pair, self.num_past_image_frames
            )[:-1]
            for sample, qa_pair in qa_batch
        ]
        inputs = self.processor.apply_chat_template(
            messages,
            tokenize=True,
            add_generation_prompt=True,
            continue_final_message=False,
            return_dict=True,
            return_tensors="pt",
            padding=True,
        ).to(self.device)

        text_config = getattr(self.vlm.config, "text_config", None)
        eos_token_ids: list[int] = []
        for eos_token_id_config in (
            self.processor.tokenizer.eos_token_id,
            getattr(text_config, "eos_token_id", None),
            getattr(self.vlm.config, "eos_token_id", None),
        ):
            configured_ids = (
                eos_token_id_config
                if isinstance(eos_token_id_config, (list, tuple))
                else (eos_token_id_config,)
            )
            for eos_token_id in configured_ids:
                if eos_token_id is not None and eos_token_id not in eos_token_ids:
                    eos_token_ids.append(int(eos_token_id))
        if not eos_token_ids:
            raise ValueError("The Qwen tokenizer/config defines no EOS token ID.")

        generated_ids = self.vlm.generate(
            **inputs,
            do_sample=False,
            max_new_tokens=max_new_tokens,
            use_cache=True,
            eos_token_id=(
                eos_token_ids[0] if len(eos_token_ids) == 1 else eos_token_ids
            ),
            pad_token_id=self.processor.tokenizer.pad_token_id,
        )
        new_token_ids = generated_ids[:, inputs["input_ids"].shape[1] :]
        raw_predictions = self.processor.batch_decode(
            new_token_ids, skip_special_tokens=True
        )
        return [
            normalize_qa_prediction(prediction, qa_pair.allowed_answers)
            for prediction, (_, qa_pair) in zip(raw_predictions, qa_batch)
        ]

    def get_collator(
        self,
        drop_cfg: dict[str, float] | None = None,
        inference_mode: bool = False,
    ) -> Callable:
        """Construct a collator with this model's temporal and camera settings.

        Args:
            drop_cfg: Optional prompt-input dropout probabilities.
            inference_mode: Omit teacher-forced assistant content when ``True``.

        Returns:
            Configured ``QwenFlowMatchingVLACollator``.
        """
        return QwenFlowMatchingVLACollator(
            self.processor,
            self.prompt_builder,
            self.past_traj_sampling,
            self.pred_traj_sampling,
            self.camera_sequence,
            self.num_past_image_frames,
            drop_cfg=drop_cfg,
            inference_mode=inference_mode,
            concat_cameras=self.concat_cameras,
            use_cape=self.use_cape,
        )


def _qa_only(sample: E2EDataSample) -> bool:
    """A question row whose dataset has no ego past (NuPlanQA). Its state input is zeros: the
    expert never trains on QA rows, and a driving row without a past is still an error."""
    return sample.qa_pair is not None and "past_traj" not in sample.agent_input


class QwenFlowMatchingVLACollator(TextTrajCollator):
    """Build joint SFT, state, and optional camera-geometry inputs.

    With CaPE enabled, every merged Qwen image patch receives source-camera
    extrinsics, normalized pinhole coordinates, and a relative frame index;
    state/action queries always use the latest ``FRONT`` pose. With CaPE off,
    neutral placeholders preserve a stable batching interface while calibration
    is neither accessed nor required.

    ``concat_cameras`` changes only how geometry is constructed. Both layouts
    ultimately produce full-sequence metadata tensors, so downstream context
    packing and ``ActionExpert.forward`` are identical.

    Args:
        processor: Qwen multimodal processor/tokenizer.
        prompt_builder: Repository prompt builder used by ``TextTrajCollator``.
        past_traj_sampling: Temporal grid expected by the state encoder.
        pred_traj_sampling: Temporal grid used for textual and continuous targets.
        camera_sequence: Camera names and ordering in the prompt. The standard
            driving setup is FRONT_LEFT, FRONT, FRONT_RIGHT.
        num_past_image_frames: Number of most-recent camera frames to include.
        drop_cfg: Optional prompt-input dropout probabilities.
        inference_mode: Whether prompt construction omits assistant targets.
        concat_cameras: If ``True``, cameras at a timestamp form one wide image;
            otherwise each camera is a separate Qwen image.
        use_cape: Construct calibrated geometry when ``True``. When ``False``,
            return neutral placeholders without reading calibration.
    """

    def __init__(
        self,
        processor,
        prompt_builder,
        past_traj_sampling: TrajectorySampling,
        pred_traj_sampling: TrajectorySampling,
        camera_sequence: list[str],
        num_past_image_frames: int,
        drop_cfg: dict[str, float] | None = None,
        inference_mode: bool = False,
        concat_cameras: bool = True,
        use_cape: bool = True,
    ):
        super().__init__(
            processor=processor,
            prompt_builder=prompt_builder,
            past_traj_sampling=past_traj_sampling,
            pred_traj_sampling=pred_traj_sampling,
            camera_sequence=camera_sequence,
            num_past_image_frames=num_past_image_frames,
            drop_cfg=drop_cfg,
            inference_mode=inference_mode,
            concat_cameras=concat_cameras,
        )
        self.use_cape = use_cape

    def _resample_past_traj(self, sample: E2EDataSample) -> torch.Tensor:
        """Resample history and guarantee the state encoder's fixed length.

        Different datasets can expose different amounts of past context. The
        shared resampling utility intentionally refuses to extrapolate earlier
        than the oldest observation, so it can return fewer than the requested
        number of points. That variable length is useful in a text prompt, but
        it cannot be stacked into a batch or consumed by the expert's fixed-size
        state MLP.

        Missing *oldest* target timestamps are therefore filled by repeating the
        oldest available ego-relative position. This edge-padding leaves every
        observed/resampled point untouched, keeps the present at the final row,
        and represents the unknown earlier motion conservatively as stationary.

        Args:
            sample: Driving sample containing a past XY trajectory and its
                sampling frequency.

        Returns:
            Ego-relative XY history shaped exactly
            ``[past_traj_sampling.num_poses, 2]``.

        Raises:
            ValueError: If resampling unexpectedly returns no points or more
                points than the configured target length.
        """
        if _qa_only(sample):
            return torch.zeros(self.past_traj_sampling.num_poses, 2)
        past_traj = super()._resample_past_traj(sample)
        target_length = self.past_traj_sampling.num_poses
        actual_length = past_traj.shape[-2]

        if actual_length == 0:
            raise ValueError("Past-trajectory resampling returned no points.")
        if actual_length > target_length:
            raise ValueError(
                "Past-trajectory resampling exceeded its configured length: "
                f"got {actual_length}, expected at most {target_length}."
            )
        if actual_length == target_length:
            return past_traj

        # The missing timestamps precede all available history, so pad on the
        # left with the oldest known pose rather than shifting recent samples.
        padding = past_traj[..., :1, :].expand(
            *past_traj.shape[:-2],
            target_length - actual_length,
            past_traj.shape[-1],
        )
        return torch.cat((padding, past_traj), dim=-2)

    @staticmethod
    def _initial_velocity(sample: E2EDataSample) -> torch.Tensor:
        """Estimate present scalar speed from the last past-trajectory segment.

        Args:
            sample: Driving sample containing ``agent_input['past_traj']`` and
                ``metadata['sampling_freq']``.

        Returns:
            Scalar speed in trajectory-distance units per second. A sample with
            fewer than two past points receives zero.
        """
        if _qa_only(sample):
            return torch.zeros(())
        past_traj = sample.agent_input["past_traj"]
        if past_traj.shape[0] < 2:
            return past_traj.new_zeros(())
        displacement = past_traj[-1] - past_traj[-2]
        return displacement.norm() * sample.metadata["sampling_freq"]

    @staticmethod
    def _camera_pose(camera: CameraData, camera_name: str) -> torch.Tensor:
        """Build the repository's camera-to-ego homogeneous transform.

        Args:
            camera: Image and calibration for one named camera.
            camera_name: Human-readable name used in validation messages.

        Returns:
            Float32 pose ``P`` shaped ``[4, 4]`` satisfying
            ``point_ego = P @ point_camera`` in homogeneous coordinates.

        Raises:
            ValueError: If calibration is missing or still uses legacy FLU axes.
        """
        calibration = camera.cal_dict
        if calibration is None:
            raise ValueError(f"Camera {camera_name!r} has no calibration for CaPE.")
        if calibration.is_flu:
            raise ValueError(
                f"Camera {camera_name!r} uses legacy FLU axes; CaPE requires the "
                "repository's optical/RDF camera convention."
            )
        pose = torch.eye(4, dtype=torch.float32)
        pose[:3, :3] = torch.as_tensor(calibration.rotation, dtype=torch.float32)
        pose[:3, 3] = torch.as_tensor(calibration.translation, dtype=torch.float32)
        return pose

    @staticmethod
    def _normalized_patch_coords(
        camera: CameraData,
        u: torch.Tensor,
        v: torch.Tensor,
    ) -> torch.Tensor:
        """Convert pixel centers to pinhole-normalized image-plane coordinates.

        Args:
            camera: Source camera whose intrinsics define the rays.
            u: Horizontal pixel centers in that camera's local image.
            v: Vertical pixel centers in that camera's local image.

        Returns:
            Coordinates ``[(u-cx)/fx, (v-cy)/fy]`` shaped ``[..., 2]``. If
            intrinsics are absent, centered ``[-1, 1]`` image coordinates are
            used as a documented fallback.
        """
        height, width = camera.image.shape[:2]
        calibration = camera.cal_dict
        if calibration is not None and calibration.intrinsics is not None:
            intrinsics = torch.as_tensor(calibration.intrinsics, dtype=torch.float32)
            fx, fy = intrinsics[0, 0], intrinsics[1, 1]
            if fx.abs() < 1e-8 or fy.abs() < 1e-8:
                raise ValueError("Camera focal lengths must be non-zero for CaPE.")
            x = (u - intrinsics[0, 2]) / fx
            y = (v - intrinsics[1, 2]) / fy
        else:
            # A calibrated pinhole ray is unavailable; retain patch location.
            x = 2.0 * u / max(int(width), 1) - 1.0
            y = 2.0 * v / max(int(height), 1) - 1.0
        return torch.stack((x, y), dim=-1)

    def _single_camera_geometry(
        self,
        camera: CameraData,
        camera_name: str,
        grid_thw: torch.Tensor,
        time_id: int,
    ) -> tuple[torch.Tensor, torch.Tensor, torch.Tensor]:
        """Create token geometry for one separately processed camera image.

        Args:
            camera: Source camera image and calibration.
            camera_name: Name associated with ``camera``.
            grid_thw: Qwen pre-merge vision grid ``[time, height, width]``.
            time_id: Relative frame index; the newest frame is zero.

        Returns:
            ``(poses, coords, times)`` with one entry per merged Qwen visual
            token, ordered time-major then row-major exactly like Qwen.
        """
        merge_size = int(getattr(self.processor.image_processor, "merge_size", 2))
        grid_t, grid_h, grid_w = (int(value) for value in grid_thw.tolist())
        if grid_h % merge_size or grid_w % merge_size:
            raise ValueError("Qwen visual grid is not divisible by its merge size.")
        merged_h, merged_w = grid_h // merge_size, grid_w // merge_size
        rows = (
            torch.arange(merged_h)
            .view(1, merged_h, 1)
            .expand(grid_t, -1, merged_w)
            .reshape(-1)
        )
        cols = (
            torch.arange(merged_w)
            .view(1, 1, merged_w)
            .expand(grid_t, merged_h, -1)
            .reshape(-1)
        )
        height, width = camera.image.shape[:2]
        u = (cols.float() + 0.5) * float(width) / merged_w
        v = (rows.float() + 0.5) * float(height) / merged_h
        coords = self._normalized_patch_coords(camera, u, v)
        pose = self._camera_pose(camera, camera_name)
        poses = pose.view(1, 4, 4).expand(coords.shape[0], -1, -1).clone()
        times = torch.full((coords.shape[0],), time_id, dtype=torch.long)
        return poses, coords, times

    def _concatenated_camera_geometry(
        self,
        camera_frame: dict[str, CameraData],
        grid_thw: torch.Tensor,
        time_id: int,
    ) -> tuple[torch.Tensor, torch.Tensor, torch.Tensor]:
        """Create token geometry for a horizontally concatenated camera image.

        Qwen resizes and patchifies the complete wide image. Each merged patch
        is assigned back to a source camera using its horizontal center and the
        original camera widths. Its horizontal coordinate is then converted from
        wide-image space into that camera's local pixel space.

        Args:
            camera_frame: Mapping from configured camera names to images and
                calibration at one timestamp.
            grid_thw: Qwen pre-merge grid for the wide image.
            time_id: Relative frame index; the newest frame is zero.

        Returns:
            ``(poses, coords, times)`` with one token-aligned row per merged
            visual patch. Boundary-straddling patches use their center camera.
        """
        merge_size = int(getattr(self.processor.image_processor, "merge_size", 2))
        grid_t, grid_h, grid_w = (int(value) for value in grid_thw.tolist())
        if grid_h % merge_size or grid_w % merge_size:
            raise ValueError("Qwen visual grid is not divisible by its merge size.")
        merged_h, merged_w = grid_h // merge_size, grid_w // merge_size
        rows = (
            torch.arange(merged_h)
            .view(1, merged_h, 1)
            .expand(grid_t, -1, merged_w)
            .reshape(-1)
        )
        cols = (
            torch.arange(merged_w)
            .view(1, 1, merged_w)
            .expand(grid_t, merged_h, -1)
            .reshape(-1)
        )

        cameras = [camera_frame[name] for name in self.camera_sequence]
        heights = {int(camera.image.shape[0]) for camera in cameras}
        if len(heights) != 1:
            raise ValueError("Horizontally concatenated cameras must share a height.")
        widths = torch.tensor(
            [int(camera.image.shape[1]) for camera in cameras], dtype=torch.float32
        )
        boundaries = torch.cat((torch.zeros(1), widths.cumsum(0)))
        global_u = (cols.float() + 0.5) * boundaries[-1] / merged_w
        camera_indices = torch.bucketize(global_u, boundaries[1:-1])

        token_count = rows.numel()
        poses = torch.empty(token_count, 4, 4, dtype=torch.float32)
        coords = torch.empty(token_count, 2, dtype=torch.float32)
        for camera_index, (camera_name, camera) in enumerate(
            zip(self.camera_sequence, cameras)
        ):
            selected = camera_indices.eq(camera_index)
            local_u = global_u[selected] - boundaries[camera_index]
            local_v = (
                (rows[selected].float() + 0.5) * float(camera.image.shape[0]) / merged_h
            )
            coords[selected] = self._normalized_patch_coords(camera, local_u, local_v)
            poses[selected] = self._camera_pose(camera, camera_name)
        times = torch.full((token_count,), time_id, dtype=torch.long)
        return poses, coords, times

    def _expert_geometry(
        self,
        batch: list[E2EDataSample],
        input_ids: torch.Tensor,
        image_grid_thw: torch.Tensor,
        qa_mask: torch.Tensor,
    ) -> tuple[torch.Tensor, torch.Tensor, torch.Tensor, torch.Tensor]:
        """Align camera poses, rays, and frame IDs with expanded image tokens.

        The processor returns one ``image_grid_thw`` row per input image, while
        the language sequence contains many repeated image placeholder tokens.
        This method expands each grid into per-token geometry and writes it at
        the corresponding image-token positions.

        Args:
            batch: Original driving samples, including camera images/calibration.
            input_ids: Padded Qwen token IDs shaped ``[B, sequence]``.
            image_grid_thw: Concatenated list of Qwen image grids across the batch.
            qa_mask: Boolean vector identifying rows whose prompt contains all
                separately processed current-camera images. Driving rows retain
                the configured concatenated/separate and temporal layout.

        Returns:
            Four full-sequence tensors:

            * camera poses ``[B, sequence, 4, 4]``;
            * normalized patch rays ``[B, sequence, 2]``;
            * relative frame IDs ``[B, sequence]``;
            * latest FRONT reference poses ``[B, 4, 4]``.

            Non-image positions contain neutral identity/zero values and are
            removed later using the visual-token mask.
        """
        batch_size, sequence_length = input_ids.shape
        qa_mask = qa_mask.to(dtype=torch.bool, device="cpu")
        if qa_mask.shape != (batch_size,):
            raise ValueError("qa_mask must contain one flag per batch row.")
        identity = torch.eye(4, dtype=torch.float32)
        token_poses = (
            identity.view(1, 1, 4, 4)
            .expand(batch_size, sequence_length, -1, -1)
            .clone()
        )
        token_coords = torch.zeros(batch_size, sequence_length, 2)
        token_times = torch.zeros(batch_size, sequence_length, dtype=torch.long)
        reference_poses = []

        grid_index = 0
        image_token_id = self.processor.tokenizer.convert_tokens_to_ids("<|image_pad|>")
        configured_image_token_id = getattr(self.processor, "image_token_id", None)
        if configured_image_token_id is not None:
            image_token_id = configured_image_token_id

        for batch_index, sample in enumerate(batch):
            if not sample.cameras:
                raise ValueError("CaPE requires at least one camera frame.")
            latest_frame = sample.cameras[-1]
            if "FRONT" not in latest_frame:
                raise ValueError("CaPE always uses FRONT, but FRONT is absent.")
            reference_poses.append(self._camera_pose(latest_frame["FRONT"], "FRONT"))

            sample_poses, sample_coords, sample_times = [], [], []
            if qa_mask[batch_index]:
                # QA always supplies all latest views as separate images,
                # independently of the driving concat_cameras configuration.
                for camera_name in qa_cameras(latest_frame):
                    if grid_index >= image_grid_thw.shape[0]:
                        raise ValueError("Missing Qwen image grid metadata.")
                    geometry = self._single_camera_geometry(
                        latest_frame[camera_name],
                        camera_name,
                        image_grid_thw[grid_index],
                        time_id=0,
                    )
                    grid_index += 1
                    sample_poses.append(geometry[0])
                    sample_coords.append(geometry[1])
                    sample_times.append(geometry[2])
            else:
                num_frames = min(len(sample.cameras), self.num_past_image_frames)
                frame_indices = range(
                    len(sample.cameras) - num_frames, len(sample.cameras)
                )
                for local_frame_index, frame_index in enumerate(frame_indices):
                    time_id = local_frame_index - num_frames + 1
                    camera_frame = sample.cameras[frame_index]
                    if self.concat_cameras:
                        if grid_index >= image_grid_thw.shape[0]:
                            raise ValueError("Missing Qwen image grid metadata.")
                        geometry = self._concatenated_camera_geometry(
                            camera_frame, image_grid_thw[grid_index], time_id
                        )
                        grid_index += 1
                        sample_poses.append(geometry[0])
                        sample_coords.append(geometry[1])
                        sample_times.append(geometry[2])
                    else:
                        for camera_name in self.camera_sequence:
                            if camera_name not in camera_frame:
                                raise ValueError(
                                    f"Camera {camera_name!r} is absent from a frame."
                                )
                            if grid_index >= image_grid_thw.shape[0]:
                                raise ValueError("Missing Qwen image grid metadata.")
                            geometry = self._single_camera_geometry(
                                camera_frame[camera_name],
                                camera_name,
                                image_grid_thw[grid_index],
                                time_id,
                            )
                            grid_index += 1
                            sample_poses.append(geometry[0])
                            sample_coords.append(geometry[1])
                            sample_times.append(geometry[2])

            poses = torch.cat(sample_poses)
            coords = torch.cat(sample_coords)
            times = torch.cat(sample_times)
            image_positions = (
                input_ids[batch_index].eq(image_token_id).nonzero().flatten()
            )
            if image_positions.numel() != poses.shape[0]:
                raise ValueError(
                    "Expanded Qwen image-token count does not match camera geometry: "
                    f"{image_positions.numel()} tokens versus {poses.shape[0]} patches."
                )
            token_poses[batch_index, image_positions] = poses
            token_coords[batch_index, image_positions] = coords
            token_times[batch_index, image_positions] = times

        if grid_index != image_grid_thw.shape[0]:
            raise ValueError(
                f"Consumed {grid_index} image grids but Qwen returned "
                f"{image_grid_thw.shape[0]}."
            )
        return token_poses, token_coords, token_times, torch.stack(reference_poses)

    @staticmethod
    def _empty_expert_geometry(
        input_ids: torch.Tensor,
    ) -> tuple[torch.Tensor, torch.Tensor, torch.Tensor, torch.Tensor]:
        """Return neutral placeholders when camera positional encoding is off.

        Args:
            input_ids: Padded Qwen token IDs shaped ``[B, sequence]``; only their
                batch and sequence dimensions are used.

        Returns:
            Identity token poses, zero coordinates, zero time IDs, and identity
            reference poses with the same shapes as calibrated geometry.
        """
        batch_size, sequence_length = input_ids.shape
        identity = torch.eye(4, dtype=torch.float32)
        token_poses = (
            identity.view(1, 1, 4, 4)
            .expand(batch_size, sequence_length, -1, -1)
            .clone()
        )
        token_coords = torch.zeros(batch_size, sequence_length, 2)
        token_times = torch.zeros(batch_size, sequence_length, dtype=torch.long)
        reference_poses = identity.view(1, 4, 4).expand(batch_size, -1, -1).clone()
        return token_poses, token_coords, token_times, reference_poses

    def __call__(self, batch: list[E2EDataSample]) -> E2EDataBatch:
        """Collate Qwen inputs and attach continuous-expert conditioning.

        Args:
            batch: Driving samples, some of which may carry tied QA pairs.

        Returns:
            ``E2EDataBatch`` whose ``model_inputs`` contains ordinary Qwen fields
            plus:

            * ``proprio_state``: flattened resampled past XY trajectory;
            * ``init_vel``: present scalar speed estimate;
            * ``visual_camera_poses``: token-aligned source poses;
            * ``visual_patch_coords``: token-aligned normalized rays;
            * ``visual_time_ids``: token-aligned frame indices;
            * ``reference_camera_pose``: latest FRONT pose.

            ``qa_mask`` is stored on the batch, outside ``model_inputs``, and
            identifies rows supervised only through the Qwen backbone.
        """
        qa_mask = torch.tensor(
            [sample.qa_pair is not None for sample in batch], dtype=torch.bool
        )
        full_messages = [
            self.prompt_builder.build_qa_message(
                sample, sample.qa_pair, self.num_past_image_frames
            )
            if sample.qa_pair is not None
            else self._get_sample_message(sample)
            for sample in batch
        ]
        collated = self._collate_messages(batch, full_messages, qa_mask=qa_mask)
        model_inputs = collated.model_inputs
        (
            model_inputs["visual_camera_poses"],
            model_inputs["visual_patch_coords"],
            model_inputs["visual_time_ids"],
            model_inputs["reference_camera_pose"],
        ) = (
            # Do not touch camera calibration in the geometry-free baseline.
            self._expert_geometry(
                batch,
                model_inputs["input_ids"],
                model_inputs["image_grid_thw"],
                qa_mask,
            )
            if self.use_cape
            else self._empty_expert_geometry(model_inputs["input_ids"])
        )
        model_inputs["proprio_state"] = torch.stack(
            [self._resample_past_traj(sample).flatten() for sample in batch]
        )
        model_inputs["init_vel"] = torch.stack(
            [self._initial_velocity(sample) for sample in batch]
        )
        return collated
