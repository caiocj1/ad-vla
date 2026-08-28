import re
from collections.abc import Callable
from typing import Any

import torch
import torch.nn.functional as F
from omegaconf import DictConfig
from transformers import GenerationConfig, AutoProcessor
from transformers.modeling_outputs import ModelOutput

from ad_vla.dataset.data_types import (
    E2EDataBatch,
    E2EDataSample,
    QuestionAnswerPair,
    TrajectorySampling,
)
from ad_vla.dataset.nuscenes.nuscenes_types import _NUSCENES_QA_ANSWERS, _NUSCENES_QA_CAMERAS, _NUSCENES_QA_SYSTEM_PROMPT
from ad_vla.models.base_vla import BaseVLA, BaseVLAConfig
from ad_vla.models.base_traj_planner import BaseTrajPlanner
from ad_vla.utils.prompts import PromptBuilder, ALL_INPUTS
from ad_vla.utils.traj_utils import resample_tensor


_PAST_TRAJ_TEXT_PREFIX = "Past ego trajectory:"
_MISSING_PAST_TRAJ_TEXT = "[X.X, X.X]"


class TextTrajVLMConfig(BaseVLAConfig):
    model_type = "text_traj_vlm"

    def __init__(
        self,
        past_traj_sampling: dict | None = None,
        pred_traj_sampling: dict | None = None,
        **kwargs: Any,
    ):
        super().__init__(**kwargs)
        self.past_traj_sampling = past_traj_sampling or {
            "num_poses": 4,
            "interval_length": 0.5,
        }
        self.pred_traj_sampling = pred_traj_sampling or {
            "num_poses": 5,
            "interval_length": 1.0,
        }


class TextTrajVLM(BaseVLA, BaseTrajPlanner):
    config_class = TextTrajVLMConfig
    base_model_prefix = "vlm"

    def __init__(self, config, pretrained_modules=None):
        BaseVLA.__init__(self, config=config, pretrained_modules=pretrained_modules)
        BaseTrajPlanner.__init__(
            self,
            past_traj_sampling=config.past_traj_sampling,
            pred_traj_sampling=config.pred_traj_sampling,
        )

        self.post_init()

    def _generation_eos_token_id(self) -> int | list[int]:
        """Return every EOS token used by the tokenizer or nested VLM config."""
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
        return eos_token_ids[0] if len(eos_token_ids) == 1 else eos_token_ids

    @staticmethod
    def _trajectory_text_from_generation(text: str) -> str:
        """Select the trajectory payload from the first completed assistant turn."""
        assistant_turn = text.split("<|im_end|>", 1)[0]
        return assistant_turn.rsplit("trajectory", 1)[-1]

    def forward(
        self,
        input_ids: torch.Tensor,
        attention_mask: torch.Tensor,
        pixel_values: torch.Tensor,
        image_grid_thw: torch.Tensor,
        labels: torch.Tensor | None = None,
        **kwargs,
    ) -> ModelOutput:
        """
        Model forward. TextTrajVLM only uses the backbone VLM.

        Args:
            input_ids (torch.Tensor): Indices of input sequence tokens in the vocabulary.
                Shape: (batch_size, sequence_length)
            attention_mask (torch.Tensor): Mask to avoid performing attention on padding token indices.
                Mask values selected in [0, 1]: 1 for tokens that are **not masked**, 0 for tokens that are **masked**.
                Shape: (batch_size, sequence_length)
            pixel_values (torch.Tensor): The flattened patch embeddings for all patches in the batch.
                Shape: (total_patches, hidden_size) or equivalent flattened dimension.
            image_grid_thw (torch.Tensor): The spatial/temporal configuration for each image in the batch.
                Contains the (Time, Height, Width) grid dimensions for each image to reconstruct the 2D/3D structure.
                Shape: (num_images, 3)
            labels (Optional[torch.Tensor]): Labels for computing the masked language modeling loss.
                Indices should be in [-100, 0, ..., config.vocab_size].
                Tokens with indices set to -100 are ignored (masked) for loss computation.
                Shape: (batch_size, sequence_length)

        Returns:
            ModelOutput: The output object containing loss (if labels provided), logits,
                hidden_states, and Qwen-specific fields like rope_deltas.
        """
        return self.vlm(
            input_ids=input_ids,
            attention_mask=attention_mask,
            pixel_values=pixel_values,
            image_grid_thw=image_grid_thw,
            labels=labels,
            **kwargs,
        )

    def predict_from_processed_inputs(
        self,
        # Model-specific inputs
        input_ids: torch.Tensor,
        pixel_values: torch.Tensor,
        image_grid_thw: torch.Tensor,
        labels: torch.Tensor | None = None,
        # General inputs (see BaseTrajPlanner)
        num_traj_samples: int = 1,
        target_sampling: TrajectorySampling | None = None,
        generate_cfg: DictConfig | None = None,
        **kwargs,
    ) -> tuple[torch.Tensor, dict[str, Any]]:
        # 1. Prepare Prompts (extract input text, remove labels if present)
        if labels is not None:
            prompt_mask = (labels == -100) & (
                input_ids != self.processor.tokenizer.pad_token_id
            )
            prompt_ids = [input_ids[i][prompt_mask[i]] for i in range(len(prompt_mask))]
        else:
            prompt_ids = list(input_ids)

        # 2. Pad and prepare generation inputs, gets updated attention mask, then get img features
        gen_inputs = self.processor.tokenizer.pad(
            {"input_ids": prompt_ids},
            padding=True,
            return_tensors="pt",
        )

        if gen_inputs["attention_mask"].bool().all():
            gen_inputs["input_ids"] = F.pad(
                gen_inputs["input_ids"],
                (1, 0),
                value=self.processor.tokenizer.pad_token_id,
            )
            gen_inputs["attention_mask"] = F.pad(
                gen_inputs["attention_mask"], (1, 0), value=0
            )

        gen_inputs.update(
            {
                "pixel_values": pixel_values,
                "image_grid_thw": image_grid_thw,
            }
        )
        gen_inputs = {k: v.to(self.device) for k, v in gen_inputs.items()}

        # 3. Generate
        with torch.no_grad():
            gen_kwargs = {
                "max_new_tokens": 512,
                "num_beams": 1,
                "pad_token_id": self.processor.tokenizer.pad_token_id,
                "eos_token_id": self._generation_eos_token_id(),
                "do_sample": False,
                "return_dict_in_generate": True,
            }
            if generate_cfg is not None:
                gen_kwargs.update(
                    generate_cfg
                )  # update with user-provided generate_cfg, use self.vlm defaults otherwise
            if num_traj_samples > 1:
                gen_kwargs["do_sample"] = (
                    True  # enable sampling if multiple trajectories are requested
                )
            gen_kwargs["num_return_sequences"] = num_traj_samples
            gen_out = self.vlm.generate(
                **gen_inputs,
                generation_config=GenerationConfig(**gen_kwargs),
            )

        # 4. Decode
        generated_ids = gen_out.sequences
        input_ids = gen_inputs["input_ids"]
        generated_ids_trimmed = generated_ids[:, input_ids.shape[1] :]
        output_texts = self.processor.batch_decode(generated_ids_trimmed)

        # 5. Parse Trajectories (Post-processing)
        parsed_preds = []
        success_preds = []
        for text in output_texts:
            # Parse the first completed assistant turn. Without this boundary, a
            # generation that continues into a hallucinated conversation can make
            # a later/truncated occurrence of "trajectory" hide a valid answer.
            traj_text = self._trajectory_text_from_generation(text)
            traj, success = self.text_to_traj(traj_text)
            parsed_preds.append(traj)
            success_preds.append(success)

        pred_trajs = torch.stack(parsed_preds).to(self.device)  # [B, T, 2]
        success_preds = torch.tensor(success_preds).to(self.device)

        # Resample if target_sampling differs from native
        pred_trajs = self._maybe_resample(pred_trajs, target_sampling)

        extra = {
            "success_preds": success_preds,
            "reasoning_trace": output_texts,
            "prompt_length": input_ids.shape[1],
            "gen_input_ids": generated_ids,
            "gen_attention_mask": (
                generated_ids != self.processor.tokenizer.pad_token_id
            ).long(),
            "gen_pixel_values": pixel_values,
            "gen_image_grid_thw": image_grid_thw,
        }
        return pred_trajs, extra

    @staticmethod
    def _normalize_qa_prediction(raw_prediction: str) -> str | None:
        """Normalize a generated answer without accepting appended junk text."""
        text = raw_prediction.lower()
        if "</think>" in text:
            text = text.rsplit("</think>", 1)[1]
        text = text.replace("<|im_end|>", " ").strip()
        normalized = text.strip(" \n\t.,:;!?\"'")
        return normalized if normalized in _NUSCENES_QA_ANSWERS else None

    @torch.inference_mode()
    def predict_qa_answers(
        self,
        qa_batch: list[tuple[E2EDataSample, QuestionAnswerPair]],
        max_new_tokens: int = 16,
    ) -> list[str | None]:
        """Answer NuScenesQA questions directly with the Qwen backbone."""
        if max_new_tokens < 1:
            raise ValueError("max_new_tokens must be at least one.")
        if not qa_batch:
            return []

        self.processor.tokenizer.padding_side = "left"
        messages = [
            TextTrajCollator._get_qa_message(sample, qa_pair)[:-1]
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

        generated_ids = self.vlm.generate(
            **inputs,
            do_sample=False,
            max_new_tokens=max_new_tokens,
            use_cache=True,
            eos_token_id=self._generation_eos_token_id(),
            pad_token_id=self.processor.tokenizer.pad_token_id,
        )
        new_token_ids = generated_ids[:, inputs["input_ids"].shape[1] :]
        raw_predictions = self.processor.batch_decode(
            new_token_ids, skip_special_tokens=True
        )
        return [
            self._normalize_qa_prediction(prediction) for prediction in raw_predictions
        ]

    def text_to_traj(self, text: str) -> tuple[torch.Tensor, bool]:
        """
        Tries to find list of lists of numbers in text string to output its corresponding tensor.
        Caps number of matches at pred_traj_sampling.num_poses in case the model hallucinates
        extra waypoints. If text has insufficient matches or if list pattern is not present,
        returns zero tensor.
        """
        # Matches numbers like: -8.686, 0.0, 1e-5
        num_pattern = r"-?\d+(?:\.\d+)?(?:[eE][-+]?\d+)?"

        # Matches pairs like: [-8.686, -0.025]
        pair_pattern = re.compile(rf"\[\s*({num_pattern})\s*,\s*({num_pattern})\s*\]")

        # Find all non-overlapping matches in the string
        matches = pair_pattern.findall(text)

        # Find first num_poses waypoints
        num_poses = self.pred_traj_sampling.num_poses
        if len(matches) >= num_poses:
            try:
                # Convert string tuples to float list
                data = [[float(x), float(y)] for x, y in matches[:num_poses]]
                return torch.tensor(data), True
            except ValueError:
                # Fallthrough to return zero tensor if float conversion fails
                pass

        # Return zero tensor if pattern not found or count is wrong
        return torch.zeros(num_poses, 2), False

    def get_collator(
        self,
        drop_cfg: dict[str, float] | None = None,
        inference_mode: bool = False,
        qa_probability: float = 0.0,
    ) -> Callable:
        """
        Model provides its own collator -> simplifies code and input processing.

        Args:
            drop_cfg: Optional per-input drop probabilities for training-time
                augmentation, e.g. {"intent": 0.3}. When None (default), all
                inputs are always included (suitable for validation).
            inference_mode: When True, the system prompt always uses the global
                use_cot setting (suitable for generation/evaluation). When False
                (default), the system prompt falls back to no-cot for samples
                that have no reasoning trace (suitable for supervised training).
            qa_probability: Probability of replacing an annotated nuScenes
                trajectory conversation with one attached QA pair. The default
                zero retains the original trajectory-only collator path.

        To do inference on a single specific E2EDataSample:
        ```
        collator = model.get_collator(inference_mode=True)
        inputs = collator([sample])
        pred_trajs, extra = model.predict_trajectories(**inputs)
        ```
        """
        return TextTrajCollator(
            self.processor,
            self.prompt_builder,
            self.past_traj_sampling,
            self.pred_traj_sampling,
            self.camera_sequence,
            self.num_past_image_frames,
            drop_cfg=drop_cfg,
            inference_mode=inference_mode,
            concat_cameras=self.concat_cameras,
            qa_probability=qa_probability,
            pad_missing_past_traj=getattr(self, "pad_missing_past_traj", False),
        )

    def compute_loss(self, batch: E2EDataBatch) -> dict[str, torch.Tensor]:
        """Compute the ordinary causal-LM loss at supervised positions only.

        This is mathematically the same objective as passing ``labels`` directly
        to Qwen: labels are shifted once, ``-100`` targets are ignored, logits
        are converted to fp32, and cross entropy is averaged over target tokens.
        Supplying tensor-valued ``logits_to_keep`` merely avoids applying the
        large vocabulary head to prompt/image positions whose shifted label is
        ignored.

        Args:
            batch: Output from ``TextTrajCollator`` with assistant-only labels.

        Returns:
            Dictionary containing the fp32 mean causal cross-entropy.
        """
        model_inputs = dict(batch.model_inputs)
        labels = model_inputs.pop("labels", None)
        if labels is None:
            raise ValueError("Text trajectory training requires language labels.")

        # Logit position i predicts label position i + 1. The final sequence
        # logit predicts an appended ignore target in Hugging Face's causal loss,
        # so labels[:, 1:] contains every target that can affect that loss.
        shifted_labels = labels[:, 1:]
        logit_positions = shifted_labels.ne(-100).any(dim=0).nonzero(as_tuple=True)[0]
        if logit_positions.numel() == 0:
            raise ValueError("Text trajectory training batch has no supervised tokens.")

        model_inputs["logits_to_keep"] = logit_positions
        outputs = self.vlm(**model_inputs)
        target_labels = shifted_labels.index_select(1, logit_positions)

        # Hugging Face's ForCausalLMLoss performs this same fp32 upcast before
        # cross entropy. Here it applies only to the compact target-position
        # tensor instead of a [batch, full_sequence, vocabulary] allocation.
        loss = F.cross_entropy(
            outputs.logits.float().reshape(-1, outputs.logits.shape[-1]),
            target_labels.to(outputs.logits.device).reshape(-1),
            ignore_index=-100,
        )
        return {"loss": loss}


class TextTrajCollator:
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
        qa_probability: float = 0.0,
        pad_missing_past_traj: bool = False,
        drop_incomplete_past_traj: bool = False,
    ):
        if not 0.0 <= qa_probability <= 1.0:
            raise ValueError("qa_probability must be between zero and one.")
        self.processor = processor
        self.prompt_builder = prompt_builder
        self.past_traj_sampling = past_traj_sampling
        self.pred_traj_sampling = pred_traj_sampling
        self.camera_sequence = camera_sequence
        self.num_past_image_frames = num_past_image_frames
        self.drop_cfg = drop_cfg
        self.inference_mode = inference_mode
        self.concat_cameras = concat_cameras
        self.qa_probability = qa_probability
        self.pad_missing_past_traj = pad_missing_past_traj
        self.drop_incomplete_past_traj = drop_incomplete_past_traj

    def _resample_past_traj(self, sample: E2EDataSample) -> torch.Tensor:
        """Resample past trajectory to model's expected input grid."""
        past_traj = sample.agent_input["past_traj"]
        dataset_sampling = TrajectorySampling(
            num_poses=past_traj.shape[0],
            interval_length=1.0 / sample.metadata["sampling_freq"],
        )
        return resample_tensor(
            past_traj,
            dataset_sampling,
            self.past_traj_sampling,
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

    def _maybe_pad_past_traj_text(
        self,
        message: list[dict[str, Any]],
        num_available_poses: int,
    ) -> None:
        """Prepend explicit text slots for unavailable oldest history poses."""
        num_missing = self.past_traj_sampling.num_poses - num_available_poses
        if num_missing <= 0:
            return

        if not self.pad_missing_past_traj:
            #if num_missing > 0:
                #print(f"Sample has incomplete past and padding is not applied.")
            return

        user_message = next(item for item in message if item["role"] == "user")
        past_item = next(
            item
            for item in user_message["content"]
            if item.get("type") == "text"
            and item.get("text", "").startswith(_PAST_TRAJ_TEXT_PREFIX)
        )
        trajectory_text = past_item["text"][len(_PAST_TRAJ_TEXT_PREFIX) :]
        available_entries = trajectory_text[1:-1]
        entries = [_MISSING_PAST_TRAJ_TEXT] * num_missing
        if available_entries:
            entries.append(available_entries)
        past_item["text"] = _PAST_TRAJ_TEXT_PREFIX + "[" + ", ".join(entries) + "]"

    def _get_sample_message(self, sample: E2EDataSample) -> list[dict[str, Any]]:
        if self.drop_cfg is not None:
            active_inputs = self.prompt_builder.sample_active_inputs(self.drop_cfg)
        else:
            active_inputs = ALL_INPUTS

        past_traj_for_prompt = self._resample_past_traj(sample)
        if (past_traj_for_prompt.shape[0] < self.past_traj_sampling.num_poses) and self.drop_incomplete_past_traj:
            active_inputs = active_inputs - {"past_traj"}
        fut_traj_for_prompt = self._get_fut_traj_for_prompt(sample)

        message = self.prompt_builder.build_message(
            cameras=sample.cameras,
            past_traj=past_traj_for_prompt,
            intent=sample.intent,
            active_inputs=active_inputs,
            camera_sequence=self.camera_sequence,
            num_past_image_frames=self.num_past_image_frames,
            reasoning_trace=sample.reasoning_trace,
            fut_traj=fut_traj_for_prompt,
            inference_mode=self.inference_mode,
            concat_cameras=self.concat_cameras,
        )
        if "past_traj" in active_inputs:
            self._maybe_pad_past_traj_text(
                message, num_available_poses=past_traj_for_prompt.shape[0]
            )
        return message

    def _select_qa_pair(self, sample: E2EDataSample) -> QuestionAnswerPair | None:
        """Randomly select one attached QA pair for this training occurrence."""
        if not sample.qa_pairs or self.qa_probability == 0.0:
            return None
        if self.qa_probability < 1.0 and torch.rand(()).item() >= self.qa_probability:
            return None
        pair_index = int(torch.randint(len(sample.qa_pairs), ()).item())
        return sample.qa_pairs[pair_index]

    @staticmethod
    def _get_qa_message(
        sample: E2EDataSample,
        qa_pair: QuestionAnswerPair,
    ) -> list[dict[str, Any]]:
        """Build the same six-view conversation used by Qwen FM-VLA QA."""
        if not sample.cameras:
            raise ValueError("NuScenesQA requires a current camera frame.")
        current_frame = sample.cameras[-1]
        missing = [name for name in _NUSCENES_QA_CAMERAS if name not in current_frame]
        if missing:
            raise ValueError(
                "NuScenesQA requires all six surround cameras; missing: "
                + ", ".join(missing)
            )

        user_content: list[dict[str, Any]] = [
            {"type": "text", "text": "Surround-view images:"}
        ]
        for camera_name in _NUSCENES_QA_CAMERAS:
            user_content.extend(
                [
                    {"type": "text", "text": f"{camera_name} camera:"},
                    {
                        "type": "image",
                        "image": current_frame[camera_name].image,
                        "input_data_format": "channels_last",
                    },
                ]
            )
        user_content.append({"type": "text", "text": f"Question: {qa_pair.question}"})
        return [
            {
                "role": "system",
                "content": [{"type": "text", "text": _NUSCENES_QA_SYSTEM_PROMPT}],
            },
            {"role": "user", "content": user_content},
            {
                "role": "assistant",
                "content": [{"type": "text", "text": qa_pair.answer}],
            },
        ]

    def __call__(self, batch: list[E2EDataSample]) -> E2EDataBatch:
        # Preserve the original trajectory-only path exactly when QA is disabled.
        if self.qa_probability == 0.0:
            full_messages = [self._get_sample_message(sample) for sample in batch]
            qa_mask = torch.zeros(len(batch), dtype=torch.bool)
            return self._collate_messages(batch, full_messages, qa_mask=qa_mask)

        qa_pairs = [self._select_qa_pair(sample) for sample in batch]
        qa_mask = torch.tensor(
            [qa_pair is not None for qa_pair in qa_pairs], dtype=torch.bool
        )
        full_messages = [
            self._get_qa_message(sample, qa_pair)
            if qa_pair is not None
            else self._get_sample_message(sample)
            for sample, qa_pair in zip(batch, qa_pairs)
        ]
        return self._collate_messages(batch, full_messages, qa_mask=qa_mask)

    def _collate_messages(
        self,
        batch: list[E2EDataSample],
        full_messages: list[list[dict[str, Any]]],
        qa_mask: torch.Tensor | None = None,
    ) -> E2EDataBatch:
        """Tokenize chats and mask every token outside assistant supervision.

        Keeping message construction separate lets specialized collators supply
        a different task prompt while reusing exactly the same Qwen-compatible
        SFT masking behavior.

        Args:
            batch: Samples represented by ``full_messages`` in matching order.
            full_messages: Complete system/user/assistant chats for each sample.

        Returns:
            Batch containing processor outputs and assistant-only labels.
        """
        if len(batch) != len(full_messages):
            raise ValueError("Each sample must have exactly one chat message.")
        if qa_mask is not None and qa_mask.shape != (len(batch),):
            raise ValueError("qa_mask must contain one flag per batch sample.")

        full_inputs = self.processor.apply_chat_template(
            full_messages,
            tokenize=True,
            add_generation_prompt=False,
            continue_final_message=False,
            return_dict=True,
            return_tensors="pt",
            padding=True,
        )

        input_ids = full_inputs["input_ids"]
        labels = input_ids.clone()

        # Mask padding tokens
        pad_id = None
        if getattr(self.processor, "tokenizer", None) is not None:
            pad_id = self.processor.tokenizer.pad_token_id
        if pad_id is not None:
            labels[input_ids == pad_id] = -100
        else:
            labels[full_inputs["attention_mask"] == 0] = -100

        # Get inputs only, to mask non-assistant tokens and for evaluation
        # Convoluted because built-in flag return_assistant_mask_tokens does not work for Qwen
        for i, msgs in enumerate(full_messages):
            cur_msg = [m for m in msgs if m["role"] != "assistant"]

            prompt = self.processor.apply_chat_template(
                [cur_msg],
                tokenize=True,
                add_generation_prompt=True,
                continue_final_message=False,
                return_dict=True,
                return_tensors="pt",
                padding=True,
            )

            prompt_len = prompt["input_ids"].shape[1]
            valid_indices = (labels[i] != -100).nonzero(as_tuple=True)[0]
            target_indices = valid_indices[:prompt_len]
            labels[i, target_indices] = -100

        full_inputs["labels"] = labels

        out = {}
        out["model_inputs"] = full_inputs
        out["samples"] = batch
        out["qa_mask"] = qa_mask

        return E2EDataBatch(**out)
