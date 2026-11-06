from dataclasses import dataclass

import torch
import torch.nn as nn
import torch.nn.functional as F
from torchvision.transforms import v2
from transformers import PreTrainedModel

from ad_vla.dataset.data_types import E2EDataBatch
from ad_vla.dataset.camera_config import qa_cameras


@dataclass(frozen=True)
class _TeacherLayout:
    images: torch.Tensor
    views_per_frame: int
    concatenate_views: bool


class SpatialForcingAuxHead(nn.Module):
    loss_name = "3d_distill"
    log_on_validation = False
    train_aux_head_only = False

    def __init__(
        self,
        model: PreTrainedModel,
        target_layer: int,
        grid_size: int,
        add_pos_embedding: bool = True,
    ):
        """
        Spatial Forcing auxiliary loss from https://arxiv.org/abs/2510.12276.

        The head keeps only the model pieces it needs after initialization:
        processor metadata, camera layout, and a forward hook on the selected VLM
        layer. The full planner stays owned by the train stage.
        """
        super().__init__()

        self.processor = model.processor
        self.camera_sequence = list(model.camera_sequence)
        self.num_past_image_frames = model.num_past_image_frames
        self.concat_cameras = model.concat_cameras
        self.target_layer = target_layer
        # Retained for compatibility with existing configs. Spatial Forcing now
        # aligns VGGT directly to each VLM image-token grid instead of forcing
        # both representations through a common fixed-size grid.
        self.grid_size = grid_size
        self.add_pos_embedding = add_pos_embedding

        if self.concat_cameras:
            print(
                "Using concat_cameras for model: will concatenate VGGT features to match."
            )

        from vggt.models.vggt import VGGT

        # Frozen VGGT teacher provides the 3D-aware feature target.
        self.vggt = VGGT.from_pretrained("facebook/VGGT-1B")
        self.vggt.to(torch.bfloat16)
        self.vggt.requires_grad_(False)
        self.vggt.eval()
        self.vggt_patch_size = int(self.vggt.aggregator.patch_size)
        vggt_hidden_size = 2 * int(self.vggt.aggregator.camera_token.shape[-1])
        print("3D foundation model loaded (VGGT).")

        # Hook the selected VLM language layer so image-token states are available after forward.
        self.captured_states = {}
        layer = model.vlm.model.language_model.layers[self.target_layer]
        self._hook = layer.register_forward_hook(
            lambda m, inp, out, idx=self.target_layer: self.captured_states.__setitem__(
                idx, out
            )
        )

        vlm_config = model.vlm.config
        hidden_size = int(vlm_config.text_config.hidden_size)
        self.spatial_merge_size = int(vlm_config.vision_config.spatial_merge_size)

        # Project VLM image-token features into VGGT feature space before cosine loss.
        self.img_fts_norm = nn.BatchNorm1d(hidden_size)
        self.img_fts_adapter = nn.Sequential(
            nn.Linear(hidden_size, vggt_hidden_size),
            nn.GELU(),
            nn.Linear(vggt_hidden_size, vggt_hidden_size),
        ).to(torch.bfloat16)

    def train(self, mode: bool = True):
        """Keep the frozen VGGT teacher in evaluation mode."""
        super().train(mode)
        self.vggt.eval()
        return self

    def compute_aux_loss(self, batch: E2EDataBatch) -> tuple[torch.Tensor, dict]:
        """Compute cosine distillation loss between VLM and VGGT image features."""
        img_fts, image_grid_hw = self.get_img_fts(batch)
        vggt_fts = self.get_vggt_fts(batch, image_grid_hw)

        losses = []
        for sample_idx, (student, teacher) in enumerate(zip(img_fts, vggt_fts)):
            if student.shape != teacher.shape:
                raise ValueError(
                    "Spatial Forcing feature shapes do not match for sample "
                    f"{sample_idx}: VLM {tuple(student.shape)} versus "
                    f"VGGT {tuple(teacher.shape)}."
                )
            losses.append(1 - F.cosine_similarity(student, teacher, dim=-1).mean())

        return torch.stack(losses).mean(), {}

    def get_vggt_fts(
        self,
        batch: E2EDataBatch,
        image_grid_hw: list[list[tuple[int, int]]],
    ) -> list[torch.Tensor]:
        """Extract VGGT features and resize them to the corresponding VLM grids."""
        device = next(self.img_fts_adapter.parameters()).device
        qa_mask = self._get_qa_mask(batch)
        teacher_layouts = [
            self._get_teacher_layout(sample, is_qa=bool(qa_mask[sample_idx]))
            for sample_idx, sample in enumerate(batch.samples)
        ]
        image_batches = [
            self._prepare_vggt_images(layout.images.to(device))
            for layout in teacher_layouts
        ]

        # VGGT requires a common view count and resolution within a batch. Group
        # compatible rows so three-view driving and all-view QA remain efficient.
        sample_groups = {}
        for sample_idx, images in enumerate(image_batches):
            sample_groups.setdefault(tuple(images.shape), []).append(sample_idx)

        features_by_sample = {}
        for sample_indices in sample_groups.values():
            grouped_images = torch.stack(
                [image_batches[sample_idx] for sample_idx in sample_indices]
            )
            grouped_features = self._extract_vggt_features(grouped_images)
            features_by_sample.update(zip(sample_indices, grouped_features.unbind(0)))

        outputs = []
        for sample_idx, (target_grids, images, layout) in enumerate(
            zip(image_grid_hw, image_batches, teacher_layouts)
        ):
            features = features_by_sample[sample_idx]
            if self.add_pos_embedding:
                features = self._add_vggt_positional_embedding(
                    features,
                    image_hw=tuple(images.shape[-2:]),
                )

            if layout.concatenate_views:
                if features.shape[0] % layout.views_per_frame != 0:
                    raise ValueError(
                        "VGGT image count is not divisible by the configured number "
                        f"of cameras for sample {sample_idx}."
                    )
                feature_grids = [
                    torch.cat(frame_features.unbind(0), dim=1)
                    for frame_features in features.split(layout.views_per_frame, dim=0)
                ]
            else:
                feature_grids = list(features.unbind(0))

            if len(feature_grids) != len(target_grids):
                raise ValueError(
                    "Spatial Forcing image count does not match the VLM prompt for "
                    f"sample {sample_idx}: expected {len(feature_grids)} image spans, "
                    f"found {len(target_grids)}."
                )

            resized = [
                self._resize_feature_grid(feature_grid, target_hw)
                for feature_grid, target_hw in zip(feature_grids, target_grids)
            ]
            outputs.append(torch.cat([feature.flatten(0, 1) for feature in resized]))

        return outputs

    @staticmethod
    def _get_qa_mask(batch: E2EDataBatch) -> torch.Tensor:
        qa_mask = batch.qa_mask
        if qa_mask is None:
            qa_mask = torch.zeros(len(batch.samples), dtype=torch.bool)
        qa_mask = qa_mask.detach().to(dtype=torch.bool, device="cpu")
        if qa_mask.shape != (len(batch.samples),):
            raise ValueError("qa_mask must contain one flag per batch sample.")
        return qa_mask

    def _get_teacher_layout(self, sample, is_qa: bool) -> _TeacherLayout:
        if not sample.cameras:
            raise ValueError("Spatial Forcing requires at least one camera frame.")

        if is_qa:
            frames = sample.cameras[-1:]
            camera_names = qa_cameras(frames[0])
            concatenate_views = False
        else:
            num_frames = min(len(sample.cameras), self.num_past_image_frames)
            frames = sample.cameras[-num_frames:]
            camera_names = self.camera_sequence
            concatenate_views = self.concat_cameras

        images = []
        for frame in frames:
            missing = [name for name in camera_names if name not in frame]
            if missing:
                raise KeyError(
                    "Spatial Forcing teacher cameras are missing prompt cameras: "
                    f"{missing}"
                )
            images.extend(frame[name].image for name in camera_names)

        first_shape = tuple(images[0].shape)
        if any(tuple(image.shape) != first_shape for image in images):
            raise ValueError(
                "All Spatial Forcing camera images within a sample must have the "
                "same shape."
            )
        return _TeacherLayout(
            images=torch.stack(images).permute(0, 3, 1, 2),
            views_per_frame=len(camera_names),
            concatenate_views=concatenate_views,
        )

    def _prepare_vggt_images(self, images: torch.Tensor) -> torch.Tensor:
        height, width = images.shape[-2:]
        resized_hw = (
            self.vggt_patch_size * (height // self.vggt_patch_size),
            self.vggt_patch_size * (width // self.vggt_patch_size),
        )
        if min(resized_hw) == 0:
            raise ValueError(
                f"Camera image size {(height, width)} is smaller than VGGT's "
                f"patch size {self.vggt_patch_size}."
            )
        return v2.Resize(resized_hw)(images)

    def _extract_vggt_features(self, images: torch.Tensor) -> torch.Tensor:
        """Run the frozen VGGT aggregator on a stack shaped [B, N, C, H, W]."""
        self.vggt.eval()
        with torch.no_grad():
            if images.is_cuda:
                with torch.amp.autocast("cuda", dtype=torch.bfloat16):
                    aggregated_tokens_list, patch_start_idx = self.vggt.aggregator(
                        images.float() / 255.0
                    )
            else:
                aggregated_tokens_list, patch_start_idx = self.vggt.aggregator(
                    images.float() / 255.0
                )

        batch_size, num_images, _, height, width = images.shape
        features = aggregated_tokens_list[-1][:, :, patch_start_idx:]
        return features.reshape(
            batch_size,
            num_images,
            height // self.vggt_patch_size,
            width // self.vggt_patch_size,
            -1,
        )

    @staticmethod
    def _resize_feature_grid(
        features: torch.Tensor,
        target_hw: tuple[int, int],
    ) -> torch.Tensor:
        features = features.permute(2, 0, 1).unsqueeze(0)
        features = F.interpolate(
            features,
            size=target_hw,
            mode="bilinear",
            align_corners=False,
        )
        return features[0].permute(1, 2, 0)

    @staticmethod
    def _add_vggt_positional_embedding(
        features: torch.Tensor,
        image_hw: tuple[int, int],
    ) -> torch.Tensor:
        """Add the fixed coordinate encoding used by the official SF code."""
        from vggt.heads.utils import create_uv_grid, position_grid_to_embed

        patch_height, patch_width = features.shape[1:3]
        image_height, image_width = image_hw
        position_grid = create_uv_grid(
            patch_width,
            patch_height,
            aspect_ratio=image_width / image_height,
            dtype=features.dtype,
            device=features.device,
        )
        position_embedding = position_grid_to_embed(
            position_grid,
            features.shape[-1],
        )
        return features + 0.1 * position_embedding.unsqueeze(0)

    def get_img_fts(
        self,
        batch: E2EDataBatch,
    ) -> tuple[list[torch.Tensor], list[list[tuple[int, int]]]]:
        """Extract and project each VLM image span using its own grid metadata."""
        hidden_states = self.captured_states.pop(self.target_layer, None)
        if hidden_states is None:
            raise RuntimeError(
                "Spatial Forcing did not capture VLM hidden states. The model "
                "forward must run before compute_aux_loss."
            )

        image_token_spans = self._get_image_token_spans(batch)
        image_grid_thw = batch.model_inputs["image_grid_thw"]
        num_spans = sum(len(spans) for spans in image_token_spans)
        if num_spans != image_grid_thw.shape[0]:
            raise ValueError(
                "Spatial Forcing found a different number of VLM image spans and "
                f"image grids: {num_spans} spans versus {image_grid_thw.shape[0]} grids."
            )

        raw_features = []
        feature_lengths = []
        image_grid_hw = [[] for _ in batch.samples]
        grid_idx = 0
        for sample_idx, spans in enumerate(image_token_spans):
            for start, end in spans:
                time, patch_height, patch_width = image_grid_thw[grid_idx].tolist()
                grid_idx += 1
                if time != 1:
                    raise ValueError(
                        "Spatial Forcing currently expects still-image Qwen grids; "
                        f"sample {sample_idx} has temporal grid size {time}."
                    )
                if (
                    patch_height % self.spatial_merge_size != 0
                    or patch_width % self.spatial_merge_size != 0
                ):
                    raise ValueError(
                        "Qwen image grid is not divisible by spatial_merge_size: "
                        f"{(patch_height, patch_width)} versus "
                        f"{self.spatial_merge_size}."
                    )

                grid_height = patch_height // self.spatial_merge_size
                grid_width = patch_width // self.spatial_merge_size
                features = hidden_states[sample_idx, start + 1 : end]
                expected_tokens = grid_height * grid_width
                if features.shape[0] != expected_tokens:
                    raise ValueError(
                        "VLM image span length does not match its Qwen grid for "
                        f"sample {sample_idx}: {features.shape[0]} tokens versus "
                        f"{expected_tokens}."
                    )

                raw_features.append(features)
                feature_lengths.append(expected_tokens)
                image_grid_hw[sample_idx].append((grid_height, grid_width))

        projected = torch.cat(raw_features, dim=0)
        projected = self.img_fts_norm(projected)
        projected = self.img_fts_adapter(projected)
        projected_by_image = projected.split(feature_lengths)

        features_by_sample = []
        image_idx = 0
        for grids in image_grid_hw:
            sample_features = []
            for grid_height, grid_width in grids:
                sample_features.append(
                    projected_by_image[image_idx].reshape(
                        grid_height,
                        grid_width,
                        -1,
                    )
                )
                image_idx += 1
            features_by_sample.append(
                torch.cat([features.flatten(0, 1) for features in sample_features])
            )

        return features_by_sample, image_grid_hw

    def _get_image_token_spans(
        self,
        batch: E2EDataBatch,
    ) -> list[list[tuple[int, int]]]:
        input_ids = batch.model_inputs["input_ids"]
        vision_start_id = self.processor.tokenizer.convert_tokens_to_ids(
            "<|vision_start|>"
        )
        vision_end_id = self.processor.tokenizer.convert_tokens_to_ids("<|vision_end|>")

        image_token_spans = []
        for sample_idx in range(input_ids.shape[0]):
            starts = (input_ids[sample_idx] == vision_start_id).nonzero(as_tuple=True)[
                0
            ]
            ends = (input_ids[sample_idx] == vision_end_id).nonzero(as_tuple=True)[0]
            if starts.numel() != ends.numel():
                raise ValueError(
                    "Mismatched vision start/end tokens for Spatial Forcing: "
                    f"sample {sample_idx} has {starts.numel()} starts and "
                    f"{ends.numel()} ends."
                )
            if starts.numel() == 0:
                raise ValueError(
                    f"Sample {sample_idx} has no image tokens for Spatial Forcing."
                )

            spans = []
            for start, end in zip(starts.tolist(), ends.tolist()):
                if end <= start + 1:
                    raise ValueError(
                        "Invalid empty vision token span for Spatial Forcing: "
                        f"sample {sample_idx}, start={start}, end={end}."
                    )
                spans.append((start, end))
            image_token_spans.append(spans)
        return image_token_spans
