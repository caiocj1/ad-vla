import torch
import torch.nn as nn
import torch.nn.functional as F
from transformers import PreTrainedModel
import matplotlib.pyplot as plt
from pytorch_lightning.loggers import CometLogger

from simple_bev import saverloader
from simple_bev.utils import vox
from simple_bev.nets.segnet import Segnet
from ad_vla.models.layers.bev_cross_attention import BEVCrossAttentionLayer
from ad_vla.dataset.data_types import E2EDataBatch
from ad_vla.utils.bev_utils import build_simple_bev_calibs


class BEVForcingAuxHead(nn.Module):
    loss_name = "bev_distill"
    log_on_validation = True

    def __init__(
        self,
        model: PreTrainedModel,
        bev_model_ckpt_path: str,
        target_layer: int,
        train_bev_head_only: bool = False,
        cross_attn_layer_ckpt_path: str | None = None,
        add_pos_embedding: bool = False,
        add_residual: bool = False,
        add_ffn: bool = False,
        num_layers: int = 1,
        num_attention_heads: int = 1,
        ffn_hidden_dim: int | None = None,
    ):
        """
        Initialization of necessary elements for BEV Forcing.
        Initializes:
            - SimpleBEV feature extractor.
            - BEV cross-attention layer to output predicted BEV features/maps.
            - Hook to capture the desired layer's hidden states from VLM.
        """
        super().__init__()

        self.processor = model.processor
        self.camera_sequence = list(model.camera_sequence)
        self.train_aux_head_only = train_bev_head_only
        extra_token_ids = getattr(model, "bev_extra_token_ids", None)
        if extra_token_ids is None and hasattr(model, "calib_token_id"):
            extra_token_ids = [model.calib_token_id]
        self.extra_token_ids = tuple(
            int(token_id) for token_id in extra_token_ids or []
        )

        # Initialize pre-trained SimpleBEV model
        scene_centroid = torch.tensor([[0.0, 1.0, 0.0]])
        bounds = (-50, 50, -5, 5, -50, 50)  # XMIN, XMAX, YMIN, YMAX, ZMIN, ZMAX
        Z, Y, X = 200, 8, 200

        self.vox_util = vox.Vox_util(
            Z,
            Y,
            X,
            scene_centroid=scene_centroid,
            bounds=bounds,
            assert_cube=False,
        )

        self.bev_model = Segnet(
            Z,
            Y,
            X,
            self.vox_util,
            use_radar=False,
            use_lidar=False,
            use_metaradar=False,
            do_rgbcompress=True,
            encoder_type="res101",
        )
        self.bev_model.requires_grad_(False)

        saverloader.load(bev_model_ckpt_path, self.bev_model)
        self.bev_model.eval()

        # Initialize BEV cross-attention layer
        self.bev_cross_attn_layer = BEVCrossAttentionLayer(
            grid_size=(Z, Y, X),
            bev_bounds=bounds,
            add_pos_embedding=add_pos_embedding,
            add_residual=add_residual,
            add_ffn=add_ffn,
            num_layers=num_layers,
            num_attention_heads=num_attention_heads,
            ffn_hidden_dim=ffn_hidden_dim,
        )
        if cross_attn_layer_ckpt_path is not None:
            state_dict = torch.load(cross_attn_layer_ckpt_path)
            self.bev_cross_attn_layer.load_state_dict(state_dict)

        # Set up hook to capture hidden states at target layer for VLM image feature extraction
        self.captured_states = {}
        self.target_layer = target_layer
        layer = model.vlm.model.language_model.layers[self.target_layer]
        self._hook = layer.register_forward_hook(
            lambda m, inp, out, idx=self.target_layer: self.captured_states.__setitem__(
                idx, out
            )
        )

        # If training head only, freeze model parameters for this training run
        if train_bev_head_only:
            model.requires_grad_(False)

    def compute_aux_loss(self, batch: E2EDataBatch) -> tuple[torch.Tensor, dict]:
        """
        Calculates BEV Forcing loss.

        Uses BEV occupancy map coming from pretrained SimpleBEV model
        to distill spatial awareness into the VLM.
        Falls back to per-sample processing when mixed datasets produce
        different teacher image shapes or different VLM image-token counts.

        Args:
            batch (E2EDataBatch): Data batch.

        Returns:
            torch.Tensor: BCELossWithLogits between BEV cross-attention layer outputs
                and occupancy map from SimpleBEV.
        """
        bev_occ_map = self.get_bev_occ_map(batch)
        image_token_spans = self._get_image_token_spans(batch)

        if self._image_token_spans_are_stackable(
            image_token_spans
        ) and self._extra_token_counts_are_stackable(batch):
            img_fts = self.get_img_fts(batch, image_token_spans=image_token_spans)
            extra_fts = self.get_extra_fts(batch)

            # Predict BEV occupancy map
            out = self.bev_cross_attn_layer(
                img_fts.float(),
                extra_hidden_states=extra_fts.float()
                if extra_fts is not None
                else None,
            ).squeeze(-1)
        else:
            img_fts_by_sample = self.get_img_fts_by_sample(
                batch,
                image_token_spans=image_token_spans,
            )
            extra_fts_by_sample = self.get_extra_fts_by_sample(batch)

            outs = []
            for img_fts, extra_fts in zip(img_fts_by_sample, extra_fts_by_sample):
                out = self.bev_cross_attn_layer(
                    img_fts.float(),
                    extra_hidden_states=(
                        extra_fts.float() if extra_fts is not None else None
                    ),
                ).squeeze(-1)
                outs.append(out)
            out = torch.cat(outs, dim=0)

        # Calculate loss on portion of BEV corresponding to front cam images
        bev_loss = F.binary_cross_entropy_with_logits(
            out,
            bev_occ_map[:, 100:, 50:150].sigmoid(),
            pos_weight=torch.tensor(20.0, device=out.device),
        )

        extra = {}
        extra["pred_bev"] = out.sigmoid()
        extra["gt_bev"] = bev_occ_map[:, 100:, 50:150].sigmoid()

        return bev_loss, extra

    def get_bev_occ_map(self, batch: E2EDataBatch) -> torch.Tensor:
        # Prefer privileged teacher cameras when datasets provide them. These
        # are fixed-size last-timestep images with matching calibrations.
        camera_frames = self._get_camera_frames(batch)
        key_list = self._get_camera_key_list(camera_frames)

        if self._camera_frames_are_stackable(camera_frames, key_list):
            return self._get_bev_occ_map_batched(camera_frames, key_list)

        bev_maps = [
            self._get_bev_occ_map_for_camera_frame(camera_frame, key_list)
            for camera_frame in camera_frames
        ]
        return torch.cat(bev_maps, dim=0)

    def _get_camera_frames(self, batch: E2EDataBatch) -> list[dict]:
        return [
            (
                sample.teacher_cameras
                if sample.teacher_cameras is not None
                else sample.cameras
            )[-1]
            for sample in batch.samples
        ]

    def _get_camera_key_list(self, camera_frames: list[dict]) -> list[str]:
        if "FRONT" not in self.camera_sequence:
            raise ValueError("BEV forcing requires FRONT in model.camera_sequence.")

        key_list = list(dict.fromkeys(self.camera_sequence))
        key_list.remove("FRONT")
        key_list.insert(0, "FRONT")

        for sample_idx, camera_frame in enumerate(camera_frames):
            missing = [key for key in key_list if key not in camera_frame]
            if missing:
                raise KeyError(
                    "BEV forcing teacher cameras are missing model input cameras "
                    f"for sample {sample_idx}: {missing}"
                )
        return key_list

    def _camera_frames_are_stackable(
        self,
        camera_frames: list[dict],
        key_list: list[str],
    ) -> bool:
        first_shape = tuple(camera_frames[0][key_list[0]].image.shape)
        for camera_frame in camera_frames:
            for key in key_list:
                if tuple(camera_frame[key].image.shape) != first_shape:
                    return False
        return True

    def _get_bev_occ_map_batched(
        self,
        camera_frames: list[dict],
        key_list: list[str],
    ) -> torch.Tensor:
        pix_T_cams = []
        cam0_T_camXs = []
        for camera_frame in camera_frames:
            sample_h, sample_w, _ = camera_frame["FRONT"].image.shape
            cal_list = [camera_frame[k].cal_dict for k in key_list]
            pix_T_cam, cam0_T_camX = build_simple_bev_calibs(
                cal_list, image_hw=(sample_h, sample_w)
            )
            pix_T_cams.append(pix_T_cam)
            cam0_T_camXs.append(cam0_T_camX)
        pix_T_cams = torch.cat(pix_T_cams, dim=0)
        cam0_T_camXs = torch.cat(cam0_T_camXs, dim=0)

        rgb_imgs = [
            torch.stack([camera_frame[k].image for k in key_list])
            for camera_frame in camera_frames
        ]
        rgb_imgs = torch.stack(rgb_imgs)
        rgb_imgs = rgb_imgs.permute(0, 1, 4, 2, 3).float() / 255.0
        rgb_imgs = rgb_imgs - 0.5

        device = next(self.bev_model.parameters()).device
        self.bev_model.eval()
        with torch.no_grad():
            bev_out = self.bev_model(
                rgb_camXs=rgb_imgs.to(device),  # [B,S,3,H,W]
                pix_T_cams=pix_T_cams.to(device),  # [B,S,4,4]
                cam0_T_camXs=cam0_T_camXs.to(device),  # [B,S,4,4]
                vox_util=self.vox_util,
                rad_occ_mem0=None,  # camera-only model
            )
        bev_seg_map = bev_out[2]
        if bev_seg_map.ndim == 4 and bev_seg_map.shape[1] == 1:
            bev_seg_map = bev_seg_map[:, 0]
        return bev_seg_map

    def _get_bev_occ_map_for_camera_frame(
        self,
        camera_frame: dict,
        key_list: list[str],
    ) -> torch.Tensor:
        sample_h, sample_w, _ = camera_frame["FRONT"].image.shape
        cal_list = [camera_frame[k].cal_dict for k in key_list]
        pix_T_cams, cam0_T_camXs = build_simple_bev_calibs(
            cal_list,
            image_hw=(sample_h, sample_w),
        )

        rgb_imgs = torch.stack([camera_frame[k].image for k in key_list])
        rgb_imgs = rgb_imgs.unsqueeze(0).permute(0, 1, 4, 2, 3).float() / 255.0
        rgb_imgs = rgb_imgs - 0.5

        device = next(self.bev_model.parameters()).device
        self.bev_model.eval()
        with torch.no_grad():
            bev_out = self.bev_model(
                rgb_camXs=rgb_imgs.to(device),  # [1,S,3,H,W]
                pix_T_cams=pix_T_cams.to(device),  # [1,S,4,4]
                cam0_T_camXs=cam0_T_camXs.to(device),  # [1,S,4,4]
                vox_util=self.vox_util,
                rad_occ_mem0=None,  # camera-only model
            )
        bev_seg_map = bev_out[2]
        if bev_seg_map.ndim == 4 and bev_seg_map.shape[1] == 1:
            bev_seg_map = bev_seg_map[:, 0]
        return bev_seg_map

    def get_img_fts(
        self,
        batch: E2EDataBatch,
        image_token_spans: list[list[tuple[int, int]]] | None = None,
    ) -> torch.Tensor:
        # Retrieve image features
        bs = len(batch.samples)

        if image_token_spans is None:
            image_token_spans = self._get_image_token_spans(batch)
        if not self._image_token_spans_are_stackable(image_token_spans):
            raise ValueError(
                "Image token spans have different lengths across the batch; "
                "use get_img_fts_by_sample for mixed-resolution batches."
            )

        # Extract hidden states of images using start and end found for each image
        # Assumes hook has been setup to capture hidden states
        img_hidden_states = []
        hidden_states = self.captured_states[self.target_layer]
        for i, spans in enumerate(image_token_spans):
            for s, e in spans:
                img_hidden_states.append(hidden_states[i, s + 1 : e])
        num_img_tok, vlm_hidden_dim = img_hidden_states[0].shape
        img_fts = torch.stack(img_hidden_states).reshape(
            bs, -1, num_img_tok, vlm_hidden_dim
        )
        return img_fts

    def get_img_fts_by_sample(
        self,
        batch: E2EDataBatch,
        image_token_spans: list[list[tuple[int, int]]] | None = None,
    ) -> list[torch.Tensor]:
        if image_token_spans is None:
            image_token_spans = self._get_image_token_spans(batch)

        hidden_states = self.captured_states[self.target_layer]
        img_fts_by_sample = []
        for i, spans in enumerate(image_token_spans):
            img_hidden_states = [hidden_states[i, s + 1 : e] for s, e in spans]
            img_fts = torch.cat(img_hidden_states, dim=0).unsqueeze(0).unsqueeze(0)
            img_fts_by_sample.append(img_fts)
        return img_fts_by_sample

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
        for i in range(input_ids.shape[0]):
            starts = (input_ids[i] == vision_start_id).nonzero(as_tuple=True)[0]
            ends = (input_ids[i] == vision_end_id).nonzero(as_tuple=True)[0]
            if starts.numel() != ends.numel():
                raise ValueError(
                    "Mismatched vision start/end tokens for BEV forcing: "
                    f"sample {i} has {starts.numel()} starts and {ends.numel()} ends."
                )
            if starts.numel() == 0:
                raise ValueError(f"Sample {i} has no image tokens for BEV forcing.")

            spans = []
            for s, e in zip(starts.tolist(), ends.tolist()):
                if e <= s + 1:
                    raise ValueError(
                        "Invalid empty vision token span for BEV forcing: "
                        f"sample {i}, start={s}, end={e}."
                    )
                spans.append((s, e))
            image_token_spans.append(spans)
        return image_token_spans

    @staticmethod
    def _image_token_spans_are_stackable(
        image_token_spans: list[list[tuple[int, int]]],
    ) -> bool:
        num_spans = len(image_token_spans[0])
        first_len = image_token_spans[0][0][1] - image_token_spans[0][0][0] - 1
        for spans in image_token_spans:
            if len(spans) != num_spans:
                return False
            for s, e in spans:
                if e - s - 1 != first_len:
                    return False
        return True

    def get_extra_fts(self, batch: E2EDataBatch) -> torch.Tensor | None:
        if not self.extra_token_ids:
            return None

        model_inputs = batch.model_inputs
        input_ids = model_inputs["input_ids"]
        token_mask = torch.zeros_like(input_ids, dtype=torch.bool)
        for token_id in self.extra_token_ids:
            token_mask |= input_ids == token_id

        if not token_mask.any():
            return None

        tokens_per_sample = token_mask.sum(dim=1)
        if not torch.all(tokens_per_sample == tokens_per_sample[0]):
            raise ValueError(
                "Extra token counts must match across the batch for BEV forcing, "
                f"got {tokens_per_sample.tolist()}."
            )

        hidden_states = self.captured_states[self.target_layer]
        extra_hidden_states = [
            hidden_states[i, token_mask[i]] for i in range(input_ids.shape[0])
        ]
        return torch.stack(extra_hidden_states, dim=0)

    def get_extra_fts_by_sample(
        self,
        batch: E2EDataBatch,
    ) -> list[torch.Tensor | None]:
        if not self.extra_token_ids:
            return [None for _ in batch.samples]

        model_inputs = batch.model_inputs
        input_ids = model_inputs["input_ids"]
        token_mask = torch.zeros_like(input_ids, dtype=torch.bool)
        for token_id in self.extra_token_ids:
            token_mask |= input_ids == token_id

        hidden_states = self.captured_states[self.target_layer]
        extra_hidden_states = []
        for i in range(input_ids.shape[0]):
            if not token_mask[i].any():
                extra_hidden_states.append(None)
            else:
                extra_hidden_states.append(hidden_states[i, token_mask[i]].unsqueeze(0))
        return extra_hidden_states

    def _extra_token_counts_are_stackable(self, batch: E2EDataBatch) -> bool:
        if not self.extra_token_ids:
            return True

        input_ids = batch.model_inputs["input_ids"]
        token_mask = torch.zeros_like(input_ids, dtype=torch.bool)
        for token_id in self.extra_token_ids:
            token_mask |= input_ids == token_id

        if not token_mask.any():
            return True

        tokens_per_sample = token_mask.sum(dim=1)
        return bool(torch.all(tokens_per_sample == tokens_per_sample[0]))

    def log_validation_outputs(
        self, logger, batch: E2EDataBatch, extra: dict, step: int
    ):
        if not isinstance(logger, CometLogger):
            return

        for i in range(len(batch.samples)):
            scenario_id = batch.samples[i].metadata["scenario_id"]
            fig, ax = plt.subplots(1, 2)
            ax[0].imshow(extra["pred_bev"][i].detach().cpu().numpy())
            ax[1].imshow(extra["gt_bev"][i].detach().cpu().numpy())
            logger.experiment.log_figure(
                figure_name="BEV/" + scenario_id,
                figure=fig,
                step=step,
            )
            plt.close(fig)

    def save_aux_model(self, output_dir: str):
        torch.save(
            self.bev_cross_attn_layer.state_dict(),
            f"{output_dir}/bev_cross_attn_layer_sd.pt",
        )
