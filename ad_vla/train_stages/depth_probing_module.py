import torch
from torchvision.transforms import v2
from omegaconf import DictConfig
import matplotlib.pyplot as plt
from pytorch_lightning.loggers import CometLogger

from ad_vla.models.base_traj_planner import BaseTrajPlanner
from ad_vla.dataset.data_types import E2EDataBatch, TrajectorySampling
from ad_vla.train_stages.base_module import BaseTrainingModule
from ad_vla.utils.depth_utils import compute_depth_loss


class DepthProbingModule(BaseTrainingModule):
    def __init__(
        self,
        model: BaseTrajPlanner,
        optim_cfg: DictConfig,
        lora_cfg: DictConfig | None = None,
        loss_cfg: DictConfig | None = None,
        val_log_every_n_steps: int = 2,
        val_target_sampling: DictConfig | dict | TrajectorySampling | None = None,
        val_generate_cfg: DictConfig | dict | None = None,
    ):
        super().__init__(
            model=model,
            optim_cfg=optim_cfg,
            lora_cfg=lora_cfg,
            loss_cfg=loss_cfg,
            val_log_every_n_steps=val_log_every_n_steps,
            val_target_sampling=val_target_sampling,
            val_generate_cfg=val_generate_cfg,
        )
        self._init_depth_probing()

    def _init_depth_probing(self):
        from vggt.models.vggt import VGGT
        from vggt.heads.dpt_head import DPTHead

        self.vggt = VGGT.from_pretrained("facebook/VGGT-1B")
        self.vggt = self.vggt.float()
        self.vggt.requires_grad_(False)
        self.vggt.eval()
        print("3D foundation model loaded (VGGT).")

        self.captured_states = {}
        self.target_layer = self.loss_cfg.target_layer
        self.probe_layers = [0, 1, 2, self.target_layer]
        for layer_num in self.probe_layers:
            layer = self.model.vlm.model.language_model.layers[layer_num]
            self._hook = layer.register_forward_hook(
                lambda m, inp, out, idx=layer_num: self.captured_states.__setitem__(
                    idx, out
                )
            )

        self.model.requires_grad_(False)
        hidden_size = self.model.config.text_config["hidden_size"]
        self.dpt_head = DPTHead(
            dim_in=hidden_size,
            patch_size=self.loss_cfg.patch_size,
            intermediate_layer_idx=[0, 1, 2, 3],
            output_dim=2,
            activation="exp",
            conf_activation="expp1",
        )

    def _shared_step(
        self, batch: E2EDataBatch
    ) -> tuple[torch.Tensor, tuple[torch.Tensor, torch.Tensor, torch.Tensor]]:
        # Extract and resize images for 14x14 patches
        images = self._get_img_batch(batch)
        images = images.to(self.model.device)
        h, w = images.shape[-2:]

        # Get images features from intermediate layers and target spatial forcing layer
        img_fts_layers = self._get_img_fts(batch, (h, w))

        # Reshape images for DPTHead patch correspondance
        patch_size = self.dpt_head.patch_size
        fts_h, fts_w = img_fts_layers[0].shape[-3:-1]
        images = v2.Resize((patch_size * fts_h, patch_size * fts_w))(images)

        # Predict depth map with DPTHead
        depth_maps, depth_conf = self.dpt_head(
            img_fts_layers, images=images.float() / 255.0, patch_start_idx=0
        )

        # Use depth map from VGGT as ground-truth
        with torch.no_grad():
            with torch.amp.autocast("cuda", dtype=torch.bfloat16):
                predictions = self.vggt(images.float() / 255.0)

        # Calculate loss
        depth_loss_dict = compute_depth_loss(
            {"depth": depth_maps, "depth_conf": depth_conf},
            {
                "depths": predictions["depth"].squeeze(),
                "point_masks": torch.ones_like(predictions["depth_conf"]).bool(),
            },
            gradient_loss_fn="grad",
            valid_range=0.98,
        )

        loss = (
            depth_loss_dict["loss_conf_depth"]
            + depth_loss_dict["loss_reg_depth"]
            + depth_loss_dict["loss_grad_depth"]
        )

        return loss, (images, depth_maps, predictions["depth"])

    def training_step(self, batch: E2EDataBatch, batch_idx: int):
        loss, _ = self._shared_step(batch)

        self.log(
            "train/loss",
            loss,
            on_step=True,
            on_epoch=False,
            batch_size=len(batch.samples),
            sync_dist=False,
        )

        return loss

    def validation_step(self, batch: E2EDataBatch, batch_idx: int):
        loss, (images, depth_maps, gt_depth_maps) = self._shared_step(batch)

        self.log(
            "val/loss",
            loss,
            on_step=False,
            on_epoch=True,
            batch_size=len(batch.samples),
            sync_dist=True,
        )

        if batch_idx % self.val_log_every_n_steps == 0:
            self._log_depth_maps(batch, images, depth_maps, gt_depth_maps)

    # Override unnecessary hooks in Depth Probing
    def on_train_end(self):
        return

    def on_validation_epoch_end(self):
        return

    def build_metrics(self):
        return

    # Auxiliar functions
    def _get_img_batch(self, batch: E2EDataBatch) -> torch.Tensor:
        bs = len(batch.samples)
        images = [
            torch.stack(
                [
                    batch.samples[i].cameras[-1][k].image
                    for k in self.model.camera_sequence
                ],
                dim=0,
            )
            for i in range(bs)
        ]
        images = torch.stack(images, dim=0)
        images = images.permute(0, 1, 4, 2, 3)
        return images

    def _get_img_fts(self, batch: E2EDataBatch, img_size: tuple) -> list[torch.Tensor]:
        bs = len(batch.samples)
        num_imgs = len(self.model.camera_sequence)
        h, w = img_size

        # Extract hidden states
        model_inputs = batch.model_inputs

        with torch.no_grad():
            self.model.vlm(**batch.model_inputs)

        image_grid_thw = model_inputs["image_grid_thw"]

        # Find start and end of each sequence of image tokens
        input_ids = model_inputs["input_ids"]
        vision_start_id = self.model.processor.tokenizer.convert_tokens_to_ids(
            "<|vision_start|>"
        )
        vision_end_id = self.model.processor.tokenizer.convert_tokens_to_ids(
            "<|vision_end|>"
        )

        idx, starts = (input_ids == vision_start_id).nonzero(as_tuple=True)
        _, ends = (input_ids == vision_end_id).nonzero(as_tuple=True)

        # Extract hidden states of images using start and end found for each image, at each layer
        img_fts_layers = []
        for layer_num in self.probe_layers:
            img_hidden_states = []
            for i, s, e in zip(idx, starts, ends):
                img_hidden_states.append(self.captured_states[layer_num][i, s + 1 : e])
            img_fts = torch.stack(img_hidden_states)
            img_fts = torch.stack(
                torch.split(img_fts, img_fts.shape[1] // 3, dim=1), dim=1
            )

            ph, pw = image_grid_thw[
                0, 1:
            ].tolist()  # Assumes all images in batch have the same format!
            if self.model.concat_cameras:
                pw = pw // 3
            img_fts_reshaped = img_fts.reshape(bs, num_imgs, ph // 2, pw // 2, -1)
            img_fts_layers.append(img_fts_reshaped.float())

        return img_fts_layers

    def _log_depth_maps(
        self,
        batch: E2EDataBatch,
        images: torch.Tensor,
        depth_maps: torch.Tensor,
        gt_depth_maps: torch.Tensor,
    ):
        for i in range(images.shape[0]):
            fig, ax = plt.subplots(1, 3)

            ax[0].imshow(images.permute(0, 1, 3, 4, 2)[i, 1].cpu().numpy())
            ax[1].imshow((1 / (depth_maps[i, 1] + 1e-6)).cpu().numpy())
            ax[2].imshow((1 / (gt_depth_maps[i, 1] + 1e-6)).cpu().numpy())

            scenario_id = batch.samples[i].metadata["scenario_id"]
            if isinstance(self.logger, CometLogger):
                self.logger.experiment.log_figure(
                    figure_name=scenario_id,
                    figure=fig,
                    step=self.global_step,
                )

            plt.close(fig)
