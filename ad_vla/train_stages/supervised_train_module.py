import torch
from omegaconf import DictConfig
from pytorch_lightning.loggers import CometLogger

from ad_vla.models.base_traj_planner import BaseTrajPlanner
from ad_vla.dataset.data_types import E2EDataBatch, TrajectorySampling
from ad_vla.train_stages.base_module import BaseTrainingModule


class SupervisedTrainModule(BaseTrainingModule):
    def __init__(
        self,
        model: BaseTrajPlanner,
        optim_cfg: DictConfig,
        lora_cfg: DictConfig | None = None,
        loss_cfg: DictConfig | None = None,
        aux_head: DictConfig | None = None,
        aux_loss_weight: float = 1.0,
        aux_warmup_steps: int | None = None,
        val_log_every_n_steps: int = 2,
        val_target_sampling: DictConfig | dict | TrajectorySampling | None = None,
    ):
        super().__init__(
            model=model,
            optim_cfg=optim_cfg,
            lora_cfg=lora_cfg,
            loss_cfg=loss_cfg,
            aux_head=aux_head,
            aux_loss_weight=aux_loss_weight,
            aux_warmup_steps=aux_warmup_steps,
            val_log_every_n_steps=val_log_every_n_steps,
            val_target_sampling=val_target_sampling,
        )

    def training_step(self, batch: E2EDataBatch, batch_idx: int):
        qa_mask = batch.model_inputs.get("qa_mask")
        if (
            qa_mask is not None
            and self.global_rank == 0
            and isinstance(self.logger, CometLogger)
        ):
            qa_rows = int(qa_mask.sum().item())
            self.logger.log_metrics(
                {
                    "train/qa_rows_rank0_batch": qa_rows,
                    "train/qa_fraction_rank0_batch": qa_rows / qa_mask.numel(),
                },
                step=self.trainer.fit_loop.epoch_loop.total_batch_idx,
            )

        # Currently delegates loss calculation to model, and returns it for training
        if self.train_aux_head_only():
            with torch.no_grad():
                loss_dict = self.model.compute_loss(batch)
        else:
            loss_dict = self.model.compute_loss(batch)

        if self.aux_head is not None:
            loss_aux, _ = self.aux_head.compute_aux_loss(batch)
            aux_loss_name = getattr(self.aux_head, "loss_name", "aux")
            loss_dict[f"loss_{aux_loss_name}"] = loss_aux

            if self.train_aux_head_only():
                loss_dict["loss"] = loss_aux
            else:
                aux_loss_weight = self.current_aux_loss_weight()
                loss_dict["loss"] += aux_loss_weight * loss_aux

                self.log(
                    "train/aux_loss_weight",
                    aux_loss_weight,
                    on_step=True,
                    on_epoch=False,
                    batch_size=len(batch.samples),
                    sync_dist=False,
                )

        for k, v in loss_dict.items():
            self.log(
                f"train/{k}",
                v,
                on_step=True,
                on_epoch=False,
                batch_size=len(batch.samples),
                sync_dist=False,
            )
        return loss_dict["loss"]

    # def validation_step(self, batch: E2EDataBatch, batch_idx: int):
    #     if self.aux_head is not None and getattr(
    #         self.aux_head, "log_on_validation", False
    #     ):
    #         self.model(**batch.model_inputs)
    #         _, extra_aux = self.aux_head.compute_aux_loss(batch)
    #         if batch_idx % self.val_log_every_n_steps == 0:
    #             self.aux_head.log_validation_outputs(
    #                 self.logger,
    #                 batch,
    #                 extra_aux,
    #                 self.global_step,
    #             )

    #     # Continue with normal validation step if also training backbone
    #     if not self.train_aux_head_only():
    #         super().validation_step(batch, batch_idx)
