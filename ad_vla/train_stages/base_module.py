import torch
import pytorch_lightning as pl
from collections.abc import Mapping
from torchmetrics import MetricCollection, SumMetric
from pytorch_lightning.loggers import CometLogger, TensorBoardLogger
import matplotlib.pyplot as plt
import os
from hydra.utils import instantiate
from omegaconf import DictConfig
from peft import LoraConfig, get_peft_model
from datetime import datetime
import json
from waymo_open_dataset.protos import (
    end_to_end_driving_submission_pb2 as wod_e2ed_submission_pb2,
)

from ad_vla.models.base_traj_planner import BaseTrajPlanner
from ad_vla.metrics.ade import ADE
from ad_vla.metrics.fde import FDE
from ad_vla.metrics.rfs import RFS
from ad_vla.dataset.waymo.waymo_utils import prepare_for_rfs, create_waymo_submission
from ad_vla.dataset.data_types import E2EDataBatch, TrajectorySampling
from ad_vla.utils.plots import plot_pred_in_cam, plot_pref_trajs
from ad_vla.dataset.waymo.waymo_dataset import WaymoE2EDataset
from ad_vla.dataset.kitscenes.kitscenes_dataset import KITScenesDataset
from ad_vla.utils.traj_utils import resample_tensor


class BaseTrainingModule(pl.LightningModule):
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
        val_target_sampling: Mapping | TrajectorySampling | None = None,
    ):
        super().__init__()
        self.save_hyperparameters(ignore=["model"])
        self.model = model

        self.optim_cfg = optim_cfg
        self.loss_cfg = loss_cfg
        self.aux_head = None
        self.aux_loss_weight = aux_loss_weight
        self.aux_warmup_steps = aux_warmup_steps
        self.val_log_every_n_steps = val_log_every_n_steps
        self.val_target_sampling = self._coerce_val_target_sampling(val_target_sampling)

        if aux_head is not None:
            self.aux_head = instantiate(aux_head, model=self.model)

        if lora_cfg is not None and not self.train_aux_head_only():
            peft_config = LoraConfig(**lora_cfg)
            self.model.vlm = get_peft_model(self.model.vlm, peft_config)
            self.model.vlm.print_trainable_parameters()

        self.build_metrics()
        self.qa_val_correct = SumMetric()
        self.qa_val_valid = SumMetric()
        self.qa_val_total = SumMetric()
        self._qa_val_seen = False

        self.test_predictions = []

    def training_step(self, batch: E2EDataBatch, batch_idx: int):
        # Child LightningModule implements training logic
        raise NotImplementedError

    def train_aux_head_only(self) -> bool:
        return bool(
            self.aux_head is not None
            and getattr(self.aux_head, "train_aux_head_only", False)
        )

    def current_aux_loss_weight(self) -> float:
        if self.aux_warmup_steps is not None and self.aux_warmup_steps > 0:
            return (
                min(1.0, self.trainer.global_step / self.aux_warmup_steps)
                * self.aux_loss_weight
            )
        return self.aux_loss_weight

    @staticmethod
    def _coerce_val_target_sampling(
        sampling: Mapping | TrajectorySampling | None,
    ) -> TrajectorySampling | None:
        if sampling is None:
            return None
        if isinstance(sampling, TrajectorySampling):
            return sampling
        if not isinstance(sampling, Mapping):
            raise TypeError(
                "val_target_sampling must be null, a mapping, or "
                f"TrajectorySampling; got {type(sampling).__name__}."
            )
        return TrajectorySampling(**dict(sampling))

    def _prepare_validation_trajectories(
        self,
        batch: E2EDataBatch,
    ) -> tuple[torch.Tensor, TrajectorySampling | None]:
        source_sampling = batch.samples[0].fut_traj_sampling
        gt_trajs = torch.stack([sample.fut_traj for sample in batch.samples])

        # Strict compatibility path: null retains the dataset tensor and sampling
        # exactly as validation did before target-grid overrides were supported.
        if self.val_target_sampling is None:
            return gt_trajs, source_sampling

        if source_sampling is None:
            raise ValueError(
                "Validation samples must define fut_traj_sampling when "
                "val_target_sampling is configured."
            )

        return (
            resample_tensor(gt_trajs, source_sampling, self.val_target_sampling),
            self.val_target_sampling,
        )

    def transfer_batch_to_device(self, batch, device, dataloader_idx):
        # QA validation batches deliberately contain a frozen annotation
        # dataclass. Keep the raw image/sample objects on CPU; the Qwen
        # processor creates tensors and predict_qa_answers moves those instead.
        if dataloader_idx == 1:
            return batch
        return super().transfer_batch_to_device(batch, device, dataloader_idx)

    def validation_step(
        self,
        batch,
        batch_idx: int,
        dataloader_idx: int = 0,
    ):
        if dataloader_idx == 1:
            self._validation_qa_step(batch)
            return
        if dataloader_idx != 0:
            raise ValueError(
                f"Unexpected validation dataloader index {dataloader_idx}."
            )

        # If training stage uses auxiliary head and it logs on validation,
        # get auxiliary head output and use its own function to properly log.
        if self.aux_head is not None and getattr(
            self.aux_head, "log_on_validation", False
        ):
            self.model(**batch.model_inputs)
            _, extra_aux = self.aux_head.compute_aux_loss(batch)
            if batch_idx % self.val_log_every_n_steps == 0:
                self.aux_head.log_validation_outputs(
                    self.logger,
                    batch,
                    extra_aux,
                    self.global_step,
                )

        # If training only auxiliary head, skip trajectory prediction.
        if self.train_aux_head_only():
            return

        model_inputs = batch.model_inputs

        gt_trajs, gt_traj_sampling = self._prepare_validation_trajectories(batch)

        with torch.no_grad():
            # model.predict_from_processed_inputs returns trajectories at
            # model.pred_traj_sampling unless gt_traj_sampling is passed.
            # Here, we want to make sure final trajectory shapes match
            # evaluation shapes from dataset itself
            predict_kwargs = {
                **model_inputs,
                "num_traj_samples": 1,
                "generate_cfg": None,  # Use default generation config from model
                "target_sampling": gt_traj_sampling,
            }
            if hasattr(self.model, "prepare_image_delta"):
                predict_kwargs["image_delta"] = self.model.prepare_image_delta(batch)
            pred_trajs, extra = self.model.predict_from_processed_inputs(
                **predict_kwargs
            )

        pred_dict = {
            "pred_trajs": pred_trajs,
            **{k: v for k, v in extra.items() if isinstance(v, torch.Tensor)},
        }
        gt_dict = {
            "gt_trajs": gt_trajs.to(pred_trajs.device),
        }
        if hasattr(batch.samples[0], "pref_trajs"):
            gt_dict.update(prepare_for_rfs(batch, self.device))

        self.val_metrics.update(pred_dict, gt_dict)

        if batch_idx % self.val_log_every_n_steps == 0:
            self._log_images(
                batch, pred_trajs.float(), gt_trajs, extra.get("reasoning_trace", [])
            )

        return

    def _validation_qa_step(self, qa_batch) -> None:
        """Accumulate exact NuScenesQA classification counts for one batch."""
        self._qa_val_seen = True
        predict_answers = getattr(self.model, "predict_qa_answers", None)
        if predict_answers is None:
            raise TypeError(
                f"{type(self.model).__name__} does not implement QA validation."
            )

        predictions = predict_answers(qa_batch)
        if len(predictions) != len(qa_batch):
            raise RuntimeError(
                "QA prediction count does not match the validation batch size."
            )
        targets = [qa_pair.answer for _, qa_pair in qa_batch]
        self.qa_val_correct.update(
            torch.tensor(
                sum(
                    prediction == target
                    for prediction, target in zip(predictions, targets)
                ),
                dtype=torch.float32,
                device=self.device,
            )
        )
        self.qa_val_valid.update(
            torch.tensor(
                sum(prediction is not None for prediction in predictions),
                dtype=torch.float32,
                device=self.device,
            )
        )
        self.qa_val_total.update(
            torch.tensor(len(targets), dtype=torch.float32, device=self.device)
        )

    def test_step(self, batch: E2EDataBatch, batch_idx: int):
        model_inputs = batch.model_inputs

        # Initialize proper trajectory sampling for test dataset
        dataset = self.trainer.test_dataloaders.dataset
        if isinstance(dataset, WaymoE2EDataset):
            gt_traj_sampling = TrajectorySampling(num_poses=20, interval_length=0.25)
        elif isinstance(dataset, KITScenesDataset):
            gt_traj_sampling = TrajectorySampling(num_poses=25, interval_length=0.2)
        else:
            raise Exception(f"Testing not implemented for dataset {dataset}.")

        # Output predictions
        with torch.no_grad():
            pred_trajs, extra = self.model.predict_from_processed_inputs(
                **model_inputs, target_sampling=gt_traj_sampling
            )

        # Save predictions to create submission at the end
        batch_size = len(batch.samples)
        for i in range(batch_size):
            if isinstance(dataset, WaymoE2EDataset):
                predicted_trajectory = wod_e2ed_submission_pb2.TrajectoryPrediction(
                    pos_x=pred_trajs[i, :, 0].cpu().numpy(),
                    pos_y=pred_trajs[i, :, 1].cpu().numpy(),
                )
                frame_name = batch.samples[i].metadata["scenario_id"]
                frame_trajectory = wod_e2ed_submission_pb2.FrameTrajectoryPredictions(
                    frame_name=frame_name,
                    trajectory=predicted_trajectory,
                )
                self.test_predictions.append(frame_trajectory)
            elif isinstance(dataset, KITScenesDataset):
                cur_pred_traj = pred_trajs[i].cpu().tolist()
                cur_pred_traj = [[round(x, 2), round(y, 2)] for x, y in cur_pred_traj]
                self.test_predictions.append(
                    {
                        "scenario_id": batch.samples[i].metadata["scenario_id"],
                        "future_trajectory": cur_pred_traj,
                    }
                )
            else:
                raise Exception(f"Testing not implemented for dataset {dataset}.")

    def configure_optimizers(self):
        trainable_params = [p for p in self.parameters() if p.requires_grad]
        parameter_group_lrs = self.optim_cfg.get("parameter_group_lrs")

        group_names = ("vlm", "action_branch")
        if parameter_group_lrs is None or all(
            parameter_group_lrs.get(name) is None for name in group_names
        ):
            optimizer = instantiate(self.optim_cfg.optimizer, params=trainable_params)
        else:
            module_groups = {
                "vlm": (getattr(self.model, "vlm", None),),
                "action_branch": tuple(
                    getattr(self.model, name, None)
                    for name in (
                        "action_expert",
                        "state_encoder",
                        "action_in_proj",
                        "action_out_proj",
                    )
                ),
            }
            ungrouped_ids = {id(param) for param in trainable_params}
            optimizer_params = []

            for group_name, modules in module_groups.items():
                group_lr = parameter_group_lrs.get(group_name)
                if group_lr is None:
                    continue
                module_param_ids = {
                    id(param)
                    for module in modules
                    if module is not None
                    for param in module.parameters()
                }
                group_params = [
                    param
                    for param in trainable_params
                    if id(param) in module_param_ids and id(param) in ungrouped_ids
                ]
                if group_params:
                    optimizer_params.append(
                        {"params": group_params, "lr": float(group_lr)}
                    )
                    ungrouped_ids.difference_update(id(param) for param in group_params)

            remaining_params = [
                param for param in trainable_params if id(param) in ungrouped_ids
            ]
            if remaining_params:
                optimizer_params.append({"params": remaining_params})

            optimizer = instantiate(
                self.optim_cfg.optimizer,
                params=optimizer_params,
                _convert_="all",
            )

        lr_scheduler = instantiate(
            self.optim_cfg.lr_scheduler,
            optimizer=optimizer,
            T_max=self.trainer.estimated_stepping_batches,
        )
        lr_interval = self.optim_cfg.lr_interval

        lr_scheduler_config = {
            "scheduler": lr_scheduler,
            "interval": lr_interval,
        }

        optimizer_config = {
            "optimizer": optimizer,
            "lr_scheduler": lr_scheduler_config,
        }

        return optimizer_config

    ####################################
    ##### METRICS, LOGGING, SAVING #####
    ####################################

    def on_validation_epoch_end(self):
        val_dict = self.val_metrics.compute()

        self.log_dict(
            val_dict, on_step=False, on_epoch=True, prog_bar=False, sync_dist=True
        )

        self.val_metrics.reset()

        if self._qa_val_seen:
            qa_total = self.qa_val_total.compute()
            self.log_dict(
                {
                    "eval_qa/accuracy": self.qa_val_correct.compute() / qa_total,
                    "eval_qa/valid_fraction": self.qa_val_valid.compute() / qa_total,
                    "eval_qa/questions": qa_total,
                },
                on_step=False,
                on_epoch=True,
                prog_bar=False,
                # SumMetric synchronizes its state before compute().
                sync_dist=False,
            )
        self.qa_val_correct.reset()
        self.qa_val_valid.reset()
        self.qa_val_total.reset()
        self._qa_val_seen = False

    def on_test_epoch_end(self) -> None:
        world_size = self.trainer.world_size

        if world_size > 1:
            # Prepare output list — ALL ranks must participate
            gathered = [None] * world_size
            torch.distributed.all_gather_object(gathered, self.test_predictions)

            if not self.trainer.is_global_zero:
                return

            # gathered is a list of lists, one per rank
            all_predictions = []
            for rank_preds in gathered:
                all_predictions.extend(rank_preds)
        else:
            all_predictions = self.test_predictions

        dataset = self.trainer.test_dataloaders.dataset
        if isinstance(dataset, WaymoE2EDataset):
            create_waymo_submission(all_predictions)
        elif isinstance(dataset, KITScenesDataset):
            timestamp = datetime.now().strftime("%d-%m-%Y_%H-%M-%S")
            os.makedirs("kit_subs/", exist_ok=True)
            with open(
                f"kit_subs/submission_{timestamp}.jsonl", "w", encoding="utf-8"
            ) as f:
                for item in all_predictions:
                    f.write(json.dumps(item) + "\n")

    def build_metrics(self):
        self.val_metrics = MetricCollection(
            {
                "ADE": ADE(),
                "FDE": FDE(),
                "RFS": RFS(),
            },
            prefix="eval/",
        )

    def _log_images(self, batch, pred_at_gt, gt_trajs, reasoning_traces):
        """Log visualization images to comet and/or disk."""
        for sample_idx, sample in enumerate(batch.samples):
            cameras = sample.cameras[-1]
            front_cam = cameras["FRONT"]
            if front_cam is None:
                continue

            cam_dict = [cameras["FRONT_LEFT"], front_cam, cameras["FRONT_RIGHT"]]
            pred_cot = (
                reasoning_traces[sample_idx]
                if sample_idx < len(reasoning_traces)
                else None
            )
            scenario_id = sample.metadata["scenario_id"]
            pref_trajs = getattr(sample, "pref_trajs", None)

            fig = plot_pred_in_cam(
                cam_dict,
                past_traj=sample.agent_input["past_traj"],
                pred_traj=pred_at_gt[sample_idx],
                fut_traj=gt_trajs[sample_idx],
                pred_cot=pred_cot,
                gt_cot=sample.reasoning_trace,
            )
            if pref_trajs is not None:
                fig = plot_pref_trajs(pref_trajs, fig)

            if isinstance(self.logger, CometLogger):
                self.logger.experiment.log_figure(
                    figure_name=scenario_id,
                    figure=fig,
                    step=self.global_step,
                )
            elif isinstance(self.logger, TensorBoardLogger):
                self.logger.experiment.add_figure(
                    tag=scenario_id,
                    figure=fig,
                    global_step=self.global_step,
                )

            plt.close(fig)
