import os
from dotenv import load_dotenv
import logging

logging.getLogger("httpx").setLevel(logging.WARNING)
logging.getLogger("httpcore").setLevel(logging.WARNING)

# ── Suppress TF/JAX CUDA warnings (see ad_vla/dataset/waymo/tf_importer.py) ─
os.environ["USE_TF"] = "0"
os.environ["USE_FLAX"] = "0"
os.environ["TF_CPP_MIN_LOG_LEVEL"] = "3"

load_dotenv()
# ──────────────────────────────────────────────────────────────────────────────

import torch

# TF32 tensor cores for FP32 matmuls (the flow-matching expert is kept in FP32; bf16 backbones
# are unaffected). Inputs are rounded to a 10-bit mantissa, accumulation stays FP32.
torch.set_float32_matmul_precision("high")
from torch.utils.data import DataLoader
from pathlib import Path

import hydra
from omegaconf import DictConfig, OmegaConf
from hydra.utils import instantiate
from pytorch_lightning.loggers import CometLogger, TensorBoardLogger
from pytorch_lightning import seed_everything
from pytorch_lightning.utilities.rank_zero import rank_zero_only

from ad_vla.train_stages.resume import (
    FullCheckpointIO,
    ResumableSampler,
    ResumeDataPosition,
    resume_checkpoint_path,
)
from ad_vla.dataset.builder import (
    QA,
    build_train_dataset,
    build_val_sets,
    check_model_trains_tasks,
    primary_val_name,
    reject_old_keys,
)
from ad_vla.utils.export_utils import finalize_training_outputs, validate_export_cfg

seed_everything(0)


@hydra.main(
    version_base=None, config_path="../ad_vla/conf", config_name="default_training"
)
def main(cfg: DictConfig):
    print(OmegaConf.to_yaml(cfg))
    if "dataloader_args" in cfg.data:
        # An old key would otherwise be ignored silently (default batch size / workers).
        raise ValueError(
            "data.dataloader_args was renamed: use data.train_dataloader_args / data.val_dataloader_args (see ad_vla/conf/README.md)."
        )
    reject_old_keys(cfg.data)

    run_dir = Path(cfg.data.save_path) / cfg.logging.version
    resume_ckpt = resume_checkpoint_path(cfg.get("resume_ckpt_path"), run_dir)

    # LOGGER SETUP
    if not cfg.logging.disable_comet:
        # A resumed run continues its Comet experiment, so its curves stay in one place.
        comet_key_path = run_dir / "comet_experiment_key.txt"
        logger = CometLogger(
            api_key=cfg.logging.comet_api_key,
            workspace=cfg.logging.comet_workspace,
            project_name=cfg.logging.comet_project_name,
            experiment_name=cfg.logging.version,
            experiment_key=comet_key_path.read_text().strip() if resume_ckpt else None,
        )
        if not resume_ckpt:
            logger.experiment.log_asset_data(OmegaConf.to_yaml(cfg), name="config.yaml")
            logger.experiment.log_parameters(OmegaConf.to_container(cfg))
            if rank_zero_only.rank == 0:
                run_dir.mkdir(parents=True, exist_ok=True)
                comet_key_path.write_text(logger.experiment.get_key())
    else:
        logger = TensorBoardLogger(".", version=cfg.logging.version)

    # LOAD MODEL
    model = instantiate(cfg.model)
    print("param dtype:", next(model.parameters()).dtype)
    total_params = sum(p.numel() for p in model.parameters())
    print(f"Total parameters: {total_params:,}")

    # PREPARE DATA
    train_dataset = build_train_dataset(cfg.data)
    val_sets = build_val_sets(cfg.data, model, training=True)

    # Prepare the training collator and dataloader
    drop_cfg_raw = cfg.data.get("drop_cfg", None)
    drop_cfg = (
        OmegaConf.to_container(drop_cfg_raw) if drop_cfg_raw is not None else None
    )
    train_collator_kwargs = {"drop_cfg": drop_cfg}

    check_model_trains_tasks(cfg.data, model)

    # Share of training samples whose past trajectory is randomly shortened (FM-VLA only).
    past_truncation_prob = float(cfg.data.get("past_traj_truncation_prob", 0.0))
    if past_truncation_prob > 0.0:
        train_collator_kwargs["past_truncation_prob"] = past_truncation_prob

    train_collator = model.get_collator(**train_collator_kwargs)

    train_sampler = ResumableSampler(train_dataset)
    train_dataloader = DataLoader(
        train_dataset,
        sampler=train_sampler,
        drop_last=True,
        collate_fn=train_collator,
        **cfg.data.train_dataloader_args,
    )

    # TRAINER SETUP, FIT AND VALIDATE
    # data.val_target_sampling (as in run_evaluation.py) overrides train_stage.val_target_sampling.
    stage_kwargs = {}
    if cfg.data.get("val_target_sampling") is not None:
        stage_kwargs["val_target_sampling"] = cfg.data.val_target_sampling
    train_stage_module = instantiate(
        cfg.train_stage, model=model, _recursive_=False, **stage_kwargs
    )

    train_stage_module.configure_val_sets(
        [val_set.name for val_set in val_sets],
        primary_val_name(cfg.data),
        qa_names=[val_set.name for val_set in val_sets if val_set.task == QA],
    )

    trainer = instantiate(cfg.trainer, logger=logger)
    validate_export_cfg(cfg, trainer)
    trainer.callbacks.append(
        ResumeDataPosition(train_sampler, cfg.data.train_dataloader_args.batch_size)
    )
    trainer.strategy.checkpoint_io = FullCheckpointIO()
    trainer.fit(
        train_stage_module,
        train_dataloader,
        [val_set.dataloader(periodic=True) for val_set in val_sets],
        ckpt_path=resume_ckpt,
    )

    # FINAL VALIDATION AND SAVING
    trainer.validate(
        train_stage_module,
        [val_set.dataloader() for val_set in val_sets],
        ckpt_path=None,
    )
    finalize_training_outputs(cfg, trainer)


if __name__ == "__main__":
    main()
