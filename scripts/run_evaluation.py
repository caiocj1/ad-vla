import logging

logging.getLogger("httpx").setLevel(logging.WARNING)
logging.getLogger("httpcore").setLevel(logging.WARNING)

import os
from dotenv import load_dotenv

# ── Suppress TF/JAX CUDA warnings ────────────────────────────────────────────
# USE_TF=0 / USE_FLAX=0 tells HuggingFace Transformers to skip TF/JAX entirely,
os.environ["USE_TF"] = "0"
os.environ["USE_FLAX"] = "0"
os.environ["TF_CPP_MIN_LOG_LEVEL"] = "3"

load_dotenv()
# ──────────────────────────────────────────────────────────────────────────────

import torch

# TF32 tensor cores for FP32 matmuls (the flow-matching expert is kept in FP32; bf16 backbones
# are unaffected). Inputs are rounded to a 10-bit mantissa, accumulation stays FP32.
torch.set_float32_matmul_precision("high")
import hydra
from omegaconf import DictConfig, OmegaConf
from hydra.utils import instantiate
from pytorch_lightning.loggers import CometLogger, TensorBoardLogger
from pytorch_lightning import seed_everything

from ad_vla.dataset.builder import (
    QA,
    build_val_sets,
    primary_val_name,
    reject_old_keys,
)

seed_everything(0)


@hydra.main(
    version_base=None, config_path="../ad_vla/conf", config_name="default_evaluation"
)
def main(cfg: DictConfig):
    print(OmegaConf.to_yaml(cfg))
    if "dataloader_args" in cfg.data:
        # An old key would otherwise be ignored silently (default batch size / workers).
        raise ValueError(
            "data.dataloader_args was renamed: use data.val_dataloader_args (see ad_vla/conf/README.md)."
        )
    reject_old_keys(cfg.data)

    # LOGGER SETUP
    if not cfg.logging.disable_comet:
        logger = CometLogger(
            api_key=cfg.logging.comet_api_key,
            workspace=cfg.logging.comet_workspace,
            project_name=cfg.logging.comet_project_name,
            experiment_name=cfg.logging.version,
        )
        logger.experiment.log_asset_data(OmegaConf.to_yaml(cfg), name="config.yaml")
        logger.experiment.log_parameters(OmegaConf.to_container(cfg))
    else:
        logger = TensorBoardLogger(".", version=cfg.logging.version)

    # LOAD MODEL
    model = instantiate(cfg.model)
    total_params = sum(p.numel() for p in model.parameters())
    print(f"Total parameters: {total_params:,}")

    # PREPARE DATA
    val_sets = build_val_sets(cfg.data, model)

    # TRAINER SETUP, FIT AND VALIDATE
    train_stage_module = instantiate(
        cfg.train_stage,
        model=model,
        val_target_sampling=cfg.data.val_target_sampling,
        _recursive_=False,
    )

    train_stage_module.configure_val_sets(
        [val_set.name for val_set in val_sets],
        primary_val_name(cfg.data),
        qa_names=[val_set.name for val_set in val_sets if val_set.task == QA],
    )

    trainer = instantiate(cfg.trainer, logger=logger)
    trainer.validate(train_stage_module, [val_set.dataloader() for val_set in val_sets])


if __name__ == "__main__":
    main()
