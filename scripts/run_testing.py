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

from torch.utils.data import DataLoader
import hydra
from omegaconf import DictConfig, OmegaConf
from hydra.utils import instantiate
from pytorch_lightning.loggers import CometLogger, TensorBoardLogger
from pytorch_lightning import seed_everything

seed_everything(0)


@hydra.main(
    version_base=None, config_path="../ad_vla/conf", config_name="default_testing"
)
def main(cfg: DictConfig):
    print(OmegaConf.to_yaml(cfg))

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
    test_dataset = instantiate(cfg.data.val_dataset)

    collator = model.get_collator(inference_mode=True)
    test_dataloader = DataLoader(
        test_dataset,
        shuffle=False,
        collate_fn=collator,
        **cfg.data.dataloader_args,
    )

    # TRAINER SETUP, FIT AND VALIDATE
    train_stage_module = instantiate(cfg.train_stage, model=model, _recursive_=False)

    trainer = instantiate(cfg.trainer, logger=logger)
    trainer.test(train_stage_module, test_dataloader)


if __name__ == "__main__":
    main()
