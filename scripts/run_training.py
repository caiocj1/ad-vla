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
from torch.utils.data import Subset, DataLoader
import hydra
from omegaconf import DictConfig, OmegaConf
from hydra.utils import instantiate
from pytorch_lightning.loggers import CometLogger, TensorBoardLogger
from pytorch_lightning import seed_everything

from ad_vla.utils.export_utils import finalize_training_outputs
from ad_vla.dataset.nuscenes.nuscenes_dataset import (
    NuScenesQAPairDataset,
    collate_nuscenes_qa_pairs,
)

seed_everything(0)


def _deterministic_qa_subset(dataset, max_questions, seed):
    """Select a fixed, representative question subset without changing order."""
    if max_questions is None or int(max_questions) >= len(dataset):
        return dataset
    max_questions = int(max_questions)
    if max_questions < 1:
        raise ValueError("data.qa_val_max_questions must be at least one or null.")

    generator = torch.Generator().manual_seed(int(seed))
    indices = torch.randperm(len(dataset), generator=generator)[:max_questions]
    # Preserve dataset order so questions from one scene remain adjacent and
    # the scene/image cache can be reused by each DataLoader worker.
    indices = indices.sort().values.tolist()
    return Subset(dataset, indices)


@hydra.main(
    version_base=None, config_path="../ad_vla/conf", config_name="default_training"
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
    print("param dtype:", next(model.parameters()).dtype)
    total_params = sum(p.numel() for p in model.parameters())
    print(f"Total parameters: {total_params:,}")

    # PREPARE DATA
    train_dataset = instantiate(
        cfg.data.train_dataset, reasoning_traces_path=cfg.data.reasoning_traces_path
    )
    val_dataset = instantiate(
        cfg.data.val_dataset, reasoning_traces_path=cfg.data.reasoning_traces_path
    )
    if cfg.data.train_subset_path is not None:
        train_subset_idxs = torch.load(cfg.data.train_subset_path)
        train_dataset = Subset(train_dataset, train_subset_idxs)

    if cfg.data.val_subset_path is not None:
        val_subset_idxs = torch.load(cfg.data.val_subset_path)
        val_dataset = Subset(val_dataset, val_subset_idxs)
        if (
            cfg.data.train_dataset.split == cfg.data.val_dataset.split
            and cfg.data.train_dataset._target_ == cfg.data.val_dataset._target_
        ):
            train_size = int(len(val_subset_idxs) * cfg.data.train_val_split)
            train_subset_idxs = torch.arange(train_size)
            train_dataset = Subset(val_dataset, train_subset_idxs)
            val_dataset = Subset(
                val_dataset, torch.arange(train_size, len(val_subset_idxs))
            )

    # Prepare dataset for in-training validations
    train_val_dataset = (
        val_dataset
        if len(val_dataset) < 1000
        else Subset(val_dataset, torch.arange(1000))
    )

    # Prepare data collators for training and validation
    drop_cfg_raw = cfg.data.get("drop_cfg", None)
    drop_cfg = (
        OmegaConf.to_container(drop_cfg_raw) if drop_cfg_raw is not None else None
    )
    train_collator_kwargs = {"drop_cfg": drop_cfg}
    qa_probability = float(cfg.data.get("qa_probability", 0.0))
    if not 0.0 <= qa_probability <= 1.0:
        raise ValueError("data.qa_probability must be between zero and one.")
    if qa_probability > 0.0:
        train_collator_kwargs["qa_probability"] = qa_probability
    train_collator = model.get_collator(**train_collator_kwargs)
    val_collator = model.get_collator(inference_mode=True)

    # Instantiate dataloaders
    train_dataloader = DataLoader(
        train_dataset,
        shuffle=True,
        drop_last=True,
        collate_fn=train_collator,
        **cfg.data.train_dataloader_args,
    )
    train_val_dataloader = DataLoader(
        train_val_dataset,
        shuffle=False,
        collate_fn=val_collator,
        **cfg.data.val_dataloader_args,
    )

    # Prepare QA separate dataloader if present
    qa_val_dataloader = None
    qa_val_dataset_cfg = cfg.data.get("qa_val_dataset")
    if qa_val_dataset_cfg is not None:
        if not hasattr(model, "predict_qa_answers"):
            raise TypeError(f"{type(model).__name__} does not implement QA validation.")
        qa_scene_dataset = instantiate(qa_val_dataset_cfg)
        qa_val_dataset = NuScenesQAPairDataset(qa_scene_dataset)
        qa_val_dataset = _deterministic_qa_subset(
            qa_val_dataset,
            cfg.data.get("qa_val_max_questions"),
            cfg.data.get("qa_val_subset_seed", 0),
        )
        qa_val_dataloader = DataLoader(
            qa_val_dataset,
            shuffle=False,
            collate_fn=collate_nuscenes_qa_pairs,
            **cfg.data.qa_val_dataloader_args,
        )
        print(
            f"QA validation dataset: {len(qa_scene_dataset):,} scenes, "
            f"{len(qa_val_dataset):,} deterministic questions."
        )

    # TRAINER SETUP, FIT AND VALIDATE
    train_stage_module = instantiate(cfg.train_stage, model=model, _recursive_=False)

    trainer = instantiate(cfg.trainer, logger=logger)
    fit_val_dataloaders = (
        train_val_dataloader
        if qa_val_dataloader is None
        else [train_val_dataloader, qa_val_dataloader]
    )
    trainer.fit(train_stage_module, train_dataloader, fit_val_dataloaders)

    # FINAL VALIDATION AND SAVING
    val_dataloader = DataLoader(
        val_dataset,
        shuffle=False,
        collate_fn=val_collator,
        **cfg.data.val_dataloader_args,
    )
    if qa_val_dataloader is None:
        final_val_dataloaders = val_dataloader
    else:
        final_qa_val_dataloader = DataLoader(
            qa_val_dataset,
            shuffle=False,
            collate_fn=collate_nuscenes_qa_pairs,
            **cfg.data.qa_val_dataloader_args,
        )
        final_val_dataloaders = [val_dataloader, final_qa_val_dataloader]
    trainer.validate(train_stage_module, final_val_dataloaders, ckpt_path=None)
    finalize_training_outputs(cfg, trainer)


if __name__ == "__main__":
    main()
