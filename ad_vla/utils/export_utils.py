from __future__ import annotations

import argparse
import os
import shutil
import subprocess
import sys
import traceback
from pathlib import Path
from typing import Any

from omegaconf import DictConfig, OmegaConf


DISTRIBUTED_ENV_KEYS = (
    "RANK",
    "LOCAL_RANK",
    "WORLD_SIZE",
    "LOCAL_WORLD_SIZE",
    "GROUP_RANK",
    "ROLE_RANK",
    "ROLE_WORLD_SIZE",
    "MASTER_ADDR",
    "MASTER_PORT",
    "NODE_RANK",
)


def finalize_training_outputs(cfg: DictConfig, trainer: Any) -> None:
    run_dir = (Path(cfg.data.save_path) / cfg.logging.version).resolve()
    checkpoint_dir = run_dir / "checkpoints"
    checkpoint_dir.mkdir(parents=True, exist_ok=True)

    final_ckpt_path = (
        checkpoint_dir
        / f"epoch={trainer.current_epoch}-step={trainer.global_step}-final.ckpt"
    )
    resolved_config_path = run_dir / "resolved_training_config.yaml"
    status_path = run_dir / "post_training_export_status.txt"

    print(f"Saving final Lightning checkpoint: {final_ckpt_path}", flush=True)
    trainer.save_checkpoint(str(final_ckpt_path), weights_only=True)

    if trainer.is_global_zero:
        status_path.unlink(missing_ok=True)
        resolved_cfg = OmegaConf.create(OmegaConf.to_container(cfg, resolve=True))
        OmegaConf.save(config=resolved_cfg, f=resolved_config_path)
        _run_export_subprocess(
            config_path=resolved_config_path,
            checkpoint_path=final_ckpt_path,
            output_dir=run_dir,
            status_path=status_path,
        )

    trainer.strategy.barrier("post_training_export")

    if status_path.exists() and status_path.read_text().startswith("error\n"):
        raise RuntimeError(status_path.read_text())


def _run_export_subprocess(
    config_path: Path,
    checkpoint_path: Path,
    output_dir: Path,
    status_path: Path,
) -> None:
    status_path.parent.mkdir(parents=True, exist_ok=True)
    command = [
        sys.executable,
        "-m",
        "ad_vla.utils.export_utils",
        "--config",
        str(config_path),
        "--checkpoint",
        str(checkpoint_path),
        "--output-dir",
        str(output_dir),
    ]

    env = os.environ.copy()
    for key in DISTRIBUTED_ENV_KEYS:
        env.pop(key, None)
    env["USE_TF"] = "0"
    env["USE_FLAX"] = "0"
    env["TF_CPP_MIN_LOG_LEVEL"] = "3"

    try:
        repo_root = Path(__file__).resolve().parents[2]
        subprocess.run(command, check=True, cwd=repo_root, env=env)
    except BaseException:
        status_path.write_text("error\n" + traceback.format_exc())
    else:
        status_path.write_text("ok\n")


def export_from_lightning_checkpoint(
    config_path: str | Path,
    checkpoint_path: str | Path,
    output_dir: str | Path,
) -> None:
    import torch
    from hydra.utils import instantiate
    from safetensors import safe_open

    config_path = Path(config_path).resolve()
    checkpoint_path = Path(checkpoint_path).resolve()
    output_dir = Path(output_dir).resolve()
    hf_output_dir = output_dir / "hf_ckpt"
    aux_output_dir = output_dir / "aux_models"
    tmp_hf_output_dir = output_dir / "hf_ckpt.tmp_export"
    tmp_aux_output_dir = output_dir / "aux_models.tmp_export"

    cfg = OmegaConf.load(config_path)
    OmegaConf.set_struct(cfg, False)
    if "ckpt_path" in cfg.model:
        cfg.model.ckpt_path = None
    if "init_mode" in cfg.model:
        cfg.model.init_mode = "scratch"

    _reset_tmp_dir(tmp_hf_output_dir)
    _reset_tmp_dir(tmp_aux_output_dir)

    print(f"Loading Lightning checkpoint: {checkpoint_path}", flush=True)
    checkpoint = torch.load(checkpoint_path, map_location="cpu", weights_only=False)
    state_dict = checkpoint["state_dict"]
    print(f"Loaded state tensors={len(state_dict)}", flush=True)

    print("Instantiating model and training stage for export.", flush=True)
    model = instantiate(cfg.model)
    train_stage = instantiate(cfg.train_stage, model=model, _recursive_=False)

    missing, unexpected = train_stage.load_state_dict(state_dict, strict=False)
    print(
        f"load_state_dict missing={len(missing)} unexpected={len(unexpected)}",
        flush=True,
    )
    if missing:
        print(f"missing sample={list(missing)[:20]}", flush=True)
    if unexpected:
        print(f"unexpected sample={list(unexpected)[:20]}", flush=True)
        raise RuntimeError(
            "Unexpected state dict keys while restoring final checkpoint."
        )

    train_stage.eval()

    vlm = getattr(train_stage.model, "vlm", None)
    if vlm is not None and hasattr(vlm, "merge_and_unload"):
        print("Merging LoRA adapters into base VLM.", flush=True)
        train_stage.model.vlm = train_stage.model.vlm.merge_and_unload()

    if hasattr(train_stage.model, "tie_weights"):
        train_stage.model.tie_weights()

    if not hasattr(train_stage.model, "save_pretrained"):
        raise RuntimeError(
            f"Model does not support save_pretrained(): {type(train_stage.model)}"
        )

    print(f"Saving Hugging Face checkpoint: {tmp_hf_output_dir}", flush=True)
    _save_pretrained_without_deepspeed_unwrap(train_stage.model, tmp_hf_output_dir)
    _validate_hf_checkpoint(tmp_hf_output_dir, safe_open)
    _publish_dir(tmp_hf_output_dir, hf_output_dir)

    aux_head = getattr(train_stage, "aux_head", None)
    if aux_head is not None and hasattr(aux_head, "save_aux_model"):
        print(f"Saving auxiliary model: {tmp_aux_output_dir}", flush=True)
        tmp_aux_output_dir.mkdir(parents=True, exist_ok=False)
        aux_head.save_aux_model(str(tmp_aux_output_dir))
        _publish_dir(tmp_aux_output_dir, aux_output_dir)

    print(f"Finished post-training export in: {output_dir}", flush=True)


def _save_pretrained_without_deepspeed_unwrap(model: Any, output_dir: Path) -> None:
    from accelerate.utils import other as accelerate_other

    original = accelerate_other.is_deepspeed_available
    accelerate_other.is_deepspeed_available = lambda: False
    try:
        model.save_pretrained(output_dir, safe_serialization=True)
    finally:
        accelerate_other.is_deepspeed_available = original


def _validate_hf_checkpoint(output_dir: Path, safe_open: Any) -> None:
    config_path = output_dir / "config.json"
    weights_path = output_dir / "model.safetensors"
    if not config_path.exists() or not weights_path.exists():
        raise RuntimeError(f"Missing expected Hugging Face files in {output_dir}")
    with safe_open(weights_path, framework="pt", device="cpu") as handle:
        keys = list(handle.keys())
    print(
        f"Validated Hugging Face checkpoint: tensors={len(keys)} "
        f"has_lora={any('lora_' in key for key in keys)}",
        flush=True,
    )


def _reset_tmp_dir(path: Path) -> None:
    if path.exists():
        shutil.rmtree(path)


def _publish_dir(tmp_dir: Path, output_dir: Path) -> None:
    if output_dir.exists():
        if any(output_dir.iterdir()):
            raise RuntimeError(
                f"Refusing to overwrite non-empty directory: {output_dir}"
            )
        output_dir.rmdir()
    tmp_dir.rename(output_dir)


def main() -> None:
    parser = argparse.ArgumentParser(
        description="Export ad_vla hf_ckpt/ and aux_models/ from a Lightning checkpoint."
    )
    parser.add_argument(
        "--config", required=True, help="Resolved training config YAML."
    )
    parser.add_argument(
        "--checkpoint", required=True, help="Final Lightning checkpoint."
    )
    parser.add_argument(
        "--output-dir", required=True, help="Training run output directory."
    )
    args = parser.parse_args()

    export_from_lightning_checkpoint(args.config, args.checkpoint, args.output_dir)


if __name__ == "__main__":
    main()
