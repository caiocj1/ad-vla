from typing import Any

import torch
from hydra.utils import get_class
from omegaconf import DictConfig, OmegaConf
from transformers import PreTrainedModel

from ad_vla.models.base_traj_planner import BaseTrajPlanner


def _resolve_torch_dtype(dtype: str | None) -> torch.dtype | None:
    if dtype is None:
        return None
    # Accept "bfloat16", "float16", "float32", etc.
    if not hasattr(torch, dtype):
        raise ValueError(f"Unknown torch dtype string: {dtype}")
    return getattr(torch, dtype)


def instantiate_model(
    class_path: str,
    init_mode: str = "auto",
    ckpt_path: str | None = None,
    torch_dtype: str | None = "bfloat16",
    attn_implementation: str | None = "flash_attention_2",
    model_args: dict[str, Any] | None = None,
    config_overrides: dict[str, Any] | None = None,
) -> PreTrainedModel | BaseTrajPlanner:
    """
    Hydra-friendly model factory.

    init_mode:
      - "auto": if ckpt_path -> from_pretrained; else if has from_pretrained_submodules -> pretrained_submodules; else scratch
      - "from_pretrained": always use from_pretrained(ckpt_path)
      - "pretrained_submodules": use config_class.from_pretrained(base_model, **args) then from_pretrained_submodules
      - "scratch": config_class(**args) then model_class(model_config)

    ``ckpt_path`` may be a local exported HF directory such as ``hf_ckpt/``.
    The from_pretrained branch uses its saved configuration; ``model_args`` is
    not forwarded as configuration overrides. Dtype and attention implementation
    are passed explicitly. Use the class matching the checkpoint architecture.

    ``config_overrides`` (from_pretrained only) updates the checkpoint's saved configuration
    before the model is built, to warm-start a modified architecture from it; dict-valued
    entries are merged one level deep, e.g. ``{"expert_cfg": {"camera_geometry": "ray_kv"}}``
    keeps the saved expert_cfg and adds camera geometry. Weights the checkpoint lacks are
    initialized by the model (FlowMatchingVLA keeps zero-initialized layers at zero).
    """
    model_args = model_args or {}
    # If someone passes a DictConfig by accident, normalize to plain dict
    if isinstance(model_args, DictConfig):
        model_args = OmegaConf.to_container(model_args, resolve=True)  # type: ignore
    if isinstance(config_overrides, DictConfig):
        config_overrides = OmegaConf.to_container(config_overrides, resolve=True)  # type: ignore

    model_class = get_class(class_path)
    dtype = _resolve_torch_dtype(torch_dtype)

    # Decide mode automatically if requested
    if init_mode == "auto":
        if ckpt_path:
            init_mode = "from_pretrained"
        elif hasattr(model_class, "from_pretrained_submodules"):
            init_mode = "pretrained_submodules"
        else:
            init_mode = "scratch"

    if init_mode == "from_pretrained":
        if not ckpt_path:
            raise ValueError("init_mode=from_pretrained requires ckpt_path")
        # Transformers uses torch_dtype kwarg; keep attn_implementation if your model supports it.
        # If your exact model doesn’t accept attn_implementation here, move it into model_args/config.
        print(
            f"Loading checkpoint: {model_class}, {dtype} dtype, {attn_implementation} attn_implementation."
        )
        load_kwargs = {}
        if config_overrides:
            config = model_class.config_class.from_pretrained(ckpt_path)
            for key, value in config_overrides.items():
                current = getattr(config, key, None)
                if isinstance(current, dict) and isinstance(value, dict):
                    value = {**current, **value}
                setattr(config, key, value)
            print(f"Config overrides on the checkpoint's config: {config_overrides}")
            load_kwargs["config"] = config
        model = model_class.from_pretrained(
            ckpt_path,
            dtype=dtype,
            attn_implementation=attn_implementation,
            **load_kwargs,
        )
        if hasattr(model, "tie_weights"):
            model.tie_weights()
        if config_overrides:
            # Modules added by the overrides are initialized by the model, not loaded: make sure
            # nothing is left uninitialized.
            bad = [
                n
                for n, t in (*model.named_parameters(), *model.named_buffers())
                if not torch.isfinite(t).all()
            ]
            if bad:
                raise ValueError(
                    f"Non-finite weights after loading {ckpt_path} with config_overrides: {bad[:5]}"
                )
        if getattr(model, "insulate_backbone", False):
            # from_pretrained leaves every loaded weight trainable; keep the backbone frozen as
            # the model's __init__ does (otherwise fine-tuning from an hf_ckpt trains the VLM).
            model.vlm.requires_grad_(False)
        if hasattr(model, "calibration_mlp"):
            # Calibration features enter this MLP as FP32; its embeddings are
            # cast back to the backbone activation dtype by the model.
            model.calibration_mlp.float()
        return model

    if config_overrides:
        raise ValueError("config_overrides applies only to init_mode=from_pretrained.")

    if init_mode == "pretrained_submodules":
        base_model = model_args.get("vlm_name_or_path", None)
        if not base_model:
            raise ValueError(
                "pretrained_submodules requires model_args.vlm_name_or_path"
            )
        model_config = model_class.config_class.from_pretrained(
            base_model,
            **model_args,
        )
        return model_class.from_pretrained_submodules(model_config)

    if init_mode == "scratch":
        model_config = model_class.config_class(**model_args)
        return model_class(model_config)

    raise ValueError(f"Unknown init_mode: {init_mode}")
