from __future__ import annotations

from pathlib import Path
from typing import Any

import pytest
import pytorch_lightning as pl
import torch
from hydra import compose, initialize_config_dir
from hydra.utils import instantiate
from huggingface_hub.errors import (
    EntryNotFoundError,
    GatedRepoError,
    HfHubHTTPError,
    LocalEntryNotFoundError,
    RepositoryNotFoundError,
)
from pytorch_lightning.callbacks import ModelCheckpoint
from torch.utils.data import DataLoader, Dataset

from ad_vla.dataset.data_types import (
    CalibrationDict,
    CameraData,
    E2EDataSample,
    TrajectorySampling,
)
from ad_vla.train_stages.grpo_module import GRPOModule


CUDA_REQUIRED = pytest.mark.skipif(
    not torch.cuda.is_available(),
    reason="CUDA is required for these integration tests.",
)
pytestmark = [pytest.mark.gpu_integration]

_CONF_DIR = Path(__file__).resolve().parent.parent / "ad_vla" / "conf"
_EXTERNAL_MODEL_ERRORS = (
    EntryNotFoundError,
    GatedRepoError,
    HfHubHTTPError,
    LocalEntryNotFoundError,
    OSError,
    RepositoryNotFoundError,
)

# Models that load a HuggingFace VLM backbone and need the tiny swap.
_VLM_MODELS = {
    "text_traj_vlm",
    "fm_vla",
}

# Model → prompt config mapping (only needed when the default drive_poutine
# is wrong for a given model).
_PROMPT_OVERRIDES: dict[str, str] = {}


class _SampleDataset(Dataset):
    def __init__(self, samples: list[E2EDataSample]) -> None:
        self._samples = samples

    def __len__(self) -> int:
        return len(self._samples)

    def __getitem__(self, idx: int) -> E2EDataSample:
        return self._samples[idx].model_copy(deep=True)


def _build_mock_samples() -> list[E2EDataSample]:
    height, width = 64, 64
    camera_names = ["FRONT_LEFT", "FRONT", "FRONT_RIGHT", "FRONT_TELE"]
    samples: list[E2EDataSample] = []

    for sample_idx in range(2):
        calibration = CalibrationDict(
            intrinsics=torch.eye(3),
            distortion=torch.zeros(5),
            translation=torch.tensor([0.0, float(sample_idx), 0.0]),
            rotation=torch.eye(3),
            original_img_size=(height, width),
            is_flu=False,
        )

        cameras = []
        for timestep in range(4):
            timestep_cameras = {}
            for camera_idx, camera_name in enumerate(camera_names):
                image_value = (timestep + 1) * (camera_idx + 2 + sample_idx) * 16
                image = torch.full(
                    (height, width, 3),
                    fill_value=image_value % 255,
                    dtype=torch.uint8,
                )
                timestep_cameras[camera_name] = CameraData(
                    image=image,
                    cal_dict=calibration.model_copy(deep=True),
                )
            cameras.append(timestep_cameras)

        phase = 0.25 * sample_idx
        lateral_scale = 0.75 + 0.15 * sample_idx
        future_scale = 1.5 + 0.1 * sample_idx

        past_x = torch.linspace(-20.0, -0.5, 16)
        past_y = lateral_scale * torch.sin(torch.linspace(0.0, torch.pi, 16) + phase)
        future_x = torch.linspace(0.5, 10.0, 20)
        future_y = future_scale * torch.sin(
            torch.linspace(0.0, torch.pi / 3.0, 20) + phase
        )

        samples.append(
            E2EDataSample(
                metadata={
                    "scenario_id": f"mock-scene-000{sample_idx + 1}",
                    "sampling_freq": 10,
                    "dataset": "mock",
                },
                agent_input={
                    "past_traj": torch.stack([past_x, past_y], dim=-1),
                },
                cameras=cameras,
                intent="GO_STRAIGHT",
                fut_traj=torch.stack([future_x, future_y], dim=-1),
                fut_traj_sampling=TrajectorySampling(
                    num_poses=20,
                    interval_length=0.25,
                ),
                reasoning_trace=(
                    "The ego vehicle should continue smoothly through the open lane "
                    f"while maintaining a stable path for sample {sample_idx + 1}."
                ),
            )
        )

    return samples


def _compose_cfg(model_name: str, overrides: list[str] | None = None):
    all_overrides = [f"model={model_name}"]

    # Swap prompt config when the model needs a different one.
    if model_name in _PROMPT_OVERRIDES:
        all_overrides.append(f"prompt={_PROMPT_OVERRIDES[model_name]}")

    # Swap the real VLM backbone for a tiny random one.
    if model_name in _VLM_MODELS:
        all_overrides.append("model.model_args.vlm_name_or_path=tiny-random/qwen3.5")

    if overrides:
        all_overrides.extend(overrides)
    with initialize_config_dir(version_base=None, config_dir=str(_CONF_DIR)):
        return compose(
            config_name="default_integration",
            overrides=all_overrides,
        )


def _instantiate_model_or_skip(cfg):
    try:
        return instantiate(cfg.model)
    except _EXTERNAL_MODEL_ERRORS as exc:
        model_id = cfg.model.model_args.get(
            "vlm_name_or_path",
            cfg.model.model_args.get("model_name_or_path", "external model"),
        )
        pytest.skip(f"Unable to load {model_id}: {exc}")


def _build_dataloader(
    cfg,
    model,
    samples: list[E2EDataSample],
    *,
    inference_mode: bool,
):
    try:
        collator = model.get_collator(inference_mode=inference_mode)
    except TypeError:
        collator = model.get_collator()

    dataset = _SampleDataset(samples)
    return DataLoader(
        dataset,
        shuffle=False,
        collate_fn=collator,
        **(
            cfg.data.val_dataloader_args
            if inference_mode
            else cfg.data.train_dataloader_args
        ),
    )


def _build_train_module(cfg, model) -> pl.LightningModule:
    module = instantiate(cfg.train_stage, model=model, _recursive_=False)
    module._log_images = lambda *args, **kwargs: None  # type: ignore[method-assign]
    return module


def _build_trainer(cfg, tmp_path: Path):
    checkpoint_callback = ModelCheckpoint(
        dirpath=str(tmp_path / "checkpoints"),
        save_top_k=0,
        save_last=False,
    )
    return instantiate(
        cfg.trainer,
        logger=False,
        callbacks=[checkpoint_callback],
        default_root_dir=str(tmp_path),
    )


def _assert_validation_metrics(results: list[dict[str, Any]]) -> None:
    assert len(results) == 1
    metrics = results[0]
    assert {"eval/ADE", "eval/FDE", "eval/RFS"} <= metrics.keys()
    assert float(metrics["eval/ADE"]) >= 0.0
    assert float(metrics["eval/FDE"]) >= 0.0
    assert float(metrics["eval/RFS"]) == pytest.approx(-1.0)


@CUDA_REQUIRED
@pytest.mark.parametrize(
    "model_name",
    [
        "text_traj_vlm",
        "fm_vla",
    ],
)
def test_single_fit_and_validate_for_trainable_models(model_name: str, tmp_path: Path):
    cfg = _compose_cfg(model_name)
    samples = _build_mock_samples()
    model = _instantiate_model_or_skip(cfg)
    module = _build_train_module(cfg, model)
    train_loader = _build_dataloader(cfg, model, samples, inference_mode=False)
    val_loader = _build_dataloader(cfg, model, samples, inference_mode=True)
    trainer = _build_trainer(cfg, tmp_path)

    trainer.fit(module, train_dataloaders=train_loader, val_dataloaders=val_loader)
    assert trainer.global_step == 1

    results = trainer.validate(module, dataloaders=val_loader, verbose=False)
    _assert_validation_metrics(results)


@CUDA_REQUIRED
def test_fm_vla_polar_actions_fit_and_validate(tmp_path: Path):
    # Polar actions require normalization stats on the model's 20 x 0.25 s grid.
    rows = ",".join(["[0.0,2.0]"] * 20)
    std_rows = ",".join(["[0.05,1.0]"] * 20)
    cfg = _compose_cfg(
        "fm_vla",
        overrides=[
            "model.model_args.action_representation=polar",
            f"++model.model_args.action_stats={{mean:[{rows}],std:[{std_rows}]}}",
        ],
    )
    samples = _build_mock_samples()
    model = _instantiate_model_or_skip(cfg)
    module = _build_train_module(cfg, model)
    train_loader = _build_dataloader(cfg, model, samples, inference_mode=False)
    val_loader = _build_dataloader(cfg, model, samples, inference_mode=True)
    trainer = _build_trainer(cfg, tmp_path)

    trainer.fit(module, train_dataloaders=train_loader, val_dataloaders=val_loader)
    assert trainer.global_step == 1

    results = trainer.validate(module, dataloaders=val_loader, verbose=False)
    _assert_validation_metrics(results)


@CUDA_REQUIRED
@pytest.mark.parametrize(
    "past_traj_overrides",
    [
        [],
        [
            "model.model_args.past_traj_sampling.num_poses=4",
            "model.model_args.past_traj_sampling.interval_length=0.5",
            "model.model_args.expert_cfg.past_traj_mode=pooled",
        ],
    ],
    ids=["per_pose_16x0.25", "pooled_4x0.5"],
)
def test_fm_vla_past_traj_tokens_and_multi_sample_eval(
    past_traj_overrides: list[str], tmp_path: Path
):
    cfg = _compose_cfg(
        "fm_vla",
        overrides=[
            "model.model_args.expert_cfg.past_traj_tokens=true",
            "model.model_args.diff_cfg.num_repeat_steps=2",
            "model.model_args.diff_cfg.num_eval_samples=4",
            *past_traj_overrides,
        ],
    )
    samples = _build_mock_samples()
    model = _instantiate_model_or_skip(cfg)
    assert model.action_expert.past_traj_encoder is not None
    module = _build_train_module(cfg, model)
    train_loader = _build_dataloader(cfg, model, samples, inference_mode=False)
    val_loader = _build_dataloader(cfg, model, samples, inference_mode=True)
    trainer = _build_trainer(cfg, tmp_path)

    trainer.fit(module, train_dataloaders=train_loader, val_dataloaders=val_loader)
    assert trainer.global_step == 1

    results = trainer.validate(module, dataloaders=val_loader, verbose=False)
    _assert_validation_metrics(results)
    assert "eval/ADE_single" in results[0]

    # The train-stage override takes precedence over the model's own eval settings.
    module.val_generate_cfg = {"num_eval_samples": 1, "eval_init": "zero"}
    results = trainer.validate(module, dataloaders=val_loader, verbose=False)
    _assert_validation_metrics(results)
    assert "eval/ADE_single" not in results[0]


@CUDA_REQUIRED
@pytest.mark.parametrize(
    "model_name",
    ["text_traj_vlm"],
)
def test_grpo_single_fit_and_validate(model_name: str, tmp_path: Path):
    cfg = _compose_cfg(model_name, overrides=["train_stage=grpo"])
    samples = _build_mock_samples()
    model = _instantiate_model_or_skip(cfg)
    module = _build_train_module(cfg, model)
    assert isinstance(module, GRPOModule)

    train_loader = _build_dataloader(cfg, model, samples, inference_mode=False)
    val_loader = _build_dataloader(cfg, model, samples, inference_mode=True)
    trainer = _build_trainer(cfg, tmp_path)

    trainer.fit(module, train_dataloaders=train_loader, val_dataloaders=val_loader)
    assert trainer.global_step == 1

    results = trainer.validate(module, dataloaders=val_loader, verbose=False)
    _assert_validation_metrics(results)
