"""Builds the datasets and dataloaders named in ``data.train``, ``data.val``, ``data.test`` and
``data.annotate``.

Each entry of those two mappings is a dataset config plus a few settings read here, so that a
config states everything about a dataset in one place::

    data:
      train:
        waymo: {size: [608, 544]}
        navsim: {size: [448, 768], img_transform: center_crop_resize}
        nuscenes_qa: {size: [384, 672], task: qa, questions_per_frame: 3}
      val:
        waymo: {size: [608, 544], subset: ${oc.env:ASSETS_DIR}/indices/waymo_pref_idx.pt}
        nuscenes_qa: {task: qa, max_questions: 1000}
      primary_val: waymo
"""

from dataclasses import dataclass
from pathlib import Path
from typing import Any, get_args, Callable

import torch
from hydra.utils import instantiate
from omegaconf import DictConfig, OmegaConf
from torch.utils.data import DataLoader, Dataset, Subset

from ad_vla.dataset.camera_config import StandardCamera
from ad_vla.dataset.concat_dataset import E2EConcatDataset
from ad_vla.dataset.qa_datasets import (
    QAPairDataset,
    QATrainDataset,
    collate_qa_pairs,
)

TRAJECTORY, QA = "trajectory", "qa"
TRANSFORM_DIR = Path(__file__).parents[1] / "conf" / "img_transform"
# Settings of an entry that the builder reads; everything else goes to the dataset.
BUILDER_KEYS = {
    "task",
    "size",
    "camera_sizes",
    "repeat",
    "subset",
    "subset_fraction",
    "max_samples",
    "max_questions",
    "questions_per_frame",
    "subset_seed",
    "dataloader_args",
}
OLD_KEYS = {
    "train_dataset",
    "val_dataset",
    "qa_val_dataset",
    "train_subset_path",
    "val_subset_path",
    "train_val_split",
    "image",
    "qa_probability",
}


@dataclass
class ValSet:
    """One validation dataset and how to load it."""

    name: str
    task: str
    dataset: Dataset
    collate_fn: Callable
    dataloader_args: dict[str, Any]
    # Sample cap for validation during training; the final validation uses the whole set.
    periodic_samples: int | None = None

    def dataloader(self, periodic: bool = False) -> DataLoader:
        dataset = self.dataset
        if periodic and self.periodic_samples is not None:
            if self.periodic_samples < len(dataset):
                dataset = Subset(dataset, torch.arange(self.periodic_samples))
        return DataLoader(
            dataset,
            shuffle=False,
            collate_fn=self.collate_fn,
            **self.dataloader_args,
        )


def seeded_subset(dataset: Dataset, size: int | None, seed: int) -> Dataset:
    """A seeded random subset in the dataset's own order, so runs validate on the same items.

    Keeping the order leaves questions of one scene adjacent, so each worker's scene and image
    cache can be reused. It isn't stratified by answer, scene or question type.
    """
    if size is None or int(size) >= len(dataset):
        return dataset
    if int(size) < 1:
        raise ValueError("max_questions must be at least one or null.")
    generator = torch.Generator().manual_seed(int(seed))
    indices = torch.randperm(len(dataset), generator=generator)[: int(size)]
    return Subset(dataset, indices.sort().values.tolist())


def _resolved(value: Any) -> Any:
    """Plain values, since a node instantiated apart from its config can't resolve references."""
    return (
        OmegaConf.to_container(value, resolve=True)
        if OmegaConf.is_config(value)
        else value
    )


def transform_config(spec: Any, size: list[int]) -> dict[str, Any]:
    """An image transform at the entry's size. ``spec`` is empty for a plain resize, a name in
    ``conf/img_transform``, ``{name: ..., <overrides>}``, or a full ``{_target_: ...}``."""
    spec = _resolved(spec) or "default"
    if isinstance(spec, str):
        spec = {"name": spec}
    if "name" in spec:
        spec = dict(spec)
        base = OmegaConf.load(TRANSFORM_DIR / f"{spec.pop('name')}.yaml")
        spec = OmegaConf.to_container(base) | spec
    return {**spec, "size": size}


def _per_camera_transform(
    entry: DictConfig, camera_sizes: dict[str, list[int]], name: str
) -> dict[str, Any]:
    """Cameras in ``camera_sizes`` get their own size, the others the entry's ``size``. Only for
    QA, since driving concatenates its cameras, which needs one size."""
    if entry.get("task", TRAJECTORY) != QA:
        raise ValueError(f"data entry {name}: camera_sizes needs task: qa.")
    if entry["size"] is None:
        raise ValueError(f"data entry {name}: camera_sizes needs a size.")
    unknown = set(camera_sizes) - set(get_args(StandardCamera))
    if unknown:
        raise ValueError(f"data entry {name}: unknown cameras {sorted(unknown)}.")
    transform = entry.get("img_transform")
    return {
        "_target_": "ad_vla.utils.img_utils.PerCameraTransform",
        "default": transform_config(transform, list(entry["size"])),
        "cameras": {
            camera: transform_config(transform, list(size))
            for camera, size in camera_sizes.items()
        },
    }


def dataset_kwargs(
    entry: DictConfig, default_split: str, name: str = ""
) -> dict[str, Any]:
    """What an entry passes to its dataset. The entry owns the image size; ``img_transform``
    only says what kind of transform runs at it."""
    if "size" not in entry:
        raise ValueError(
            f"data entry {name} needs size: [height, width], or size: null for native resolution."
        )
    kwargs = {
        key: _resolved(entry[key])
        for key in entry
        if key not in BUILDER_KEYS | {"img_transform"}
    }
    kwargs.setdefault("split", default_split)
    transform = entry.get("img_transform")
    camera_sizes = _resolved(entry.get("camera_sizes"))
    if camera_sizes:
        kwargs["img_transform"] = _per_camera_transform(entry, camera_sizes, name)
    elif entry["size"] is not None:
        kwargs["img_transform"] = transform_config(transform, list(entry["size"]))
    elif transform is not None:
        raise ValueError(f"data entry {name}: img_transform needs a size.")
    else:
        kwargs["img_transform"] = None
    return kwargs


def _with_subset(dataset: Dataset, entry: DictConfig) -> Dataset:
    """``subset`` selects saved indices, then ``subset_fraction`` a [start, end) share of them,
    which splits one set into training and validation parts."""
    if entry.get("subset") is not None:
        dataset = Subset(dataset, torch.load(entry["subset"]))
    if entry.get("subset_fraction") is not None:
        start, end = entry["subset_fraction"]
        dataset = Subset(
            dataset, torch.arange(int(len(dataset) * start), int(len(dataset) * end))
        )
    return dataset


def reject_old_keys(data_cfg: DictConfig) -> None:
    """Old fields would be ignored silently, so a run could use different data than it says."""
    used = sorted(OLD_KEYS & set(data_cfg))
    if used:
        raise ValueError(
            f"data.{used[0]} was replaced by named entries that own their size, such as "
            "data.train.<name> or data.val.<name> (see ad_vla/conf/README.md)."
        )


def single_entry(entries: DictConfig | None, where: str) -> tuple[str, DictConfig]:
    """The name and config of a mapping that holds exactly one dataset."""
    if not entries or len(entries) != 1:
        raise ValueError(f"{where} must name exactly one dataset.")
    return next(iter(entries.items()))


def build_single_dataset(
    entries: DictConfig | None, default_split: str, where: str
) -> Dataset:
    """The one dataset of ``data.test`` or ``data.annotate``."""
    name, entry = single_entry(entries, where)
    return _with_subset(instantiate(dataset_kwargs(entry, default_split, name)), entry)


def _check_qa_cameras(where: str, entry: DictConfig) -> None:
    """QA shows every camera a sample has, so an entry that loads fewer would answer from less."""
    if entry.get("sensors", "all_cams") != "all_cams":
        raise ValueError(f"{where} has task qa, which needs sensors: all_cams.")


def _check_train_entry(name: str, entry: DictConfig) -> None:
    """``task: qa`` alone decides what an entry trains on, so anything that disagrees is an error."""
    where = f"data.train.{name}"
    if entry.get("task", TRAJECTORY) == QA:
        _check_qa_cameras(where, entry)
        if not entry.get("qa_annotations_path"):
            raise ValueError(f"{where} has task qa but no qa_annotations_path.")
        # With question groups, more groups give more distinct items than repeating them; with
        # questions_per_frame: null every question already has its own item, so repeat shows each
        # question that many times per epoch.
        if "repeat" in entry and entry.get("questions_per_frame", 1) is not None:
            raise ValueError(
                f"{where}: use questions_per_frame instead of repeat "
                "(repeat only with questions_per_frame: null)."
            )
        return
    if entry.get("qa_annotations_path"):
        raise ValueError(f"{where} has qa_annotations_path but not task: qa.")
    if "questions_per_frame" in entry:
        raise ValueError(f"{where} has questions_per_frame but not task: qa.")


def check_model_trains_tasks(data_cfg: DictConfig, model: Any) -> None:
    """A collator without QA support would train QA items on the wrong prompt."""
    for name, entry in data_cfg.train.items():
        if entry.get("task", TRAJECTORY) == QA and not getattr(
            model, "trains_qa", False
        ):
            raise TypeError(
                f"data.train.{name} has task qa, which {type(model).__name__} can't train on."
            )
    if getattr(model, "num_past_image_frames", 1) > 1 and not getattr(
        model, "qa_past_frames", False
    ):
        for section in ("train", "val"):
            for name, entry in (data_cfg.get(section) or {}).items():
                if entry.get("task", TRAJECTORY) == QA:
                    raise TypeError(
                        f"data.{section}.{name}: {type(model).__name__} can't show several "
                        "frames in a QA prompt."
                    )


def build_train_dataset(data_cfg: DictConfig) -> Dataset:
    """The entries of ``data.train``, in order. A ``task: qa`` entry draws its questions."""
    if not data_cfg.get("train"):
        raise ValueError("data.train is empty: name at least one dataset.")
    parts: list[Dataset] = []
    for name, entry in data_cfg.train.items():
        _check_train_entry(name, entry)
        kwargs = dataset_kwargs(entry, "train", name)
        kwargs["reasoning_traces_path"] = data_cfg.get("reasoning_traces_path")
        dataset = instantiate(kwargs)
        if entry.get("task", TRAJECTORY) == QA:
            dataset = QATrainDataset(
                dataset,
                entry.get("questions_per_frame", 1),
                int(entry.get("subset_seed", 0)),
            )
        dataset = _with_subset(dataset, entry)
        parts.extend([dataset] * int(entry.get("repeat", 1)))
    return parts[0] if len(parts) == 1 else E2EConcatDataset(parts)


def primary_val_name(data_cfg: DictConfig) -> str | None:
    """The trajectory set behind the plain ``eval/*`` metrics and the best checkpoint."""
    names = [
        name
        for name, entry in data_cfg.val.items()
        if entry.get("task", TRAJECTORY) == TRAJECTORY
    ]
    primary = data_cfg.get("primary_val")
    if primary is not None:
        if primary not in names:
            raise ValueError(
                f"data.primary_val {primary} isn't a trajectory set of data.val."
            )
        return primary
    if len(names) > 1:
        raise ValueError(f"data.primary_val must pick one of {names}.")
    return names[0] if names else None


def build_val_sets(
    data_cfg: DictConfig, model: Any, training: bool = False
) -> list[ValSet]:
    """The entries of ``data.val``, in order. ``training`` applies the periodic sample cap."""
    if not data_cfg.get("val"):
        raise ValueError("data.val is empty: name at least one dataset.")
    default_args = OmegaConf.to_container(data_cfg.val_dataloader_args, resolve=True)
    periodic_default = data_cfg.get("val_periodic_samples") if training else None

    val_sets = []
    for name, entry in data_cfg.val.items():
        task = entry.get("task", TRAJECTORY)
        if "questions_per_frame" in entry:
            raise ValueError(
                f"data.val.{name}: QA validation has one item per question; use max_questions."
            )
        kwargs = dataset_kwargs(entry, "val", name)
        loader_args = default_args | OmegaConf.to_container(
            entry.get("dataloader_args") or OmegaConf.create({}), resolve=True
        )
        if task == TRAJECTORY:
            kwargs["reasoning_traces_path"] = data_cfg.get("reasoning_traces_path")
            val_sets.append(
                ValSet(
                    name=name,
                    task=task,
                    dataset=_with_subset(instantiate(kwargs), entry),
                    collate_fn=model.get_collator(inference_mode=True),
                    dataloader_args=loader_args,
                    periodic_samples=(
                        entry.get("max_samples", periodic_default) if training else None
                    ),
                )
            )
        elif task == QA:
            if not hasattr(model, "predict_qa_answers"):
                raise TypeError(
                    f"{type(model).__name__} does not implement QA validation."
                )
            _check_qa_cameras(f"data.val.{name}", entry)
            scenes = instantiate(kwargs)
            questions = seeded_subset(
                QAPairDataset(scenes),
                entry.get("max_questions"),
                entry.get("subset_seed", 0),
            )
            # Pinning would turn the batch's (sample, qa_pair) tuples into lists, and QA batches
            # hold raw samples anyway: their tensors are made and moved at prediction time.
            loader_args["pin_memory"] = False
            print(
                f"QA validation {name}: {len(scenes):,} scenes, "
                f"{len(questions):,} deterministic questions."
            )
            val_sets.append(
                ValSet(
                    name=name,
                    task=task,
                    dataset=questions,
                    collate_fn=collate_qa_pairs,
                    dataloader_args=loader_args,
                )
            )
        else:
            raise ValueError(f"data.val.{name}: unknown task {task}.")
    return val_sets
