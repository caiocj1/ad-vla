from typing import Any

from torch.utils.data import ConcatDataset, Dataset


class E2EConcatDataset(ConcatDataset):
    """Thin ConcatDataset wrapper that fits the repo's Hydra dataset interface."""

    def __init__(
        self,
        datasets: list[Dataset],
        split: str = "train",
        img_transform: Any | None = None,
        **_: Any,
    ) -> None:
        if not datasets:
            raise ValueError("E2EConcatDataset requires at least one dataset")
        super().__init__(datasets)
        self.split = split
        self.img_transform = img_transform

        if img_transform is not None:
            for dataset in self.datasets:
                if (
                    hasattr(dataset, "img_transform")
                    and getattr(dataset, "img_transform") is None
                ):
                    dataset.img_transform = img_transform
