from torch.utils.data import Dataset
from typing import Any
from collections.abc import Callable
import pandas as pd
import os
import torch
import torch.nn.functional as F

from ad_vla.dataset.data_types import E2EDataSample
from ad_vla.dataset.camera_config import SENSOR_CONFIGS


class BaseDataset(Dataset):
    """Abstract base class for all driving datasets."""

    def __init__(
        self,
        root: str | None = None,
        split: str = "train",
        sensors: str = "front_cam_only",
        img_transform: Callable | None = None,
        reasoning_traces_path: str | None = None,
    ):
        """Initialize the common dataset interface.

        Child datasets must implement :meth:`_get_sample` and
        :meth:`_clean_sample` to cast raw backend data into
        :class:`E2EDataSample` instances.

        Args:
            root: Root directory of the underlying dataset. Can be None for HF datasets.
            split: Dataset split (``\"train\"``, ``\"val\"``, or ``\"test\"``).
            sensors: Camera configuration key defined in ``SENSOR_CONFIGS``.
            img_transform: Optional image transform applied to each camera image.
            reasoning_traces_path: Optional directory containing
                ``*_annotations.jsonl`` files with reasoning traces.
        """
        assert split in ["train", "val", "test"]
        assert sensors in SENSOR_CONFIGS, (
            f"Unknown sensor config '{sensors}'. "
            f"Choose from {list(SENSOR_CONFIGS.keys())}"
        )

        self.root = root
        self.split = split
        self.sensors = sensors
        self.img_transform = img_transform
        if reasoning_traces_path is None:
            self.reasoning_trace_df = None
        else:
            try:
                self.reasoning_trace_df = pd.read_json(
                    os.path.join(reasoning_traces_path, f"{split}_annotations.jsonl"),
                    lines=True,
                )
            except FileNotFoundError:
                print(
                    f"Warning: Reasoning traces file not found at {reasoning_traces_path} for split {split}. Proceeding without reasoning traces."
                )
                self.reasoning_trace_df = None

    @staticmethod
    def _copy_calibration_value(value: Any) -> Any:
        if isinstance(value, torch.Tensor):
            return value.clone()
        if hasattr(value, "copy"):
            return value.copy()
        return value

    @classmethod
    def _copy_calibration(cls, cal_dict: dict[str, Any]) -> dict[str, Any]:
        return {k: cls._copy_calibration_value(v) for k, v in cal_dict.items()}

    @classmethod
    def _resize_calibration(
        cls,
        cal_dict: dict[str, Any],
        from_size: tuple[int, int],
        to_size: tuple[int, int],
    ) -> dict[str, Any]:
        """Return calibration adjusted from one image size to another."""
        from_h, from_w = from_size
        to_h, to_w = to_size
        resized = cls._copy_calibration(cal_dict)

        scale_x = float(to_w) / float(from_w)
        scale_y = float(to_h) / float(from_h)

        intrinsics = resized.get("intrinsics")
        if intrinsics is not None:
            intrinsics = cls._copy_calibration_value(intrinsics)
            intrinsics[0, 0] *= scale_x
            intrinsics[0, 2] *= scale_x
            intrinsics[1, 1] *= scale_y
            intrinsics[1, 2] *= scale_y
            resized["intrinsics"] = intrinsics

        # F-theta radius coefficients are in pixel units. The current datasets
        # using them are not resized by default; this keeps the metadata coherent
        # if a resize is configured later.
        ftheta_scale = (scale_x * scale_y) ** 0.5
        fw_poly = resized.get("fw_poly")
        if fw_poly is not None:
            fw_poly = cls._copy_calibration_value(fw_poly)
            fw_poly *= ftheta_scale
            resized["fw_poly"] = fw_poly

        bw_poly = resized.get("bw_poly")
        if bw_poly is not None:
            bw_poly = cls._copy_calibration_value(bw_poly)
            for degree in range(1, len(bw_poly)):
                bw_poly[degree] /= ftheta_scale**degree
            resized["bw_poly"] = bw_poly

        resized["original_img_size"] = (int(to_h), int(to_w))
        return resized

    @classmethod
    def _crop_calibration(
        cls,
        cal_dict: dict[str, Any],
        *,
        left: int = 0,
        top: int = 0,
        cropped_size: tuple[int, int],
    ) -> dict[str, Any]:
        """Return calibration adjusted for a crop from the top-left origin."""
        cropped = cls._copy_calibration(cal_dict)
        intrinsics = cropped.get("intrinsics")
        if intrinsics is not None:
            intrinsics = cls._copy_calibration_value(intrinsics)
            intrinsics[0, 2] -= left
            intrinsics[1, 2] -= top
            cropped["intrinsics"] = intrinsics
        cropped["original_img_size"] = tuple(int(v) for v in cropped_size)
        return cropped

    def _apply_img_transform(
        self,
        image: torch.Tensor,
        cal_dict: dict[str, Any],
    ) -> tuple[torch.Tensor, dict[str, Any]]:
        """Apply this dataset's image transform and keep calibration coherent."""
        if self.img_transform is None:
            return image, cal_dict

        if not callable(self.img_transform):
            raise TypeError(
                "img_transform must be a callable transform, got "
                f"{type(self.img_transform).__name__}"
            )

        if getattr(self.img_transform, "requires_calibration", False):
            result = self.img_transform(image, cal_dict)
            if not isinstance(result, tuple) or len(result) != 2:
                raise TypeError(
                    "Calibration-aware img_transform must return (image, cal_dict)"
                )
            return result

        # If doesn't require calibration, assumes img_transform is a simple Resize
        raw_size = tuple(image.shape[:2])
        transformed = self.img_transform(image.permute(2, 0, 1)).permute(1, 2, 0)
        return transformed, self._resize_calibration(
            cal_dict,
            raw_size,
            tuple(transformed.shape[:2]),
        )

    def _make_teacher_camera_data(
        self,
        image: torch.Tensor,
        cal_dict: dict[str, Any],
    ) -> dict[str, Any]:
        """Build a fixed-size teacher camera image with matching calibration."""
        if not isinstance(image, torch.Tensor):
            image = torch.as_tensor(image)
        if image.ndim != 3 or image.shape[-1] != 3:
            raise ValueError(f"Teacher image must be HWC RGB, got {image.shape}")

        raw_size = tuple(image.shape[:2])
        target_size = self.teacher_image_size
        chw = image.permute(2, 0, 1).unsqueeze(0).float()
        resized = F.interpolate(
            chw,
            size=target_size,
            mode="bilinear",
            align_corners=False,
        )[0].permute(1, 2, 0)

        if image.dtype == torch.uint8:
            resized = resized.round().clamp(0, 255).to(image.dtype)
        else:
            resized = resized.to(image.dtype)

        return {
            "image": resized,
            "cal_dict": self._resize_calibration(cal_dict, raw_size, target_size),
        }

    # ------------------------------------------------------------------
    # Camera resolution helper
    # ------------------------------------------------------------------
    def _resolve_cameras(
        self,
        available_standard_cams: set[str],
    ) -> tuple[str, ...]:
        """Resolve the sensor config to a validated tuple of standard camera names.

        Args:
            available_standard_cams: The set of standard camera names that the
                concrete dataset actually provides.

        Returns:
            Tuple of standard camera names to load for this dataset instance.

        Raises:
            ValueError: If the sensor config requires cameras not available in
                the dataset.
        """
        config = SENSOR_CONFIGS[self.sensors]

        if not config:
            # "all_cams" sentinel → load everything the dataset has
            return tuple(sorted(available_standard_cams))

        missing = set(config) - available_standard_cams
        if missing:
            raise ValueError(
                f"Sensor config '{self.sensors}' requires cameras {missing} "
                f"but this dataset only has {sorted(available_standard_cams)}"
            )
        return config

    def __getitem__(self, idx: int) -> E2EDataSample:
        scenario = self._get_sample(idx)
        sample = self._clean_sample(scenario)

        if isinstance(sample, E2EDataSample):
            return sample
        return E2EDataSample(**sample)

    def __len__(self) -> int:
        raise NotImplementedError

    def _get_sample(self, idx: int) -> Any:
        """
        Loads a traffic scenario from the raw data.
        For NAVSIM, output type is dict extracted from navsim.common.dataclasses.Scene
        For WOD-E2E, output type is wod_e2ed_pb2.E2EDFrame
        """
        raise NotImplementedError

    def _clean_sample(self, scenario: Any) -> E2EDataSample | dict[str, Any]:
        """
        Transforms raw traffic scenario into E2EDataSample or
        dict parseable into E2EDataSample.
        """
        raise NotImplementedError
