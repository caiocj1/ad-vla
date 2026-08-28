"""nuScenes dataset adapter for the common AD-VLA sample format.

The expected data layout is ``root/version`` where ``version`` is usually
``v1.0-trainval`` for train/val splits and ``v1.0-test`` for test. Samples are
filtered through the official nuScenes scene splits. Ego trajectories are
sampled at the native 2 Hz sample cadence, expressed in the current ego frame,
and future trajectories are omitted for the test split.

Camera calibrations are emitted in the repository's canonical convention:
``translation`` is the sensor origin in ego coordinates, ``rotation`` is the
camera-to-ego rotation, and the camera axes are optical/RDF with +x right,
+y down, and +z forward through the lens.
"""

import json
from functools import lru_cache
from pathlib import Path
from typing import Any

import numpy as np
import torch
from nuscenes.nuscenes import NuScenes
from nuscenes.utils.splits import create_splits_scenes
from PIL import Image
from pyquaternion import Quaternion
from torch.utils.data import Dataset

from ad_vla.dataset.base_dataset import BaseDataset
from ad_vla.dataset.data_types import (
    E2EDataSample,
    QuestionAnswerPair,
    TrajectorySampling,
)
from ad_vla.dataset.nuscenes.nuscenes_types import NUSCENES_CAM_LABELS


_NUSCENES_AVAILABLE_CAMS = set(NUSCENES_CAM_LABELS.keys())


class NuScenesQAPairDataset(Dataset):
    """Flatten a QA-annotated nuScenes dataset to one item per question.

    The wrapped :class:`NuScenesDataset` still owns scene loading and image
    transforms. This view only makes question selection deterministic: each
    item returns one scene sample together with exactly one attached QA pair.
    Items stay grouped by sample token so the small per-worker cache avoids
    reloading the same six images for adjacent questions.
    """

    def __init__(self, scene_dataset: "NuScenesDataset") -> None:
        if not isinstance(scene_dataset, NuScenesDataset):
            raise TypeError("scene_dataset must be a NuScenesDataset instance.")
        if not scene_dataset.qa_by_sample_token:
            raise ValueError(
                "NuScenesQAPairDataset requires qa_annotations_path on its "
                "scene dataset."
            )

        self.scene_dataset = scene_dataset
        sample_index_by_token = {
            sample["token"]: index for index, sample in enumerate(scene_dataset.samples)
        }
        missing_tokens = (
            scene_dataset.qa_by_sample_token.keys() - sample_index_by_token.keys()
        )
        if missing_tokens:
            raise ValueError(
                f"{len(missing_tokens)} QA-annotated sample tokens are absent "
                "from the configured nuScenes split/filter."
            )

        self.items = [
            (sample_index_by_token[sample_token], qa_pair)
            for sample_token, qa_pairs in scene_dataset.qa_by_sample_token.items()
            for qa_pair in qa_pairs
        ]
        if not self.items:
            raise ValueError("The configured NuScenesQA dataset contains no questions.")

    def __len__(self) -> int:
        return len(self.items)

    @lru_cache(maxsize=2)
    def _load_sample(self, sample_index: int) -> E2EDataSample:
        return self.scene_dataset[sample_index]

    def __getitem__(self, index: int) -> tuple[E2EDataSample, QuestionAnswerPair]:
        sample_index, qa_pair = self.items[index]
        return self._load_sample(sample_index), qa_pair


def collate_nuscenes_qa_pairs(
    batch: list[tuple[E2EDataSample, QuestionAnswerPair]],
) -> list[tuple[E2EDataSample, QuestionAnswerPair]]:
    """Keep QA samples as Python objects for model-side prompt construction."""
    return batch


class NuScenesDataset(BaseDataset):
    """Expose nuScenes samples as ``E2EDataSample`` objects.

    ``root`` should contain a nuScenes version directory such as
    ``v1.0-trainval``. The wrapper builds one sample per selected nuScenes
    sample token, optionally requiring full past/future context. Past
    trajectories include the current ego pose before being reversed into
    chronological order; future trajectories are converted into the current ego
    frame and then drop the current pose.

    Camera names are resolved through ``NUSCENES_CAM_LABELS`` into nuScenes raw
    camera channels. Each camera calibration follows the shared
    ``CalibrationDict`` convention used by projection and BEV utilities:
    optical/RDF camera axes, camera-to-ego rotation, and translation in ego
    meters.
    """

    def __init__(
        self,
        root: str,
        split: str,
        sensors: str = "front_cam_only",
        img_transform: torch.nn.Module | None = None,
        reasoning_traces_path: str | None = None,
        qa_annotations_path: str | None = None,
        num_past_steps: int = 8,
        num_future_steps: int = 10,
        require_full_past: bool = True,
        require_full_future: bool = True,
        max_samples: int | None = None,
        version: str | None = None,
        **kwargs: Any,
    ) -> None:
        super().__init__(root, split, sensors, img_transform, reasoning_traces_path)

        self.standard_cameras = self._resolve_cameras(_NUSCENES_AVAILABLE_CAMS)
        self.num_past_steps = num_past_steps
        self.num_future_steps = num_future_steps
        self.require_full_past = require_full_past
        self.require_full_future = require_full_future
        self.sampling_freq = 2
        self.teacher_image_size = (480, 848)
        self.qa_by_sample_token = self._load_qa_annotations(
            qa_annotations_path, expected_split=split
        )

        if version is None:
            path_split = "v1.0-test" if split == "test" else "v1.0-trainval"
        else:
            path_split = version
        self.version = path_split

        self.nusc = NuScenes(
            version=path_split,
            dataroot=str(Path(root) / path_split),
            verbose=True,
        )

        split_scene_names = set(create_splits_scenes()[split])
        self.samples = []
        for scene in self.nusc.scene:
            if scene["name"] not in split_scene_names:
                continue
            for sample in self._iter_scene_samples(scene):
                if self.require_full_past and not self._has_full_past(sample):
                    continue
                if (
                    self.split != "test"
                    and self.require_full_future
                    and not self._has_full_future(sample)
                ):
                    continue
                self.samples.append(sample)
                if max_samples is not None and len(self.samples) >= max_samples:
                    return

    def __len__(self) -> int:
        return len(self.samples)

    def _get_sample(self, idx: int) -> dict[str, Any]:
        return self.samples[idx]

    def _iter_scene_samples(self, scene: dict[str, Any]):
        token = scene["first_sample_token"]
        while token:
            sample = self.nusc.get("sample", token)
            yield sample
            token = sample["next"]

    def _has_full_future(self, sample: dict[str, Any]) -> bool:
        cur = sample
        for _ in range(self.num_future_steps):
            token = cur["next"]
            if not token:
                return False
            cur = self.nusc.get("sample", token)
        return True

    def _has_full_past(self, sample: dict[str, Any]) -> bool:
        cur = sample
        for _ in range(self.num_past_steps):
            token = cur["prev"]
            if not token:
                return False
            cur = self.nusc.get("sample", token)
        return True

    def _clean_sample(self, sample: dict[str, Any]) -> E2EDataSample:
        current_ego_pose = self._get_sample_ego_pose(sample)

        past_global = self._collect_ego_positions(
            sample, steps=self.num_past_steps, direction="prev"
        )
        past_traj = self._global_to_current_ego(past_global, current_ego_pose)[:, :2]
        past_traj = past_traj[::-1].copy()

        info: dict[str, Any] = {
            "agent_input": {"past_traj": torch.from_numpy(past_traj).float()},
            "intent": "UNKNOWN",
            "metadata": self._get_metadata(sample),
            "cameras": [{}],
            "teacher_cameras": [{}],
            # In nuScenes, scenario_id and sample token are the same identifier.
            "qa_pairs": self.qa_by_sample_token.get(sample["token"]),
        }

        if self.split != "test":
            future_global = self._collect_ego_positions(
                sample, steps=self.num_future_steps, direction="next"
            )
            future_traj = self._global_to_current_ego(future_global, current_ego_pose)[
                :, :2
            ]
            future_traj = future_traj[1:].copy()
            if future_traj.shape[0] > 0:
                info["fut_traj"] = torch.from_numpy(future_traj).float()
                info["fut_traj_sampling"] = TrajectorySampling(
                    num_poses=future_traj.shape[0],
                    interval_length=1.0 / self.sampling_freq,
                )
            info["intent"] = self._infer_intent(future_traj)

        for cam_name in self.standard_cameras:
            raw_image, cal_dict = self._get_camera_tensor_and_calib(sample, cam_name)
            info["teacher_cameras"][-1][cam_name] = self._make_teacher_camera_data(
                raw_image,
                cal_dict,
            )
            image, cal_dict = self._apply_img_transform(raw_image, cal_dict)
            info["cameras"][-1][cam_name] = {
                "image": image,
                "cal_dict": cal_dict,
            }

        if self.reasoning_trace_df is not None:
            try:
                info["reasoning_trace"] = self.reasoning_trace_df[
                    self.reasoning_trace_df.scenario_id
                    == info["metadata"]["scenario_id"]
                ].text.item()
            except ValueError:
                info["reasoning_trace"] = None

        return E2EDataSample(**info)

    @staticmethod
    def _load_qa_annotations(
        annotations_path: str | None,
        expected_split: str,
    ) -> dict[str, tuple[QuestionAnswerPair, ...]]:
        """Load and index one NuScenesQA JSON file by nuScenes sample token.

        The JSON is read once when the dataset is constructed. Every retained
        annotation is reduced to the fields used by training and grouped so
        ``_clean_sample`` can attach it in constant time.

        Args:
            annotations_path: Path to ``NuScenes_{split}_questions.json``. When
                ``None``, QA support is disabled without changing dataset output.
            expected_split: nuScenes split requested from this dataset instance.

        Returns:
            Mapping from official sample token to all QA pairs for that sample.

        Raises:
            ValueError: If the annotation split disagrees with the dataset split
                or if a question is missing its required token/question/answer.
        """
        if annotations_path is None:
            return {}

        path = Path(annotations_path).expanduser()
        with path.open(encoding="utf-8") as annotations_file:
            payload = json.load(annotations_file)

        annotation_split = payload.get("info", {}).get("split")
        if annotation_split is not None and annotation_split != expected_split:
            raise ValueError(
                f"NuScenesQA split {annotation_split!r} does not match dataset "
                f"split {expected_split!r} for {path}."
            )

        grouped: dict[str, list[QuestionAnswerPair]] = {}
        for index, annotation in enumerate(payload.get("questions", [])):
            try:
                sample_token = str(annotation["sample_token"])
                pair = QuestionAnswerPair(
                    question=str(annotation["question"]).strip(),
                    answer=str(annotation["answer"]).strip(),
                    num_hop=(
                        int(annotation["num_hop"])
                        if annotation.get("num_hop") is not None
                        else None
                    ),
                    template_type=annotation.get("template_type"),
                )
            except KeyError as error:
                raise ValueError(
                    f"NuScenesQA annotation {index} in {path} is missing {error}."
                ) from error
            if not sample_token or not pair.question or not pair.answer:
                raise ValueError(
                    f"NuScenesQA annotation {index} in {path} has an empty "
                    "sample token, question, or answer."
                )
            grouped.setdefault(sample_token, []).append(pair)

        return {token: tuple(pairs) for token, pairs in grouped.items()}

    def _get_metadata(self, sample: dict[str, Any]) -> dict[str, Any]:
        scene = self.nusc.get("scene", sample["scene_token"])
        log = self.nusc.get("log", scene["log_token"])
        return {
            "scenario_id": sample["token"],
            "scene_token": sample["scene_token"],
            "scene_name": scene["name"],
            "log_token": scene["log_token"],
            "log_location": log.get("location"),
            "timestamp": sample["timestamp"],
            "sampling_freq": self.sampling_freq,
        }

    def _get_sample_ego_pose(self, sample: dict[str, Any]) -> dict[str, Any]:
        lidar_sd = self.nusc.get("sample_data", sample["data"]["LIDAR_TOP"])
        return self.nusc.get("ego_pose", lidar_sd["ego_pose_token"])

    def _collect_ego_positions(
        self,
        sample: dict[str, Any],
        steps: int,
        direction: str,
    ) -> np.ndarray:
        positions = []
        cur = sample
        for _ in range(steps + 1):
            ego_pose = self._get_sample_ego_pose(cur)
            positions.append(np.asarray(ego_pose["translation"], dtype=np.float64))
            token = cur[direction]
            if not token:
                break
            cur = self.nusc.get("sample", token)
        return np.stack(positions)

    @staticmethod
    def _global_to_current_ego(
        points_global: np.ndarray,
        current_ego_pose: dict[str, Any],
    ) -> np.ndarray:
        points = points_global - np.asarray(
            current_ego_pose["translation"], dtype=np.float64
        )
        ego_from_global = Quaternion(
            current_ego_pose["rotation"]
        ).inverse.rotation_matrix
        return (ego_from_global @ points.T).T.astype(np.float32)

    @staticmethod
    def _infer_intent(future_traj: np.ndarray) -> str:
        if future_traj.shape[0] == 0:
            return "UNKNOWN"
        final_x, final_y = future_traj[-1]
        if abs(float(final_y)) > 2.0 and abs(float(final_y)) > 0.25 * abs(
            float(final_x)
        ):
            return "GO_LEFT" if final_y > 0 else "GO_RIGHT"
        return "GO_STRAIGHT"

    def _get_camera_tensor_and_calib(
        self,
        sample: dict[str, Any],
        cam_name: str,
    ) -> tuple[torch.Tensor, dict[str, Any]]:
        raw_cam_name = NUSCENES_CAM_LABELS[cam_name]
        sample_data_token = sample["data"][raw_cam_name]
        sample_data = self.nusc.get("sample_data", sample_data_token)

        with Image.open(self.nusc.get_sample_data_path(sample_data_token)) as image:
            image = image.convert("RGB")
        image_tensor = torch.from_numpy(np.asarray(image).copy())

        calibrated_sensor = self.nusc.get(
            "calibrated_sensor", sample_data["calibrated_sensor_token"]
        )
        intrinsics = torch.tensor(
            calibrated_sensor["camera_intrinsic"], dtype=torch.float32
        )
        rotation = torch.tensor(
            Quaternion(calibrated_sensor["rotation"]).rotation_matrix,
            dtype=torch.float32,
        )
        translation = torch.tensor(
            calibrated_sensor["translation"], dtype=torch.float32
        )

        cal_dict = {
            "intrinsics": intrinsics,
            "distortion": None,
            "translation": translation,
            "rotation": rotation,
            "original_img_size": tuple(int(v) for v in image_tensor.shape[:2]),
            "is_flu": False,
            "timestamp": sample_data["timestamp"],
            "filename": sample_data["filename"],
        }
        return image_tensor, cal_dict
