import torch
from typing import Any
from datasets import load_dataset
from torchvision.transforms.functional import pil_to_tensor

from ad_vla.dataset.base_dataset import BaseDataset
from ad_vla.dataset.data_types import E2EDataSample, TrajectorySampling
from ad_vla.dataset.kitscenes.kitscenes_types import (
    STD_CAM_NAME_TO_KIT,
    KIT_CAM_NAME_TO_STD,
    KIT_CAM_PARAMS,
)

_KIT_AVAILABLE_CAMS = set(STD_CAM_NAME_TO_KIT.keys())


def _kit_to_repo_extrinsics(
    rotation_world_to_camera: torch.Tensor,
    translation_world_to_camera: torch.Tensor,
) -> tuple[torch.Tensor, torch.Tensor]:
    """Convert KITScenes world->camera extrinsics to the repo's sensor->ego form."""
    rotation_sensor_to_ego = rotation_world_to_camera.T
    translation_sensor_to_ego = -rotation_sensor_to_ego @ translation_world_to_camera
    return rotation_sensor_to_ego, translation_sensor_to_ego


class KITScenesDataset(BaseDataset):
    def __init__(
        self,
        root: str | None = None,
        split: str = "train",
        sensors: str = "front_cam_only",
        img_transform: torch.nn.Module | None = None,
        reasoning_traces_path: str | None = None,
        **kwargs: Any,
    ):
        super().__init__(root, split, sensors, img_transform, reasoning_traces_path)

        self.standard_cameras = set(self._resolve_cameras(_KIT_AVAILABLE_CAMS))

        self.ds = load_dataset("KIT-MRT/KITScenes-LongTail")[split]

        self.sampling_freq = 5
        self.teacher_image_size = (720, 1280)

    def __len__(self) -> int:
        return len(self.ds)

    def _get_sample(self, idx: int) -> dict:
        return self.ds[idx]

    def _clean_sample(self, sample: dict) -> E2EDataSample:
        info = {}

        info["cameras"] = []
        cam_names = [k for k in sample.keys() if "frames_camera" in k]

        info["cameras"].append({})
        info["teacher_cameras"] = [{}]

        for cam_name in cam_names:
            std_cam_name = KIT_CAM_NAME_TO_STD[cam_name]
            if std_cam_name not in self.standard_cameras:
                continue

            cam_data: dict[str, Any] = {}
            image = pil_to_tensor(sample[cam_name][-1]).permute(1, 2, 0)
            raw_size = tuple(image.shape[:2])

            raw_rotation = torch.from_numpy(KIT_CAM_PARAMS[std_cam_name.lower()]["R"])
            raw_translation = torch.from_numpy(
                KIT_CAM_PARAMS[std_cam_name.lower()]["t"]
            )
            rotation, translation = _kit_to_repo_extrinsics(
                raw_rotation, raw_translation
            )

            cal_dict = {}
            cal_dict["intrinsics"] = torch.from_numpy(
                KIT_CAM_PARAMS[std_cam_name.lower()]["K"]
            )
            cal_dict["distortion"] = None
            cal_dict["translation"] = translation
            cal_dict["rotation"] = rotation
            cal_dict["is_flu"] = False
            cal_dict["original_img_size"] = raw_size
            cal_dict["projection_z_offset"] = -1.5
            cal_dict["projection_min_depth"] = 0.0

            info["teacher_cameras"][-1][std_cam_name] = self._make_teacher_camera_data(
                image, cal_dict
            )

            image, cal_dict = self._apply_img_transform(image, cal_dict)

            # _, w, _ = image.shape
            # left = w // 5
            # image = image[:, left : -w // 5]
            # cal_dict = self._crop_calibration(
            #     cal_dict,
            #     left=left,
            #     cropped_size=tuple(image.shape[:2]),
            # )

            cam_data["image"] = image
            cam_data["cal_dict"] = cal_dict

            info["cameras"][-1][std_cam_name] = cam_data

        info["metadata"] = {}
        info["metadata"]["scenario_id"] = sample["scenario_id"]
        info["metadata"]["scenario_type"] = sample["scenario_type"]
        info["metadata"]["sampling_freq"] = self.sampling_freq

        info["intent"] = (
            sample["driving_instruction"]
            .upper()
            .replace(" ", "_")
            .replace("TURN", "GO")
        )
        if (
            "STRAIGHT" in info["intent"]
            or "USE" in info["intent"]
            or "OVERTAKE" in info["intent"]
        ):
            info["intent"] = "GO_STRAIGHT"

        info["reasoning_trace"] = str(sample["reasoning"]["english"])

        info["agent_input"] = {}
        info["agent_input"]["past_traj"] = torch.tensor(sample["trajectory"]["past"])[
            1:
        ]
        cur_pos = info["agent_input"]["past_traj"][-1:]
        info["agent_input"]["past_traj"] = info["agent_input"]["past_traj"] - cur_pos

        def has_traj(traj: list[list[float]]) -> bool:
            return len(traj) > 1

        if has_traj(sample["trajectory"]["expert_like"]):
            info["fut_traj"] = (
                torch.tensor(sample["trajectory"]["expert_like"]) - cur_pos
            )
            info["fut_traj_sampling"] = TrajectorySampling(
                num_poses=info["fut_traj"].shape[0],
                interval_length=1 / self.sampling_freq,
            )

        info["special_trajs"] = {}
        for traj_type, traj in sample["trajectory"].items():
            if traj_type in ["past", "expert_like"]:
                continue

            if has_traj(traj):
                info["special_trajs"][traj_type] = torch.tensor(traj)[..., :2] - cur_pos

        return E2EDataSample(**info)
