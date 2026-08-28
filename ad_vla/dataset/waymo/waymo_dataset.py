import os
import json
import pickle
from typing import Any

import numpy as np
import torch

from ad_vla.dataset.waymo.tf_importer import import_tf_silent

import_tf_silent()
import tensorflow as tf

from ad_vla.dataset.base_dataset import BaseDataset
from ad_vla.dataset.data_types import E2EDataSample, TrajectorySampling
from ad_vla.dataset.waymo.waymo_types import (
    WAYMO_IDX2CAM,
    WAYMO_INTENT_DICT,
    WAYMO_SCENARIO_CLUSTERS,
)

from waymo_open_dataset.protos import end_to_end_driving_data_pb2 as wod_e2ed_pb2

# Available standard cameras in Waymo (UNKNOWN excluded)
_WAYMO_AVAILABLE_CAMS = {v for v in WAYMO_IDX2CAM.values() if v != "UNKNOWN"}

# Waymo stores camera extrinsics in a camera-local FLU basis
# (x forward, y left, z up). The rest of the datasets store camera
# extrinsics in an optical/RDF basis (x right, y down, z forward), so convert
# Waymo's sensor->ego rotation before saving it in E2EDataSample.
_WAYMO_FLU_FROM_OPTICAL = torch.tensor(
    [
        [0.0, 0.0, 1.0],
        [-1.0, 0.0, 0.0],
        [0.0, -1.0, 0.0],
    ],
    dtype=torch.float32,
)


class WaymoE2EDataset(BaseDataset):
    """Dataset wrapper exposing Waymo E2E frames as ``E2EDataSample``."""

    def __init__(
        self,
        root: str,
        split: str,
        sensors: str = "front_cam_only",
        img_transform: torch.nn.Module | None = None,
        reasoning_traces_path: str | None = None,
        **kwargs: Any,
    ) -> None:
        """Initialize the Waymo E2E dataset.

        Args:
            root: Path to index files produced by ``waymo_preprocess.py``.
            split: Data split (``\"train\"``, ``\"val\"``, or ``\"test\"``).
            sensors: Camera configuration key defined in ``SENSOR_CONFIGS``.
            img_transform: Optional image transform applied to each camera image.
            reasoning_traces_path: Optional directory with ``*_annotations.jsonl``
                files containing reasoning traces.
        """
        super().__init__(root, split, sensors, img_transform, reasoning_traces_path)

        # Resolve standard camera names via central config
        self.standard_cameras = set(self._resolve_cameras(_WAYMO_AVAILABLE_CAMS))

        index_path = os.path.join(root, f"wod_e2e_{split}_index.pkl")
        with open(index_path, "rb") as f:
            self.index = pickle.load(f)

        clusters_path = os.path.join(root, "val_sequence_name_to_scenario_cluster.json")
        with open(clusters_path, "r") as f:
            self.val_scenario_cluster = json.load(f)

        self.file_handles = {}

        self.sampling_freq = 4
        self.teacher_image_size = (1072, 976)

    def __len__(self) -> int:
        return len(self.index)

    def _read_raw_bytes(self, path: str, offset: int, length: int) -> bytes:
        if path not in self.file_handles:
            self.file_handles[path] = open(path, "rb")

        f = self.file_handles[path]
        f.seek(offset)

        f.read(12)
        data = f.read(length)
        return data

    def _get_sample(self, idx: int) -> wod_e2ed_pb2.E2EDFrame:
        record_info = self.index[idx]

        raw_bytes = self._read_raw_bytes(
            record_info["path"], record_info["offset"], record_info["length"]
        )

        scenario = wod_e2ed_pb2.E2EDFrame()
        scenario.ParseFromString(raw_bytes)

        return scenario

    def _clean_sample(self, scenario: wod_e2ed_pb2.E2EDFrame) -> E2EDataSample:
        """Reformat a Waymo proto into a valid ``E2EDataSample``."""
        info: dict[str, Any] = {}

        info["agent_input"] = {}
        info["agent_input"]["past_traj"] = torch.tensor(
            np.stack([scenario.past_states.pos_x, scenario.past_states.pos_y], axis=-1)
        )
        info["agent_input"]["past_vel"] = torch.tensor(
            np.stack([scenario.past_states.vel_x, scenario.past_states.vel_y], axis=-1)
        )
        info["agent_input"]["past_acc"] = torch.tensor(
            np.stack(
                [scenario.past_states.accel_x, scenario.past_states.accel_y], axis=-1
            )
        )

        info["intent"] = WAYMO_INTENT_DICT[scenario.intent]

        if hasattr(scenario, "future_states") and len(scenario.future_states.pos_x) > 0:
            fut_poses_xyz = torch.tensor(
                np.stack(
                    [
                        scenario.future_states.pos_x,
                        scenario.future_states.pos_y,
                        scenario.future_states.pos_z,
                    ],
                    axis=-1,
                )
            )
            info["fut_traj"] = fut_poses_xyz[..., :2]  # [T, 2] x, y only
            info["fut_traj_sampling"] = TrajectorySampling(
                num_poses=info["fut_traj"].shape[0],
                interval_length=1.0 / self.sampling_freq,
            )

        info["metadata"] = {}
        info["metadata"]["scenario_id"] = scenario.frame.context.name
        info["metadata"]["sampling_freq"] = self.sampling_freq

        # Add preference trajectories to samples that have them
        if hasattr(scenario, "preference_trajectories"):
            pref_trajs = scenario.preference_trajectories

            if len(pref_trajs) > 0 and pref_trajs[0].preference_score != -1.0:
                info["pref_trajs"] = []
                for i in range(len(pref_trajs)):
                    traj = torch.tensor(
                        np.stack([pref_trajs[i].pos_x, pref_trajs[i].pos_y], axis=-1)
                    )
                    info["pref_trajs"].append((traj, pref_trajs[i].preference_score))

                scenario_cluster = self.val_scenario_cluster[
                    info["metadata"]["scenario_id"].split("-")[0]
                ]["scenario_cluster"]
                info["metadata"]["scenario_cluster"] = WAYMO_SCENARIO_CLUSTERS[
                    scenario_cluster
                ]

        # Process camera data — only keep resolved cameras
        info["cameras"] = [{}]
        info["teacher_cameras"] = [{}]

        image_list = []
        calibration_list = []

        for index, image_content in enumerate(scenario.frame.images):
            # Decode the raw image string and convert to numpy type.
            calibration = scenario.frame.context.camera_calibrations[index]
            image = tf.io.decode_image(image_content.image).numpy()
            image_list.append(image)
            calibration_list.append(calibration)

        for img, cam_dict in zip(image_list, calibration_list):
            cam_name = WAYMO_IDX2CAM[cam_dict.name]

            # Skip cameras not in the resolved set
            if cam_name not in self.standard_cameras:
                continue

            extrinsic = torch.tensor(
                cam_dict.extrinsic.transform, dtype=torch.float32
            ).reshape(4, 4)
            translation = extrinsic[:3, -1]
            rotation = extrinsic[:3, :3] @ _WAYMO_FLU_FROM_OPTICAL.to(extrinsic.device)
            intrinsic = torch.tensor(cam_dict.intrinsic)
            intrinsics = intrinsic[:4]
            distortion = intrinsic[4:]
            new_intrinsics = torch.eye(3)
            idxs = [(0, 0), (1, 1), (0, 2), (1, 2)]
            for i, idx in enumerate(idxs):
                new_intrinsics[idx] = intrinsics[i]
            intrinsics = new_intrinsics

            info["cameras"][-1][cam_name] = {}

            raw_size = tuple(img.shape[:2])
            cal_dict = {
                "intrinsics": intrinsics,
                "distortion": distortion,
                "translation": translation,
                "rotation": rotation,
                "original_img_size": raw_size,
                "is_flu": False,
            }

            image = torch.tensor(img)
            info["teacher_cameras"][-1][cam_name] = self._make_teacher_camera_data(
                image,
                cal_dict,
            )

            image, cal_dict = self._apply_img_transform(image, cal_dict)
            info["cameras"][-1][cam_name]["image"] = image
            info["cameras"][-1][cam_name]["cal_dict"] = cal_dict

        if self.reasoning_trace_df is not None:
            try:
                reasoning_trace = self.reasoning_trace_df[
                    self.reasoning_trace_df.scenario_id
                    == info["metadata"]["scenario_id"]
                ].text.item()
                info["reasoning_trace"] = reasoning_trace
            except ValueError:
                info["reasoning_trace"] = None

        return E2EDataSample(**info)
