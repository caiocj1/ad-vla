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
from ad_vla.dataset.intent_labels import load_intent_table
from ad_vla.dataset.data_types import (
    E2EDataSample,
    QuestionAnswerPair,
    TrajectorySampling,
)
from ad_vla.dataset.waymo.waymo_qa import load_waymo_qa
from ad_vla.utils.intent_utils import intent7_from_path
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
        num_past_frames: int = 0,
        past_frame_stride: int = 5,
        clip_index_path: str | None = None,
        qa_annotations_path: str | None = None,
        agent_boxes_path: str | None = None,
        intent_labels_path: str | None = None,
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
            agent_boxes_path: Optional ``AgentBoxStore`` directory with detections and tracks.
            num_past_frames: Past timesteps added to ``sample.cameras``, oldest first. Near the
                start of a clip the list is shorter; frames are never repeated.
            past_frame_stride: Spacing of past frames, in 10 Hz frames (5 = 0.5 s).
            intent_labels_path: Table of 7-class intents keyed by dataset index
                (``ad_vla.dataset.intent_labels``), used instead of the official intent.
            clip_index_path: Pickle from ``build_waymo_clip_index.py``, required if
                ``num_past_frames > 0`` or ``qa_annotations_path`` is set.
            qa_annotations_path: Waymo QA JSONL file, attached to ``sample.qa_pairs``.
        """
        super().__init__(
            root,
            split,
            sensors,
            img_transform,
            reasoning_traces_path,
            agent_boxes_path=agent_boxes_path,
        )

        # Resolve standard camera names via central config
        self.standard_cameras = set(self._resolve_cameras(_WAYMO_AVAILABLE_CAMS))

        index_path = os.path.join(root, f"wod_e2e_{split}_index.pkl")
        with open(index_path, "rb") as f:
            self.index = pickle.load(f)

        clusters_path = os.path.join(root, "val_sequence_name_to_scenario_cluster.json")
        with open(clusters_path, "r") as f:
            self.val_scenario_cluster = json.load(f)

        self.file_handles = {}

        self.intent_table = load_intent_table(intent_labels_path, by_index=True)
        if self.intent_table is not None:
            print(self.intent_table.report(f"Waymo {split}", range(len(self.index))))

        self.sampling_freq = 4
        self.teacher_image_size = (1072, 976)

        self.qa_by_index, self._qa_index_by_scenario, self.qa_dropped = {}, {}, 0
        if qa_annotations_path is not None:
            self.qa_by_index, self._qa_index_by_scenario, self.qa_dropped = (
                load_waymo_qa(qa_annotations_path, split, clip_index_path)
            )
            print(
                f"Waymo QA {split}: {sum(map(len, self.qa_by_index.values())):,} questions on "
                f"{len(self.qa_by_index):,} frames, {self.qa_dropped} without a frame."
            )

        self.past_indices = None
        if num_past_frames > 0:
            if clip_index_path is None:
                raise ValueError("num_past_frames > 0 needs clip_index_path")
            self.past_indices = self._build_past_indices(
                clip_index_path, num_past_frames, past_frame_stride
            )

    def qa_pairs_by_index(
        self, strict: bool = False
    ) -> dict[int, tuple[QuestionAnswerPair, ...]]:
        """Questions of each sample. See the base class."""
        if strict and self.qa_dropped:
            raise ValueError(
                f"{self.qa_dropped} QA annotations have no frame in Waymo {self.split}."
            )
        return self.qa_by_index

    def _build_past_indices(
        self, clip_index_path: str, num_past: int, stride: int
    ) -> np.ndarray:
        """Return [N, num_past] dataset indices of each sample's past frames, nearest first, -1 if absent."""
        with open(clip_index_path, "rb") as f:
            clips = pickle.load(f)
        past_indices = np.full((len(self.index), num_past), -1, dtype=np.int32)
        for frames in clips.values():
            by_frame = dict(frames)
            for frame, idx in frames:
                for k in range(num_past):
                    past_indices[idx, k] = by_frame.get(frame - (k + 1) * stride, -1)
        return past_indices

    def __getitem__(self, idx: int) -> E2EDataSample:
        past_scenarios = []
        if self.past_indices is not None:
            for past_idx in self.past_indices[idx]:
                if past_idx < 0:  # no older frames either: stop at the first gap
                    break
                past_scenarios.append(self._get_sample(int(past_idx)))
        return self._clean_sample(self._get_sample(idx), past_scenarios[::-1], idx)

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

    def _clean_sample(
        self,
        scenario: wod_e2ed_pb2.E2EDFrame,
        past_scenarios: list[wod_e2ed_pb2.E2EDFrame] = (),
        idx: int | None = None,
    ) -> E2EDataSample:
        """Reformat a Waymo proto into a valid ``E2EDataSample``; past scenarios are oldest first.
        ``idx`` (the dataset index) looks up the intent table, when there is one."""
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

        if self.intent_table is not None:
            table_intent = None if idx is None else self.intent_table.get(idx)
            info["intent"] = table_intent or intent7_from_path(
                info.get("fut_traj", np.zeros((0, 2))),
                float(
                    np.hypot(
                        scenario.past_states.vel_x[-1], scenario.past_states.vel_y[-1]
                    )
                ),
                official=info["intent"],
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
        cameras, teacher_cameras = self._load_cameras(scenario, with_teacher=True)
        info["cameras"] = [
            self._load_cameras(p, with_teacher=False)[0] for p in past_scenarios
        ]
        info["cameras"].append(cameras)
        info["teacher_cameras"] = [teacher_cameras]

        if self.reasoning_trace_df is not None:
            try:
                reasoning_trace = self.reasoning_trace_df[
                    self.reasoning_trace_df.scenario_id
                    == info["metadata"]["scenario_id"]
                ].text.item()
                info["reasoning_trace"] = reasoning_trace
            except ValueError:
                info["reasoning_trace"] = None

        qa_index = self._qa_index_by_scenario.get(info["metadata"]["scenario_id"])
        if qa_index is not None:
            info["qa_pairs"] = self.qa_by_index[qa_index]

        stored_agents = self._stored_agent_info(info["metadata"]["scenario_id"])
        if stored_agents is not None:
            info["agent_info"] = stored_agents

        return E2EDataSample(**info)

    def _load_cameras(
        self, scenario: wod_e2ed_pb2.E2EDFrame, with_teacher: bool
    ) -> tuple[dict[str, Any], dict[str, Any]]:
        """Decode one frame's resolved cameras; teacher images only if ``with_teacher``."""
        cameras: dict[str, Any] = {}
        teacher_cameras: dict[str, Any] = {}

        image_list = []
        calibration_list = []

        for index, image_content in enumerate(scenario.frame.images):
            calibration = scenario.frame.context.camera_calibrations[index]
            # Skip decoding cameras outside the resolved set (JPEG decoding dominates loading).
            if WAYMO_IDX2CAM[calibration.name] not in self.standard_cameras:
                continue
            # Decode the raw image string and convert to numpy type.
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

            cameras[cam_name] = {}

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
            if with_teacher:
                teacher_cameras[cam_name] = self._make_teacher_camera_data(
                    image, cal_dict
                )

            image, cal_dict = self._apply_img_transform(image, cal_dict, cam_name)
            cameras[cam_name]["image"] = image
            cameras[cam_name]["cal_dict"] = cal_dict

        return cameras, teacher_cameras
