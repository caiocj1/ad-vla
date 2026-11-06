from collections.abc import Callable
from dataclasses import asdict, is_dataclass
from importlib.resources import files
from pathlib import Path
from typing import Any

import numpy as np
import torch
import yaml
from navsim.common.dataloader import SceneLoader, Scene
from navsim.common.dataclasses import SceneFilter, SensorConfig
from navsim.common.enums import BoundingBoxIndex

from ad_vla.dataset.agent_boxes import coarse_type, wrap_angle
from ad_vla.dataset.base_dataset import BaseDataset
from ad_vla.dataset.data_types import E2EDataSample, TrajectorySampling
from ad_vla.dataset.intent_labels import load_intent_table
from ad_vla.dataset.navsim.navsim_types import NAVSIM_CAM_LABELS, NAVSIM_INTENT_DICT
from ad_vla.utils.intent_utils import intent7_from_path


# Reverse mapping: standard name → NAVSIM raw name
_STD_TO_RAW = {v: k for k, v in NAVSIM_CAM_LABELS.items()}

# All raw camera attribute names used by SensorConfig (lidar excluded)
_ALL_RAW_CAMS = {
    "cam_f0",
    "cam_l0",
    "cam_r0",
    "cam_l1",
    "cam_r1",
    "cam_l2",
    "cam_b0",
    "cam_r2",
}


# A car standing still: under this speed now and within this distance over the scene's future.
STILL_SPEED_MPS = 0.5
STILL_DISTANCE_M = 1.0


def is_still_scene(frames: list[dict[str, Any]], current: int) -> bool:
    """Whether the car is stopped at frame ``current`` and stays put until the scene ends."""
    if np.hypot(*frames[current]["ego_dynamic_state"][:2]) >= STILL_SPEED_MPS:
        return False
    start = frames[current]["ego2global"][:2, 3]
    return all(
        np.hypot(*(frame["ego2global"][:2, 3] - start)) < STILL_DISTANCE_M
        for frame in frames[current + 1 :]
    )


def still_scenes_to_drop(
    scenes: dict[str, list[dict[str, Any]]], current: int, keep_every: int
) -> set[str]:
    """Tokens to leave out so each stop keeps its first scene and every ``keep_every``-th after it.

    A car about to pull away is not still, so those scenes all stay.
    """
    if keep_every < 1:
        raise ValueError("still_keep_every must be at least 1")
    ordered = sorted(
        scenes,
        key=lambda token: (
            scenes[token][current]["log_name"],
            scenes[token][current]["timestamp"],
        ),
    )
    dropped, position, log = set(), 0, None
    for token in ordered:
        frames = scenes[token]
        still = is_still_scene(frames, current)
        if not still or frames[current]["log_name"] != log:
            position = 0
        log = frames[current]["log_name"]
        if still:
            if position % keep_every:
                dropped.add(token)
            position += 1
    return dropped


class NAVSIMDataset(BaseDataset):
    """Dataset wrapper that exposes Navsim scenes as ``E2EDataSample`` objects."""

    def __init__(
        self,
        root: str,
        split: str,
        sensors: str = "front_cam_only",
        img_transform: Callable | None = None,
        reasoning_traces_path: str | None = None,
        num_history_frames: int = 9,
        num_future_frames: int = 10,
        max_scenes: int | None = None,
        frame_interval: int = 5,
        log_split_yaml: str | None = None,
        has_route: bool = True,
        num_image_frames: int | None = None,
        still_keep_every: int | None = None,
        intent_labels_path: str | None = None,
        **kwargs,
    ) -> None:
        """
        Initialize the Navsim dataset.

        Args:
            root: Root directory containing ``navsim_logs`` and ``sensor_blobs``.
            split: Dataset split (``\"train\"``, ``\"val\"``, or ``\"test\"``).
            sensors: Camera configuration key defined in ``SENSOR_CONFIGS``.
            img_transform: Optional image transform applied to each camera image.
            reasoning_traces_path: Optional directory with ``*_annotations.jsonl``
                files containing reasoning traces.
            num_history_frames: Number of history frames to load per scene.
            num_future_frames: Number of future frames to load per scene.
            max_scenes: Optional cap on the number of scenes loaded by Navsim.
            num_image_frames: Load camera images only for the latest N history frames
                (None: every history frame). Decoding dominates loading time, and the
                models use only the latest frame(s); the ego history is unaffected.
            still_keep_every: Keep one in this many scenes while the car stands still
                (None: all). Stops are long and their scenes nearly identical.
            intent_labels_path: Table of 7-class intents keyed by scene token
                (``ad_vla.dataset.intent_labels``), used instead of the official
                ``driving_command`` (None: the official command).
        """
        super().__init__(root, split, sensors, img_transform, reasoning_traces_path)
        if (
            num_image_frames is not None
            and not 1 <= num_image_frames <= num_history_frames
        ):
            raise ValueError("num_image_frames must be in [1, num_history_frames]")
        self.num_history_frames = num_history_frames
        self.num_image_frames = num_image_frames

        # Resolve standard camera names via central config
        available_std = set(NAVSIM_CAM_LABELS.values())
        self.standard_cameras = self._resolve_cameras(available_std)
        # Raw names needed for data loading
        self._raw_cameras = tuple(_STD_TO_RAW[s] for s in self.standard_cameras)

        # Setup scene filter for correct split
        scene_filter = SceneFilter(
            num_history_frames=num_history_frames,
            num_future_frames=num_future_frames,
            max_scenes=max_scenes,
            frame_interval=frame_interval,
            has_route=has_route,
        )

        if log_split_yaml is None:
            yaml_path = files("ad_vla.dataset.navsim").joinpath(
                "default_train_val_test_log_split.yaml"
            )
        else:
            yaml_path = Path(log_split_yaml)
        with open(yaml_path, "r") as file:
            logs = yaml.safe_load(file)
        scene_filter.log_names = logs[split + "_logs"]

        # Configure sensors: enable only the resolved cameras
        self.sensor_config = self._build_sensor_config()

        # Prepare scene loader
        if split in ["train", "val"]:
            path_split = "trainval"
        scene_loader = SceneLoader(
            Path(root) / f"navsim_logs/{path_split}",
            Path(root) / f"sensor_blobs/{path_split}",
            scene_filter,
            sensor_config=self.sensor_config,
        )

        self.num_future_frames = num_future_frames

        self.scene_loader = scene_loader
        self.tokens = list(scene_loader.tokens)
        self.tokens.sort()
        if still_keep_every is not None:
            dropped = still_scenes_to_drop(
                scene_loader.scene_frames_dicts,
                num_history_frames - 1,
                still_keep_every,
            )
            self.tokens = [token for token in self.tokens if token not in dropped]
            print(
                f"NavSim {split}: kept {len(self.tokens):,} scenes, "
                f"left out {len(dropped):,} of a car standing still."
            )

        self.intent_table = load_intent_table(intent_labels_path)
        if self.intent_table is not None:
            print(self.intent_table.report(f"NavSim {split}", self.tokens))

        self.sampling_freq = 2
        self.teacher_image_size = (480, 848)

    def _build_sensor_config(self) -> SensorConfig:
        """Build a ``SensorConfig`` enabling only the resolved cameras."""
        enabled = set(self._raw_cameras)
        if self.num_image_frames is None:
            frames: bool | list[int] = True
        else:
            # SensorConfig takes history iteration indices; the current frame is the last one.
            frames = list(
                range(
                    self.num_history_frames - self.num_image_frames,
                    self.num_history_frames,
                )
            )
        return SensorConfig(
            **{cam: (frames if cam in enabled else False) for cam in _ALL_RAW_CAMS},
            lidar_pc=False,
        )

    def __len__(self) -> int:
        """Return the number of available Navsim scenes."""
        return len(self.tokens)

    def _get_sample(self, idx: int) -> dict[str, Any]:
        """Load a raw Navsim scene and convert it to a Python dict.

        Args:
            idx: Scene index in the token list.

        Returns:
            Dictionary representation of the scene produced by ``extractor``.
        """
        tok = self.tokens[idx]
        scenario = self.scene_loader.get_scene_from_token(tok)
        scenario = self.extractor(scenario)
        return scenario

    def _clean_sample(self, scenario: dict[str, Any]) -> E2EDataSample:
        """Convert a Navsim scene dict into an ``E2EDataSample``.

        Args:
            scenario: Scene dictionary returned by :meth:`extractor`.

        Returns:
            Parsed ``E2EDataSample`` instance.
        """
        info: dict[str, Any] = {}

        info["intent"] = NAVSIM_INTENT_DICT[
            scenario["agent_input"]["ego_statuses"][-1]["driving_command"].argmax()
        ]

        camera_info = scenario["agent_input"]["cameras"]
        info["cameras"] = [{} for _ in range(len(camera_info))]
        info["teacher_cameras"] = [{}]
        teacher_timestep = len(camera_info) - 1

        info["agent_input"] = {}
        ego_statuses = scenario["agent_input"].pop("ego_statuses")
        past_pos = []
        past_vel = []
        past_acc = []
        for stat in ego_statuses:
            past_pos.append(torch.from_numpy(stat["ego_pose"][:2]))
            past_vel.append(torch.from_numpy(stat["ego_velocity"][:2]))
            past_acc.append(torch.from_numpy(stat["ego_acceleration"][:2]))
        info["agent_input"]["past_traj"] = torch.stack(past_pos, dim=0)
        info["agent_input"]["past_vel"] = torch.stack(past_vel, dim=0)
        info["agent_input"]["past_acc"] = torch.stack(past_acc, dim=0)

        for t in range(len(camera_info)):
            for cam_name in list(camera_info[t].keys()):
                cam_data: dict[str, Any] = {}
                raw_cam_entry = camera_info[t][cam_name]
                raw_image = raw_cam_entry.pop("image")
                cal_dict = {
                    k: torch.from_numpy(v)
                    for k, v in raw_cam_entry.items()
                    if isinstance(v, np.ndarray)
                }
                cal_dict["translation"] = cal_dict.pop(
                    "sensor2lidar_translation", torch.zeros(3)
                )
                cal_dict["rotation"] = cal_dict.pop(
                    "sensor2lidar_rotation", torch.zeros(3, 3)
                )
                cal_dict["original_img_size"] = raw_cam_entry["original_img_size"]
                cal_dict["is_flu"] = False

                if t == teacher_timestep:
                    info["teacher_cameras"][-1][cam_name] = (
                        self._make_teacher_camera_data(raw_image, cal_dict)
                    )

                cam_data["image"], cam_data["cal_dict"] = self._apply_img_transform(
                    raw_image,
                    cal_dict,
                    cam_name,
                )

                info["cameras"][t][cam_name] = cam_data

        # History frames outside num_image_frames have no images.
        info["cameras"] = [frame for frame in info["cameras"] if frame]

        info["metadata"] = {
            "scenario_id": scenario.pop("scene_token"),
            "log_name": scenario.pop("log_name"),
            "map_name": scenario.pop("map_name"),
            "sampling_freq": self.sampling_freq,
        }

        if "future_trajectory" in scenario:
            fut_poses = torch.from_numpy(scenario["future_trajectory"].pop("poses"))
            info["fut_traj"] = fut_poses[..., :2]  # [T, 2] x, y only
            info["fut_traj_sampling"] = TrajectorySampling(
                num_poses=info["fut_traj"].shape[0],
                interval_length=1.0 / self.sampling_freq,
            )

        if self.intent_table is not None:
            info["intent"] = self.intent_table.get(
                info["metadata"]["scenario_id"]
            ) or intent7_from_path(
                info.get("fut_traj", np.zeros((0, 2))),
                float(torch.linalg.norm(info["agent_input"]["past_vel"][-1])),
                official=info["intent"],
            )

        info["agent_info"] = {
            "pos": torch.from_numpy(scenario["agent_trajs"])[..., :2],
            "mask": torch.from_numpy(scenario["agent_mask"]),
            "types": scenario["agent_names"],
            **scenario["agent_boxes"],
        }

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

    def extractor(self, scenario: Any) -> dict[str, Any]:
        """Extract a Navsim ``Scene`` into a plain Python dictionary.

        Args:
            scenario: Navsim :class:`~navsim.common.dataclasses.Scene` instance.

        Returns:
            Dictionary containing scene metadata, agent input, and future
            trajectory with cameras renamed to standard names.
        """

        def _to_py(obj: Any) -> Any:
            if is_dataclass(obj):
                return asdict(obj)
            return obj

        agent_trajs, agent_mask, agent_names = self.get_all_agents_future_trajectories(
            scenario
        )

        sample = {
            "scene_token": scenario.scene_metadata.initial_token,
            "log_name": scenario.scene_metadata.log_name,
            "map_name": scenario.scene_metadata.map_name,
            "agent_input": _to_py(scenario.get_agent_input()),
            "future_trajectory": _to_py(scenario.get_future_trajectory()),
            "agent_trajs": agent_trajs,
            "agent_boxes": self._extract_agent_boxes(scenario),
            "agent_mask": agent_mask,
            "agent_names": agent_names,
        }

        # Remove unused sensors
        del sample["agent_input"]["lidars"]

        # Rename resolved cameras to standard names, drop the rest
        raw_to_keep = set(self._raw_cameras)
        for frame in sample["agent_input"]["cameras"]:
            for raw_cam in list(frame.keys()):
                # Frames outside num_image_frames come without an image.
                if raw_cam in raw_to_keep and frame[raw_cam]["image"] is not None:
                    original_size = frame[raw_cam]["image"].shape[:2]
                    frame[raw_cam]["image"] = torch.tensor(frame[raw_cam]["image"])
                    std_name = NAVSIM_CAM_LABELS[raw_cam]
                    frame[std_name] = frame.pop(raw_cam)
                    frame[std_name]["original_img_size"] = original_size
                else:
                    del frame[raw_cam]

        return sample

    def get_all_agents_future_trajectories(
        self,
        scene: Scene,
    ) -> tuple[np.ndarray, np.ndarray]:
        """
        Returns ground-truth trajectories for all agents visible at the current frame,
        expressed in the current frame's ego-local coordinate system (ego at origin).

        Args:
            scene:      NAVSIM Scene object.

        Returns:
            trajs: np.ndarray of shape [N, T+1, 2]
                N = number of agents at current frame
                T+1 = current timestep (t=0) + T future steps
                2 = (x, y) in current ego-local coords
            mask:  np.ndarray of shape [N, T+1], dtype bool
                True  → agent is visible and data is valid at that timestep
                False → agent has left the scene or the frame doesn't exist
        """
        current_idx = scene.scene_metadata.num_history_frames - 1
        current_ann = scene.frames[current_idx].annotations
        current_ego_pose = scene.frames[
            current_idx
        ].ego_status.ego_pose  # [x, y, heading] global

        N = len(current_ann.track_tokens)
        T_plus_1 = self.num_future_frames + 1

        trajs = np.zeros((N, T_plus_1, 2), dtype=np.float32)
        mask = np.zeros((N, T_plus_1), dtype=bool)
        names = list(current_ann.names)

        # t=0: current frame — agents are already in current ego-local coords
        trajs[:, 0, 0] = current_ann.boxes[:, BoundingBoxIndex.X]
        trajs[:, 0, 1] = current_ann.boxes[:, BoundingBoxIndex.Y]
        mask[:, 0] = True  # all N agents are visible at t=0 by definition

        # t=1..T: future frames
        for t in range(1, T_plus_1):
            future_idx = current_idx + t
            if future_idx >= len(scene.frames):
                break  # scene ended, mask stays False for remaining steps

            future_ann = scene.frames[future_idx].annotations
            future_ego_pose = scene.frames[
                future_idx
            ].ego_status.ego_pose  # [x, y, heading] global

            # Precompute transform: future_ego_local → current_ego_local
            # Step 1: future_ego_local → global
            #   p_global = R(future_heading) @ p_future_local + future_ego_xy
            # Step 2: global → current_ego_local
            #   p_current_local = R(-current_heading) @ (p_global - current_ego_xy)
            dx = future_ego_pose[0] - current_ego_pose[0]
            dy = future_ego_pose[1] - current_ego_pose[1]
            dh = future_ego_pose[2]  # future ego heading in global frame
            ch = current_ego_pose[2]  # current ego heading in global frame

            cos_dh, sin_dh = np.cos(dh), np.sin(dh)
            cos_ch, sin_ch = np.cos(ch), np.sin(ch)

            # Build the composite 2x2 rotation and 2D translation
            # p_current_local = R_c^T @ (R_f @ p_future_local + t_f_in_global)
            # where t_f_in_global = [dx, dy]
            R_f = np.array([[cos_dh, -sin_dh], [sin_dh, cos_dh]])
            R_c_T = np.array(
                [[cos_ch, sin_ch], [-sin_ch, cos_ch]]
            )  # transpose = inverse for rotation matrix
            R_composite = R_c_T @ R_f  # [2, 2]
            t_composite = R_c_T @ np.array([dx, dy])  # [2]

            # Build a lookup: track_token → index in future_ann
            future_token_to_idx = {
                tok: i for i, tok in enumerate(future_ann.track_tokens)
            }

            for agent_i, token in enumerate(current_ann.track_tokens):
                if token not in future_token_to_idx:
                    continue  # agent left the scene, mask stays False

                fut_i = future_token_to_idx[token]
                p_fut_local = future_ann.boxes[
                    fut_i, [BoundingBoxIndex.X, BoundingBoxIndex.Y]
                ]

                p_curr_local = R_composite @ p_fut_local + t_composite

                trajs[agent_i, t] = p_curr_local
                mask[agent_i, t] = True

        return trajs, mask, names

    def _extract_agent_boxes(self, scene: Any) -> dict[str, Any]:
        """``AgentInfo`` box fields of the current frame's agents, with their future boxes."""
        current_idx = scene.scene_metadata.num_history_frames - 1
        current_ann = scene.frames[current_idx].annotations
        if getattr(current_ann, "is_missing", False):
            return {"boxes": None, "source": "gt"}
        idx = [
            BoundingBoxIndex.X, BoundingBoxIndex.Y, BoundingBoxIndex.Z, BoundingBoxIndex.LENGTH,
            BoundingBoxIndex.WIDTH, BoundingBoxIndex.HEIGHT, BoundingBoxIndex.HEADING,
        ]  # fmt: skip
        boxes = np.asarray(current_ann.boxes, dtype=np.float32)[:, idx]
        tokens = list(current_ann.track_tokens)
        fut_boxes = np.zeros((len(tokens), self.num_future_frames, 7), dtype=np.float32)
        fut_mask = np.zeros((len(tokens), self.num_future_frames), dtype=bool)
        x0, y0, h0 = scene.frames[current_idx].ego_status.ego_pose
        cos0, sin0 = np.cos(h0), np.sin(h0)
        for t in range(self.num_future_frames):
            future_idx = current_idx + t + 1
            if future_idx >= len(scene.frames):
                break
            ann = scene.frames[future_idx].annotations
            if getattr(ann, "is_missing", False):
                continue
            x1, y1, h1 = scene.frames[future_idx].ego_status.ego_pose
            cos1, sin1 = np.cos(h1), np.sin(h1)
            lookup = {tok: i for i, tok in enumerate(ann.track_tokens)}
            for a, tok in enumerate(tokens):
                i = lookup.get(tok)
                if i is None:
                    continue
                b = np.asarray(ann.boxes[i], dtype=np.float64)[idx]
                gx = x1 + cos1 * b[0] - sin1 * b[1] - x0
                gy = y1 + sin1 * b[0] + cos1 * b[1] - y0
                fut_boxes[a, t] = (
                    cos0 * gx + sin0 * gy, -sin0 * gx + cos0 * gy, b[2], b[3], b[4], b[5],
                    wrap_angle(b[6] + h1 - h0),
                )  # fmt: skip
                fut_mask[a, t] = True
        names = list(current_ann.names)
        return {
            "boxes": torch.from_numpy(boxes),
            "track_ids": [str(t) for t in tokens],
            "coarse_types": [coarse_type(n) for n in names],
            "source": "gt",
            "fut_boxes": torch.from_numpy(fut_boxes),
            "fut_mask": torch.from_numpy(fut_mask),
            "fut_interval_s": 1.0 / self.sampling_freq,
        }
