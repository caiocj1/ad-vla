"""PyTorch Dataset wrapper for PhysicalAI-AV dataset, compatible with BaseDataset / E2EDataSample."""

import os
from collections import OrderedDict
from pathlib import Path
from typing import Any

import cv2
import numpy as np
import scipy.spatial.transform as spt
import torch

from ad_vla.dataset.agent_boxes import coarse_type, wrap_angle
from ad_vla.dataset.base_dataset import BaseDataset
from ad_vla.dataset.data_types import E2EDataSample, TrajectorySampling
from ad_vla.dataset.physical_ai.physical_ai_types import PHYSAI_CAM_LABELS
from ad_vla.dataset.intent_labels import load_intent_table
from ad_vla.utils.intent_utils import intent7_from_path, intent_from_path


def ftheta_to_pinhole_maps(
    fw_poly: np.ndarray,
    intrinsics: np.ndarray,
    height: int,
    half_fov_deg: float,
    vertical_half_fov_deg: float,
) -> tuple[np.ndarray, np.ndarray, np.ndarray]:
    """Maps for ``cv2.remap`` from a virtual pinhole view to an f-theta lens, and the view's K.

    The virtual camera keeps the real one's position, axes and focal length, so extrinsics stay
    valid and only K changes. The principal point is centred, as the images models expect: an
    off-centre one cost WildDet3D both recall and accuracy.
    """
    focal, cx, cy = intrinsics[0, 0], intrinsics[0, 2], intrinsics[1, 2]
    # Pixel radius -> angle off the axis, by inverting the lens polynomial numerically.
    theta_grid = np.radians(np.arange(0.0, 90.0, 0.1))
    radius_grid = np.polynomial.polynomial.polyval(theta_grid, fw_poly)
    # A fixed vertical field gives every sensor kit the same aspect ratio. The principal point sits
    # at very different heights across kits, so "as much as the lens sees" would not.
    theta_vertical = min(
        np.radians(vertical_half_fov_deg),
        np.interp(cy, radius_grid, theta_grid),
        np.interp(height - cy, radius_grid, theta_grid),
    )
    width_out = int(round(2 * focal * np.tan(np.radians(half_fov_deg))))
    height_out = int(round(2 * focal * np.tan(theta_vertical)))
    half_width, half_height = width_out / 2, height_out / 2
    new_k = np.array(
        [[focal, 0.0, half_width], [0.0, focal, half_height], [0.0, 0.0, 1.0]],
        dtype=np.float32,
    )
    u, v = np.meshgrid(
        np.arange(width_out, dtype=np.float64), np.arange(height_out, dtype=np.float64)
    )
    x, y = (u - half_width) / focal, (v - half_height) / focal
    rho = np.hypot(x, y)
    radius = np.polynomial.polynomial.polyval(np.arctan(rho), fw_poly)
    scale = np.divide(radius, rho, out=np.full_like(rho, focal), where=rho > 1e-9)
    return (
        (cx + x * scale).astype(np.float32),
        (cy + y * scale).astype(np.float32),
        new_k,
    )


# Rig (rear axle) to vehicle centre and half sizes, to recognise the ego's own track in the autolabels.
EGO_CENTER_X_M = 1.4
EGO_HALF_LENGTH_M = 2.45
EGO_HALF_WIDTH_M = 1.0


class PhysicalAIDataset(BaseDataset):
    """Dataset for PhysicalAI-AV that streams or loads data locally.

    Provides multi-camera video frames and egomotion via the ``physical_ai_av``
    library.  Each sample is returned as an :class:`E2EDataSample`.

    Args:
        root: Cache directory for downloaded data.  Falls back to
            ``PHYSICAL_AI_CACHE_DIR`` then ``HF_HOME`` env vars if empty.
        split: One of ``"train"``, ``"val"``, ``"test"``.  Used to look for a
            split-specific clip-ID file (``{root}/{split}.parquet`` or
            ``{root}/{split}.txt``).  If no such file exists, **all** available
            clips are loaded from the dataset API.
        sensors: Camera configuration – any key from
            ``camera_config.SENSOR_CONFIGS`` (e.g. ``"front_cam_only"``,
            ``"front_three"``, ``"front_four_tele"``, ``"all_cams"``).
        img_transform: Optional torchvision transform applied to each image.
        clip_ids: Explicit list of clip IDs.  When provided, used directly
            without any file lookup.  Otherwise ``_load_clip_ids`` searches
            for ``{root}/{split}.parquet`` / ``.txt``, then falls back to
            fetching all clips from the dataset API.
        clip_ids_path: Explicit ``.txt``, ``.parquet``, or manifest ``.json``
            file containing clip IDs.  This is useful when *root* points to the
            HuggingFace cache and subset metadata lives elsewhere.
        t0_us: Reference timestamp in microseconds (default 5.1 s into clip).
        sample_t0_us: Optional list of timestamps.  When provided, every clip ID
            yields one sample per timestamp.
        num_history_steps: Number of past trajectory steps (default 16 = 1.6 s @ 10 Hz).
        num_future_steps: Number of future trajectory steps (default 64 = 6.4 s @ 10 Hz).
        time_step_s: Time between consecutive trajectory points in seconds (default 0.1 s).
        num_frames: Number of video frames per camera (default 4). Frames before the clip's start
            are left out, so the list can be shorter.
        frame_spacing_s: Time between those frames; ``None`` uses ``time_step_s``.
        stream: If ``True`` (default), stream from HuggingFace; otherwise use
            cached data from *root* / ``HF_HOME``.
        hf_token: HuggingFace token.  Falls back to ``HF_TOKEN`` env var.
        revision: Optional PhysicalAI HuggingFace snapshot revision.  Pin this
            for offline/local-cache training to avoid remote revision lookup.
        include_future: Whether to include future trajectory in the sample.
        undistort: Return pinhole images, as the other datasets have, instead of the raw f-theta
            ones. The view keeps each camera's position, axes and focal length, with a centred
            principal point, so ``intrinsics`` is a true pinhole matrix and ``fw_poly`` /
            ``bw_poly`` are gone from ``cal_dict``.
        undistort_half_fov_deg: Horizontal half field of view of the undistorted view.
        undistort_vertical_half_fov_deg: Its vertical half field of view. The default is what the
            ``hyperion_8.1`` lens can show below the axis, so both kits get the same framing and
            aspect; a lens that can't reach it clamps it.
        intent_from_future: Derive the high-level intent from the future trajectory
            (``ad_vla.utils.intent_utils.intent_from_path``) instead of ``UNKNOWN``.
        intent_labels_path: Table of 7-class intents keyed by ``<clip_id>__<t0_us>``
            (``ad_vla.dataset.intent_labels``); replaces ``intent_from_future``.
        max_video_readers: Maximum number of decoded video reader handles to
            keep open per dataset worker.  ``None`` keeps all readers.
    """

    def __init__(
        self,
        root: str,
        split: str,
        sensors: str = "front_four_tele",
        img_transform: torch.nn.Module | None = None,
        reasoning_traces_path: str | None = None,
        # PhysicalAI-specific parameters
        clip_ids: list[str] | None = None,
        clip_ids_path: str | None = None,
        t0_us: int = 5_100_000,
        sample_t0_us: list[int] | int | None = None,
        num_history_steps: int = 16,
        num_future_steps: int = 64,
        time_step_s: float = 0.1,
        num_frames: int = 4,
        frame_spacing_s: float | None = None,
        stream: bool = True,
        hf_token: str | None = None,
        revision: str | None = None,
        include_future: bool = True,
        max_video_readers: int | None = 32,
        intent_from_future: bool = False,
        intent_labels_path: str | None = None,
        agent_labels_dir: str | None = None,
        agent_future_steps: int = 10,
        agent_future_interval_s: float = 0.5,
        undistort: bool = False,
        undistort_half_fov_deg: float = 35.0,
        undistort_vertical_half_fov_deg: float = 19.5,
        **kwargs: Any,
    ) -> None:
        super().__init__(root, split, sensors, img_transform, reasoning_traces_path)

        self.undistort = undistort
        self.undistort_half_fov_deg = undistort_half_fov_deg
        self.undistort_vertical_half_fov_deg = undistort_vertical_half_fov_deg
        self._undistort_maps: dict[tuple, tuple] = {}
        self.agent_labels_dir = agent_labels_dir
        self.agent_future_steps = agent_future_steps
        self.agent_future_interval_s = agent_future_interval_s
        self.t0_us = t0_us
        self.sample_t0_us = self._resolve_sample_t0_us(t0_us, sample_t0_us)
        self.num_history_steps = num_history_steps
        self.num_future_steps = num_future_steps
        self.time_step_s = time_step_s
        self.num_frames = num_frames
        self.frame_spacing_s = (
            time_step_s if frame_spacing_s is None else frame_spacing_s
        )
        self.stream = stream
        self.revision = revision
        self.include_future = include_future
        self.max_video_readers = max_video_readers
        # PhysicalAI has no route/intent labels; optionally derive them from the future
        # trajectory with the rule shared with deployment (ad_vla/utils/intent_utils.py).
        self.intent_from_future = intent_from_future
        # Table of 7-class intents keyed by "<clip_id>__<t0_us>" (ad_vla.dataset.intent_labels);
        # it replaces intent_from_future.
        self.intent_table = load_intent_table(intent_labels_path)

        # Resolve HF token: explicit arg > HF_TOKEN env var
        self.hf_token = hf_token or os.environ.get("HF_TOKEN")

        # Resolve cache dir: explicit root > PHYSICAL_AI_CACHE_DIR > HF_HOME
        if not self.root:
            self.root = os.environ.get(
                "PHYSICAL_AI_CACHE_DIR",
                os.environ.get("HF_HOME", ""),
            )

        # Resolve which cameras to load via the central config + this
        # dataset's raw-to-standard mapping.
        # _resolve_cameras validates and returns standard names; we also
        # build the reverse map (standard → raw) for data-loading methods.
        self._std_to_raw: dict[str, str] = {v: k for k, v in PHYSAI_CAM_LABELS.items()}
        available_std = set(PHYSAI_CAM_LABELS.values())
        self.standard_cameras: tuple[str, ...] = self._resolve_cameras(available_std)
        self.camera_names: tuple[str, ...] = tuple(
            self._std_to_raw[s] for s in self.standard_cameras
        )

        # Lazy-init handles
        self._avdi = None
        self._video_readers: OrderedDict[tuple[str, str], Any] = OrderedDict()

        # Load clip IDs: explicit list or split file / API fallback
        if clip_ids is not None and clip_ids_path is not None:
            raise ValueError("Provide either clip_ids or clip_ids_path, not both")
        self.clip_ids = (
            list(clip_ids)
            if clip_ids is not None
            else self._load_clip_ids(clip_ids_path)
        )
        self.sample_index: list[tuple[str, int]] = [
            (clip_id, t0_us) for clip_id in self.clip_ids for t0_us in self.sample_t0_us
        ]
        if self.intent_table is not None:
            print(
                self.intent_table.report(
                    f"PhysicalAI {split}", (f"{c}__{t}" for c, t in self.sample_index)
                )
            )

        # Sampling frequency in Hz (10 Hz by default)
        self.sampling_freq = round(1.0 / self.time_step_s)
        self.teacher_image_size = (720, 1280)

    # ------------------------------------------------------------------
    # Lazy AVDI accessor
    # ------------------------------------------------------------------
    @property
    def avdi(self):
        """Lazily initialise ``PhysicalAIAVDatasetInterface``."""
        if self._avdi is None:
            import physical_ai_av

            kwargs: dict[str, Any] = {}
            if self.hf_token:
                kwargs["token"] = self.hf_token
            if self.revision:
                kwargs["revision"] = self.revision
            if self.root:
                kwargs["cache_dir"] = str(self.root)
            self._avdi = physical_ai_av.PhysicalAIAVDatasetInterface(**kwargs)
        return self._avdi

    # ------------------------------------------------------------------
    # Clip ID loading
    # ------------------------------------------------------------------
    @staticmethod
    def _resolve_sample_t0_us(
        t0_us: int,
        sample_t0_us: list[int] | int | None,
    ) -> list[int]:
        if sample_t0_us is None:
            values = [t0_us]
        elif isinstance(sample_t0_us, int):
            values = [sample_t0_us]
        else:
            values = list(sample_t0_us)

        if not values:
            raise ValueError("sample_t0_us must contain at least one timestamp")

        return [int(value) for value in values]

    def _load_clip_ids(self, clip_ids_path: str | None = None) -> list[str]:
        """Load clip IDs from a split-specific file in *root*, or from the API.

        Lookup order:
        1. Explicit ``clip_ids_path`` when provided.
        2. ``{root}/{split}.parquet`` – reads the ``clip_id`` column (or index).
        3. ``{root}/{split}.txt`` – one clip ID per line.
        4. Falls back to fetching **all** clip IDs from the dataset API.
        """
        if clip_ids_path is not None:
            path = Path(clip_ids_path)
            if not path.is_file():
                raise FileNotFoundError(f"clip_ids_path does not exist: {path}")
            return self._read_clip_ids_file(path)

        if self.root:
            root_path = Path(self.root)

            # Try parquet first
            parquet_file = root_path / f"{self.split}.parquet"
            if parquet_file.is_file():
                return self._read_clip_ids_file(parquet_file)

            # Try plain text
            txt_file = root_path / f"{self.split}.txt"
            if txt_file.is_file():
                return self._read_clip_ids_file(txt_file)

        # No split file found → fetch all clip IDs from the dataset
        all_ids = self.avdi.clip_index.index.tolist()
        all_ids.sort()
        return all_ids

    @staticmethod
    def _read_clip_ids_file(path: Path) -> list[str]:
        suffix = path.suffix.lower()

        if suffix == ".parquet":
            import pandas as pd

            df = pd.read_parquet(path)
            if "clip_id" in df.columns:
                return [str(clip_id) for clip_id in df["clip_id"].tolist()]
            return [str(clip_id) for clip_id in df.index.tolist()]

        if suffix == ".json":
            import json

            with path.open() as f:
                data = json.load(f)

            if "clip_ids" in data:
                return [str(clip_id) for clip_id in data["clip_ids"]]
            if "entries" in data:
                return [str(entry["clip_id"]) for entry in data["entries"]]
            raise ValueError(
                f"JSON clip ID file must contain 'clip_ids' or 'entries': {path}"
            )

        return [
            line.strip()
            for line in path.read_text().strip().split("\n")
            if line.strip()
        ]

    # ------------------------------------------------------------------
    # Dataset interface
    # ------------------------------------------------------------------
    def __len__(self) -> int:
        return len(self.sample_index)

    def _get_sample(self, idx: int) -> dict[str, Any]:
        """Load raw egomotion, camera frames, and calibration for one clip."""
        clip_id, t0_us = self.sample_index[idx]

        ego_data = self._load_egomotion(clip_id, t0_us)
        frames_per_cam, frame_timestamps = self._load_frames(clip_id, t0_us)
        calibrations = self._load_all_calibrations(clip_id)
        if self.undistort:
            frames_per_cam, calibrations = self._undistort_views(
                frames_per_cam, calibrations
            )

        return {
            "clip_id": clip_id,
            "t0_us": t0_us,
            "agent_info": self._load_agent_info(clip_id, t0_us),
            "ego": ego_data,
            "frames_per_cam": frames_per_cam,
            "frame_timestamps": frame_timestamps,
            "calibrations": calibrations,
        }

    def _clean_sample(self, scenario: dict[str, Any]) -> E2EDataSample:
        """Convert raw data dict into :class:`E2EDataSample`."""
        info: dict[str, Any] = {}

        # -- agent_input -----------------------------------------------
        ego = scenario["ego"]
        # past_traj: take x, y from history (drop z)
        info["agent_input"] = {
            "past_traj": ego["history_xyz"][:, :2],  # [T, 2]
        }

        if scenario.get("agent_info") is not None:
            info["agent_info"] = scenario["agent_info"]

        # -- intent (set below from the future trajectory if enabled) ---
        info["intent"] = "UNKNOWN"

        # -- cameras ---------------------------------------------------
        # cameras is List[Dict[str, CameraData-like dict]]
        # one entry per timestep, each mapping standard camera name -> {image, cal_dict}
        frames_per_cam = scenario[
            "frames_per_cam"
        ]  # dict[raw_name -> (num_frames, H, W, 3)]
        calibrations = scenario["calibrations"]  # dict[raw_name -> cal_dict kwargs]

        num_frames = next(iter(frames_per_cam.values())).shape[0]
        cameras: list[dict[str, Any]] = [{} for _ in range(num_frames)]
        info["teacher_cameras"] = [{}]
        teacher_timestep = num_frames - 1

        for raw_name, frames_np in frames_per_cam.items():
            std_name = PHYSAI_CAM_LABELS[raw_name]
            base_cal_kwargs = calibrations[raw_name]

            for t in range(num_frames):
                image = torch.from_numpy(frames_np[t])  # (H, W, 3) uint8
                if t == teacher_timestep:
                    info["teacher_cameras"][-1][std_name] = (
                        self._make_teacher_camera_data(image, base_cal_kwargs)
                    )

                image, cal_kwargs = self._apply_img_transform(
                    image, base_cal_kwargs, std_name
                )

                cameras[t][std_name] = {
                    "image": image,
                    "cal_dict": cal_kwargs,
                }

        info["cameras"] = cameras

        # -- future trajectory -----------------------------------------
        if "future_xyz" in ego:
            info["fut_traj"] = ego["future_xyz"][:, :2]  # [T, 2]
            info["fut_traj_sampling"] = TrajectorySampling(
                num_poses=info["fut_traj"].shape[0],
                interval_length=1.0 / self.sampling_freq,
            )
            if self.intent_from_future:
                info["intent"] = intent_from_path(info["fut_traj"])
        if self.intent_table is not None:
            fut = info.get("fut_traj", np.zeros((0, 2)))
            first = (
                np.asarray(fut[0][:2], dtype=np.float64) if len(fut) else np.zeros(2)
            )
            speed = float(np.hypot(*first)) * self.sampling_freq
            info["intent"] = self.intent_table.get(
                f"{scenario['clip_id']}__{scenario['t0_us']}"
            ) or intent7_from_path(fut, speed)

        # -- metadata --------------------------------------------------
        scenario_id = scenario["clip_id"]
        if len(self.sample_t0_us) > 1:
            scenario_id = f"{scenario_id}__t0_{scenario['t0_us']}"

        info["metadata"] = {
            "scenario_id": scenario_id,
            "clip_id": scenario["clip_id"],
            "sampling_freq": self.sampling_freq,
            "t0_us": scenario["t0_us"],
        }

        if self.reasoning_trace_df is not None:
            try:
                reasoning_trace = self.reasoning_trace_df[
                    self.reasoning_trace_df.scenario_id == info["metadata"]["clip_id"]
                ].text.item()
                info["reasoning_trace"] = reasoning_trace
            except ValueError:
                info["reasoning_trace"] = None

        return E2EDataSample(**info)

    # ------------------------------------------------------------------
    # Undistortion
    # ------------------------------------------------------------------
    def _undistort_views(
        self, frames_per_cam: dict[str, np.ndarray], calibrations: dict[str, dict]
    ) -> tuple[dict[str, np.ndarray], dict[str, dict]]:
        """Remap f-theta frames to a pinhole view, with the matching calibration."""
        out_frames: dict[str, np.ndarray] = {}
        out_calibrations: dict[str, dict] = {}
        for cam_name, frames in frames_per_cam.items():
            calibration = calibrations[cam_name]
            # Calibrations differ little across the fleet, so a few maps serve many clips.
            key = (
                cam_name,
                frames.shape[1],
                calibration["fw_poly"].numpy().tobytes(),
                calibration["intrinsics"].numpy().tobytes(),
            )
            if key not in self._undistort_maps:
                if len(self._undistort_maps) >= 256:
                    self._undistort_maps.clear()
                self._undistort_maps[key] = ftheta_to_pinhole_maps(
                    calibration["fw_poly"].numpy().astype(np.float64),
                    calibration["intrinsics"].numpy().astype(np.float64),
                    frames.shape[1],
                    self.undistort_half_fov_deg,
                    self.undistort_vertical_half_fov_deg,
                )
            map_x, map_y, intrinsics = self._undistort_maps[key]
            out_frames[cam_name] = np.stack(
                [
                    cv2.remap(
                        frame,
                        map_x,
                        map_y,
                        cv2.INTER_LINEAR,
                        borderMode=cv2.BORDER_CONSTANT,
                    )
                    for frame in frames
                ]
            )
            pinhole = {
                k: v for k, v in calibration.items() if k not in ("fw_poly", "bw_poly")
            }
            pinhole["intrinsics"] = torch.from_numpy(intrinsics.copy())
            pinhole["original_img_size"] = map_x.shape
            out_calibrations[cam_name] = pinhole
        return out_frames, out_calibrations

    # ------------------------------------------------------------------
    # Agent boxes (obstacle.offline autolabels)
    # ------------------------------------------------------------------
    def _load_agent_info(self, clip_id: str, t0_us: int) -> dict[str, Any] | None:
        """``AgentInfo`` at ``t0_us`` from the clip's autolabels; ``boxes=None`` if the clip has none.

        Rows are in the rig frame at their own time, about 10 Hz per track and not synchronized, so
        positions are interpolated per track through the world frame. A track counts at a time only
        if it has rows within one label gap (0.3 s) on either side.
        """
        if self.agent_labels_dir is None:
            return None
        rows = self._read_agent_labels(clip_id)
        if rows is None:
            return {"boxes": None, "source": "autolabel"}
        times = t0_us + (
            np.arange(self.agent_future_steps + 1) * self.agent_future_interval_s * 1e6
        ).astype(np.int64)
        rows = rows[
            (rows["t"] >= t0_us - 1_000_000) & (rows["t"] <= times[-1] + 1_000_000)
        ]
        egomotion = self._load_egomotion_interpolator(clip_id)
        pose0 = egomotion(np.array([t0_us], dtype=np.int64)).pose
        rot0_inv, trans0 = pose0.rotation[0].inv(), pose0.translation[0]
        ids, names, boxes, fut, mask = [], [], [], [], []
        max_gap_us = 300_000
        for track_id, track in rows.groupby("track_id", sort=False):
            track = track.sort_values("t")
            t = track["t"].to_numpy(
                copy=True
            )  # parquet buffers are read-only, which scipy refuses
            if not (t[0] <= t0_us <= t[-1]):
                continue
            # Rig at the row's time -> world -> rig at t0.
            poses = egomotion(t).pose
            local = track[["center_x", "center_y", "center_z"]].to_numpy(
                dtype=np.float64, copy=True
            )
            at_t0 = rot0_inv.apply(
                poses.rotation.apply(local) + poses.translation - trans0
            )
            local_yaw = 2.0 * np.arctan2(
                track["orientation_z"].to_numpy(), track["orientation_w"].to_numpy()
            )
            yaw = np.unwrap(
                local_yaw + (rot0_inv * poses.rotation).as_euler("ZYX")[:, 0]
            )
            size = track[["size_x", "size_y", "size_z"]].to_numpy(dtype=np.float64)
            state = np.stack(
                [np.interp(times, t, v) for v in (*at_t0.T, *size.T)]
                + [wrap_angle(np.interp(times, t, yaw))],
                axis=-1,
            ).astype(np.float32)
            right = np.searchsorted(t, times).clip(1, len(t) - 1)
            valid = (
                (times >= t[0])
                & (times <= t[-1])
                & (t[right] - t[right - 1] <= max_gap_us)
            )
            if not valid[0]:
                continue
            # Some clips label the ego vehicle itself: nothing else has its centre inside the ego's footprint.
            if (
                abs(state[0, 0] - EGO_CENTER_X_M) < EGO_HALF_LENGTH_M
                and abs(state[0, 1]) < EGO_HALF_WIDTH_M
            ):
                continue
            ids.append(str(track_id))
            names.append(str(track["label_class"].iloc[0]))
            boxes.append(state[0])
            fut.append(state[1:])
            mask.append(valid[1:])
        if not ids:
            return {"boxes": torch.zeros(0, 7), "track_ids": [], "types": [], "coarse_types": [], "source": "autolabel",
                    "fut_boxes": torch.zeros(0, self.agent_future_steps, 7), "fut_mask": torch.zeros(0, self.agent_future_steps, dtype=torch.bool),
                    "fut_interval_s": self.agent_future_interval_s}  # fmt: skip
        return {
            "boxes": torch.from_numpy(np.stack(boxes)),
            "track_ids": ids,
            "types": names,
            "coarse_types": [coarse_type(n) for n in names],
            "source": "autolabel",
            "fut_boxes": torch.from_numpy(np.stack(fut)),
            "fut_mask": torch.from_numpy(np.stack(mask)),
            "fut_interval_s": self.agent_future_interval_s,
        }

    def _read_agent_labels(self, clip_id: str):
        """The clip's ``obstacle.offline`` rows, or None if it has no labels. The zip handle is cached."""
        import io
        import zipfile

        import pandas as pd

        chunk = self.avdi.get_clip_chunk(clip_id)
        path = os.path.join(
            self.agent_labels_dir, f"obstacle.offline.chunk_{chunk:04d}.zip"
        )
        if not hasattr(self, "_label_zips"):
            self._label_zips = {}
        if chunk not in self._label_zips:
            if not os.path.exists(path):
                return None
            self._label_zips[chunk] = zipfile.ZipFile(path)
        try:
            data = self._label_zips[chunk].read(f"{clip_id}.obstacle.offline.parquet")
        except KeyError:
            return None
        rows = pd.read_parquet(io.BytesIO(data))
        rows["t"] = rows["timestamp_us"].astype(
            "int64"
        )  # uint64 wraps around when subtracting
        return rows

    # ------------------------------------------------------------------
    # Egomotion loading (adapted from reference code)
    # ------------------------------------------------------------------
    def _load_egomotion_interpolator(self, clip_id: str):
        """Load egomotion for *clip_id*, returning an interpolator.

        We replicate the logic of ``avdi.get_clip_feature`` for egomotion
        but call ``.copy()`` on DataFrame columns so that the resulting
        numpy arrays are **writable** — pyarrow-backed parquet frames
        return read-only buffers which cause scipy ``Rotation.from_quat``
        to fail.
        """
        import io
        import zipfile
        import pandas as pd
        from physical_ai_av import egomotion as ego_mod

        feature = self.avdi.features.LABELS.EGOMOTION
        chunk_filename = self.avdi.features.get_chunk_feature_filename(
            self.avdi.get_clip_chunk(clip_id),
            feature,
        )
        clip_files_in_zip = self.avdi.features.get_clip_files_in_zip(clip_id, feature)

        with self.avdi.open_file(chunk_filename, maybe_stream=self.stream) as f:
            with zipfile.ZipFile(f, "r") as zf:
                egomotion_df = pd.read_parquet(
                    io.BytesIO(zf.read(clip_files_in_zip["egomotion"]))
                )

        # ---- Make columns writable so scipy can operate in-place ----
        for col in egomotion_df.columns:
            egomotion_df[col] = egomotion_df[col].to_numpy(dtype="float64", copy=True)

        timestamps = egomotion_df["timestamp"].to_numpy().copy()
        ego_state = ego_mod.EgomotionState.from_egomotion_df(egomotion_df)
        return ego_state.create_interpolator(timestamps)

    def _load_egomotion(self, clip_id: str, t0_us: int) -> dict[str, torch.Tensor]:
        """Load egomotion, interpolate, and transform to ego-local frame at *t0*."""
        egomotion_interp = self._load_egomotion_interpolator(clip_id)

        time_step_us = int(self.time_step_s * 1_000_000)

        # History timestamps: [..., t0-0.2s, t0-0.1s, t0]
        history_offsets = np.arange(
            -(self.num_history_steps - 1) * time_step_us,
            time_step_us // 2,
            time_step_us,
            dtype=np.int64,
        )
        history_ts = t0_us + history_offsets

        ego_history = egomotion_interp(history_ts)
        history_xyz = ego_history.pose.translation  # (N, 3)
        history_quat = ego_history.pose.rotation.as_quat()  # (N, 4)

        # Transform to local frame at t0
        t0_xyz = history_xyz[-1].copy()
        t0_rot = spt.Rotation.from_quat(history_quat[-1])
        t0_rot_inv = t0_rot.inv()

        history_xyz_local = t0_rot_inv.apply(history_xyz - t0_xyz)

        result: dict[str, torch.Tensor] = {
            "history_xyz": torch.from_numpy(history_xyz_local).float(),
        }

        # Future trajectory
        if self.include_future:
            future_offsets = np.arange(
                time_step_us,
                int((self.num_future_steps + 0.5) * time_step_us),
                time_step_us,
                dtype=np.int64,
            )
            future_ts = t0_us + future_offsets

            ego_future = egomotion_interp(future_ts)
            future_xyz = ego_future.pose.translation
            future_xyz_local = t0_rot_inv.apply(future_xyz - t0_xyz)

            result["future_xyz"] = torch.from_numpy(future_xyz_local).float()

        return result

    # ------------------------------------------------------------------
    # Frame loading
    # ------------------------------------------------------------------
    def _load_frames(
        self, clip_id: str, t0_us: int
    ) -> tuple[dict[str, np.ndarray], dict[str, np.ndarray]]:
        """Decode video frames for all configured cameras.

        Returns:
            frames_per_cam: ``{raw_camera_name: np.ndarray (num_frames, H, W, 3)}``
            timestamps_per_cam: ``{raw_camera_name: np.ndarray (num_frames,) int64}``
        """
        spacing_us = int(self.frame_spacing_s * 1_000_000)
        frame_ts = np.array(
            [
                t0_us - (self.num_frames - 1 - i) * spacing_us
                for i in range(self.num_frames)
            ],
            dtype=np.int64,
        )
        frame_ts = frame_ts[frame_ts >= 0]

        frames_per_cam: dict[str, np.ndarray] = {}
        timestamps_per_cam: dict[str, np.ndarray] = {}

        for cam_name in self.camera_names:
            cache_key = (clip_id, cam_name)
            if cache_key not in self._video_readers:
                reader = self.avdi.get_clip_feature(
                    clip_id,
                    cam_name,
                    maybe_stream=self.stream,
                )
                self._video_readers[cache_key] = reader
            else:
                reader = self._video_readers.pop(cache_key)
                self._video_readers[cache_key] = reader

            self._trim_video_readers()

            # frames: (num_frames, H, W, 3) uint8
            frames, actual_ts = reader.decode_images_from_timestamps(frame_ts)

            frames_per_cam[cam_name] = frames
            timestamps_per_cam[cam_name] = actual_ts.astype(np.int64)

        return frames_per_cam, timestamps_per_cam

    def _trim_video_readers(self) -> None:
        if self.max_video_readers is None:
            return

        while len(self._video_readers) > self.max_video_readers:
            _, reader = self._video_readers.popitem(last=False)
            if hasattr(reader, "close"):
                reader.close()

    # ------------------------------------------------------------------
    # Calibration loading & conversion
    # ------------------------------------------------------------------
    def _load_all_calibrations(self, clip_id: str) -> dict[str, dict]:
        """Load and convert calibration for every configured camera.

        Returns a dict mapping raw camera names to ``CalibrationDict``-compatible
        keyword dicts (including extra f-theta fields).
        """
        # Fetch raw calibration DataFrames for this clip
        intrinsics_df = self.avdi.get_clip_feature(
            clip_id,
            self.avdi.features.CALIBRATION.CAMERA_INTRINSICS,
            maybe_stream=self.stream,
        )
        extrinsics_df = self.avdi.get_clip_feature(
            clip_id,
            self.avdi.features.CALIBRATION.SENSOR_EXTRINSICS,
            maybe_stream=self.stream,
        )

        result: dict[str, dict] = {}
        for cam_name in self.camera_names:
            result[cam_name] = self._convert_calibration(
                cam_name, intrinsics_df, extrinsics_df
            )

        return result

    @staticmethod
    def _convert_calibration(
        camera_name: str,
        intrinsics_obj,
        extrinsics_obj,
    ) -> dict:
        """Convert f-theta calibration to a ``CalibrationDict``-compatible dict.

        Returns dict with the required fields (intrinsics, distortion, rotation,
        translation, original_img_size, is_flu) **plus** extra f-theta fields
        (fw_poly, bw_poly) that pass through via ``extra='allow'``.
        """
        # -- Intrinsics (f-theta) --------------------------------------
        cam_model = intrinsics_obj.camera_models[camera_name]
        cx, cy = cam_model.principal_point
        width = cam_model.width
        height = cam_model.height

        # th2r = forward (angle->radius), r2th = backward (radius->angle)
        # Pad/truncate to 5 coefficients for consistency
        fw_coef = cam_model.th2r.coef
        bw_coef = cam_model.r2th.coef
        fw_poly = np.zeros(5, dtype=np.float32)
        bw_poly = np.zeros(5, dtype=np.float32)
        fw_poly[: min(len(fw_coef), 5)] = fw_coef[:5]
        bw_poly[: min(len(bw_coef), 5)] = bw_coef[:5]

        # Approximate pinhole intrinsic matrix using linear coefficient
        focal = float(fw_poly[1]) if fw_poly[1] != 0.0 else 1.0
        intrinsics = torch.tensor(
            [[focal, 0.0, float(cx)], [0.0, focal, float(cy)], [0.0, 0.0, 1.0]],
            dtype=torch.float32,
        )

        # -- Extrinsics ------------------------------------------------
        pose = extrinsics_obj.sensor_poses[camera_name]
        rotation = torch.from_numpy(pose.rotation.as_matrix().astype(np.float32))
        translation = torch.from_numpy(pose.translation.astype(np.float32))

        return {
            "intrinsics": intrinsics,
            "distortion": torch.zeros(5, dtype=torch.float32),
            "rotation": rotation,
            "translation": translation,
            "original_img_size": (int(height), int(width)),
            "is_flu": False,  # PhysicalAI extrinsics give RDF camera frame (z = optical axis)
            # Extra f-theta fields (stored via extra='allow')
            "fw_poly": torch.from_numpy(fw_poly),
            "bw_poly": torch.from_numpy(bw_poly),
        }

    # ------------------------------------------------------------------
    # Resource management
    # ------------------------------------------------------------------
    def close(self) -> None:
        """Close all video readers and release resources."""
        for reader in self._video_readers.values():
            if hasattr(reader, "close"):
                reader.close()
        self._video_readers.clear()

    def __enter__(self):
        return self

    def __exit__(self, *args):
        self.close()

    def __del__(self):
        try:
            self.close()
        except Exception:
            pass
