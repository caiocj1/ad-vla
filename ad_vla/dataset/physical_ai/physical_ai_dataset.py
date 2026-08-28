"""PyTorch Dataset wrapper for PhysicalAI-AV dataset, compatible with BaseDataset / E2EDataSample."""

import os
from collections import OrderedDict
from pathlib import Path
from typing import Any

import numpy as np
import scipy.spatial.transform as spt
import torch

from ad_vla.dataset.base_dataset import BaseDataset
from ad_vla.dataset.data_types import E2EDataSample, TrajectorySampling
from ad_vla.dataset.physical_ai.physical_ai_types import PHYSAI_CAM_LABELS


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
        num_frames: Number of video frames per camera (default 4).
        stream: If ``True`` (default), stream from HuggingFace; otherwise use
            cached data from *root* / ``HF_HOME``.
        hf_token: HuggingFace token.  Falls back to ``HF_TOKEN`` env var.
        revision: Optional PhysicalAI HuggingFace snapshot revision.  Pin this
            for offline/local-cache training to avoid remote revision lookup.
        include_future: Whether to include future trajectory in the sample.
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
        stream: bool = True,
        hf_token: str | None = None,
        revision: str | None = None,
        include_future: bool = True,
        max_video_readers: int | None = 32,
        **kwargs: Any,
    ) -> None:
        super().__init__(root, split, sensors, img_transform, reasoning_traces_path)

        self.t0_us = t0_us
        self.sample_t0_us = self._resolve_sample_t0_us(t0_us, sample_t0_us)
        self.num_history_steps = num_history_steps
        self.num_future_steps = num_future_steps
        self.time_step_s = time_step_s
        self.num_frames = num_frames
        self.stream = stream
        self.revision = revision
        self.include_future = include_future
        self.max_video_readers = max_video_readers

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

        return {
            "clip_id": clip_id,
            "t0_us": t0_us,
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

        # -- intent ----------------------------------------------------
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

                image, cal_kwargs = self._apply_img_transform(image, base_cal_kwargs)

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
        time_step_us = int(self.time_step_s * 1_000_000)
        frame_ts = np.array(
            [
                t0_us - (self.num_frames - 1 - i) * time_step_us
                for i in range(self.num_frames)
            ],
            dtype=np.int64,
        )

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
