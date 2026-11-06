"""Closed-loop AlpaSim states as RL training samples (no ground-truth future).

Each ``.npz`` is one model call recorded by the AlpaSim driver and joined with its rollout log
(the extraction isn't part of this repository). The sample is rebuilt exactly as the driver's
``FmvlaPolicy._sample`` builds it (camera JPEG -> RGB -> ``ImageResize`` on CHW -> HWC, simulator
camera ids renamed to the training names, intent from the command one-hot, past trajectory as
recorded, at least two poses), so the policy sees in training what it saw in the simulator.
The reward context (drivable raster, recorded ego path, actor boxes, ego box) is attached as
``metadata["rl_ctx"]`` for ``ad_vla.rewards.sim_pdm_reward.SimPDMReward``.
"""

from __future__ import annotations

import glob
import io
import os
from collections.abc import Sequence

import numpy as np
import torch
from PIL import Image
from torch.utils.data import Dataset

from ad_vla.dataset.data_types import CameraData, E2EDataSample
from ad_vla.dataset.navsim.navsim_types import NAVSIM_INTENT_DICT
from ad_vla.utils.intent_utils import INTENTS_7
from ad_vla.utils.img_utils import ImageResize

# Simulator logical camera id -> training camera name, per driver rig (ttvlm_driver/policy.py CAMERA_RIGS);
# a state's ``camera_rig`` names its rig (absent in the first NavTest states: "navsim").
CAMERA_RIGS = {
    "navsim": {"CAM_L0": "FRONT_LEFT", "CAM_F0": "FRONT", "CAM_R0": "FRONT_RIGHT"},
    "physical_ai": {
        "camera_cross_left_120fov": "FRONT_LEFT",
        "camera_front_wide_120fov": "FRONT",
        "camera_cross_right_120fov": "FRONT_RIGHT",
    },
}

RL_CTX_KEYS = (
    "path_xy",
    "drivable_origin",
    "drivable_res",
    "agent_corners",
    "agent_valid",
    "agent_times_s",
    "ego_length",
    "ego_width",
    "rig_to_center_xy",
    "velocity",
)


class AlpaSimStateDataset(Dataset):
    """RL states from one or more extracted state directories.

    Args:
        state_dirs: Directories (or glob patterns) of extracted ``.npz`` states.
        state_list: Instead of ``state_dirs``, a text file with one state path per line,
            optionally followed by a tab and a group name (``metadata["probe_group"]``, used
            by the GRPO probe to report groups separately); ``#`` starts a comment line.
        image_size: ``[height, width]`` of the model input, as the driver's ``TTVLM_IMAGE_SIZE``.
        past_interval_s: Spacing of the recorded past trajectory: the checkpoint's
            ``past_traj_sampling.interval_length`` the driver used.
        max_states: Optional cap (the first ``max_states`` files in sorted order).
        centerline_dirs: Optional directories of lane-centreline files named as the states: their ``centerline_xy`` and ``center_halfwidth`` join
            the reward context (the reward's ``pk_path="centerline"`` and centring terms).
        split: Unused; accepted for the training script's dataset interface.
        reasoning_traces_path: Unused; accepted for the training script's dataset interface.
    """

    def __init__(
        self,
        state_dirs: str | Sequence[str] | None = None,
        state_list: str | None = None,
        image_size: Sequence[int] = (448, 768),
        past_interval_s: float = 0.25,
        max_states: int | None = None,
        centerline_dirs: str | Sequence[str] | None = None,
        split: str | None = None,
        reasoning_traces_path: str | None = None,
        **kwargs,
    ) -> None:
        self.centerlines: dict[str, str] = {}
        for d in (
            [centerline_dirs]
            if isinstance(centerline_dirs, str)
            else list(centerline_dirs or [])
        ):
            for name in os.listdir(d):
                self.centerlines.setdefault(name, os.path.join(d, name))
        if (state_dirs is None) == (state_list is None):
            raise ValueError("Pass exactly one of state_dirs and state_list.")
        files = []
        self.groups: dict[str, str] = {}
        if state_list is not None:
            with open(state_list) as handle:
                for line in handle:
                    line = line.strip()
                    if not line or line.startswith("#"):
                        continue
                    path, _, group = line.partition("\t")
                    files.append(path)
                    if group:
                        self.groups[path] = group
            if not files:
                raise FileNotFoundError(f"No states listed in {state_list}")
        else:
            dirs = [state_dirs] if isinstance(state_dirs, str) else list(state_dirs)
            for pattern in dirs:
                for d in sorted(glob.glob(pattern)) or [pattern]:
                    files.extend(sorted(glob.glob(os.path.join(d, "*.npz"))))
            if not files:
                raise FileNotFoundError(f"No .npz states under {dirs}")
        self.files = files[:max_states] if max_states else files
        self.resize = ImageResize(list(image_size))
        self.past_interval_s = float(past_interval_s)

    def __len__(self) -> int:
        return len(self.files)

    def _camera(self, jpeg: np.ndarray) -> CameraData:
        with Image.open(io.BytesIO(jpeg.tobytes())) as decoded:
            rgb = np.asarray(decoded.convert("RGB")).copy()
        rgb = torch.from_numpy(np.ascontiguousarray(rgb))
        resized = self.resize(rgb.permute(2, 0, 1)).permute(1, 2, 0).contiguous()
        return CameraData(image=resized)

    def __getitem__(self, idx: int) -> E2EDataSample:
        path = self.files[idx]
        with np.load(path) as data:
            state = {k: data[k] for k in data.files}
        rig = str(state["camera_rig"]) if "camera_rig" in state else "navsim"
        cameras = {
            name: self._camera(state[f"jpeg_{cam}"])
            for cam, name in CAMERA_RIGS[rig].items()
        }
        past = torch.as_tensor(state["past_traj_xy"], dtype=torch.float32)
        if past.shape[0] < 2:
            # As FmvlaPolicy._sample: the collator reads the speed from the last two poses.
            past = torch.cat([past[:1].expand(2 - past.shape[0], -1), past])
        shape = tuple(int(v) for v in state["drivable_shape"])
        drivable = np.unpackbits(state["drivable"])[: shape[0] * shape[1]].reshape(
            shape
        )
        rl_ctx = {k: state[k] for k in RL_CTX_KEYS}
        # Added to the extractor later (human progress over the plan horizon); absent in older states.
        for key in ("human_progress_m", "human_covered_s"):
            if key in state:
                rl_ctx[key] = float(state[key])
        rl_ctx["drivable"] = drivable.astype(bool)
        if (
            "road_edge" in state
        ):  # maps with road edges and no road areas (PhysicalAI); see the reward's DAC
            rl_ctx["road_edge"] = (
                np.unpackbits(state["road_edge"])[: shape[0] * shape[1]]
                .reshape(shape)
                .astype(bool)
            )
        rl_ctx["past_traj_xy"] = past.numpy()
        rl_ctx["past_interval_s"] = self.past_interval_s
        centerline = self.centerlines.get(os.path.basename(path))
        if centerline is not None:
            with np.load(centerline) as data:
                rl_ctx["centerline_xy"] = data["centerline_xy"]
                rl_ctx["center_halfwidth"] = data["center_halfwidth"]
        return E2EDataSample(
            metadata={
                "scenario_id": os.path.basename(path)[: -len(".npz")],
                "sampling_freq": 1.0 / self.past_interval_s,
                "rl_ctx": rl_ctx,
                "probe_group": self.groups.get(path),
            },
            agent_input={"past_traj": past},
            cameras=[cameras],
            # 4 elements: route command (NavSim order); 7: TTVLM_INTENT_RULE=intent7 (INTENTS_7).
            intent=(
                INTENTS_7
                if len(state["command_one_hot"]) == len(INTENTS_7)
                else NAVSIM_INTENT_DICT
            )[int(np.argmax(state["command_one_hot"]))],
        )
