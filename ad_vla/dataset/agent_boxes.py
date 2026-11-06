"""Shared helpers for agent boxes in ``AgentInfo``: coarse classes and frame conversions."""

from __future__ import annotations

import json
import os

import numpy as np
import torch

_COARSE_BY_NAME = {
    # NavSim
    "vehicle": "vehicle", "pedestrian": "person", "bicycle": "cyclist",
    # PhysicalAI
    "automobile": "vehicle", "heavy_truck": "vehicle", "bus": "vehicle", "trailer": "vehicle",
    "other_vehicle": "vehicle", "train_or_tram_car": "vehicle", "person": "person", "stroller": "person",
    "rider": "cyclist",
    # WildDet3D prompts
    "car": "vehicle",
}  # fmt: skip


def coarse_type(name: str) -> str:
    """Map a source's raw class name to vehicle / person / cyclist / other."""
    if name.startswith("human.pedestrian"):  # nuScenes
        return "person"
    if name in ("vehicle.bicycle", "vehicle.motorcycle"):
        return "cyclist"
    if name.startswith("vehicle."):
        return "vehicle"
    return _COARSE_BY_NAME.get(name, "other")


def wrap_angle(a: np.ndarray) -> np.ndarray:
    return (a + np.pi) % (2.0 * np.pi) - np.pi


def box_corners_xyz(boxes: torch.Tensor) -> torch.Tensor:
    """Corners [N, 8, 3] of boxes [N, 7]: bottom face first (front-left, front-right, rear-right, rear-left)."""
    x, y, z, length, width, height, yaw = boxes.unbind(-1)
    sx = boxes.new_tensor([1, 1, -1, -1, 1, 1, -1, -1]) / 2
    sy = boxes.new_tensor([1, -1, -1, 1, 1, -1, -1, 1]) / 2
    sz = boxes.new_tensor([-1, -1, -1, -1, 1, 1, 1, 1]) / 2
    dx, dy = sx * length[..., None], sy * width[..., None]
    cos, sin = torch.cos(yaw)[..., None], torch.sin(yaw)[..., None]
    return torch.stack(
        [
            x[..., None] + dx * cos - dy * sin,
            y[..., None] + dx * sin + dy * cos,
            z[..., None] + sz * height[..., None],
        ],
        dim=-1,
    )


def camera_boxes_to_ego(
    boxes3d: np.ndarray, rotation: np.ndarray, translation: np.ndarray
) -> np.ndarray:
    """WildDet3D boxes [N, 10] (x, y, z, w, l, h, qw, qx, qy, qz; optical frame) to ego boxes [N, 7].

    ``p_ego = rotation @ p_camera + translation``. WildDet3D's length runs along the box's local x axis.
    """
    from scipy.spatial.transform import Rotation

    if len(boxes3d) == 0:
        return np.zeros((0, 7), dtype=np.float32)
    center = boxes3d[:, :3] @ rotation.T + translation
    q = boxes3d[:, [7, 8, 9, 6]]  # scipy order: x, y, z, w
    forward_cam = Rotation.from_quat(q).as_matrix()[:, :, 0]
    forward_ego = forward_cam @ rotation.T
    yaw = np.arctan2(forward_ego[:, 1], forward_ego[:, 0])
    return np.stack(
        [
            center[:, 0],
            center[:, 1],
            center[:, 2],
            boxes3d[:, 4],
            boxes3d[:, 3],
            boxes3d[:, 5],
            yaw,
        ],
        axis=-1,
    ).astype(np.float32)


def boxes_to_bev(
    boxes: torch.Tensor,
    x_range: tuple[float, float] = (0.0, 60.0),
    y_range: tuple[float, float] = (-30.0, 30.0),
    resolution: float = 0.5,
) -> torch.Tensor:
    """Occupancy [num_x, num_y] of ego-frame boxes [N, 7]; i grows forward, j grows left, a cell is its centre.

    The same map on every dataset, whatever the source of the boxes. Boxes smaller than a cell can miss every centre.
    """
    num_x, num_y = (
        round((x_range[1] - x_range[0]) / resolution),
        round((y_range[1] - y_range[0]) / resolution),
    )
    x = x_range[0] + (torch.arange(num_x, dtype=torch.float32) + 0.5) * resolution
    y = y_range[0] + (torch.arange(num_y, dtype=torch.float32) + 0.5) * resolution
    occupancy = torch.zeros(num_x, num_y, dtype=torch.bool)
    for box in boxes.float():
        dx, dy = x[:, None] - box[0], y[None, :] - box[1]
        cos, sin = torch.cos(box[6]), torch.sin(box[6])
        along, across = dx * cos + dy * sin, -dx * sin + dy * cos
        occupancy |= (along.abs() <= box[3] / 2) & (across.abs() <= box[4] / 2)
    return occupancy


class AgentBoxStore:
    """Read-only detections and tracks of one split, memory-mapped; see ``merge_agent_boxes.py`` for the layout.

    Looks samples up by ``metadata["scenario_id"]`` with a binary search, so workers share one sorted id file instead
    of each building a dictionary over hundreds of thousands of samples. A sample the store doesn't hold gives
    ``boxes=None``.
    """

    def __init__(self, root: str, split: str, min_score: float = 0.5) -> None:
        directory = os.path.join(root, split)
        load = lambda name: np.load(
            os.path.join(directory, f"{name}.npy"), mmap_mode="r"
        )  # noqa: E731
        self.ids = load("scenario_ids")
        self.offsets = load("offsets")
        self.boxes = load("boxes")
        self.scores = load("scores")
        self.class_ids = load("class_ids")
        self.track_ids = load("track_ids")
        self.fut_boxes = load("fut_boxes")
        self.fut_mask = load("fut_mask")
        self.visible = load("visible_bev")
        with open(os.path.join(directory, "meta.json")) as f:
            self.meta = json.load(f)
        self.min_score = min_score

    def __len__(self) -> int:
        return len(self.ids)

    def agent_info(self, scenario_id: str) -> dict:
        i = int(np.searchsorted(self.ids, scenario_id))
        if i >= len(self.ids) or self.ids[i] != scenario_id:
            return {"boxes": None, "source": self.meta["source"]}
        a, b = int(self.offsets[i]), int(self.offsets[i + 1])
        keep = np.flatnonzero(np.asarray(self.scores[a:b]) >= self.min_score) + a
        names = self.meta["class_names"]
        types = [names[int(c)] for c in self.class_ids[keep]]
        return {
            "boxes": torch.from_numpy(np.array(self.boxes[keep], dtype=np.float32)),
            "scores": torch.from_numpy(np.array(self.scores[keep], dtype=np.float32)),
            "track_ids": [f"{scenario_id}:{int(t)}" for t in self.track_ids[keep]],
            "types": types,
            "coarse_types": [coarse_type(t) for t in types],
            "source": self.meta["source"],
            "fut_boxes": torch.from_numpy(
                np.array(self.fut_boxes[keep], dtype=np.float32)
            ),
            "fut_mask": torch.from_numpy(np.array(self.fut_mask[keep], dtype=bool)),
            "fut_interval_s": float(self.meta["fut_interval_s"]),
            "visible_bev": torch.from_numpy(
                np.unpackbits(np.array(self.visible[i]))
                .reshape(self.meta["bev_shape"])
                .astype(bool)
            ),
        }
