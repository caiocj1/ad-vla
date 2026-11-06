"""Offline tracking of per-frame detections into the future tracks of ``AgentInfo``.

Pure numpy, so it can be scored on datasets with ground truth before it is trusted on Waymo. A chain is a list of
frames a fixed interval apart. Boxes are moved into a common frame with the ego poses, associated with a constant
velocity prediction, then each anchor frame gets its agents' future boxes in its own ego frame.
"""

from __future__ import annotations

from dataclasses import dataclass, field

import numpy as np
from scipy.interpolate import CubicSpline
from scipy.optimize import linear_sum_assignment

# Largest distance (m) between a prediction and a detection, per class; a first association has no velocity to
# predict with, so it also allows the fastest plausible motion over the interval.
GATE_M = {0: 4.0, 1: 1.5}
MAX_SPEED_MPS = {0: 35.0, 1: 4.0}
FIRST_STEP_LATERAL_M = 2.0


def wrap(a):
    return (a + np.pi) % (2.0 * np.pi) - np.pi


def align_2d(src: np.ndarray, dst: np.ndarray) -> tuple[float, np.ndarray]:
    """Rotation angle and translation with ``dst = R src + t`` (Kabsch). Rotation 0 if the points barely spread."""
    src_mean, dst_mean = src.mean(0), dst.mean(0)
    a, b = src - src_mean, dst - dst_mean
    if np.sqrt((a**2).sum(1).max(initial=0.0)) < 1.0:
        return 0.0, dst_mean - src_mean
    angle = float(
        np.arctan2((a[:, 0] * b[:, 1] - a[:, 1] * b[:, 0]).sum(), (a * b).sum())
    )
    c, s = np.cos(angle), np.sin(angle)
    return angle, dst_mean - np.array(
        [c * src_mean[0] - s * src_mean[1], s * src_mean[0] + c * src_mean[1]]
    )


def waymo_relative_pose(
    past_a: np.ndarray,
    future_a: np.ndarray | None,
    past_b: np.ndarray,
    offset_steps: float,
) -> tuple[float, np.ndarray]:
    """Pose of frame b's ego in frame a's ego; b is ``offset_steps`` state steps later (0.25 s each).

    Both records hold the logged ego path in their own frame, so the same world points appear in both, and a rigid
    fit gives the transform. ``past_*`` runs oldest first and ends at the frame itself (its last point is the
    origin); ``future_a`` starts one step later. Returns (yaw, xy) with ``p_a = R(yaw) p_b + xy``. No heading field is
    needed, which the Waymo records don't have.
    """
    path_a = past_a if future_a is None else np.vstack([past_a, future_a])
    now_a, now_b = len(past_a) - 1, len(past_b) - 1
    # Point j of b is at state step j - now_b + offset_steps + now_a on a's path, a fraction between states
    # when the frames aren't a whole number of steps apart: a's path is interpolated there.
    steps_a = np.arange(len(past_b)) - now_b + offset_steps + now_a
    keep = (steps_a >= 0) & (steps_a <= len(path_a) - 1)
    grid = np.arange(len(path_a))
    on_a = np.stack(
        [
            CubicSpline(grid, path_a[:, 0])(steps_a[keep]),
            CubicSpline(grid, path_a[:, 1])(steps_a[keep]),
        ],
        -1,
    )
    return align_2d(past_b[keep], on_a)


def dedupe_boxes(
    boxes: np.ndarray, scores: np.ndarray, classes: np.ndarray, radius_m: float = 2.0
) -> np.ndarray:
    """Indices kept after removing lower-scored boxes of the same class within ``radius_m``: neighbouring cameras overlap."""
    keep: list[int] = []
    for i in np.argsort(-scores):
        close = [
            k
            for k in keep
            if classes[k] == classes[i]
            and np.hypot(*(boxes[k, :2] - boxes[i, :2])) < radius_m
        ]
        if not close:
            keep.append(int(i))
    return np.asarray(keep, dtype=np.int64)


@dataclass
class Track:
    frames: list[int] = field(default_factory=list)
    boxes: list[np.ndarray] = field(
        default_factory=list
    )  # [7] in the chain's common frame
    detections: list[int] = field(
        default_factory=list
    )  # index of the detection in its frame
    cls: int = 0


def to_common(boxes: np.ndarray, yaw: float, xy: np.ndarray) -> np.ndarray:
    out = boxes.copy()
    c, s = np.cos(yaw), np.sin(yaw)
    out[:, 0] = c * boxes[:, 0] - s * boxes[:, 1] + xy[0]
    out[:, 1] = s * boxes[:, 0] + c * boxes[:, 1] + xy[1]
    out[:, 6] = wrap(boxes[:, 6] + yaw)
    return out


def from_common(boxes: np.ndarray, yaw: float, xy: np.ndarray) -> np.ndarray:
    out = boxes.copy()
    c, s = np.cos(yaw), np.sin(yaw)
    dx, dy = boxes[..., 0] - xy[0], boxes[..., 1] - xy[1]
    out[..., 0] = c * dx + s * dy
    out[..., 1] = -s * dx + c * dy
    out[..., 6] = wrap(boxes[..., 6] - yaw)
    return out


def track_chain(
    boxes: list[np.ndarray],
    classes: list[np.ndarray],
    poses: list[tuple[float, np.ndarray]],
    interval_s: float,
    max_missed: int = 2,
) -> list[Track]:
    """Frame-by-frame optimal association. ``boxes[f]`` [N, 7] in frame f's ego, ``poses[f]`` frame f in the common frame."""
    tracks: list[Track] = []
    active: list[Track] = []
    for f, (frame_boxes, frame_cls) in enumerate(zip(boxes, classes)):
        common = to_common(frame_boxes, *poses[f])
        matched = set()
        if len(active) and len(common):
            cost = np.full((len(active), len(common)), 1e6)
            for t, track in enumerate(active):
                gap = f - track.frames[-1]
                last = track.boxes[-1][:2]
                if len(track.frames) >= 2:
                    velocity = (last - track.boxes[-2][:2]) / (
                        track.frames[-1] - track.frames[-2]
                    )
                    predicted, gate = (
                        last + velocity * gap,
                        GATE_M.get(track.cls, 4.0) * (1 + 0.5 * (gap - 1)),
                    )
                else:
                    predicted, gate = (
                        last,
                        GATE_M.get(track.cls, 4.0)
                        + MAX_SPEED_MPS.get(track.cls, 35.0) * interval_s * gap,
                    )
                offset = common[:, :2] - predicted
                dist = np.linalg.norm(offset, axis=1)
                ok = (dist < gate) & (frame_cls == track.cls)
                if len(track.frames) < 2:
                    # Without a velocity, a long first step must run along the box, or a track jumps between parked cars.
                    heading = track.boxes[-1][6]
                    lateral = np.abs(
                        -np.sin(heading) * offset[:, 0] + np.cos(heading) * offset[:, 1]
                    )
                    ok &= (dist < GATE_M.get(track.cls, 4.0)) | (
                        lateral < FIRST_STEP_LATERAL_M
                    )
                cost[t, ok] = dist[ok]
            for t, d in zip(*linear_sum_assignment(cost)):
                if cost[t, d] < 1e5:
                    track = active[t]
                    track.frames.append(f)
                    track.boxes.append(common[d])
                    track.detections.append(int(d))
                    matched.add(int(d))
        for d in range(len(common)):
            if d not in matched:
                track = Track(
                    frames=[f], boxes=[common[d]], detections=[d], cls=int(frame_cls[d])
                )
                tracks.append(track)
                active.append(track)
        active = [t for t in active if f - t.frames[-1] <= max_missed]
    return tracks


def smooth_track(track: Track, window: int = 3) -> np.ndarray:
    """Box per observed frame: median-filtered position and yaw, and the track's median size (the detector's size jitters)."""
    boxes = np.stack(track.boxes)
    out = boxes.copy()
    out[:, 3:6] = np.median(boxes[:, 3:6], axis=0)
    n = len(boxes)
    if n >= 3:
        half = window // 2
        for i in range(n):
            lo, hi = max(0, i - half), min(n, i + half + 1)
            out[i, :3] = np.median(boxes[lo:hi, :3], axis=0)
    # A detector's yaw can flip by 180 deg between frames: follow the track's own direction when it moves.
    yaw = np.unwrap(boxes[:, 6] * 2.0) / 2.0
    out[:, 6] = wrap(yaw)
    return out


def anchors_from_tracks(
    tracks: list[Track],
    num_frames: int,
    poses: list[tuple[float, np.ndarray]],
    num_future: int,
    num_detections: list[int],
    min_length: int = 2,
) -> list[dict[str, np.ndarray]]:
    """Per frame: ``fut_boxes`` [N, num_future, 7] and ``fut_mask`` for its N detections, in its own ego frame.

    Interpolates over missed frames. A track of a single detection has unknown future (mask False), and every step
    after a track's last observation is unknown, never free.
    """
    out = [
        {
            "fut_boxes": np.zeros((n, num_future, 7), dtype=np.float32),
            "fut_mask": np.zeros((n, num_future), dtype=bool),
            "track_ids": np.full(n, -1, dtype=np.int32),
        }
        for n in num_detections
    ]
    for track_id, track in enumerate(tracks):
        if len(track.frames) < min_length:
            for f, d in zip(track.frames, track.detections):
                out[f]["track_ids"][d] = track_id
            continue
        smooth = smooth_track(track)
        frames = np.asarray(track.frames)
        full_frames = np.arange(frames[0], frames[-1] + 1)
        full = np.stack(
            [np.interp(full_frames, frames, smooth[:, k]) for k in range(6)], axis=-1
        )
        full_yaw = wrap(np.interp(full_frames, frames, np.unwrap(smooth[:, 6])))
        full = np.concatenate([full, full_yaw[:, None]], axis=-1)
        for f, d in zip(track.frames, track.detections):
            out[f]["track_ids"][d] = track_id
            steps = np.arange(f + 1, min(f + num_future, frames[-1]) + 1)
            if len(steps):
                out[f]["fut_boxes"][d, steps - f - 1] = from_common(
                    full[steps - frames[0]], *poses[f]
                )
                out[f]["fut_mask"][d, steps - f - 1] = True
    return out


# ---------------------------------------------------------------------------------------------------------------------
# Dense tracking: every frame (10 Hz on Waymo), with variable time steps.
# ---------------------------------------------------------------------------------------------------------------------
DENSE_BASE_GATE_M = {0: 2.0, 1: 1.0}
DENSE_GATE_PER_RANGE = {0: 0.04, 1: 0.02}  # detector error grows with range
VELOCITY_WINDOW_S = 1.0
SMOOTH_HALF_WINDOW_S = 0.5
STATIONARY_M = 1.5


def _predict(track: Track, times: np.ndarray, now: float) -> tuple[np.ndarray, bool]:
    """Predicted xy at ``now`` and whether it used a velocity: a line through the last second, which needs some spread."""
    frames = np.asarray(track.frames)
    t = times[frames]
    recent = t >= t[-1] - VELOCITY_WINDOW_S
    if recent.sum() >= 3 and t[recent][-1] - t[recent][0] >= 0.25:
        xy = np.stack([track.boxes[i][:2] for i in np.flatnonzero(recent)])
        origin = t[recent].mean()
        slope = ((t[recent] - origin)[:, None] * (xy - xy.mean(0))).sum(0) / (
            (t[recent] - origin) ** 2
        ).sum()
        return xy.mean(0) + slope * (now - origin), True
    return track.boxes[-1][:2], False


def track_dense(
    boxes: list[np.ndarray],
    classes: list[np.ndarray],
    poses: list[tuple[float, np.ndarray]],
    times: np.ndarray,
    max_gap_s: float = 1.0,
) -> list[Track]:
    """Associates every frame. ``boxes[f]`` [N, 7] in frame f's ego, ``poses[f]`` frame f in the common frame, ``times`` in s."""
    tracks: list[Track] = []
    active: list[Track] = []
    for f, (frame_boxes, frame_cls) in enumerate(zip(boxes, classes)):
        common = to_common(frame_boxes, *poses[f])
        rng = np.linalg.norm(frame_boxes[:, :2], axis=1)
        matched = set()
        if len(active) and len(common):
            cost = np.full((len(active), len(common)), 1e6)
            for t, track in enumerate(active):
                dt = times[f] - times[track.frames[-1]]
                predicted, has_velocity = _predict(track, times, times[f])
                offset = common[:, :2] - predicted
                dist = np.linalg.norm(offset, axis=1)
                base = (
                    DENSE_BASE_GATE_M.get(track.cls, 2.0)
                    + DENSE_GATE_PER_RANGE.get(track.cls, 0.04) * rng
                )
                if has_velocity:
                    ok = dist < base * (1.0 + dt)
                else:
                    heading = track.boxes[-1][6]
                    lateral = np.abs(
                        -np.sin(heading) * offset[:, 0] + np.cos(heading) * offset[:, 1]
                    )
                    fast = dist < base + MAX_SPEED_MPS.get(track.cls, 35.0) * dt
                    ok = (dist < base) | (fast & (lateral < FIRST_STEP_LATERAL_M))
                ok &= frame_cls == track.cls
                cost[t, ok] = dist[ok]
            for t, d in zip(*linear_sum_assignment(cost)):
                if cost[t, d] < 1e5:
                    active[t].frames.append(f)
                    active[t].boxes.append(common[d])
                    active[t].detections.append(int(d))
                    matched.add(int(d))
        for d in range(len(common)):
            if d not in matched:
                track = Track(
                    frames=[f], boxes=[common[d]], detections=[d], cls=int(frame_cls[d])
                )
                tracks.append(track)
                active.append(track)
        active = [t for t in active if times[f] - times[t.frames[-1]] <= max_gap_s]
    return tracks


def smooth_dense(track: Track, times: np.ndarray) -> np.ndarray:
    """Boxes per observation: a local line through positions, held still if the track never really moved.

    Size is the track's median. Yaw follows the motion when moving, else the track's median axis (a detector's yaw can
    flip by 180 deg, which doesn't change a rectangle).
    """
    boxes = np.stack(track.boxes)
    t = times[np.asarray(track.frames)]
    out = boxes.copy()
    out[:, 3:6] = np.median(boxes[:, 3:6], axis=0)
    axis = 0.5 * np.arctan2(
        np.median(np.sin(2 * boxes[:, 6])), np.median(np.cos(2 * boxes[:, 6]))
    )
    out[:, 6] = axis
    if len(t) < 3:
        return out
    centre = t - t.mean()
    slope = (centre[:, None] * (boxes[:, :3] - boxes[:, :3].mean(0))).sum(0) / max(
        (centre**2).sum(), 1e-6
    )
    if np.linalg.norm(slope[:2]) * (t[-1] - t[0]) < STATIONARY_M:
        out[:, :3] = np.median(boxes[:, :3], axis=0)
        return out
    speed = np.zeros(len(t))
    heading = np.zeros(len(t))
    for i, now in enumerate(t):
        near = np.abs(t - now) <= SMOOTH_HALF_WINDOW_S
        tn = t[near] - now
        xyz = boxes[near, :3]
        if near.sum() >= 3 and tn.max() - tn.min() >= 0.2:
            line = np.polyfit(tn, xyz, 1)  # [2, 3]: slope, value at now
            out[i, :3] = line[1]
            speed[i] = np.linalg.norm(line[0][:2])
            heading[i] = np.arctan2(line[0][1], line[0][0])
        else:
            out[i, :3] = xyz.mean(0)
    moving = speed > 2.0
    out[moving, 6] = heading[moving]
    return out


def anchors_dense(
    tracks: list[Track],
    times: np.ndarray,
    poses: list[tuple[float, np.ndarray]],
    num_detections: list[int],
    step_times: list[np.ndarray | None],
    min_observations: int = 3,
) -> list[dict[str, np.ndarray] | None]:
    """Per frame with ``step_times[f]`` (absolute future times, in s): fut_boxes, fut_mask, track_ids for its detections.

    A step after a track's last observation is unknown, never free; a track with too few observations has no future.
    ``poses`` here is the frame the output is expressed in, which may differ from the tracking pose by a sensor offset.
    """
    out: list[dict[str, np.ndarray] | None] = []
    for f, n in enumerate(num_detections):
        if step_times[f] is None:
            out.append(None)
            continue
        k = len(step_times[f])
        out.append(
            {
                "fut_boxes": np.zeros((n, k, 7), dtype=np.float32),
                "fut_mask": np.zeros((n, k), dtype=bool),
                "track_ids": np.full(n, -1, dtype=np.int32),
            }
        )
    for track_id, track in enumerate(tracks):
        wanted = [
            (f, d) for f, d in zip(track.frames, track.detections) if out[f] is not None
        ]
        if not wanted:
            continue
        for f, d in wanted:
            out[f]["track_ids"][d] = track_id
        if len(track.frames) < min_observations:
            continue
        smooth = smooth_dense(track, times)
        t = times[np.asarray(track.frames)]
        yaw = np.unwrap(smooth[:, 6], period=np.pi)
        for f, d in wanted:
            steps = step_times[f]
            known = (steps > times[f]) & (steps <= t[-1] + 1e-6)
            if not known.any():
                continue
            state = np.stack(
                [np.interp(steps[known], t, smooth[:, c]) for c in range(6)], -1
            )
            state = np.concatenate(
                [state, np.interp(steps[known], t, yaw)[:, None]], -1
            ).astype(np.float32)
            out[f]["fut_boxes"][d, known] = from_common(state, *poses[f])
            out[f]["fut_mask"][d, known] = True
    return out
