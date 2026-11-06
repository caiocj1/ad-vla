"""High-level driving intent derived from a path, shared by training labels and deployment.

The same rule labels a training sample from its future trajectory and turns a navigation route
into the matching command at inference (e.g. the AlpaSim driver), so the model sees intents
with the same meaning in both places.

Rule: take the point the path reaches after `max_distance_m` of arc length (its last point when
None or when the path is shorter). It is a turn when its lateral offset exceeds both
`lateral_threshold_m` and `lateral_ratio` times its forward distance; otherwise straight.
With the defaults and no max distance this is the rule nuScenes training labels always used.

For a future trajectory spanning the prediction horizon, the last point is where the car is
after that horizon. For a route, `intent_from_route` stops at the distance the car covers over
the same horizon at its current speed, so both look equally far ahead.

Paths are in the ego frame at the current time: x forward, y left, in metres, not including
the current pose (which is the origin).
"""

from collections.abc import Sequence

import numpy as np
import torch

# Strings used in prompts (NAVSIM_INTENT_DICT values); order matches AlpaSim's DriveCommand.
INTENTS = ("GO_LEFT", "GO_STRAIGHT", "GO_RIGHT", "UNKNOWN")

DEFAULT_LATERAL_THRESHOLD_M = 2.0
DEFAULT_LATERAL_RATIO = 0.25


def _as_xy(path: np.ndarray | torch.Tensor | Sequence) -> np.ndarray:
    """Path as a float64 [N, 2] array (extra columns such as z dropped)."""
    if isinstance(path, torch.Tensor):
        path = path.detach().cpu().numpy()
    xy = np.asarray(path, dtype=np.float64)
    if xy.size == 0:
        return xy.reshape(0, 2)
    if xy.ndim != 2 or xy.shape[1] < 2:
        raise ValueError(f"path must have shape [N, >=2], got {xy.shape}")
    return xy[:, :2]


def point_at_distance(path_xy: np.ndarray, distance_m: float) -> np.ndarray:
    """Point after `distance_m` of arc length from the origin along the path (clamped to its end)."""
    points = np.concatenate([np.zeros((1, 2)), path_xy], axis=0)
    cumulative = np.concatenate(
        [[0.0], np.cumsum(np.linalg.norm(np.diff(points, axis=0), axis=1))]
    )
    if distance_m >= cumulative[-1]:
        return points[-1]
    i = int(np.searchsorted(cumulative, distance_m, side="right"))
    span = cumulative[i] - cumulative[i - 1]
    frac = 0.0 if span <= 0 else (distance_m - cumulative[i - 1]) / span
    return points[i - 1] + frac * (points[i] - points[i - 1])


def intent_from_path(
    path_xy: np.ndarray | torch.Tensor | Sequence,
    max_distance_m: float | None = None,
    lateral_threshold_m: float = DEFAULT_LATERAL_THRESHOLD_M,
    lateral_ratio: float = DEFAULT_LATERAL_RATIO,
) -> str:
    """GO_LEFT / GO_RIGHT / GO_STRAIGHT from where the path ends up; UNKNOWN for an empty path."""
    xy = _as_xy(path_xy)
    if xy.shape[0] == 0:
        return "UNKNOWN"
    x, y = xy[-1] if max_distance_m is None else point_at_distance(xy, max_distance_m)
    if abs(y) > lateral_threshold_m and abs(y) > lateral_ratio * abs(x):
        return "GO_LEFT" if y > 0 else "GO_RIGHT"
    return "GO_STRAIGHT"


def intent_from_route(
    route_xy: np.ndarray | torch.Tensor | Sequence,
    speed_mps: float,
    horizon_s: float = 5.0,
    min_distance_m: float = 10.0,
    lateral_threshold_m: float = DEFAULT_LATERAL_THRESHOLD_M,
    lateral_ratio: float = DEFAULT_LATERAL_RATIO,
) -> str:
    """Intent for a navigation route, matching labels made from `horizon_s` of future trajectory.

    Looks `speed * horizon_s` metres along the route (at least `min_distance_m`, so a car
    stopped before a turn still gets the turn).
    """
    distance = max(float(speed_mps) * horizon_s, min_distance_m)
    return intent_from_path(
        route_xy,
        max_distance_m=distance,
        lateral_threshold_m=lateral_threshold_m,
        lateral_ratio=lateral_ratio,
    )


# --- 7 classes ------------------------------------------------------------------------------
# Same rule as the precomputed label tables: datasets with ``intent_labels_path`` read those tables and use this function only for a
# sample the table lacks; a deployment applies it to the navigation route.

INTENTS_7 = (
    "FOLLOW_LANE",
    "TURN_LEFT",
    "TURN_RIGHT",
    "LANE_CHANGE_LEFT",
    "LANE_CHANGE_RIGHT",
    "U_TURN",
    "UNKNOWN",
)
# The 4-class commands (official NavSim / Waymo labels, AlpaSim's DriveCommand) used when the path
# is too short to tell.
OFFICIAL_TO_INTENT_7 = {
    "GO_LEFT": "TURN_LEFT",
    "GO_RIGHT": "TURN_RIGHT",
    "GO_STRAIGHT": "FOLLOW_LANE",
}
# Look-ahead distance: speed times LOOKAHEAD_S, clipped to the two values below.
LOOKAHEAD_S = 5.0
LOOKAHEAD_MIN_M = 40.0
LOOKAHEAD_MAX_M = 80.0
TURN_DEG = 45.0  # heading at the look-ahead point
TURN_SIDE_RATIO = 0.5  # or the point this far to the side, as a share of its distance
U_TURN_DEG = 135.0
LANE_CHANGE_MIN_M = 2.5  # sideways at the look-ahead point
LANE_CHANGE_MAX_M = 6.0
LANE_CHANGE_BACK_DEG = 15.0  # heading back near the start...
# ...and under this share of a constant arc's: an S, not a bend.
LANE_CHANGE_ARC_SHARE = 0.5
MIN_MOVE_M = 3.0  # a shorter path says nothing
HEADING_CHORD_M = 8  # metres of path averaged for the heading at the look-ahead point


def _resample_1m(xy: np.ndarray, max_s: float = 160.0) -> tuple[np.ndarray, float]:
    """1 m polyline from the origin, jitter under 0.2 m dropped; stops where the car reverses."""
    pts = np.vstack([np.zeros((1, 2)), xy])
    keep = [pts[0]]
    for p in pts[1:]:
        if np.hypot(*(p - keep[-1])) >= 0.2:
            keep.append(p)
    pts = np.array(keep)
    if len(pts) < 2:
        return np.zeros((1, 2)), 0.0
    s = np.concatenate([[0.0], np.cumsum(np.hypot(*np.diff(pts, axis=0).T))])
    grid = np.arange(0.0, min(s[-1], max_s) + 1e-6, 1.0)
    res = np.stack(
        [np.interp(grid, s, pts[:, 0]), np.interp(grid, s, pts[:, 1])], axis=-1
    )
    total = float(s[-1])
    if len(res) >= 2 and res[1, 0] < -0.2:  # reversing now: no intent to read
        return np.zeros((1, 2)), 0.0
    # The path ends where the car reverses direction: a cusp is not a turn.
    if len(res) >= 3:
        d = np.diff(res, axis=0)
        norms = np.maximum(np.hypot(*d[:-1].T) * np.hypot(*d[1:].T), 1e-9)
        cusp = np.where((d[:-1] * d[1:]).sum(1) / norms < np.cos(np.radians(100)))[0]
        if len(cusp):
            res = res[: cusp[0] + 2]
            total = float(len(res) - 1)
    return res, total


def intent7_from_path(
    path_xy: np.ndarray | torch.Tensor | Sequence,
    speed_mps: float,
    official: str | None = None,
) -> str:
    """One of ``INTENTS_7`` for a path in the current ego frame (x forward, y left, metres, origin
    not included): a logged future (as long as available) or a navigation route.

    The path is read at ``clip(speed * 5 s, 40, 80)`` m along it. A turn is a heading of at least
    45 deg there, or the point at least half its distance off to the side; a U-turn 135 deg. A lane
    change is 2.5-6 m sideways with the heading back under 15 deg and under half a constant arc's.
    When the path is too short to reach that point, a manoeuvre it already shows counts; otherwise
    the 4-class ``official`` command decides (left / right are turns, straight is follow), else
    ``FOLLOW_LANE`` when the path is at least 3 m long, else ``UNKNOWN``.
    """
    res, total = _resample_1m(_as_xy(path_xy))
    look = min(max(float(speed_mps) * LOOKAHEAD_S, LOOKAHEAD_MIN_M), LOOKAHEAD_MAX_M)
    i = int(min(round(look), len(res) - 1))
    if len(res) < 2 or i < 1:
        x = y = psi = 0.0
        short = True
    else:
        d = np.diff(res[: i + 1], axis=0)
        heading = np.unwrap(np.arctan2(d[:, 1], d[:, 0]))
        x, y = float(res[i, 0]), float(res[i, 1])
        psi = float(np.degrees(heading[-HEADING_CHORD_M:].mean()))
        short = total + 0.5 < look

    a = abs(psi)
    arc = np.degrees(2 * np.arctan2(abs(y), max(abs(x), 1e-3)))
    intent = "FOLLOW_LANE"
    if (
        LANE_CHANGE_MIN_M <= abs(y) <= LANE_CHANGE_MAX_M
        and a <= LANE_CHANGE_BACK_DEG
        and a <= LANE_CHANGE_ARC_SHARE * arc
    ):
        intent = "LANE_CHANGE_LEFT" if y > 0 else "LANE_CHANGE_RIGHT"
    side = abs(y) / max(np.hypot(x, y), 1e-3) >= TURN_SIDE_RATIO
    if a >= TURN_DEG:
        intent = "TURN_LEFT" if psi > 0 else "TURN_RIGHT"
    elif side:
        intent = "TURN_LEFT" if y > 0 else "TURN_RIGHT"
    if a >= U_TURN_DEG:
        intent = "U_TURN"
    if (short and intent == "FOLLOW_LANE") or total < MIN_MOVE_M:
        intent = "UNKNOWN"

    if intent != "UNKNOWN":
        return intent
    if official in OFFICIAL_TO_INTENT_7:
        return OFFICIAL_TO_INTENT_7[official]
    return "FOLLOW_LANE" if total >= MIN_MOVE_M else "UNKNOWN"
