"""Shared intent rule (ad_vla/utils/intent_utils.py): training labels and route commands."""

import numpy as np
import pytest
import torch

from ad_vla.utils.intent_utils import (
    INTENTS,
    intent_from_path,
    intent_from_route,
    point_at_distance,
)


def _old_nuscenes_rule(future_traj: np.ndarray) -> str:
    """nuScenes' _infer_intent before it delegated to intent_from_path."""
    if future_traj.shape[0] == 0:
        return "UNKNOWN"
    final_x, final_y = future_traj[-1]
    if abs(float(final_y)) > 2.0 and abs(float(final_y)) > 0.25 * abs(float(final_x)):
        return "GO_LEFT" if final_y > 0 else "GO_RIGHT"
    return "GO_STRAIGHT"


def test_matches_previous_nuscenes_labels():
    rng = np.random.default_rng(0)
    for _ in range(2000):
        path = np.cumsum(rng.normal([2.0, 0.0], [2.0, 1.5], size=(10, 2)), axis=0)
        assert intent_from_path(path) == _old_nuscenes_rule(path)


def test_basic_cases_and_inputs():
    assert intent_from_path(np.zeros((0, 2))) == "UNKNOWN"
    assert intent_from_path([[10.0, 0.5], [20.0, 1.0]]) == "GO_STRAIGHT"
    assert intent_from_path([[5.0, 2.0], [8.0, 6.0]]) == "GO_LEFT"
    assert intent_from_path(torch.tensor([[5.0, -2.0], [8.0, -6.0]])) == "GO_RIGHT"
    # Far ahead with a small lateral share is still straight (lane-level drift).
    assert intent_from_path([[40.0, 3.0]]) == "GO_STRAIGHT"
    # Extra columns (e.g. heading) are ignored.
    assert intent_from_path([[5.0, 6.0, 1.2]]) == "GO_LEFT"
    assert set(INTENTS) == {"GO_LEFT", "GO_STRAIGHT", "GO_RIGHT", "UNKNOWN"}


def test_point_at_distance_interpolates_from_origin():
    path = np.array([[10.0, 0.0], [10.0, 10.0]])
    np.testing.assert_allclose(point_at_distance(path, 5.0), [5.0, 0.0])
    np.testing.assert_allclose(point_at_distance(path, 15.0), [10.0, 5.0])
    np.testing.assert_allclose(point_at_distance(path, 100.0), [10.0, 10.0])


def test_route_looks_as_far_as_the_horizon():
    # Straight for 30 m, then a left turn: AlpaSim-like route, waypoints ~4 m apart.
    straight = np.stack([np.arange(4.0, 31.0, 4.0), np.zeros(7)], axis=1)
    turn = np.stack([np.full(10, 30.0), np.arange(4.0, 41.0, 4.0)], axis=1)
    route = np.concatenate([straight, turn])
    # 4 m/s x 5 s = 20 m: still on the straight part.
    assert intent_from_route(route, speed_mps=4.0) == "GO_STRAIGHT"
    # 10 m/s x 5 s = 50 m: 20 m into the turn.
    assert intent_from_route(route, speed_mps=10.0) == "GO_LEFT"
    # Stopped: the minimum look-ahead (10 m) still applies, here straight.
    assert intent_from_route(route, speed_mps=0.0) == "GO_STRAIGHT"
    with pytest.raises(ValueError):
        intent_from_path(np.zeros((3,)))


# --- 7 classes ------------------------------------------------------------------------------

from ad_vla.utils.intent_utils import INTENTS_7, intent7_from_path  # noqa: E402


def _arc(radius: float, length: float, left: bool = True) -> np.ndarray:
    """Points every 0.5 m along a constant turn starting at the origin, heading +x."""
    t = np.arange(0.5, length + 1e-6, 0.5) / radius
    xy = np.stack([radius * np.sin(t), radius * (1 - np.cos(t))], axis=-1)
    return xy if left else xy * [1, -1]


def _lane_change(offset: float, length: float = 80.0) -> np.ndarray:
    """Straight with a smooth sideways move of `offset` metres between x = 5 and 30."""
    x = np.arange(0.5, length + 1e-6, 0.5)
    u = np.clip((x - 5) / 25, 0, 1)
    return np.stack([x, offset * (3 * u**2 - 2 * u**3)], axis=-1)


def test_intent7_classes():
    straight = np.stack([np.arange(1.0, 81.0), np.zeros(80)], axis=-1)
    assert intent7_from_path(straight, speed_mps=10.0) == "FOLLOW_LANE"
    assert intent7_from_path(_arc(20, 60), speed_mps=5.0) == "TURN_LEFT"
    assert intent7_from_path(_arc(20, 60, left=False), speed_mps=5.0) == "TURN_RIGHT"
    assert intent7_from_path(_arc(10, 60), speed_mps=5.0) == "U_TURN"
    assert intent7_from_path(_lane_change(3.5), speed_mps=8.0) == "LANE_CHANGE_LEFT"
    assert intent7_from_path(_lane_change(-3.5), speed_mps=8.0) == "LANE_CHANGE_RIGHT"
    assert set(INTENTS_7) == {
        "FOLLOW_LANE",
        "TURN_LEFT",
        "TURN_RIGHT",
        "LANE_CHANGE_LEFT",
        "LANE_CHANGE_RIGHT",
        "U_TURN",
        "UNKNOWN",
    }


def test_intent7_bend_is_not_a_lane_change():
    # A constant curve 3.5 m to the side at 40 m ends with its heading still turned: follow, not
    # a lane change (the S test), and too gentle for a turn.
    radius = 40.0**2 / (2 * 3.5)
    assert intent7_from_path(_arc(radius, 80), speed_mps=8.0) == "FOLLOW_LANE"
    # Too far to the side for a lane change, still straight at the end: no lane change either.
    assert intent7_from_path(_lane_change(8.0), speed_mps=8.0) == "FOLLOW_LANE"


def test_intent7_short_paths():
    # 20 m, look-ahead 50 m
    short = np.stack([np.arange(1.0, 21.0), np.zeros(20)], axis=-1)
    assert intent7_from_path(short, speed_mps=10.0) == "FOLLOW_LANE"
    assert intent7_from_path(short, speed_mps=10.0, official="GO_RIGHT") == "TURN_RIGHT"
    assert intent7_from_path(short, speed_mps=10.0, official="UNKNOWN") == "FOLLOW_LANE"
    # A turn the short path already shows wins over the official command.
    assert (
        intent7_from_path(_arc(10, 20), speed_mps=10.0, official="GO_STRAIGHT")
        == "TURN_LEFT"
    )
    stopped = np.array([[0.5, 0.0], [1.0, 0.0]])
    assert intent7_from_path(stopped, speed_mps=0.0) == "UNKNOWN"
    assert intent7_from_path(stopped, speed_mps=0.0, official="GO_LEFT") == "TURN_LEFT"
    assert intent7_from_path(np.zeros((0, 2)), speed_mps=0.0) == "UNKNOWN"
    reversing = np.stack([-np.arange(1.0, 11.0), np.zeros(10)], axis=-1)
    assert intent7_from_path(reversing, speed_mps=2.0) == "UNKNOWN"


def test_intent7_accepts_tensors_and_extra_columns():
    path = np.concatenate([_lane_change(3.5), np.zeros((160, 1))], axis=-1)
    assert (
        intent7_from_path(torch.from_numpy(path).float(), speed_mps=8.0)
        == "LANE_CHANGE_LEFT"
    )
