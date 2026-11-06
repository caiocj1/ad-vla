"""Leaving out most scenes of a car standing still, without touching the ones about to move."""

import numpy as np
import pytest

from ad_vla.dataset.navsim.navsim_dataset import (
    is_still_scene,
    still_scenes_to_drop,
)

CURRENT = 1


def pose(x):
    matrix = np.eye(4)
    matrix[0, 3] = x
    return matrix


def scene(xs, speed=0.0, log="a", time=0):
    """Frames at positions ``xs``: frame CURRENT is now, the later ones are its future."""
    return [
        {
            "ego2global": pose(x),
            "ego_dynamic_state": [speed, 0.0, 0.0, 0.0],
            "log_name": log,
            "timestamp": time + i,
        }
        for i, x in enumerate(xs)
    ]


@pytest.mark.parametrize(
    "xs, speed, still",
    [
        ([0, 0, 0, 0], 0.0, True),
        ([0, 0, 0.5, 0.9], 0.2, True),
        ([0, 0, 0, 1.5], 0.0, False),  # about to pull away
        ([0, 0, 0, 0], 0.6, False),  # moving now
    ],
)
def test_still_means_stopped_now_and_for_the_whole_future(xs, speed, still):
    assert is_still_scene(scene(xs, speed), CURRENT) is still


def test_a_stop_keeps_its_first_scene_and_every_nth_after_it():
    scenes = {f"t{i:02d}": scene([0, 0, 0], time=i * 100) for i in range(25)}
    dropped = still_scenes_to_drop(scenes, CURRENT, 10)
    kept = sorted(set(scenes) - dropped)
    assert kept == ["t00", "t10", "t20"]


def test_moving_scenes_and_those_about_to_move_all_stay():
    scenes = {
        "go": scene([0, 0, 0, 3], time=0),
        "move": scene([0, 5, 10], 9.0, time=100),
    }
    assert not still_scenes_to_drop(scenes, CURRENT, 10)


def test_each_stop_and_each_log_starts_counting_again():
    scenes = {
        "a0": scene([0, 0, 0], time=0),
        "a1": scene([0, 0, 0], time=100),
        "a2": scene([0, 5, 10], 9.0, time=200),
        "a3": scene([0, 0, 0], time=300),
        "b0": scene([0, 0, 0], log="b", time=0),
        "b1": scene([0, 0, 0], log="b", time=100),
    }
    assert still_scenes_to_drop(scenes, CURRENT, 10) == {"a1", "b1"}


def test_keep_every_one_keeps_everything_and_zero_is_refused():
    scenes = {f"t{i}": scene([0, 0, 0], time=i * 100) for i in range(5)}
    assert not still_scenes_to_drop(scenes, CURRENT, 1)
    with pytest.raises(ValueError):
        still_scenes_to_drop(scenes, CURRENT, 0)
