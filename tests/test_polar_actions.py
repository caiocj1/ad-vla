import math

import pytest
import torch

from ad_vla.utils.action_utils import (
    POLAR_MIN_SPEED,
    decode_actions,
    encode_actions,
    polar_to_traj,
    traj_to_polar,
)

DT = 0.25


def _arc(num_poses: int, speed: float, curvature: float) -> torch.Tensor:
    """Constant-speed, constant-curvature path from the origin, heading +x."""
    s = speed * DT * torch.arange(1, num_poses + 1, dtype=torch.float64)
    if curvature == 0:
        return torch.stack([s, torch.zeros_like(s)], dim=-1)
    theta = curvature * s
    return torch.stack([theta.sin() / curvature, (1 - theta.cos()) / curvature], dim=-1)


def test_straight_line():
    traj = _arc(20, speed=10.0, curvature=0.0)
    actions = traj_to_polar(traj, dt=DT)
    torch.testing.assert_close(actions[:, 0], torch.zeros(20, dtype=torch.float64))
    torch.testing.assert_close(
        actions[:, 1], torch.full((20,), 2.5, dtype=torch.float64)
    )
    torch.testing.assert_close(polar_to_traj(actions), traj)


@pytest.mark.parametrize("curvature", [0.05, -0.1])
def test_arc_round_trip_and_constant_turn(curvature: float):
    traj = _arc(20, speed=8.0, curvature=curvature)
    actions = traj_to_polar(traj, dt=DT)
    # Chord directions on a circle: the first step turns by half the per-step angle.
    step_angle = curvature * 8.0 * DT
    torch.testing.assert_close(
        actions[0, 0], torch.tensor(step_angle / 2, dtype=torch.float64)
    )
    torch.testing.assert_close(
        actions[1:, 0], torch.full((19,), step_angle, dtype=torch.float64)
    )
    torch.testing.assert_close(polar_to_traj(actions), traj)


def test_stationary_keeps_ego_heading():
    traj = torch.zeros(20, 2, dtype=torch.float64)
    traj += 1e-3 * torch.randn_like(traj)  # sensor noise below the speed threshold
    actions = traj_to_polar(traj, dt=DT)
    assert (actions[:, 0] == 0).all()
    assert (actions[:, 1].abs() < POLAR_MIN_SPEED * DT).all()


def test_stop_then_go_holds_heading_while_stopped():
    moving = _arc(8, speed=6.0, curvature=0.05)
    stopped = moving[-1:].expand(6, 2) + torch.tensor(
        [[1e-3, -1e-3]], dtype=torch.float64
    )
    resumed = stopped[-1] + torch.tensor([[0.0, 1.5], [0.0, 3.0]], dtype=torch.float64)
    traj = torch.cat([moving, stopped, resumed])
    actions = traj_to_polar(traj, dt=DT)

    assert (actions[8:14, 0] == 0).all()
    # Resuming to +y: one turn relative to the heading held from before the stop.
    held = actions[:8, 0].sum()
    torch.testing.assert_close(actions[14, 0], math.pi / 2 - held)
    # Only the slow steps are misplaced, by at most their own length each.
    assert (polar_to_traj(actions) - traj).norm(
        dim=-1
    ).max() < 6 * 2 * POLAR_MIN_SPEED * DT


def test_reversing_is_negative_distance_not_a_half_turn():
    traj = torch.tensor([[-1.0, 0.0], [-2.0, -0.1]], dtype=torch.float64)
    actions = traj_to_polar(traj, dt=DT)
    assert actions[0, 0] == 0 and actions[0, 1] == pytest.approx(-1.0)
    assert actions[1, 0].abs() < 0.2 and actions[1, 1] < 0
    torch.testing.assert_close(polar_to_traj(actions), traj)


def test_backward_jitter_while_creeping_stays_small():
    traj = _arc(20, speed=1.0, curvature=0.0)  # 0.25 m per step
    traj[10] = traj[9] - torch.tensor([0.06, 0.0], dtype=torch.float64)  # one step back
    actions = traj_to_polar(traj, dt=DT)
    assert actions[:, 0].abs().max() < 1e-9
    torch.testing.assert_close(polar_to_traj(actions), traj)


def test_batched_and_low_precision():
    traj = torch.stack([_arc(20, 5.0, 0.02), _arc(20, 12.0, -0.03)]).float()
    actions = traj_to_polar(traj.bfloat16(), dt=DT)
    assert actions.shape == (2, 20, 2) and actions.dtype == torch.bfloat16
    torch.testing.assert_close(
        polar_to_traj(traj_to_polar(traj, dt=DT)), traj, atol=1e-4, rtol=1e-4
    )


@pytest.mark.parametrize("representation", ["trajs", "polar"])
def test_encode_decode_dispatch(representation: str):
    traj = _arc(20, 8.0, 0.04)
    actions = encode_actions(traj, representation, dt=DT)
    torch.testing.assert_close(decode_actions(actions, representation, dt=DT), traj)


def test_acc_curv_needs_initial_velocity():
    with pytest.raises(ValueError, match="initial_velocity"):
        encode_actions(_arc(20, 8.0, 0.0), "acc_curv", dt=DT)
