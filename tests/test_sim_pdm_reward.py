"""SimPDMReward on synthetic reward contexts (CPU)."""

import math
from types import SimpleNamespace

import numpy as np
import pytest
import torch

from ad_vla.rewards.sim_pdm_reward import (
    SimPDMReward,
    boxes_overlap,
    poly_smooth,
    project_on_path,
)

T, DT = 20, 0.25
SPEED = 8.0  # m/s


def _ctx(agents=None, half_road_m=5.0):
    """Straight road along +x (|y| < half_road_m), recorded path on the centre line."""
    res = 0.25
    xs = np.arange(-16.0, 80.0, res) + res / 2
    ys = np.arange(-32.0, 32.0, res) + res / 2
    drivable = np.abs(ys)[None, :].repeat(len(xs), 0) < half_road_m
    corners = np.zeros((24, 0, 4, 2), dtype=np.float32)
    valid = np.zeros((24, 0), dtype=bool)
    if agents is not None:
        corners, valid = agents
    past = np.stack([np.arange(-7, 1) * SPEED * DT, np.zeros(8)], axis=1).astype(
        np.float32
    )
    return {
        "path_xy": np.stack([np.arange(-20.0, 120.0), np.zeros(140)], axis=1).astype(
            np.float32
        ),
        "drivable": drivable,
        "drivable_origin": np.array([-16.0, -32.0], dtype=np.float32),
        "drivable_res": np.float32(res),
        "agent_corners": corners,
        "agent_valid": valid,
        "agent_times_s": (np.arange(1, 25) * 0.25).astype(np.float32),
        "ego_length": np.float32(5.0),
        "ego_width": np.float32(2.0),
        "rig_to_center_xy": np.array([1.5, 0.0], dtype=np.float32),
        "velocity": np.array([SPEED, 0.0], dtype=np.float32),
        "past_traj_xy": past,
        "past_interval_s": DT,
    }


def _straight(speed=SPEED, y=0.0):
    t = torch.arange(1, T + 1, dtype=torch.float32) * DT
    return torch.stack([speed * t, torch.full_like(t, y)], dim=-1)


def _drift(final_y):
    plan = _straight()
    plan[:, 1] = final_y * (torch.arange(1, T + 1) / T) ** 2
    return plan


def _static_agent(x, y, length=4.5, width=2.0):
    c = np.array(
        [
            [x + length / 2, y + width / 2],
            [x + length / 2, y - width / 2],
            [x - length / 2, y - width / 2],
            [x - length / 2, y + width / 2],
        ]
    )
    corners = np.repeat(c[None, None], 24, axis=0).astype(np.float32)
    return corners, np.ones((24, 1), dtype=bool)


def _score(plans, ctx, **kwargs):
    reward = SimPDMReward(**kwargs)
    batch = SimpleNamespace(samples=[SimpleNamespace(metadata={"rl_ctx": ctx})])
    out = reward(torch.stack(plans)[None], batch)[0]
    return out, reward.last_terms


def test_straight_lane_keeping_plan_scores_one():
    r, terms = _score([_straight()], _ctx())
    assert r.item() == pytest.approx(1.0, abs=1e-4)
    assert terms["dac"] == 1.0 and terms["nc"] == 1.0 and terms["comfort"] == 1.0


def test_drift_lowers_path_keeping_and_offroad_lowers_dac():
    r, _ = _score([_straight(), _drift(2.0), _drift(8.0)], _ctx())
    assert r[0] > r[1] > r[2]
    _, terms = _score([_drift(8.0)], _ctx())
    assert terms["dac"] < 0.8


def test_binary_dac_zeroes_any_offroad_plan():
    r, terms = _score([_drift(8.0)], _ctx(), dac_mode="binary")
    assert r.item() == 0.0 and terms["dac"] == 0.0 and 0.0 < terms["dac_frac"] < 0.8
    r, terms = _score([_straight()], _ctx(), dac_mode="binary")
    assert (
        r.item() == pytest.approx(1.0, abs=1e-4)
        and terms["dac"] == terms["dac_frac"] == 1.0
    )


def _edge_ctx(edge_y):
    """PhysicalAI-style map: lanes |y| < 2 m (no road areas), a road edge along y = edge_y."""
    ctx = _ctx(half_road_m=2.0)
    ys = np.arange(-32.0, 32.0, 0.25) + 0.125
    ctx["road_edge"] = (
        np.abs(ys - edge_y)[None, :].repeat(ctx["drivable"].shape[0], 0) < 0.2
    )
    return ctx


def test_road_edge_dac_fails_only_where_the_footprint_touches_an_edge():
    # Leaving the lanes onto the shoulder (edge at y = 6 m) is not off-road; crossing the edge is.
    r, terms = _score([_drift(3.0)], _edge_ctx(6.0), dac_mode="binary")
    assert terms["dac"] == 1.0 and r.item() > 0.5
    r, terms = _score([_drift(8.0)], _edge_ctx(6.0), dac_mode="binary")
    assert terms["dac"] == 0.0 and r.item() == 0.0
    # Outside the lanes, an edge crossing a long side of the box between a corner and the side
    # midpoint (box x in [-1, 4] m, |y| <= 1 m; edge stub at x = 2.6 m) is seen.
    ctx = _edge_ctx(20.0)
    ctx["drivable"][:] = False
    ctx["road_edge"][:] = False
    xs = np.arange(-16.0, 80.0, 0.25) + 0.125
    ys = np.arange(-32.0, 32.0, 0.25) + 0.125
    creep = torch.zeros(T, 2)
    creep[:, 0] = 0.01 * torch.arange(1, T + 1)
    _, terms = _score([creep], ctx, dac_mode="binary")
    assert terms["dac"] == 1.0
    ctx["road_edge"][np.ix_(np.abs(xs - 2.6) < 0.2, (ys > 0.5) & (ys < 1.5))] = True
    _, terms = _score([creep], ctx, dac_mode="binary")
    assert terms["dac"] == 0.0


def _shifted(ctx, shift_y):
    """Move the map (not the plan) by shift_y: the plans start at the ego, so offset plans would cut across."""
    ctx["drivable_origin"] = ctx["drivable_origin"] + np.array(
        [0.0, shift_y], dtype=np.float32
    )
    return ctx


def test_drivable_margin_grows_with_the_distance_to_the_road_edge():
    # Road |y - c| < 5 m with c = -3.6 / -3.9 m, box half width 1 m: the plan at y = 0 keeps ~0.4 / ~0.1 m to the edge.
    margins = []
    for c in (0.0, -3.6, -3.9):
        ctx = _shifted(_ctx(), c)
        _, terms = _score(
            [_straight()],
            ctx,
            dac_mode="binary",
            smooth="none",
            w_dm=5.0,
            dm_full_m=0.5,
        )
        assert terms["dac"] == 1.0
        margins.append(terms["margin_m"])
    assert "_dm_edt" in ctx  # distance transform cached in the context
    assert margins[0] == pytest.approx(2.0)  # capped in the logged term
    assert margins[1] == pytest.approx(0.4, abs=0.15) and margins[2] == pytest.approx(
        0.1, abs=0.15
    )
    # Human path at y = 0, 0.4 m from the edge: PK alone prefers it; with DM a plan 0.3 m further in wins, and one
    # 0.2 m closer to the edge loses more.
    plans = [_straight(y=0.0), _straight(y=-0.3), _straight(y=0.2)]
    kw = dict(dac_mode="binary", smooth="none", w_c=0.0)
    r_no, t_no = _score(plans, _shifted(_ctx(), -3.6), **kw)
    r_dm, t_dm = _score(plans, _shifted(_ctx(), -3.6), w_dm=5.0, **kw)
    assert r_no[0] > r_no[1] and r_no[0] > r_no[2]
    assert r_dm[1] > r_dm[0] > r_dm[2]
    assert "dm" in t_dm and "dm" not in t_no and "margin_m" not in t_no


def test_drivable_margin_is_zero_off_road_and_ignores_the_first_second():
    r, terms = _score([_drift(8.0)], _ctx(), dac_mode="binary", w_dm=5.0)
    assert r.item() == 0.0 and terms["dm"] == 0.0
    # Road centre at y = -3.8 m (edge at y = 1.2 m): the plan stays 0.2 m from the edge for 0.5 s, then moves to the
    # centre by 1 s. The margin is measured from 1 s on only.
    ctx = _shifted(_ctx(), -3.8)
    plan = _straight()
    plan[2, 1], plan[3:, 1] = -1.9, -3.8
    kw = dict(dac_mode="binary", smooth="none", w_dm=5.0)
    _, terms = _score([plan], ctx, dm_start_s=1.0, **kw)
    assert terms["dm"] == 1.0
    _, terms = _score([plan], ctx, dm_start_s=0.25, **kw)
    assert terms["dm"] == pytest.approx(0.4, abs=0.3) and terms["dm"] < 1.0


def test_drivable_margin_uses_the_road_edge_on_edge_maps():
    # PhysicalAI-style map: leaving the lanes (|y| < 2 m) onto the shoulder keeps the margin to the edge (y = 6 m).
    _, terms = _score([_drift(3.0)], _edge_ctx(6.0), dac_mode="binary", w_dm=5.0)
    assert terms["dac"] == 1.0 and terms["margin_m"] > 1.0 and terms["dm"] == 1.0
    # Map moved by -4.5 m (lanes y in (-6.5, -2.5), edge at y = 1.5 m): the plan at y = 0 is off the lanes, not on
    # the edge (DAC holds), ~0.1-0.25 m from it.
    _, terms = _score(
        [_straight()],
        _shifted(_edge_ctx(6.0), -4.5),
        dac_mode="binary",
        smooth="none",
        w_dm=5.0,
    )
    assert terms["dac"] == 1.0 and 0.0 < terms["dm"] < 0.6


def test_drivable_margin_rejects_tiered_mode_and_bad_values():
    with pytest.raises(ValueError):
        SimPDMReward(mode="tiered", w_dm=1.0)
    with pytest.raises(ValueError):
        SimPDMReward(w_dm=-1.0)
    with pytest.raises(ValueError):
        SimPDMReward(w_dm=1.0, dm_full_m=0.0)


def test_corridor_zeroes_plans_far_from_the_human_path():
    r, terms = _score([_drift(3.0)], _ctx(), corridor_m=4.0)
    assert terms["corridor"] == 1.0 and r.item() > 0.5
    r, terms = _score([_drift(4.5)], _ctx(), corridor_m=4.0)
    assert terms["corridor"] == 0.0 and r.item() == 0.0
    _, terms = _score([_drift(4.5)], _ctx())
    assert terms["corridor"] == 1.0
    # A fast plan running past the end of the stored path (here 20 m ahead) is not judged there.
    ctx = _ctx()
    ctx["path_xy"] = ctx["path_xy"][:41]  # x in [-20, 20] m
    r, terms = _score([_straight(speed=20.0)], ctx, corridor_m=4.0)
    assert terms["corridor"] == 1.0 and terms["pk"] == pytest.approx(1.0)


def test_collision_ahead_and_not_at_fault_cases():
    agent = _static_agent(25.0, 0.0)
    r, terms = _score([_straight()], _ctx(agent))
    assert terms["collision_rate"] == 1.0 and r.item() == 0.0  # binary NC, as in PDM
    # Stopping before the agent: no collision, less progress.
    stop = _straight()
    stop[:, 0] = stop[:, 0].clamp(max=15.0)
    r_stop, terms = _score([stop], _ctx(agent), w_c=0.0)
    assert terms["collision_rate"] == 0.0
    # An actor overlapping the ego from behind is not the ego's fault.
    behind = _static_agent(-2.0, 0.0)
    _, terms = _score([_straight()], _ctx(behind))
    assert terms["collision_rate"] == 0.0


def _moving_agent(x0, y, speed, length=4.5, width=2.0):
    corners = []
    for k in range(1, 25):
        corners.append(_static_agent(x0 + speed * k * 0.25, y, length, width)[0][0, 0])
    return np.stack(corners)[:, None].astype(np.float32), np.ones((24, 1), dtype=bool)


def test_late_collision_loses_to_slower_clean_plan():
    # An earlier failure: a fast plan that only touches the lead car at the end must not tie
    # with a slower clean one (graded NC gave 0.9 x its reward).
    agent = _static_agent(
        41.0, 0.0
    )  # reached by the 8 m/s plan (front at ~42 m) in the last step
    fast, slow = _straight(8.0), _straight(6.4)
    r, terms = _score([fast, slow], _ctx(agent), w_c=0.0)
    assert r[0].item() == 0.0 and r[1].item() > 0.5


def test_ttc_penalizes_closing_in_without_contact():
    # Lead car 2 m/s slower, starting 17 m ahead: no contact within 5 s, but less than 1 s of
    # headway at the end; a lead car at the same speed keeps the gap.
    _, slow_lead = _score([_straight()], _ctx(_moving_agent(17.0, 0.0, SPEED - 2.0)))
    assert slow_lead["collision_rate"] == 0.0 and slow_lead["ttc"] == 0.0
    _, same_speed = _score([_straight()], _ctx(_moving_agent(17.0, 0.0, SPEED)))
    assert same_speed["ttc"] == 1.0
    # Adjacent lane: no TTC violation.
    _, adjacent = _score([_straight()], _ctx(_moving_agent(17.0, 3.7, SPEED - 2.0)))
    assert adjacent["ttc"] == 1.0


def test_progress_is_relative_to_the_best_candidate():
    _, terms_fast = _score([_straight(8.0)], _ctx())
    _, terms_mix = _score([_straight(8.0), _straight(4.0)], _ctx())
    assert terms_fast["ep"] == pytest.approx(1.0)
    assert terms_mix["ep"] == pytest.approx(0.75, abs=0.01)


def test_progress_is_capped_at_the_human_pace():
    ctx = _ctx()
    ctx["human_progress_m"] = 30.0  # 6 m/s over 5 s
    r, _ = _score([_straight(6.0), _straight(8.0)], ctx, w_c=0.0)
    assert r[0].item() == pytest.approx(
        r[1].item(), abs=1e-4
    )  # faster than the human earns nothing
    _, terms = _score([_straight(8.0)], ctx)
    assert terms["ep"] == pytest.approx(1.0) and terms["progress_rel"] == pytest.approx(
        40.0 / 30.0, abs=0.01
    )
    _, terms = _score([_straight(3.0)], ctx)
    assert terms["ep"] == pytest.approx(0.5, abs=0.01)
    # Group reference on request (and for states without human progress).
    _, terms = _score([_straight(8.0), _straight(4.0)], ctx, ep_reference="group")
    assert terms["ep"] == pytest.approx(0.75, abs=0.01)


def test_progress_beyond_the_human_pace_is_penalized():
    ctx = _ctx()
    ctx["human_progress_m"] = (
        30.0  # free up to max(1.1 x 30, 30 + 2) = 33 m, EP 0 at 33 + 0.4 x 30 = 45 m
    )
    kwargs = dict(ep_over_ratio=1.1)
    for speed, ep in ((6.0, 1.0), (6.6, 1.0), (8.0, 1.0 - 7.0 / 12.0), (10.0, 0.0)):
        _, terms = _score([_straight(speed)], ctx, **kwargs)
        assert terms["ep"] == pytest.approx(ep, abs=0.01), speed
    # Slow human (starting from rest): the 2 m margin applies, not 1.1 x.
    ctx["human_progress_m"] = 10.0
    _, terms = _score([_straight(2.4)], ctx, **kwargs)
    assert terms["ep"] == pytest.approx(1.0, abs=0.01)
    # Off by default, and never with the group reference.
    _, terms = _score([_straight(10.0)], ctx)
    assert terms["ep"] == pytest.approx(1.0)
    _, terms = _score(
        [_straight(10.0), _straight(8.0)], ctx, ep_reference="group", **kwargs
    )
    assert terms["ep"] == pytest.approx(0.9, abs=0.01)


def test_harsh_braking_fails_comfort():
    brake = _straight(0.5)  # 8 m/s -> 0.5 m/s within the first step
    _, terms = _score([brake], _ctx(), smooth="none")
    assert terms["comfort"] == 0.0


def _turn(speed, yaw_rate, straight_steps=4):
    """Unicycle plan at constant speed: straight, then turning at ``yaw_rate`` from step ``straight_steps``."""
    pos, yaw, out = torch.zeros(2), 0.0, []
    for k in range(T):
        if k >= straight_steps:
            yaw += yaw_rate * DT
        pos = pos + speed * DT * torch.tensor([math.cos(yaw), math.sin(yaw)])
        out.append(pos)
    return torch.stack(out)


def _past_at(ctx, speed):
    ctx["past_traj_xy"] = np.stack(
        [np.arange(-7, 1) * speed * DT, np.zeros(8)], axis=1
    ).astype(np.float32)
    ctx["velocity"] = np.array([speed, 0.0], dtype=np.float32)
    return ctx


def test_comfort_checks_yaw_acceleration():
    # 2 m/s, yaw rate 0 -> 0.8 rad/s within one step: yaw rate, lateral acceleration (1.6) and jerk magnitude
    # (~6.4) within PDM's limits, yaw acceleration 3.2 > 1.93 rad/s^2.
    plan = _turn(2.0, 0.8)
    _, terms = _score([plan], _past_at(_ctx(), 2.0), smooth="none")
    assert terms["comfort"] == 0.0
    _, terms = _score(
        [plan],
        _past_at(_ctx(), 2.0),
        smooth="none",
        comfort_limits={"max_abs_yaw_accel": math.inf},
    )
    assert terms["comfort"] == 1.0


def test_comfort_checks_jerk_magnitude():
    # 20 m/s, yaw rate 0 -> 0.2 rad/s: lateral acceleration 0 -> 4 m/s^2 in one step (lateral jerk ~16 > 8.37);
    # longitudinal jerk, yaw rate (0.2) and yaw acceleration (0.8) within the limits.
    plan = _turn(20.0, 0.2)
    _, terms = _score([plan], _past_at(_ctx(), 20.0), smooth="none")
    assert terms["comfort"] == 0.0
    _, terms = _score(
        [plan],
        _past_at(_ctx(), 20.0),
        smooth="none",
        comfort_limits={"max_abs_mag_jerk": math.inf},
    )
    assert terms["comfort"] == 1.0


def test_comfort_limits_are_pdms_six():
    from ad_vla.rewards.sim_pdm_reward import COMFORT_LIMITS

    assert COMFORT_LIMITS == {
        "min_lon_accel": -4.05,
        "max_lon_accel": 2.40,
        "max_abs_lat_accel": 4.89,
        "max_abs_mag_jerk": 8.37,
        "max_abs_lon_jerk": 4.13,
        "max_abs_yaw_rate": 0.95,
        "max_abs_yaw_accel": 1.93,
    }


def _lane_ctx(center_y, half=1.75):
    """Human path along y = 0 (a lane line), lane centreline at y = center_y."""
    ctx = _ctx()
    ctx["centerline_xy"] = np.stack(
        [np.arange(-20.0, 120.0), np.full(140, center_y)], axis=1
    ).astype(np.float32)
    ctx["center_halfwidth"] = np.full(140, half, dtype=np.float32)
    return ctx


def test_centerline_path_keeping_prefers_the_lane_centre():
    ctx = _lane_ctx(center_y=1.75)
    plans = [_straight(y=0.0), _straight(y=1.75)]
    r, _ = _score(plans, ctx)  # default: PK against the human path
    assert r[0] > r[1]
    r, _ = _score(plans, ctx, pk_path="centerline", pk_sigma_m=0.5)
    assert r[1] > r[0]
    _, on_line = _score(plans[:1], ctx)
    _, centred = _score(plans[1:], ctx)
    assert on_line["straddle"] == 1.0 and centred["straddle"] == 0.0
    assert on_line["center_lateral_m"] == pytest.approx(1.75, abs=1e-3)
    assert (
        centred["center_lateral_m"] < 0.25
    )  # the poly smoothing starts the plan at the ego (y = 0)


def test_centring_terms_absent_without_a_centerline():
    _, terms = _score([_straight()], _ctx(), pk_path="centerline")
    assert (
        "center_lateral_m" not in terms
        and "straddle" not in terms
        and terms["pk"] == pytest.approx(1.0)
    )


def test_geometry_helpers():
    a = torch.tensor([[1.0, 1.0], [1.0, -1.0], [-1.0, -1.0], [-1.0, 1.0]])
    assert boxes_overlap(a, a + 1.5).item()
    assert not boxes_overlap(a, a + 2.5).item()
    lateral, arc = project_on_path(
        torch.tensor([[3.0, 2.0]]), torch.tensor([[0.0, 0.0], [10.0, 0.0]])
    )
    assert lateral.item() == pytest.approx(2.0) and arc.item() == pytest.approx(3.0)
    line = torch.stack([_straight()] * 2)
    torch.testing.assert_close(poly_smooth(line, DT), line, atol=1e-4, rtol=0)


# Tiered mode (FlowGRPO v10): failures negative and ranked, passes in [0, 1].
TIERED = dict(
    mode="tiered",
    dac_mode="binary",
    corridor_m=4.0,
    ttc_horizon_s=1.5,
    ep_over_ratio=1.1,
)


def test_tiered_failures_rank_below_passes_and_collisions_lowest():
    agent = _static_agent(
        41.0, 0.0
    )  # the 8 m/s plan touches it in the last steps, the 6.4 m/s plan does not
    r, terms = _score([_straight(8.0), _straight(6.4)], _ctx(agent), **TIERED)
    assert -1.0 <= r[0].item() <= -0.8 and r[1].item() >= 0.0
    r_off, _ = _score([_drift(8.0)], _ctx(), **TIERED)  # off-road on the 5 m half road
    assert -0.7 <= r_off[0].item() <= -0.5
    r_cor, t_cor = _score(
        [_drift(4.6)], _ctx(half_road_m=20.0), **TIERED
    )  # corridor exit, still on the road
    assert t_cor["corridor"] == 0.0 and -0.75 <= r_cor[0].item() <= -0.55
    assert r[0].item() < min(r_off[0].item(), r_cor[0].item())
    assert terms["fail_rate"] == 0.5


def test_tiered_later_failure_ranks_above_earlier_one():
    near, far = (
        _score([_straight()], _ctx(_static_agent(20.0, 0.0)), **TIERED),
        _score([_straight()], _ctx(_static_agent(35.0, 0.0)), **TIERED),
    )
    assert near[0][0].item() < far[0][0].item() < -0.8


def test_tiered_clearance_penalizes_closing_on_a_slow_lead():
    # Lead 3 m/s slower, centre 24 m ahead: bumper-to-bumper gap 17.75 m -> 2.75 m after 5 s (no contact), below the
    # 2 m + 0.6 s x 8 m/s = 6.8 m wanted -> clearance ~0.4.
    _, slow = _score(
        [_straight()], _ctx(_moving_agent(24.0, 0.0, SPEED - 3.0)), **TIERED
    )
    assert slow["collision_rate"] == 0.0 and slow["clearance"] < 0.6
    _, same = _score([_straight()], _ctx(_moving_agent(20.0, 0.0, SPEED)), **TIERED)
    assert same["clearance"] == 1.0
    _, side = _score(
        [_straight()], _ctx(_moving_agent(10.0, 3.5, SPEED - 3.0)), **TIERED
    )  # adjacent lane
    assert side["clearance"] == 1.0


def test_tiered_centring_prefers_converging_to_the_lane_centre():
    ctx = _ctx(half_road_m=10.0)
    ctx["path_xy"][:, 1] = 1.0  # human path and lane centre 1 m to the left of the ego
    ctx["centerline_xy"] = ctx["path_xy"].copy()
    ctx["center_halfwidth"] = np.full(len(ctx["path_xy"]), 1.75, dtype=np.float32)
    stay = _straight()  # keeps the 1 m offset
    t = torch.arange(1, T + 1, dtype=torch.float32) * DT
    converge = _straight()
    converge[:, 1] = 1.0 * torch.clamp(t / 2.5, max=1.0)  # reaches the centre in 2.5 s
    reward = SimPDMReward(**TIERED)
    r_stay, t_stay = reward.score(stay[None], ctx)
    r_conv, t_conv = reward.score(converge[None], ctx)
    assert (
        t_conv["centring"] > t_stay["centring"] + 0.2 and r_conv.item() > r_stay.item()
    )
    assert t_stay["center_lateral_m"] == pytest.approx(1.0, abs=0.1)


def test_tiered_without_centerline_falls_back_to_the_human_path():
    r, terms = _score([_straight(), _drift(2.0)], _ctx(), **TIERED)
    assert (
        np.isnan(terms.get("center_lateral_m", float("nan")))
        or "center_lateral_m" not in terms
    )
    assert r[0].item() > r[1].item() >= 0.0


def test_tiered_ignores_the_path_past_the_end_of_the_recording():
    # A plan that drifts 4.6 m off the stored path late in the horizon: a corridor exit while the human path is
    # recorded, not when the recording ended after 2 s (the stored path beyond is the route extension).
    ctx = _ctx(half_road_m=20.0)
    ctx["human_progress_m"], ctx["human_covered_s"] = 40.0, 5.0
    r_full, t_full = SimPDMReward(**TIERED).score(_drift(4.6)[None], ctx)
    ctx["human_covered_s"] = 2.0
    r_short, t_short = SimPDMReward(**TIERED).score(_drift(4.6)[None], ctx)
    assert t_full["corridor"] == 0.0 and r_full.item() < 0.0
    assert t_short["corridor"] == 1.0 and r_short.item() >= 0.0
