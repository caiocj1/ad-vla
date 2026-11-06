import math

import torch


def _to_tensor_like(value: float | torch.Tensor, ref: torch.Tensor) -> torch.Tensor:
    """Convert a scalar/tensor hyperparameter to something broadcastable with ref."""
    return torch.as_tensor(value, device=ref.device, dtype=ref.dtype)


def _wrap_to_pi(angle: torch.Tensor) -> torch.Tensor:
    """Map angles to [-pi, pi) so that differences do not jump by 2*pi."""
    two_pi = 2.0 * math.pi
    return (angle + math.pi).remainder(two_pi) - math.pi


def _unwrap_angle(angle: torch.Tensor) -> torch.Tensor:
    """Unwrap an angle sequence along the last dimension."""
    if angle.shape[-1] <= 1:
        return angle

    # Convert each consecutive jump to the shortest signed angular change,
    # then rebuild the curve from the first angle.
    delta = _wrap_to_pi(angle[..., 1:] - angle[..., :-1])
    return torch.cat([angle[..., :1], angle[..., :1] + delta.cumsum(dim=-1)], dim=-1)


def _infer_heading_from_xy(
    full_xy: torch.Tensor,
    *,
    min_forward_delta: float,
) -> tuple[torch.Tensor, torch.Tensor]:
    """Infer a robust yaw target from an ego-frame XY path.

    Alpamayo gets heading from dataset rotations. In this repo utility we only
    have XY, so we must infer a tangent direction. Raw per-segment atan2 is too
    brittle: one tiny backward/noisy delta in an otherwise straight path looks
    like a pi-radian yaw target and can create a huge fake turn.

    We use a small wider secant instead of a one-interval delta. That skips
    over bunched points when the vehicle nearly stops, while the forward clamp
    preserves the previous fix for tiny/reverse x motion in ego-frame driving.
    """
    n = full_xy.shape[-2]
    window = min(2, max(n - 1, 1))

    tangent = torch.empty_like(full_xy)
    if n == 1:
        tangent.fill_(0)
        tangent[..., 0] = 1.0
    else:
        idx = torch.arange(n, device=full_xy.device)
        left_idx = (idx - window).clamp_min(0)
        right_idx = (idx + window).clamp_max(n - 1)
        left = full_xy.index_select(dim=-2, index=left_idx)
        right = full_xy.index_select(dim=-2, index=right_idx)
        tangent = right - left

    tangent_norm = tangent.norm(dim=-1)
    tangent_x = tangent[..., 0].clamp_min(min_forward_delta)
    heading = torch.atan2(tangent[..., 1], tangent_x)
    heading[..., 0] = 0.0
    return _unwrap_angle(heading), tangent_norm


def _finite_difference_matrix(
    n: int,
    order: int,
    *,
    device: torch.device,
    dtype: torch.dtype,
) -> torch.Tensor:
    """Build a finite-difference matrix D so D @ x is an order-th difference."""
    if order < 1:
        raise ValueError(f"order must be >= 1, got {order}")

    rows = n - order
    if rows <= 0:
        return torch.zeros(0, n, device=device, dtype=dtype)

    # Coefficients for finite differences:
    # order 1: [-1, 1]
    # order 2: [1, -2, 1]
    # order 3: [-1, 3, -3, 1]
    # This is (-1) ** (order - k) * binomial(order, k).
    coeffs = [((-1.0) ** (order - k)) * math.comb(order, k) for k in range(order + 1)]

    d = torch.zeros(rows, n, device=device, dtype=dtype)
    row_ids = torch.arange(rows, device=device)
    for offset, coeff in enumerate(coeffs):
        d[row_ids, row_ids + offset] = coeff
    return d


def _smooth_penalty_matrix(
    n: int,
    order: int,
    lam: float,
    dt: float,
    *,
    device: torch.device,
    dtype: torch.dtype,
) -> torch.Tensor:
    """Return lambda * D.T @ D with the same time scaling Alpamayo uses."""
    d = _finite_difference_matrix(n, order, device=device, dtype=dtype)
    if d.numel() == 0 or lam == 0.0:
        return torch.zeros(n, n, device=device, dtype=dtype)

    # A first difference approximates dx/dt, a second difference approximates
    # d2x/dt2, etc. Squaring that derivative gives the dt ** (2 * order)
    # denominator. Alpamayo applies the same scaling.
    scale = lam / (dt ** (2 * order))
    return scale * (d.transpose(-1, -2) @ d)


def _solve_sequence_fit(
    *,
    slope: torch.Tensor,
    target: torch.Tensor,
    data_weight: torch.Tensor | None = None,
    smooth_order: int,
    smooth_lambda: float,
    ridge: float,
    dt: float,
) -> torch.Tensor:
    """Solve min_x ||slope * x - target||^2 + smoothness + ridge."""
    n = target.shape[-1]
    device = target.device
    dtype = target.dtype

    smooth = _smooth_penalty_matrix(
        n,
        smooth_order,
        smooth_lambda,
        dt,
        device=device,
        dtype=dtype,
    )
    eye = torch.eye(n, device=device, dtype=dtype)

    if data_weight is None:
        data_weight = torch.ones_like(target)

    # Since the data term is diagonal, A.T @ W @ A is diag(w * slope^2), and
    # A.T @ W @ b is w * slope * target. Broadcasting handles batch dims.
    lhs = torch.diag_embed(data_weight * slope * slope) + smooth + ridge * eye
    rhs = data_weight * slope * target
    return torch.linalg.solve(lhs, rhs.unsqueeze(-1)).squeeze(-1)


def _solve_sequence_fit_with_initial(
    *,
    initial: torch.Tensor,
    target: torch.Tensor,
    data_weight: torch.Tensor | None = None,
    smooth_order: int,
    smooth_lambda: float,
    ridge: float,
    dt: float,
) -> torch.Tensor:
    """Solve for [initial, x_1, ..., x_N] while keeping initial fixed."""
    n = target.shape[-1]
    device = target.device
    dtype = target.dtype

    smooth = _smooth_penalty_matrix(
        n + 1,
        smooth_order,
        smooth_lambda,
        dt,
        device=device,
        dtype=dtype,
    )
    eye = torch.eye(n, device=device, dtype=dtype)

    if data_weight is None:
        data_weight = torch.ones_like(target)

    # The unknowns are x_1..x_N. The smoothness matrix was built for the full
    # sequence [x_0, x_1..x_N], so we move the known x_0 contribution to RHS.
    lhs = torch.diag_embed(data_weight) + smooth[1:, 1:] + ridge * eye
    rhs = data_weight * target - smooth[1:, 0] * initial.unsqueeze(-1)
    solved = torch.linalg.solve(lhs, rhs.unsqueeze(-1)).squeeze(-1)
    return torch.cat([initial.unsqueeze(-1), solved], dim=-1)


def _confidence_from_distance(distance: torch.Tensor, threshold: float) -> torch.Tensor:
    """Return a soft data weight that goes to zero near stationary intervals."""
    if threshold <= 0:
        return torch.ones_like(distance)
    return distance / (distance + _to_tensor_like(threshold, distance))


def _solve_velocity_from_xy_and_heading(
    dxy: torch.Tensor,
    theta: torch.Tensor,
    initial_velocity: torch.Tensor,
    *,
    dt: float,
    smooth_lambda: float,
    ridge: float,
) -> torch.Tensor:
    """Infer v_0..v_T from waypoint deltas and headings.

    The unicycle position update is integrated with the trapezoid rule:

        delta_xy_t = dt / 2 * (v_t * u_t + v_{t+1} * u_{t+1})
        u_t = [cos(theta_t), sin(theta_t)]

    The unknowns are v_1..v_T. v_0 is fixed to initial_velocity.
    """
    *lead, n, _ = dxy.shape
    device = dxy.device
    dtype = dxy.dtype

    cos_theta = torch.cos(theta)
    sin_theta = torch.sin(theta)

    # We multiply both sides by 2 / dt:
    #   v_t * u_t + v_{t+1} * u_{t+1} = 2 * delta_xy_t / dt
    # This gives two linear equations per interval: one for x and one for y.
    a_data = torch.zeros(*lead, 2 * n, n, device=device, dtype=dtype)
    b_data = (2.0 / dt * dxy).reshape(*lead, 2 * n)

    interval_ids = torch.arange(n, device=device)
    x_rows = 2 * interval_ids
    y_rows = x_rows + 1

    # v_{t+1} is unknown for every interval t and maps to column t.
    a_data[..., x_rows, interval_ids] = cos_theta[..., 1:]
    a_data[..., y_rows, interval_ids] = sin_theta[..., 1:]

    if n > 1:
        # v_t is also unknown for intervals t > 0 and maps to column t - 1.
        prev_cols = interval_ids[:-1]
        a_data[..., x_rows[1:], prev_cols] = cos_theta[..., 1:-1]
        a_data[..., y_rows[1:], prev_cols] = sin_theta[..., 1:-1]

    # For interval 0, v_0 is fixed, so its contribution moves to RHS.
    b_data[..., 0] -= initial_velocity * cos_theta[..., 0]
    b_data[..., 1] -= initial_velocity * sin_theta[..., 0]

    with torch.amp.autocast(device_type="cuda", enabled=False):
        ata = a_data.transpose(-1, -2) @ a_data
        rhs = (a_data.transpose(-1, -2) @ b_data.unsqueeze(-1)).squeeze(-1)

    smooth = _smooth_penalty_matrix(
        n + 1,
        3,
        smooth_lambda,
        dt,
        device=device,
        dtype=dtype,
    )
    eye = torch.eye(n, device=device, dtype=dtype)

    # The D.T @ D matrix is for [v_0, v_1..v_T]. Keep v_0 fixed and solve the
    # rest, exactly like the heading solve above.
    lhs = ata + smooth[1:, 1:] + ridge * eye
    rhs = rhs - smooth[1:, 0] * initial_velocity.unsqueeze(-1)
    velocity_tail = torch.linalg.solve(lhs, rhs.unsqueeze(-1)).squeeze(-1)
    return torch.cat([initial_velocity.unsqueeze(-1), velocity_tail], dim=-1)


def traj_to_action(
    traj_xy: torch.Tensor,
    *,
    dt: float = 0.25,
    initial_velocity: float | torch.Tensor = 0.0,
    accel_mean: float = 0.0,
    accel_std: float = 1.0,
    curvature_mean: float = 0.0,
    curvature_std: float = 1.0,
    theta_lambda: float = 1e-6,
    theta_ridge: float = 1e-8,
    velocity_lambda: float = 1e-6,
    velocity_ridge: float = 1e-4,
    accel_lambda: float = 1e-4,
    accel_ridge: float = 1e-4,
    curvature_lambda: float = 1e-4,
    curvature_ridge: float = 1e-4,
    heading_min_forward_delta: float = 1e-3,
    heading_confidence_distance: float = 0.5,
    curvature_confidence_distance: float = 0.25,
    return_states: bool = False,
) -> torch.Tensor | tuple[torch.Tensor, dict[str, torch.Tensor]]:
    """Convert ego-frame XY waypoints to [acceleration, curvature] actions.

    Args:
        traj_xy: Future trajectory with shape [..., T, 2]. The current ego
            pose is assumed to be [0, 0], with x forward and y left.
        dt: Time spacing between waypoints, in seconds; XY coordinates are meters.
        initial_velocity: Speed at the current timestep, in m/s. Alpamayo estimates
            this from history; this utility defaults to 0 because it only sees
            the future XY trajectory.
        heading_min_forward_delta: Minimum forward tangent used when inferring
            heading from XY. This prevents tiny backward/noisy deltas from
            being interpreted as a 180-degree turn.
        heading_confidence_distance: Tangent distance scale used to downweight
            inferred heading targets around bunched/stationary points.
        curvature_confidence_distance: Segment distance scale used to downweight
            curvature observations when the vehicle is nearly stationary.
        *_mean, *_std: Optional normalization constants for the two action
            channels. Defaults leave actions unnormalized.
        *_lambda, *_ridge: Smoothing and ridge terms for the direct solves.
        return_states: If True, also return intermediate theta/velocity/etc.

    Returns:
        actions with shape [..., T, 2], where actions[..., 0] is acceleration
        and actions[..., 1] is curvature. Before optional normalization, units
        are m/s^2 and 1/m respectively. Heading/velocity/action smoothing means
        conversion back to XY is not guaranteed to reproduce the input exactly.
    """
    if traj_xy.shape[-1] != 2:
        raise ValueError(f"traj_xy must have shape [..., T, 2], got {traj_xy.shape}")
    if traj_xy.shape[-2] < 1:
        raise ValueError("traj_xy must contain at least one waypoint")

    orig_dtype = traj_xy.dtype
    work_dtype = (
        torch.float32
        if traj_xy.dtype in (torch.float16, torch.bfloat16)
        else traj_xy.dtype
    )
    traj_xy = traj_xy.to(work_dtype)

    batch_shape = traj_xy.shape[:-2]
    device = traj_xy.device
    dtype = traj_xy.dtype

    v0 = _to_tensor_like(initial_velocity, traj_xy).expand(batch_shape)

    # Step 1: add the current ego pose. The future trajectory starts at t=dt,
    # but the unicycle rollout starts at t=0 from x=0, y=0.
    origin = torch.zeros(*batch_shape, 1, 2, device=device, dtype=dtype)
    full_xy = torch.cat([origin, traj_xy], dim=-2)
    dxy = full_xy[..., 1:, :] - full_xy[..., :-1, :]

    # Step 2: get a raw heading from the XY path. This replaces the rotation
    # input Alpamayo has. We use central path tangents rather than raw segment
    # directions so a tiny backward/noisy delta does not become a fake pi turn.
    theta0 = torch.zeros(batch_shape, device=device, dtype=dtype)
    raw_theta, heading_tangent_distance = _infer_heading_from_xy(
        full_xy,
        min_forward_delta=heading_min_forward_delta,
    )
    heading_weight = _confidence_from_distance(
        heading_tangent_distance[..., 1:],
        heading_confidence_distance,
    )

    # Step 3: smooth heading with theta_0 fixed to zero. This gives theta
    # values at the current pose and at every future waypoint.
    theta = _solve_sequence_fit_with_initial(
        initial=theta0,
        target=raw_theta[..., 1:],
        data_weight=heading_weight,
        smooth_order=3,
        smooth_lambda=theta_lambda,
        ridge=theta_ridge,
        dt=dt,
    )

    # Step 4: solve for velocities that explain the observed XY deltas under
    # trapezoidal unicycle integration, with v_0 fixed.
    velocity = _solve_velocity_from_xy_and_heading(
        dxy,
        theta,
        v0,
        dt=dt,
        smooth_lambda=velocity_lambda,
        ridge=velocity_ridge,
    )

    # Step 5: acceleration is the velocity derivative. We solve a smoothed
    # version of a_t ~= (v_{t+1} - v_t) / dt instead of taking the raw finite
    # difference, because raw finite differences amplify waypoint noise.
    accel_target = (velocity[..., 1:] - velocity[..., :-1]) / dt
    accel = _solve_sequence_fit(
        slope=torch.ones_like(accel_target),
        target=accel_target,
        smooth_order=2,
        smooth_lambda=accel_lambda,
        ridge=accel_ridge,
        dt=dt,
    )

    # Step 6: curvature is heading change per distance traveled. The distance
    # for each interval under constant acceleration is:
    #   s_t = dt * v_t + 0.5 * dt^2 * a_t
    # Then kappa_t approximately satisfies:
    #   s_t * kappa_t = theta_{t+1} - theta_t
    dtheta = theta[..., 1:] - theta[..., :-1]
    distance = dt * velocity[..., :-1] + 0.5 * (dt**2) * accel
    curvature = _solve_sequence_fit(
        slope=distance,
        target=dtheta,
        data_weight=_confidence_from_distance(
            dxy.norm(dim=-1),
            curvature_confidence_distance,
        ),
        smooth_order=2,
        smooth_lambda=curvature_lambda,
        ridge=curvature_ridge,
        dt=dt,
    )

    # Step 7: normalize the control channels if statistics are provided.
    accel_norm = (accel - _to_tensor_like(accel_mean, accel)) / _to_tensor_like(
        accel_std, accel
    )
    curvature_norm = (
        curvature - _to_tensor_like(curvature_mean, curvature)
    ) / _to_tensor_like(curvature_std, curvature)
    action = torch.stack([accel_norm, curvature_norm], dim=-1).to(orig_dtype)

    if not return_states:
        return action

    states = {
        "theta": theta.to(orig_dtype),
        "velocity": velocity.to(orig_dtype),
        "accel": accel.to(orig_dtype),
        "curvature": curvature.to(orig_dtype),
        "distance": distance.to(orig_dtype),
        "heading_weight": heading_weight.to(orig_dtype),
    }
    return action, states


def action_to_traj(
    action: torch.Tensor,
    *,
    dt: float = 0.25,
    initial_velocity: float | torch.Tensor = 0.0,
    accel_mean: float = 0.0,
    accel_std: float = 1.0,
    curvature_mean: float = 0.0,
    curvature_std: float = 1.0,
    return_states: bool = False,
) -> torch.Tensor | tuple[torch.Tensor, dict[str, torch.Tensor]]:
    """Roll out [acceleration, curvature] actions into ego-frame XY waypoints.

    Args:
        action: Tensor with shape [..., T, 2]. Channel 0 is acceleration and
            channel 1 is curvature, optionally normalized.
        dt: Time spacing between waypoints.
        initial_velocity: Speed at t=0. Use the same value used by
            traj_to_action if you want an approximate round trip.
        *_mean, *_std: Normalization constants matching traj_to_action.
        return_states: If True, also return velocity and heading sequences.

    Returns:
        Future XY trajectory with shape [..., T, 2].
    """
    if action.shape[-1] != 2:
        raise ValueError(f"action must have shape [..., T, 2], got {action.shape}")
    if action.shape[-2] < 1:
        raise ValueError("action must contain at least one timestep")

    orig_dtype = action.dtype
    work_dtype = (
        torch.float32
        if action.dtype in (torch.float16, torch.bfloat16)
        else action.dtype
    )
    action = action.to(work_dtype)

    batch_shape = action.shape[:-2]
    device = action.device
    dtype = action.dtype

    accel = action[..., 0] * _to_tensor_like(accel_std, action) + _to_tensor_like(
        accel_mean, action
    )
    curvature = action[..., 1] * _to_tensor_like(
        curvature_std, action
    ) + _to_tensor_like(curvature_mean, action)

    v0 = _to_tensor_like(initial_velocity, action).expand(batch_shape)

    # Integrate acceleration:
    #   v_{t+1} = v_t + dt * a_t
    velocity = torch.cat(
        [v0.unsqueeze(-1), v0.unsqueeze(-1) + torch.cumsum(accel * dt, dim=-1)],
        dim=-1,
    )

    # Integrate curvature into heading. Curvature is dtheta / distance, and
    # distance over the interval is dt * v_t + 0.5 * dt^2 * a_t.
    distance = dt * velocity[..., :-1] + 0.5 * (dt**2) * accel
    theta0 = torch.zeros(batch_shape, device=device, dtype=dtype)
    theta = torch.cat(
        [
            theta0.unsqueeze(-1),
            theta0.unsqueeze(-1) + torch.cumsum(curvature * distance, dim=-1),
        ],
        dim=-1,
    )

    # Integrate position with the trapezoid rule. This averages the start and
    # end velocity vectors of each interval, giving smoother rollouts than a
    # purely left- or right-endpoint Euler update.
    half_dt = 0.5 * dt
    dx = (
        velocity[..., :-1] * torch.cos(theta[..., :-1])
        + velocity[..., 1:] * torch.cos(theta[..., 1:])
    ) * half_dt
    dy = (
        velocity[..., :-1] * torch.sin(theta[..., :-1])
        + velocity[..., 1:] * torch.sin(theta[..., 1:])
    ) * half_dt
    traj_xy = torch.stack([dx.cumsum(dim=-1), dy.cumsum(dim=-1)], dim=-1).to(orig_dtype)

    if not return_states:
        return traj_xy

    states = {
        "theta": theta.to(orig_dtype),
        "velocity": velocity.to(orig_dtype),
        "accel": accel.to(orig_dtype),
        "curvature": curvature.to(orig_dtype),
        "distance": distance.to(orig_dtype),
    }
    return traj_xy, states


# Below this speed (m/s) a step's direction is noise, so polar actions keep the previous heading.
POLAR_MIN_SPEED = 0.2


def traj_to_polar(
    traj_xy: torch.Tensor,
    *,
    dt: float,
    min_speed: float = POLAR_MIN_SPEED,
) -> torch.Tensor:
    """Convert ego-frame XY waypoints to per-step [delta_heading, step_distance] actions.

    Each step is expressed in the frame of the previous one: turn by delta_heading, then
    move step_distance along the new heading. The heading starts at the current ego heading
    (+x) and follows the chords between consecutive waypoints.

    A step pointing backward relative to the current heading (reversing, or stop-and-go
    jitter) is encoded as a negative distance with a small turn rather than a ~pi turn, so
    delta_heading stays within [-pi/2, pi/2]. This assumes forward turns stay below 90
    degrees per step, which holds for driving at these step lengths.

    Args:
        traj_xy: Future trajectory [..., T, 2], current pose at [0, 0], x forward, y left.
        dt: Time spacing between waypoints, in seconds.
        min_speed: Steps slower than this (m/s) have no reliable direction, so they keep
            the previous heading (delta_heading = 0). Decoding places them along that heading,
            so the round trip is exact except for these steps.

    Returns:
        actions [..., T, 2]: channel 0 is delta heading (rad), channel 1 is signed step
        distance (m).
    """
    if traj_xy.shape[-1] != 2:
        raise ValueError(f"traj_xy must have shape [..., T, 2], got {traj_xy.shape}")

    orig_dtype = traj_xy.dtype
    work_dtype = (
        torch.float32
        if traj_xy.dtype in (torch.float16, torch.bfloat16)
        else traj_xy.dtype
    )
    traj_xy = traj_xy.to(work_dtype)

    origin = traj_xy.new_zeros(*traj_xy.shape[:-2], 1, 2)
    step = torch.diff(torch.cat([origin, traj_xy], dim=-2), dim=-2)
    distance = step.norm(dim=-1)
    step_heading = torch.atan2(step[..., 1], step[..., 0])
    moving = distance >= min_speed * dt

    # Sequential over steps: whether a step is backward depends on the heading so far.
    heading = torch.zeros_like(distance[..., 0])
    delta_heading, signed_distance = [], []
    for i in range(distance.shape[-1]):
        turn = _wrap_to_pi(step_heading[..., i] - heading)
        backward = turn.abs() > math.pi / 2
        turn = torch.where(backward, _wrap_to_pi(turn + math.pi), turn)
        turn = torch.where(moving[..., i], turn, torch.zeros_like(turn))
        delta_heading.append(turn)
        signed_distance.append(
            torch.where(backward, -distance[..., i], distance[..., i])
        )
        heading = heading + turn

    actions = torch.stack(
        [torch.stack(delta_heading, dim=-1), torch.stack(signed_distance, dim=-1)],
        dim=-1,
    )
    return actions.to(orig_dtype)


def polar_to_traj(action: torch.Tensor) -> torch.Tensor:
    """Roll out [delta_heading, signed step_distance] actions into ego-frame XY waypoints.

    Inverse of traj_to_polar.
    """
    if action.shape[-1] != 2:
        raise ValueError(f"action must have shape [..., T, 2], got {action.shape}")

    orig_dtype = action.dtype
    work_dtype = (
        torch.float32
        if action.dtype in (torch.float16, torch.bfloat16)
        else action.dtype
    )
    action = action.to(work_dtype)

    heading = action[..., 0].cumsum(dim=-1)
    step = action[..., 1:] * torch.stack([heading.cos(), heading.sin()], dim=-1)
    return step.cumsum(dim=-2).to(orig_dtype)


ACTION_REPRESENTATIONS = ("trajs", "acc_curv", "polar")


def encode_actions(
    traj_xy: torch.Tensor,
    representation: str,
    *,
    dt: float,
    initial_velocity: float | torch.Tensor | None = None,
) -> torch.Tensor:
    """Ego-frame XY waypoints [..., T, 2] -> raw (unnormalized) actions [..., T, 2]."""
    if representation == "trajs":
        return traj_xy
    if representation == "acc_curv":
        if initial_velocity is None:
            raise ValueError("acc_curv actions need initial_velocity.")
        return traj_to_action(traj_xy, dt=dt, initial_velocity=initial_velocity)
    if representation == "polar":
        return traj_to_polar(traj_xy, dt=dt)
    raise ValueError(f"Unknown action representation {representation!r}.")


def decode_actions(
    actions: torch.Tensor,
    representation: str,
    *,
    dt: float,
    initial_velocity: float | torch.Tensor | None = None,
) -> torch.Tensor:
    """Raw actions [..., T, 2] -> ego-frame XY waypoints [..., T, 2]; inverse of encode_actions."""
    if representation == "trajs":
        return actions
    if representation == "acc_curv":
        if initial_velocity is None:
            raise ValueError("acc_curv actions need initial_velocity.")
        return action_to_traj(actions, dt=dt, initial_velocity=initial_velocity)
    if representation == "polar":
        return polar_to_traj(actions)
    raise ValueError(f"Unknown action representation {representation!r}.")


__all__ = [
    "ACTION_REPRESENTATIONS",
    "POLAR_MIN_SPEED",
    "action_to_traj",
    "decode_actions",
    "encode_actions",
    "polar_to_traj",
    "traj_to_action",
    "traj_to_polar",
]
