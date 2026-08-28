import warnings

import torch
from torchcubicspline import natural_cubic_spline_coeffs, NaturalCubicSpline

from ad_vla.dataset.data_types import TrajectorySampling


def resample_tensor(
    tensor: torch.Tensor,
    source: TrajectorySampling,
    target: TrajectorySampling,
    is_past: bool = False,
) -> torch.Tensor:
    """
    Resample a raw trajectory tensor from one temporal grid to another.

    Uses physical timestamps for cubic spline interpolation, with a fast path
    for exact subsampling when source steps are an integer multiple of target steps.
    Future trajectories use timestamps [dt, 2dt, ..., Ndt]. Past trajectories
    use timestamps ending at the present: [-(N-1)dt, ..., -dt, 0].

    Args:
        tensor: Tensor of shape [*, T_source, D].
        source: Temporal specification of the input tensor.
        target: Target temporal specification to resample to.
        is_past: If True, use past-trajectory semantics: re-anchor the final
            point to zero, interpolate on a time grid ending at 0, and never
            extrapolate before the available history.

    Returns:
        Resampled tensor of shape [*, T_target, D]. For past trajectories with
        target history older than the source provides, returns only the
        available target timestamps, so the time dimension can be smaller than
        T_target.
    """
    if is_past:
        return _resample_past_tensor(tensor, source, target)

    src_steps = source.num_poses
    tgt_steps = target.num_poses

    if src_steps == tgt_steps and source.interval_length == target.interval_length:
        return tensor

    # Fast path: exact subsampling (e.g. 64@10Hz -> 16@10Hz won't hit this,
    # but 20@4Hz -> 5@4Hz would if 20 % 5 == 0)
    if src_steps > tgt_steps and src_steps % tgt_steps == 0:
        # Check that the time horizons match (same total duration)
        src_horizon = src_steps * source.interval_length
        tgt_horizon = tgt_steps * target.interval_length
        if abs(src_horizon - tgt_horizon) < 1e-6:
            dt = src_steps // tgt_steps
            return tensor[..., dt - 1 :: dt, :]

    # Warn if target horizon exceeds source horizon (cubic extrapolation is unstable)
    src_horizon = src_steps * source.interval_length
    tgt_horizon = tgt_steps * target.interval_length
    if tgt_horizon > src_horizon + 1e-6:
        warnings.warn(
            f"Target horizon ({tgt_horizon:.2f}s) exceeds source horizon "
            f"({src_horizon:.2f}s). The cubic spline will extrapolate beyond "
            f"the source data using the last polynomial segment, which can "
            f"diverge quickly and produce unreliable results.",
            stacklevel=2,
        )

    # General case: cubic spline interpolation using physical timestamps
    # Source timestamps: [dt, 2*dt, ..., N*dt]
    source_t = torch.linspace(
        source.interval_length,
        src_steps * source.interval_length,
        src_steps,
        device=tensor.device,
        dtype=tensor.dtype,
    )
    # Target timestamps: [dt, 2*dt, ..., M*dt]
    target_t = torch.linspace(
        target.interval_length,
        tgt_steps * target.interval_length,
        tgt_steps,
        device=tensor.device,
        dtype=tensor.dtype,
    )

    # Resample using spline (torchcubicspline handles batch size automatically)
    coeffs = natural_cubic_spline_coeffs(source_t, tensor)
    spline = NaturalCubicSpline(coeffs)
    resampled = spline.evaluate(target_t)
    return resampled


def _resample_past_tensor(
    tensor: torch.Tensor,
    source: TrajectorySampling,
    target: TrajectorySampling,
) -> torch.Tensor:
    src_steps = source.num_poses
    tgt_steps = target.num_poses

    # Past trajectories are ego-relative history and must end exactly at the
    # present pose. Re-anchor defensively even if the dataset is slightly noisy.
    tensor = tensor - tensor[..., -1:, :]

    if src_steps == 1:
        return torch.zeros_like(tensor)

    source_t = (
        torch.arange(src_steps, device=tensor.device, dtype=tensor.dtype)
        - (src_steps - 1)
    ) * source.interval_length
    target_t = (
        torch.arange(tgt_steps, device=tensor.device, dtype=tensor.dtype)
        - (tgt_steps - 1)
    ) * target.interval_length

    # Do not extrapolate beyond available history. Text prompts can accept
    # "at most" the requested number of past points, so older unavailable target
    # timestamps are dropped instead of invented.
    target_t = target_t[target_t >= source_t[0]]

    idx_right = torch.searchsorted(source_t, target_t).clamp(1, src_steps - 1)
    idx_left = idx_right - 1

    left_t = source_t[idx_left]
    right_t = source_t[idx_right]
    weight = (target_t - left_t) / (right_t - left_t)
    weight = weight.reshape((1,) * (tensor.ndim - 2) + (-1, 1))

    left = tensor.index_select(dim=-2, index=idx_left)
    right = tensor.index_select(dim=-2, index=idx_right)
    resampled = left + (right - left) * weight
    resampled[..., -1, :] = 0
    return resampled
