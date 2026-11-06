"""Per-image-token camera geometry for the FM-VLA action expert (``expert_cfg.camera_geometry``).

For every merged Qwen visual token, the collator describes the ray the token looks along:

* the viewing ray through the token's centre pixel, in the ego frame (unit vector);
* the camera centre in the ego frame, in meters;
* the pinhole-normalized image coordinates ``((u - cx) / fx, (v - cy) / fy)``;
* the relative frame index (0 for the newest frame, -1 for the one before, ...).

Calibrations follow ``CalibrationDict``: camera-to-ego extrinsics with an optical (RDF)
camera frame. Cameras with an f-theta model (PhysicalAI: extra ``fw_poly`` / ``bw_poly``
fields) are unprojected with it; the others with the pinhole intrinsics, after removing the
Brown-Conrady ``distortion`` when present.

Token order follows ``PromptBuilder``: frames oldest to newest; per frame either one image of
the cameras concatenated left to right in ``camera_sequence`` order, or one image per camera.
Within an image, merged tokens are row-major over the merged grid, as Qwen emits them.
"""

from typing import Any, Sequence

import torch

from ad_vla.dataset.data_types import CameraData

# Raw per-token feature layout of `visual_geom` (the model normalizes and embeds it).
GEOMETRY_FEATURES = (
    "ray_x",
    "ray_y",
    "ray_z",
    "center_x",
    "center_y",
    "center_z",
    "norm_u",
    "norm_v",
    "frame",
)
NUM_GEOMETRY_FEATURES = len(GEOMETRY_FEATURES)

_UNDISTORT_ITERS = 10


def camera_to_ego(cal_dict: Any) -> tuple[torch.Tensor, torch.Tensor]:
    """(rotation [3, 3], translation [3]) with ``p_ego = rotation @ p_camera + translation``."""
    if cal_dict.is_flu:
        raise ValueError(
            "Camera geometry needs the optical (RDF) camera convention, got is_flu=True."
        )
    rotation = torch.as_tensor(cal_dict.rotation, dtype=torch.float32)
    translation = torch.as_tensor(cal_dict.translation, dtype=torch.float32)
    return rotation, translation


def _polyval(coefs: torch.Tensor, x: torch.Tensor) -> torch.Tensor:
    """sum_k coefs[k] * x**k (Horner)."""
    out = torch.zeros_like(x)
    for coef in coefs.flip(0):
        out = out * x + coef
    return out


def _undistort(
    x: torch.Tensor, y: torch.Tensor, distortion: torch.Tensor
) -> tuple[torch.Tensor, torch.Tensor]:
    """Invert Brown-Conrady [k1, k2, p1, p2, k3] on normalized coordinates (fixed-point, as OpenCV)."""
    k1, k2, p1, p2, k3 = distortion.tolist()
    xu, yu = x.clone(), y.clone()
    for _ in range(_UNDISTORT_ITERS):
        r2 = xu * xu + yu * yu
        radial = 1 + r2 * (k1 + r2 * (k2 + r2 * k3))
        dx = 2 * p1 * xu * yu + p2 * (r2 + 2 * xu * xu)
        dy = p1 * (r2 + 2 * yu * yu) + 2 * p2 * xu * yu
        xu = (x - dx) / radial
        yu = (y - dy) / radial
    return xu, yu


def _ftheta_polys(cal_dict: Any) -> tuple[torch.Tensor, torch.Tensor] | None:
    """The f-theta (fisheye) forward / backward polynomials of a calibration, or None when it
    is a pinhole camera (no usable polynomials)."""
    fw_poly = getattr(cal_dict, "fw_poly", None)
    bw_poly = getattr(cal_dict, "bw_poly", None)
    if fw_poly is None or bw_poly is None:
        return None
    fw_poly = torch.as_tensor(fw_poly, dtype=torch.float32)
    bw_poly = torch.as_tensor(bw_poly, dtype=torch.float32)
    if not bw_poly.any() or fw_poly.numel() < 2 or fw_poly[1] == 0:
        return None
    return fw_poly, bw_poly


def unproject_pixels(
    cal_dict: Any, u: torch.Tensor, v: torch.Tensor
) -> tuple[torch.Tensor, torch.Tensor]:
    """Unit camera-frame rays [N, 3] and pinhole-normalized coordinates [N, 2] for pixels (u, v).

    Pixel coordinates use the intrinsics' convention (pixel centres at integer coordinates).

    F-theta: the dataset's pinhole approximation has fx = fy = fw_poly[1] in the polynomial's
    pixel units, and image resizes / crops rescale fx, fy (and possibly the polynomials, see
    BaseDataset._resize_calibration) consistently. The polynomial radius of a pixel is therefore
    fw_poly[1] * |((u - cx) / fx, (v - cy) / fy)| in every case, and bw_poly maps it to the
    angle from the optical axis.
    """
    intrinsics = torch.as_tensor(cal_dict.intrinsics, dtype=torch.float32)
    fx, fy, cx, cy = (
        intrinsics[0, 0],
        intrinsics[1, 1],
        intrinsics[0, 2],
        intrinsics[1, 2],
    )
    if fx.abs() < 1e-8 or fy.abs() < 1e-8:
        raise ValueError("Camera focal lengths must be non-zero.")
    x = (u.float() - cx) / fx
    y = (v.float() - cy) / fy
    coords = torch.stack((x, y), dim=-1)

    polys = _ftheta_polys(cal_dict)
    if polys is not None:
        fw_poly, bw_poly = polys
        rho = torch.sqrt(x * x + y * y)
        theta = _polyval(bw_poly, fw_poly[1] * rho)
        safe_rho = rho.clamp_min(1e-12)
        sin_theta = torch.sin(theta)
        rays = torch.stack(
            (
                torch.where(rho > 1e-12, sin_theta * x / safe_rho, torch.zeros_like(x)),
                torch.where(rho > 1e-12, sin_theta * y / safe_rho, torch.zeros_like(y)),
                torch.cos(theta),
            ),
            dim=-1,
        )
        return rays, coords

    distortion = getattr(cal_dict, "distortion", None)
    if distortion is not None and torch.as_tensor(distortion).any():
        x, y = _undistort(x, y, torch.as_tensor(distortion, dtype=torch.float32))
    rays = torch.stack((x, y, torch.ones_like(x)), dim=-1)
    return rays / rays.norm(dim=-1, keepdim=True), coords


def camera_token_features(
    camera: CameraData,
    u: torch.Tensor,
    v: torch.Tensor,
    frame: int,
) -> torch.Tensor:
    """[N, NUM_GEOMETRY_FEATURES] features for pixels (u, v) of one camera image."""
    cal_dict = camera.cal_dict
    rays_cam, coords = unproject_pixels(cal_dict, u, v)
    rotation, translation = camera_to_ego(cal_dict)
    rays_ego = rays_cam @ rotation.T
    n = rays_ego.shape[0]
    return torch.cat(
        (
            rays_ego,
            translation.expand(n, 3),
            coords,
            torch.full((n, 1), float(frame)),
        ),
        dim=-1,
    )


def _merged_grid(
    grid_thw: torch.Tensor, merge_size: int
) -> tuple[torch.Tensor, torch.Tensor, int, int]:
    """Row and column indices [T * H' * W'] of the merged tokens of one image, in Qwen's order."""
    grid_t, grid_h, grid_w = (int(value) for value in grid_thw.tolist())
    if grid_h % merge_size or grid_w % merge_size:
        raise ValueError("Qwen visual grid is not divisible by its merge size.")
    merged_h, merged_w = grid_h // merge_size, grid_w // merge_size
    rows = (
        torch.arange(merged_h)
        .view(1, merged_h, 1)
        .expand(grid_t, merged_h, merged_w)
        .reshape(-1)
    )
    cols = (
        torch.arange(merged_w)
        .view(1, 1, merged_w)
        .expand(grid_t, merged_h, merged_w)
        .reshape(-1)
    )
    return rows, cols, merged_h, merged_w


def _has_calibration(camera: CameraData) -> bool:
    """Whether the camera carries a calibration with intrinsics."""
    return camera.cal_dict is not None and camera.cal_dict.intrinsics is not None


def image_token_features(
    cameras: Sequence[CameraData],
    grid_thw: torch.Tensor,
    merge_size: int,
    frame: int,
    require_calibration: bool = True,
) -> tuple[torch.Tensor, torch.Tensor]:
    """Features [N, F] and validity [N] of the merged tokens of one image made of `cameras`
    concatenated left to right (a single camera for separate-camera prompts).

    Qwen resizes the whole image before patchifying; a token's centre is mapped back to the
    source pixels by its fractional position. Tokens straddling a camera boundary use the
    camera containing their centre.
    """
    rows, cols, merged_h, merged_w = _merged_grid(grid_thw, merge_size)
    heights = {int(camera.image.shape[0]) for camera in cameras}
    if len(heights) != 1:
        raise ValueError("Horizontally concatenated cameras must share a height.")
    height = heights.pop()
    widths = torch.tensor(
        [int(camera.image.shape[1]) for camera in cameras], dtype=torch.float32
    )
    boundaries = torch.cat((torch.zeros(1), widths.cumsum(0)))
    # Continuous position of the token centre in source pixels ([0, W) covers the image).
    global_x = (cols.float() + 0.5) * boundaries[-1] / merged_w
    y = (rows.float() + 0.5) * height / merged_h
    camera_index = torch.bucketize(global_x, boundaries[1:-1], right=True)

    features = torch.zeros(rows.numel(), NUM_GEOMETRY_FEATURES)
    valid = torch.zeros(rows.numel(), dtype=torch.bool)
    for index, camera in enumerate(cameras):
        selected = camera_index == index
        if not selected.any():
            continue
        if not _has_calibration(camera):
            if require_calibration:
                raise ValueError(
                    "Camera geometry needs calibration (cal_dict with intrinsics) for every camera."
                )
            continue
        # Pixel-centre convention of the intrinsics: pixel i spans [i - 0.5, i + 0.5).
        u = global_x[selected] - boundaries[index] - 0.5
        v = y[selected] - 0.5
        features[selected] = camera_token_features(camera, u, v, frame)
        valid[selected] = True
    return features, valid


def token_geometry(
    samples_cameras: Sequence[Sequence[dict[str, CameraData]]],
    camera_sequence: Sequence[str],
    num_past_image_frames: int,
    concat_cameras: bool,
    input_ids: torch.Tensor,
    image_grid_thw: torch.Tensor,
    image_token_id: int,
    merge_size: int,
    require_calibration: bool = True,
) -> tuple[torch.Tensor, torch.Tensor]:
    """Token-aligned geometry for a collated batch.

    Args:
        samples_cameras: Per sample, its camera frames (oldest first), as ``sample.cameras``.
        camera_sequence, num_past_image_frames, concat_cameras: As given to PromptBuilder.
        input_ids: Padded prompt token ids [B, S].
        image_grid_thw: The processor's grids, one row per image, batch-major.
        image_token_id: Id of the image placeholder token.
        merge_size: The vision merger's spatial merge size.
        require_calibration: Raise when a camera lacks calibration; otherwise its tokens are
            marked invalid.

    Returns:
        ``visual_geom`` [B, S, NUM_GEOMETRY_FEATURES] (zeros off the image tokens) and
        ``visual_geom_mask`` [B, S] (True on image tokens with geometry).
    """
    batch_size, seq_len = input_ids.shape
    geom = torch.zeros(batch_size, seq_len, NUM_GEOMETRY_FEATURES)
    mask = torch.zeros(batch_size, seq_len, dtype=torch.bool)
    grid_index = 0
    for b, cameras in enumerate(samples_cameras):
        num_frames = min(len(cameras), num_past_image_frames)
        feats, valids = [], []
        frames = list(cameras[len(cameras) - num_frames :])
        # (frame_id, cameras of one image), in the prompt's image order.
        images = (
            [
                (t - num_frames + 1, [frame[name] for name in camera_sequence])
                for t, frame in enumerate(frames)
            ]
            if concat_cameras
            else [
                (t - num_frames + 1, [frame[name]])
                for name in camera_sequence
                for t, frame in enumerate(frames)
            ]
        )
        for frame_id, image_cameras in images:
            if grid_index >= image_grid_thw.shape[0]:
                raise ValueError("Fewer image grids than images in the prompts.")
            f, v = image_token_features(
                image_cameras,
                image_grid_thw[grid_index],
                merge_size,
                frame_id,
                require_calibration,
            )
            grid_index += 1
            feats.append(f)
            valids.append(v)
        positions = (input_ids[b] == image_token_id).nonzero().flatten()
        num_tokens = sum(f.shape[0] for f in feats)
        if positions.numel() != num_tokens:
            raise ValueError(
                f"Sample {b}: {positions.numel()} image tokens in the prompt but {num_tokens} "
                "image patches from the camera layout."
            )
        if num_tokens:
            geom[b, positions] = torch.cat(feats)
            mask[b, positions] = torch.cat(valids)
    if grid_index != image_grid_thw.shape[0]:
        raise ValueError(
            f"Used {grid_index} image grids but the processor returned {image_grid_thw.shape[0]}."
        )
    if not torch.isfinite(geom).all():
        raise ValueError("Non-finite camera geometry; check the calibrations.")
    return geom, mask
