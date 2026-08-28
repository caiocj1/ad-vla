import copy
import math
from dataclasses import dataclass

import numpy as np
import torch
import torch.nn as nn
import torch.nn.functional as F
from torchvision.transforms import v2
from torchvision.transforms.v2 import functional as TVF


def _clone_calibration(cal_dict):
    if cal_dict is None:
        return None
    if hasattr(cal_dict, "model_copy"):
        return cal_dict.model_copy(deep=True)
    return copy.deepcopy(cal_dict)


def _cal_get(cal_dict, key, default=None):
    if isinstance(cal_dict, dict):
        return cal_dict.get(key, default)
    return getattr(cal_dict, key, default)


def _cal_set(cal_dict, key, value):
    if isinstance(cal_dict, dict):
        cal_dict[key] = value
    else:
        setattr(cal_dict, key, value)


def _transform_intrinsics_for_crop_resize(
    intrinsics,
    *,
    top: int,
    left: int,
    crop_h: int,
    crop_w: int,
    out_h: int,
    out_w: int,
):
    sx = float(out_w) / float(crop_w)
    sy = float(out_h) / float(crop_h)

    if isinstance(intrinsics, torch.Tensor):
        pixel_transform = intrinsics.new_tensor(
            [
                [sx, 0.0, -sx * float(left)],
                [0.0, sy, -sy * float(top)],
                [0.0, 0.0, 1.0],
            ]
        )
        return pixel_transform @ intrinsics.clone()

    intrinsics_np = np.asarray(intrinsics).copy()
    pixel_transform = np.array(
        [
            [sx, 0.0, -sx * float(left)],
            [0.0, sy, -sy * float(top)],
            [0.0, 0.0, 1.0],
        ],
        dtype=intrinsics_np.dtype,
    )
    return pixel_transform @ intrinsics_np


def update_calibration_for_crop_resize(
    cal_dict,
    *,
    top: int,
    left: int,
    crop_h: int,
    crop_w: int,
    out_h: int,
    out_w: int,
):
    new_cal = _clone_calibration(cal_dict)
    if new_cal is None:
        return None

    intrinsics = _cal_get(new_cal, "intrinsics")
    if intrinsics is not None:
        intrinsics = _transform_intrinsics_for_crop_resize(
            intrinsics,
            top=top,
            left=left,
            crop_h=crop_h,
            crop_w=crop_w,
            out_h=out_h,
            out_w=out_w,
        )
        _cal_set(new_cal, "intrinsics", intrinsics)

    _cal_set(new_cal, "original_img_size", (int(out_h), int(out_w)))
    return new_cal


def _ego_delta_rotation(
    yaw_rad: float,
    pitch_rad: float,
    roll_rad: float,
    *,
    like,
):
    """Create ``Rz(yaw) @ Ry(pitch) @ Rx(roll)`` in the ego FLU frame."""
    cy, sy = math.cos(yaw_rad), math.sin(yaw_rad)
    cp, sp = math.cos(pitch_rad), math.sin(pitch_rad)
    cr, sr = math.cos(roll_rad), math.sin(roll_rad)
    values = [
        [cy * cp, cy * sp * sr - sy * cr, cy * sp * cr + sy * sr],
        [sy * cp, sy * sp * sr + cy * cr, sy * sp * cr - cy * sr],
        [-sp, cp * sr, cp * cr],
    ]
    if isinstance(like, torch.Tensor):
        return like.new_tensor(values)
    return np.asarray(values, dtype=np.asarray(like).dtype)


def update_calibration_for_camera_augmentation(
    cal_dict,
    *,
    top: int,
    left: int,
    crop_h: int,
    crop_w: int,
    out_h: int,
    out_w: int,
    yaw_rad: float = 0.0,
    pitch_rad: float = 0.0,
    roll_rad: float = 0.0,
):
    """Update calibration for crop-resize and rotation about the camera center.

    The repository stores camera-to-ego extrinsics, so the synthetic mounting
    rotation left-multiplies the existing rotation. Translation and distortion
    are preserved because the camera center does not move and the lens model is
    unchanged.
    """
    new_cal = update_calibration_for_crop_resize(
        cal_dict,
        top=top,
        left=left,
        crop_h=crop_h,
        crop_w=crop_w,
        out_h=out_h,
        out_w=out_w,
    )
    if new_cal is None:
        return None

    rotation = _cal_get(new_cal, "rotation")
    if rotation is not None and any(
        angle != 0.0 for angle in (yaw_rad, pitch_rad, roll_rad)
    ):
        delta = _ego_delta_rotation(
            yaw_rad,
            pitch_rad,
            roll_rad,
            like=rotation,
        )
        _cal_set(new_cal, "rotation", delta @ rotation)
    return new_cal


# Convert legacy camera-local FLU vectors into the canonical optical/RDF basis:
# x right, y down, z forward.
_FLU_FROM_RDF = torch.tensor(
    [
        [0.0, 0.0, 1.0],
        [-1.0, 0.0, 0.0],
        [0.0, -1.0, 0.0],
    ],
    dtype=torch.float32,
)


def _camera_to_ego_rdf_rotation(cal_dict):
    """Return a camera-to-ego rotation expressed in optical/RDF axes."""
    rotation = _cal_get(cal_dict, "rotation")
    if rotation is None:
        raise ValueError("Calibration has no rotation.")
    if not bool(_cal_get(cal_dict, "is_flu", False)):
        return rotation

    if isinstance(rotation, torch.Tensor):
        flu_from_rdf = _FLU_FROM_RDF.to(
            device=rotation.device,
            dtype=rotation.dtype,
        )
    else:
        rotation = np.asarray(rotation)
        flu_from_rdf = _FLU_FROM_RDF.numpy().astype(rotation.dtype, copy=False)
    return rotation @ flu_from_rdf


def _dst_to_src_rotation_homography(src_cal, dst_cal):
    """Build the destination-to-source pinhole homography for pure rotation."""
    src_intrinsics = _cal_get(src_cal, "intrinsics")
    dst_intrinsics = _cal_get(dst_cal, "intrinsics")
    if src_intrinsics is None or dst_intrinsics is None:
        raise ValueError("Camera rotation augmentation requires intrinsics.")

    src_rotation = _camera_to_ego_rdf_rotation(src_cal)
    dst_rotation = _camera_to_ego_rdf_rotation(dst_cal)
    if isinstance(src_intrinsics, torch.Tensor):
        device, dtype = src_intrinsics.device, src_intrinsics.dtype
        dst_intrinsics = torch.as_tensor(
            dst_intrinsics,
            device=device,
            dtype=dtype,
        )
        src_rotation = torch.as_tensor(src_rotation, device=device, dtype=dtype)
        dst_rotation = torch.as_tensor(dst_rotation, device=device, dtype=dtype)
        return (
            src_intrinsics
            @ src_rotation.transpose(-1, -2)
            @ dst_rotation
            @ torch.linalg.inv(dst_intrinsics)
        )

    src_intrinsics = np.asarray(src_intrinsics)
    dst_intrinsics = np.asarray(dst_intrinsics)
    src_rotation = np.asarray(src_rotation)
    dst_rotation = np.asarray(dst_rotation)
    return (
        src_intrinsics
        @ src_rotation.T
        @ dst_rotation
        @ np.linalg.inv(dst_intrinsics)
    )


def _warp_hwc_by_dst_to_src_homography(
    img: torch.Tensor,
    homography,
    *,
    out_h: int,
    out_w: int,
    padding_mode: str = "border",
) -> torch.Tensor:
    """Warp an HWC image with a destination-to-source homography."""
    if img.ndim != 3 or img.shape[-1] != 3:
        raise ValueError(f"Expected HWC RGB image, got {tuple(img.shape)}")

    src_h, src_w, _ = img.shape
    original_dtype = img.dtype
    was_float = img.is_floating_point()
    image_chw = (img if was_float else img.float()).permute(2, 0, 1).unsqueeze(0)
    homography = torch.as_tensor(
        homography,
        device=image_chw.device,
        dtype=image_chw.dtype,
    )

    ys, xs = torch.meshgrid(
        torch.arange(out_h, device=image_chw.device, dtype=image_chw.dtype),
        torch.arange(out_w, device=image_chw.device, dtype=image_chw.dtype),
        indexing="ij",
    )
    dst_pixels = torch.stack((xs, ys, torch.ones_like(xs)), dim=-1)
    src_pixels = dst_pixels @ homography.transpose(0, 1)
    depth = src_pixels[..., 2]
    valid_depth = depth > 1e-8
    safe_depth = torch.where(valid_depth, depth, torch.ones_like(depth))
    src_x = src_pixels[..., 0] / safe_depth
    src_y = src_pixels[..., 1] / safe_depth

    # Pixel-center normalization for grid_sample(..., align_corners=False).
    x_norm = 2.0 * (src_x + 0.5) / float(src_w) - 1.0
    y_norm = 2.0 * (src_y + 0.5) / float(src_h) - 1.0
    outside = torch.full_like(x_norm, 2.0)
    x_norm = torch.where(valid_depth, x_norm, outside)
    y_norm = torch.where(valid_depth, y_norm, outside)
    grid = torch.stack((x_norm, y_norm), dim=-1).unsqueeze(0)

    warped = F.grid_sample(
        image_chw,
        grid,
        mode="bilinear",
        padding_mode=padding_mode,
        align_corners=False,
    )[0].permute(1, 2, 0)
    if not was_float:
        if original_dtype == torch.uint8:
            warped = warped.round().clamp(0, 255)
        warped = warped.to(original_dtype)
    return warped


@dataclass(frozen=True, slots=True)
class CameraGeometryAugParams:
    top: int
    left: int
    crop_h: int
    crop_w: int
    yaw_rad: float
    pitch_rad: float
    roll_rad: float


@dataclass(frozen=True, slots=True)
class ColorAugParams:
    apply: bool
    brightness: float = 1.0
    contrast: float = 1.0
    saturation: float = 1.0
    hue: float = 0.0


class ImageResize(nn.Module):
    def __init__(self, size: list[int]):
        super().__init__()
        self.h, self.w = size
        self.resize = v2.Compose(
            [
                v2.ToImage(),
                v2.Resize((self.h, self.w)),
            ]
        )

    def forward(self, img):
        return self.resize(img)


class CenterCropResize(nn.Module):
    requires_calibration = True

    def __init__(
        self,
        size: list[int],
        p_transform: float,
        h_crop_ratio: float,
        w_crop_ratio: float,
    ):
        super().__init__()
        self.h, self.w = size
        self.resize = v2.Resize((self.h, self.w))
        self.p_transform = p_transform
        self.h_crop_ratio = h_crop_ratio
        self.w_crop_ratio = w_crop_ratio
        self.jitter = v2.ColorJitter(
            brightness=0.3, contrast=0.3, saturation=0.3, hue=0.05
        )

    def _sample_crop(self, h: int, w: int):
        w_crop_ratio = float(torch.rand(()) * self.w_crop_ratio)
        h_crop_ratio = float(torch.rand(()) * self.h_crop_ratio)

        top = int(h * h_crop_ratio / 2.0)
        left = int(w * w_crop_ratio / 2.0)
        crop_h = h - 2 * top
        crop_w = w - 2 * left
        return top, left, crop_h, crop_w

    def forward(self, img, cal_dict):
        h, w, c = img.shape
        if torch.rand(()) < self.p_transform:
            top, left, crop_h, crop_w = self._sample_crop(h, w)
            apply_jitter = True
        else:
            top, left, crop_h, crop_w = 0, 0, h, w
            apply_jitter = False

        cropped = img[top : top + crop_h, left : left + crop_w, :]
        resized = self.resize(cropped.permute(2, 0, 1)).permute(1, 2, 0)
        if apply_jitter:
            resized = self.jitter(resized.permute(2, 0, 1)).permute(1, 2, 0)
        new_cal = update_calibration_for_crop_resize(
            cal_dict,
            top=top,
            left=left,
            crop_h=crop_h,
            crop_w=crop_w,
            out_h=self.h,
            out_w=self.w,
        )
        return resized, new_cal


class CameraRigAugment(nn.Module):
    """Augment camera projection geometry and keep calibration synchronized.

    Geometry augmentation applies an aspect-preserving random crop, resizes to
    ``size``, and optionally synthesizes a small rotation about the camera's
    optical center. Appearance jitter does not alter calibration.
    """

    requires_calibration = True

    def __init__(
        self,
        size: list[int],
        p_geometry: float = 0.75,
        max_crop_ratio: float = 0.10,
        max_yaw_deg: float = 3.0,
        max_pitch_deg: float = 2.0,
        max_roll_deg: float = 1.0,
        p_color: float = 0.80,
        brightness: float = 0.15,
        contrast: float = 0.15,
        saturation: float = 0.15,
        hue: float = 0.03,
        padding_mode: str = "border",
    ):
        super().__init__()
        if len(size) != 2 or any(int(value) <= 0 for value in size):
            raise ValueError("size must contain two positive integers [height, width].")
        if not 0.0 <= p_geometry <= 1.0:
            raise ValueError("p_geometry must be between zero and one.")
        if not 0.0 <= p_color <= 1.0:
            raise ValueError("p_color must be between zero and one.")
        if not 0.0 <= max_crop_ratio < 1.0:
            raise ValueError("max_crop_ratio must be in [0, 1).")
        if min(max_yaw_deg, max_pitch_deg, max_roll_deg) < 0.0:
            raise ValueError("Maximum rotation magnitudes must be non-negative.")
        if min(brightness, contrast, saturation) < 0.0:
            raise ValueError("Color jitter magnitudes must be non-negative.")
        if not 0.0 <= hue <= 0.5:
            raise ValueError("hue must be between zero and 0.5.")
        if padding_mode not in {"zeros", "border", "reflection"}:
            raise ValueError(
                "padding_mode must be one of: zeros, border, reflection."
            )

        self.h, self.w = (int(value) for value in size)
        self.resize = v2.Resize((self.h, self.w), antialias=True)
        self.p_geometry = float(p_geometry)
        self.max_crop_ratio = float(max_crop_ratio)
        self.max_yaw_deg = float(max_yaw_deg)
        self.max_pitch_deg = float(max_pitch_deg)
        self.max_roll_deg = float(max_roll_deg)
        self.p_color = float(p_color)
        self.brightness = float(brightness)
        self.contrast = float(contrast)
        self.saturation = float(saturation)
        self.hue = float(hue)
        self.padding_mode = padding_mode

    @staticmethod
    def _uniform(low: float, high: float) -> float:
        return float(torch.empty(()).uniform_(low, high))

    def sample_geometry_params(self, h: int, w: int) -> CameraGeometryAugParams:
        """Sample one crop and ego-frame mounting rotation."""
        if float(torch.rand(())) >= self.p_geometry:
            return CameraGeometryAugParams(
                top=0,
                left=0,
                crop_h=h,
                crop_w=w,
                yaw_rad=0.0,
                pitch_rad=0.0,
                roll_rad=0.0,
            )

        crop_fraction = self._uniform(0.0, self.max_crop_ratio)
        crop_h = max(1, int(round(h * (1.0 - crop_fraction))))
        crop_w = max(1, int(round(w * (1.0 - crop_fraction))))
        max_top = h - crop_h
        max_left = w - crop_w
        top = int(torch.randint(max_top + 1, ()).item()) if max_top > 0 else 0
        left = int(torch.randint(max_left + 1, ()).item()) if max_left > 0 else 0
        return CameraGeometryAugParams(
            top=top,
            left=left,
            crop_h=crop_h,
            crop_w=crop_w,
            yaw_rad=math.radians(
                self._uniform(-self.max_yaw_deg, self.max_yaw_deg)
            ),
            pitch_rad=math.radians(
                self._uniform(-self.max_pitch_deg, self.max_pitch_deg)
            ),
            roll_rad=math.radians(
                self._uniform(-self.max_roll_deg, self.max_roll_deg)
            ),
        )

    def sample_color_params(self) -> ColorAugParams:
        """Sample deterministic factors to apply together to one image."""
        if float(torch.rand(())) >= self.p_color:
            return ColorAugParams(apply=False)
        return ColorAugParams(
            apply=True,
            brightness=self._uniform(1.0 - self.brightness, 1.0 + self.brightness),
            contrast=self._uniform(1.0 - self.contrast, 1.0 + self.contrast),
            saturation=self._uniform(1.0 - self.saturation, 1.0 + self.saturation),
            hue=self._uniform(-self.hue, self.hue),
        )

    @staticmethod
    def _apply_color(img_hwc: torch.Tensor, params: ColorAugParams) -> torch.Tensor:
        if not params.apply:
            return img_hwc
        image_chw = img_hwc.permute(2, 0, 1)
        image_chw = TVF.adjust_brightness(image_chw, params.brightness)
        image_chw = TVF.adjust_contrast(image_chw, params.contrast)
        image_chw = TVF.adjust_saturation(image_chw, params.saturation)
        image_chw = TVF.adjust_hue(image_chw, params.hue)
        return image_chw.permute(1, 2, 0)

    @staticmethod
    def _validate_geometry_params(
        params: CameraGeometryAugParams,
        image_h: int,
        image_w: int,
    ) -> None:
        if params.top < 0 or params.left < 0:
            raise ValueError("Crop offsets must be non-negative.")
        if params.crop_h <= 0 or params.crop_w <= 0:
            raise ValueError("Crop dimensions must be positive.")
        if params.top + params.crop_h > image_h:
            raise ValueError("Vertical crop exceeds the source image.")
        if params.left + params.crop_w > image_w:
            raise ValueError("Horizontal crop exceeds the source image.")

    def forward(
        self,
        img,
        cal_dict,
        *,
        geometry_params: CameraGeometryAugParams | None = None,
        color_params: ColorAugParams | None = None,
    ):
        if not isinstance(img, torch.Tensor):
            img = torch.as_tensor(img)
        if img.ndim != 3 or img.shape[-1] != 3:
            raise ValueError(f"Expected HWC RGB image, got {tuple(img.shape)}")

        image_h, image_w, _ = img.shape
        if geometry_params is None:
            geometry_params = self.sample_geometry_params(image_h, image_w)
        if color_params is None:
            color_params = self.sample_color_params()
        self._validate_geometry_params(geometry_params, image_h, image_w)
        params = geometry_params

        cropped = img[
            params.top : params.top + params.crop_h,
            params.left : params.left + params.crop_w,
            :,
        ]
        resized = self.resize(cropped.permute(2, 0, 1)).permute(1, 2, 0)

        src_cal = update_calibration_for_crop_resize(
            cal_dict,
            top=params.top,
            left=params.left,
            crop_h=params.crop_h,
            crop_w=params.crop_w,
            out_h=self.h,
            out_w=self.w,
        )
        has_rotation = any(
            angle != 0.0
            for angle in (params.yaw_rad, params.pitch_rad, params.roll_rad)
        )
        if has_rotation:
            dst_cal = update_calibration_for_camera_augmentation(
                cal_dict,
                top=params.top,
                left=params.left,
                crop_h=params.crop_h,
                crop_w=params.crop_w,
                out_h=self.h,
                out_w=self.w,
                yaw_rad=params.yaw_rad,
                pitch_rad=params.pitch_rad,
                roll_rad=params.roll_rad,
            )
            homography = _dst_to_src_rotation_homography(src_cal, dst_cal)
            resized = _warp_hwc_by_dst_to_src_homography(
                resized,
                homography,
                out_h=self.h,
                out_w=self.w,
                padding_mode=self.padding_mode,
            )
            new_cal = dst_cal
        else:
            new_cal = src_cal

        return self._apply_color(resized, color_params), new_cal
