"""Camera geometry for the FM-VLA action expert (expert_cfg.camera_geometry)."""

import math

import numpy as np
import pytest
import torch

from ad_vla.dataset.base_dataset import BaseDataset
from ad_vla.dataset.data_types import CalibrationDict, CameraData
from ad_vla.models.layers.kv_prefix_action_expert import (
    CameraGeometryKV,
    KVPrefixActionExpert,
)
from ad_vla.utils.camera_geometry import (
    NUM_GEOMETRY_FEATURES,
    camera_token_features,
    token_geometry,
    unproject_pixels,
)
from ad_vla.utils.img_utils import update_calibration_for_crop_resize

# Optical (right, down, forward) camera axes -> ego (forward, left, up), for a camera looking forward.
R_FRONT = torch.tensor([[0.0, 0.0, 1.0], [-1.0, 0.0, 0.0], [0.0, -1.0, 0.0]])


def _rz(yaw: float) -> torch.Tensor:
    c, s = math.cos(yaw), math.sin(yaw)
    return torch.tensor([[c, -s, 0.0], [s, c, 0.0], [0.0, 0.0, 1.0]])


def _pinhole(
    yaw=0.0, size=(64, 96), f=50.0, translation=(1.5, 0.0, 1.6), distortion=None
):
    h, w = size
    return CalibrationDict(
        intrinsics=torch.tensor(
            [[f, 0.0, (w - 1) / 2], [0.0, f, (h - 1) / 2], [0.0, 0.0, 1.0]]
        ),
        distortion=None
        if distortion is None
        else torch.tensor(distortion, dtype=torch.float32),
        rotation=_rz(yaw) @ R_FRONT,
        translation=torch.tensor(translation),
        original_img_size=size,
        is_flu=False,
    )


def _ftheta_cal(size=(1080, 1920), f=1000.0, c3=-40.0):
    """Synthetic f-theta camera, built like PhysicalAIDataset._convert_calibration."""
    h, w = size
    fw_poly = np.array([0.0, f, 0.0, c3, 0.0], dtype=np.float32)
    theta = np.linspace(0.0, 1.2, 400)
    radius = np.polyval(fw_poly[::-1], theta)
    bw_poly = np.polyfit(radius, theta, 4)[::-1].astype(np.float32)
    return {
        "intrinsics": torch.tensor([[f, 0.0, w / 2], [0.0, f, h / 2], [0.0, 0.0, 1.0]]),
        "distortion": torch.zeros(5),
        "rotation": R_FRONT.clone(),
        "translation": torch.tensor([1.7, 0.0, 1.5]),
        "original_img_size": (h, w),
        "is_flu": False,
        "fw_poly": torch.from_numpy(fw_poly),
        "bw_poly": torch.from_numpy(bw_poly),
    }


def _camera(cal, size=(64, 96)):
    return CameraData(image=torch.zeros(*size, 3, dtype=torch.uint8), cal_dict=cal)


@pytest.mark.parametrize("yaw", [0.0, math.pi / 2, -math.pi / 2, 0.7])
def test_principal_point_ray_is_the_optical_axis(yaw):
    cal = _pinhole(yaw=yaw)
    cx, cy = cal.intrinsics[0, 2], cal.intrinsics[1, 2]
    feats = camera_token_features(_camera(cal), cx.view(1), cy.view(1), frame=0)
    assert torch.allclose(feats[0, :3], cal.rotation[:, 2], atol=1e-6)
    assert torch.allclose(feats[0, 3:6], cal.translation)
    assert torch.allclose(feats[0, 6:8], torch.zeros(2), atol=1e-6)


def test_ray_sides_in_ego_frame():
    front, left, right = _pinhole(0.0), _pinhole(math.pi / 2), _pinhole(-math.pi / 2)
    cx, cy = float(front.intrinsics[0, 2]), float(front.intrinsics[1, 2])
    u = torch.tensor([cx + 20, cx - 20, cx, cx])
    v = torch.tensor([cy, cy, cy + 20, cy - 20])
    rays = camera_token_features(_camera(front), u, v, 0)[:, :3]
    assert rays[0, 1] < 0 and rays[1, 1] > 0  # image right -> ego right (-y)
    assert rays[2, 2] < 0 and rays[3, 2] > 0  # image down -> ego down (-z)
    assert (rays[:, 0] > 0).all()
    centre = torch.tensor([cx]), torch.tensor([cy])
    assert (
        camera_token_features(_camera(left), *centre, 0)[0, 1] > 0.99
    )  # looks along +y
    assert camera_token_features(_camera(right), *centre, 0)[0, 1] < -0.99


def test_pinhole_rays_are_unit_and_match_the_intrinsics():
    cal = _pinhole()
    u, v = torch.tensor([0.0, 95.0, 30.0]), torch.tensor([0.0, 63.0, 10.0])
    rays, coords = unproject_pixels(cal, u, v)
    assert torch.allclose(rays.norm(dim=-1), torch.ones(3), atol=1e-6)
    expected = torch.stack(
        ((u - cal.intrinsics[0, 2]) / 50.0, (v - cal.intrinsics[1, 2]) / 50.0), -1
    )
    assert torch.allclose(coords, expected)
    assert torch.allclose(rays[:, :2] / rays[:, 2:], expected, atol=1e-6)


def test_distortion_is_removed():
    distortion = [-0.2, 0.05, 0.001, -0.002, 0.0]
    cal = _pinhole(distortion=distortion)
    x, y = torch.tensor([0.3, -0.25]), torch.tensor([-0.2, 0.1])
    k1, k2, p1, p2, k3 = distortion
    r2 = x * x + y * y
    radial = 1 + k1 * r2 + k2 * r2**2 + k3 * r2**3
    xd = x * radial + 2 * p1 * x * y + p2 * (r2 + 2 * x * x)
    yd = y * radial + p1 * (r2 + 2 * y * y) + 2 * p2 * x * y
    u = xd * 50.0 + cal.intrinsics[0, 2]
    v = yd * 50.0 + cal.intrinsics[1, 2]
    rays, _ = unproject_pixels(cal, u, v)
    assert torch.allclose(rays[:, 0] / rays[:, 2], x, atol=1e-5)
    assert torch.allclose(rays[:, 1] / rays[:, 2], y, atol=1e-5)


def test_ftheta_angle_follows_the_polynomial():
    cal = CalibrationDict(**_ftheta_cal())
    cx, cy = float(cal.intrinsics[0, 2]), float(cal.intrinsics[1, 2])
    theta = torch.tensor([0.0, 0.3, 0.7, 1.0])
    radius = torch.from_numpy(
        np.polyval(cal.fw_poly.numpy()[::-1], theta.numpy())
    ).float()
    rays, _ = unproject_pixels(cal, cx + radius, torch.full_like(radius, cy))
    assert torch.allclose(torch.acos(rays[:, 2].clamp(-1, 1)), theta, atol=2e-3)
    assert (rays[1:, 0] > 0).all() and torch.allclose(
        rays[:, 1], torch.zeros(4), atol=1e-6
    )
    # Wide angles differ clearly from the dataset's pinhole approximation (fx = fw_poly[1]).
    pinhole_theta = torch.atan(radius / cal.intrinsics[0, 0])
    assert (torch.acos(rays[3, 2]) - pinhole_theta[3]).abs() > 0.1


def test_ftheta_rays_survive_both_resize_paths():
    raw = _ftheta_cal()
    (h, w), (oh, ow) = raw["original_img_size"], (448, 768)
    sx, sy = ow / w, oh / h
    # BaseDataset resize rescales the polynomials; the crop-resize transform keeps them.
    by_dataset = CalibrationDict(
        **BaseDataset._resize_calibration(raw, (h, w), (oh, ow))
    )
    by_transform = CalibrationDict(
        **update_calibration_for_crop_resize(
            raw, top=0, left=0, crop_h=h, crop_w=w, out_h=oh, out_w=ow
        )
    )
    raw = CalibrationDict(**raw)
    u_raw, v_raw = (
        torch.tensor([100.0, 960.0, 1800.0]),
        torch.tensor([50.0, 540.0, 1000.0]),
    )
    # Same physical point in resized pixel coordinates, in the convention the repository's
    # intrinsics updates use (coordinates scale by sx, sy).
    u, v = u_raw * sx, v_raw * sy
    reference, _ = unproject_pixels(raw, u_raw, v_raw)
    for cal in (by_dataset, by_transform):
        rays, _ = unproject_pixels(cal, u, v)
        assert torch.allclose(rays, reference, atol=1e-4)


def _fake_prompt(token_counts, image_token_id=7, pad=3):
    """[1, S] input ids: left padding, then text and image tokens."""
    ids = [0] * pad + [1, 2]
    for count in token_counts:
        ids += [5] + [image_token_id] * count + [6]
    ids += [8, 9]
    return torch.tensor([ids])


def _frame(cals, size=(64, 96)):
    return {name: _camera(cal, size) for name, cal in cals.items()}


def test_concatenated_and_separate_layouts_agree():
    names = ["FRONT_LEFT", "FRONT", "FRONT_RIGHT"]
    frame = _frame(
        {
            "FRONT_LEFT": _pinhole(0.9),
            "FRONT": _pinhole(0.0),
            "FRONT_RIGHT": _pinhole(-0.9),
        }
    )
    # 64 x 96 cameras, 16 px patches merged 2 x 2: 2 x 3 merged tokens each; concatenated 2 x 9.
    separate_ids = _fake_prompt([6, 6, 6])
    geom_sep, mask_sep = token_geometry(
        [[frame]], names, 1, False, separate_ids, torch.tensor([[1, 4, 6]] * 3), 7, 2
    )
    concat_ids = _fake_prompt([18])
    geom_cat, mask_cat = token_geometry(
        [[frame]], names, 1, True, concat_ids, torch.tensor([[1, 4, 18]]), 7, 2
    )
    sep = geom_sep[0][mask_sep[0]].view(
        3, 2, 3, NUM_GEOMETRY_FEATURES
    )  # camera, row, col
    cat = geom_cat[0][mask_cat[0]].view(
        2, 3, 3, NUM_GEOMETRY_FEATURES
    )  # row, camera, col
    assert torch.allclose(sep, cat.permute(1, 0, 2, 3), atol=1e-5)
    assert mask_sep[0].sum() == 18 and (mask_sep[0] == (separate_ids[0] == 7)).all()
    assert (geom_sep[0][~mask_sep[0]] == 0).all()


def test_frames_are_indexed_newest_zero():
    cals = {"FRONT": _pinhole()}
    frames = [_frame(cals), _frame(cals), _frame(cals)]
    geom, mask = token_geometry(
        [frames],
        ["FRONT"],
        2,
        True,
        _fake_prompt([6, 6]),
        torch.tensor([[1, 4, 6]] * 2),
        7,
        2,
    )
    assert geom[0][mask[0], 8].tolist() == [-1.0] * 6 + [0.0] * 6


def test_token_count_mismatch_raises():
    frame = _frame({"FRONT": _pinhole()})
    with pytest.raises(ValueError, match="image tokens"):
        token_geometry(
            [[frame]],
            ["FRONT"],
            1,
            True,
            _fake_prompt([5]),
            torch.tensor([[1, 4, 6]]),
            7,
            2,
        )


def test_missing_calibration():
    frame = {"FRONT": CameraData(image=torch.zeros(64, 96, 3, dtype=torch.uint8))}
    args = (
        [[frame]],
        ["FRONT"],
        1,
        True,
        _fake_prompt([6]),
        torch.tensor([[1, 4, 6]]),
        7,
        2,
    )
    with pytest.raises(ValueError, match="calibration"):
        token_geometry(*args)
    geom, mask = token_geometry(*args, require_calibration=False)
    assert not mask.any() and (geom == 0).all()


def _kv(batch=2, seq=10, heads=2, head_dim=16, blocks=3, dtype=torch.bfloat16):
    return [
        (
            torch.randn(batch, heads, seq, head_dim).to(dtype),
            torch.randn(batch, heads, seq, head_dim).to(dtype),
        )
        for _ in range(blocks)
    ]


def test_geometry_kv_is_exactly_the_identity_at_init():
    torch.manual_seed(0)
    geometry = CameraGeometryKV(
        3, 2, 16, rotary_dim=4, num_features=NUM_GEOMETRY_FEATURES
    )
    prompt_kv = _kv()
    geom = torch.randn(2, 10, NUM_GEOMETRY_FEATURES)
    mask = torch.rand(2, 10) > 0.5
    for (k, v), (k2, v2) in zip(prompt_kv, geometry(prompt_kv, geom, mask)):
        assert torch.equal(k2, k.float()) and torch.equal(v2, v.float())


def test_geometry_kv_only_touches_image_tokens_and_non_rotary_key_dims():
    torch.manual_seed(0)
    geometry = CameraGeometryKV(
        3, 2, 16, rotary_dim=4, num_features=NUM_GEOMETRY_FEATURES
    )
    for p in geometry.parameters():
        torch.nn.init.normal_(p)
    prompt_kv = _kv()
    geom = torch.randn(2, 10, NUM_GEOMETRY_FEATURES)
    mask = torch.zeros(2, 10, dtype=torch.bool)
    mask[:, 3:7] = True
    for (k, v), (k2, v2) in zip(prompt_kv, geometry(prompt_kv, geom, mask)):
        k, v = k.float(), v.float()
        assert torch.equal(k2[:, :, ~mask[0]], k[:, :, ~mask[0]]) and torch.equal(
            v2[:, :, ~mask[0]], v[:, :, ~mask[0]]
        )
        assert torch.equal(k2[..., :4], k[..., :4])
        assert not torch.equal(k2[:, :, mask[0], 4:], k[:, :, mask[0], 4:])
        assert not torch.equal(v2[:, :, mask[0]], v[:, :, mask[0]])


def _expert(camera_geometry_cfg=None):
    return KVPrefixActionExpert(
        num_poses=5,
        action_dim=2,
        hidden_size=32,
        intermediate_size=64,
        num_layers=3,
        num_heads=4,
        num_kv_heads=2,
        head_dim=16,
        time_embed_dim=16,
        rms_norm_eps=1e-6,
        past_traj_cfg={
            "num_poses": 4,
            "interval_length": 0.5,
            "features": ("xy", "vel"),
            "mode": "per_pose",
            "normalization": 20.0,
            "missing_token": True,
        },
        camera_geometry_cfg=camera_geometry_cfg,
    )


def test_expert_without_geometry_is_unchanged_and_geometry_is_built_last():
    torch.manual_seed(0)
    plain = _expert()
    torch.manual_seed(0)
    with_geometry = _expert({"rotary_dim": 4, "num_features": NUM_GEOMETRY_FEATURES})
    assert plain.camera_geometry is None
    assert not any(k.startswith("camera_geometry") for k in plain.state_dict())
    geometry_keys = {
        k for k in with_geometry.state_dict() if k.startswith("camera_geometry.")
    }
    assert geometry_keys and set(with_geometry.state_dict()) - geometry_keys == set(
        plain.state_dict()
    )
    for key, value in plain.state_dict().items():
        assert torch.equal(value, with_geometry.state_dict()[key]), key


def test_geometry_kv_has_no_buffers():
    # from_pretrained re-creates unloaded non-persistent buffers as uninitialized memory, so a
    # warm start (checkpoint without geometry) must not depend on any buffer.
    geometry = CameraGeometryKV(
        3, 2, 16, rotary_dim=4, num_features=NUM_GEOMETRY_FEATURES
    )
    assert not list(geometry.buffers())
