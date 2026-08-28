import numpy as np
import cv2
import torch
import matplotlib.pyplot as plt

from ad_vla.dataset.data_types import CalibrationDict, E2EDataSample, CameraData


def project_trajectory_to_image(
    image: torch.Tensor,
    cal_dict: CalibrationDict,
    ego_traj: torch.Tensor,
    color="r",
    z_offset: float | None = None,
):
    """
    Projects an ego-frame trajectory onto an image.

    Calibration rotations are expected to be saved in optical/RDF camera axes:
    x right, y down, z forward.

    Args:
        image: Tensor (H, W, C)
        cal_dict: Calibration data
        ego_traj: Trajectory points (N, 2) or (N, 3)
    """

    def to_numpy(t):
        if isinstance(t, torch.Tensor):
            return t.detach().cpu().numpy()
        return t

    # 1. Image Prep
    img_np = to_numpy(image)
    if img_np.ndim == 3 and img_np.shape[0] == 3:
        img_np = np.transpose(img_np, (1, 2, 0))
    img_draw = np.ascontiguousarray(img_np.copy()).astype(np.uint8)
    curr_h, curr_w = img_draw.shape[:2]

    if cal_dict.is_flu:
        raise ValueError(
            "project_trajectory_to_image expects optical/RDF calibrations "
            "(is_flu=False). Rebuild legacy FLU samples with the normalized "
            "dataset loaders before plotting."
        )

    # 2. Extrinsics (Ego -> optical/RDF camera frame)
    R_s2l = to_numpy(cal_dict.rotation)
    t_s2l = to_numpy(cal_dict.translation)
    R_l2s = R_s2l.T
    t_l2s = -R_l2s @ t_s2l

    # 3. Intrinsics
    K = to_numpy(cal_dict.intrinsics).copy().astype(np.float32)

    # 4. Prepare Object Points (Lidar Frame)
    if z_offset is None:
        z_offset = float(getattr(cal_dict, "projection_z_offset", 0.0))

    traj_np = to_numpy(ego_traj)

    if traj_np.shape[-1] == 2:
        N = len(traj_np)
        z_vals = np.full((N, 1), z_offset)
        object_points_3d = np.hstack([traj_np, z_vals]).astype(np.float32)
    else:
        object_points_3d = traj_np.astype(np.float32)

    # 5. Transform: ego frame -> optical/RDF camera frame.
    points_cam_optical = object_points_3d @ R_l2s.T + t_l2s

    # 6. Filter: discard points behind the camera (Z <= min_depth).
    min_depth = float(getattr(cal_dict, "projection_min_depth", 1.5))
    valid_mask = points_cam_optical[:, 2] > min_depth
    if not np.any(valid_mask):
        return img_draw, np.array([])
    points_to_project = points_cam_optical[valid_mask]

    # 7. Project
    use_ftheta = hasattr(cal_dict, "fw_poly") and cal_dict.fw_poly is not None

    if use_ftheta:
        # ---- F-theta (fisheye) projection ----
        # Polynomial: f(θ) = k0 + k1*θ + k2*θ² + k3*θ³ + k4*θ⁴
        fw_poly = to_numpy(cal_dict.fw_poly).astype(np.float64)
        # cx, cy are stored in the approximate pinhole intrinsics matrix
        cx = float(to_numpy(cal_dict.intrinsics)[0, 2])
        cy = float(to_numpy(cal_dict.intrinsics)[1, 2])

        pts = points_to_project.astype(np.float64)
        norms = np.linalg.norm(pts, axis=1, keepdims=True)
        rays = pts / norms
        rx, ry, rz = rays[:, 0], rays[:, 1], rays[:, 2]

        rp_norm = np.sqrt(rx**2 + ry**2)
        rp_norm = np.maximum(rp_norm, 1e-10)

        theta = np.arccos(np.clip(rz, -1.0, 1.0))
        f_val = (
            fw_poly[0]
            + fw_poly[1] * theta
            + fw_poly[2] * theta**2
            + fw_poly[3] * theta**3
            + fw_poly[4] * theta**4
        )

        u = cx + f_val * rx / rp_norm
        v = cy + f_val * ry / rp_norm
        image_points = np.stack([u, v], axis=1)
    else:
        # ---- Pinhole projection (no distortion) ----
        # NAVSIM images are stored undistorted; applying distortion coefficients
        # via cv2.projectPoints warps off-axis points inward (barrel distortion),
        # causing trajectories to incorrectly appear in side cameras.
        # This matches NAVSIM's own _transform_pcs_to_images reference.
        r_vec = np.zeros((3, 1), dtype=np.float32)
        t_vec = np.zeros((3, 1), dtype=np.float32)
        D_zero = np.zeros(5, dtype=np.float32)
        image_points, _ = cv2.projectPoints(points_to_project, r_vec, t_vec, K, D_zero)
        image_points = image_points.reshape(-1, 2)

    # 9. Visualize — only draw segments and points whose projected coordinates fall
    # within (or just outside) the image, to avoid wild lines from off-FoV points.
    margin = 10
    in_bounds = (
        (image_points[:, 0] >= -margin)
        & (image_points[:, 0] < curr_w + margin)
        & (image_points[:, 1] >= -margin)
        & (image_points[:, 1] < curr_h + margin)
    )
    points_int = image_points.astype(np.int32)
    color_rgb = {"r": (255, 0, 0), "b": (0, 0, 255)}.get(color, (255, 0, 0))

    for i in range(len(points_int) - 1):
        if in_bounds[i] and in_bounds[i + 1]:
            cv2.line(
                img_draw, tuple(points_int[i]), tuple(points_int[i + 1]), color_rgb, 2
            )
    for i, p in enumerate(points_int):
        if in_bounds[i]:
            cv2.circle(img_draw, (p[0], p[1]), 4, color_rgb, -1)

    return img_draw, image_points


def plot_pred_in_cam(
    cam_dicts: list[CameraData],
    past_traj: torch.Tensor,
    pred_traj: torch.Tensor | None = None,
    fut_traj: torch.Tensor | None = None,
    intent: str | None = None,
    pred_cot: str | None = None,
    gt_cot: str | None = None,
):
    past_traj = past_traj.cpu().numpy()
    pred_traj = pred_traj.cpu().numpy() if pred_traj is not None else None
    fut_traj = fut_traj.cpu().numpy() if fut_traj is not None else None

    # Layout: 4-panel when both CoTs, 3-panel when only pred_cot, 2-panel otherwise
    if pred_cot is not None and gt_cot is not None:
        fig, ax = plt.subplot_mosaic("AA\nBC\nBD", figsize=(10, 14))
    elif pred_cot is not None:
        fig, ax = plt.subplot_mosaic("AA\nBC", figsize=(10, 10))
    else:
        fig, ax = plt.subplot_mosaic("AAB", figsize=(13, 5))

    result_img_list = []
    for cam_data in cam_dicts:
        image = cam_data.image
        cal_dict = cam_data.cal_dict
        result_img = image.cpu().numpy()
        # Use consistent colors: GT = red, Pred = blue
        # Always show GT if available
        if fut_traj is not None:
            result_img, _ = project_trajectory_to_image(
                result_img, cal_dict, fut_traj, color="r"
            )
        # Only show prediction if explicitly provided
        if pred_traj is not None:
            result_img, _ = project_trajectory_to_image(
                result_img, cal_dict, pred_traj, color="b"
            )
        result_img_list.append(result_img)
    final_img = np.concatenate(result_img_list, axis=1)

    # Add intent to camera view title
    cam_title = "Camera View"
    if intent is not None:
        cam_title += f" | Intent: {intent}"
    ax["A"].imshow(final_img)
    ax["A"].set_title(cam_title)

    # BEV view with y pointing forward (swap x and y, negate lateral to fix orientation)
    # Trajectory is (x, y) where typically x=forward, y=lateral (positive=left, negative=right)
    # To make y point forward, we plot (-y, x) to keep correct lateral orientation
    delta = 5
    y_min = past_traj[:, 0].min() - delta  # x becomes y (forward)
    y_max = past_traj[:, 0].max() + delta
    x_min = (
        -past_traj[:, 1].max() - delta
    )  # -y becomes x (lateral, negated to fix orientation)
    x_max = -past_traj[:, 1].min() + delta

    # Past trajectory in gray - swap axes and negate lateral for y-forward
    ax["B"].plot(
        -past_traj[:, 1],
        past_traj[:, 0],
        "--",
        label="past",
        alpha=0.7,
        color="gray",
        markersize=4.5,
    )
    # GT trajectory in red (same as camera view) - always show if available
    if fut_traj is not None:
        ax["B"].plot(
            -fut_traj[:, 1],
            fut_traj[:, 0],
            "o-",
            label="gt",
            alpha=0.5,
            color="red",
            markersize=4.5,
        )
        y_min = min(y_min, fut_traj[:, 0].min() - delta, -5)
        y_max = max(y_max, fut_traj[:, 0].max() + delta, 15)
        x_min = min(x_min, -fut_traj[:, 1].max() - delta, -4)
        x_max = max(x_max, -fut_traj[:, 1].min() + delta, 4)
    # Pred trajectory in blue (same as camera view) - only if provided
    if pred_traj is not None:
        ax["B"].plot(
            -pred_traj[:, 1],
            pred_traj[:, 0],
            "x-",
            label="pred",
            alpha=0.7,
            color="blue",
            markersize=4.5,
        )
        y_min = min(y_min, pred_traj[:, 0].min() - delta, -5)
        y_max = max(y_max, pred_traj[:, 0].max() + delta, 15)
        x_min = min(x_min, -pred_traj[:, 1].max() - delta, -4)
        x_max = max(x_max, -pred_traj[:, 1].min() + delta, 4)

    ax["B"].set_xlim(x_min, x_max)
    ax["B"].set_ylim(y_min, y_max)
    ax["B"].set_xlabel("Lateral (m)")
    ax["B"].set_ylabel("Forward (m)")
    ax["B"].grid(alpha=0.3)
    ax["B"].set_title("BEV View")
    ax["B"].legend(loc="upper right")

    # Chain-of-thought reasoning panels
    if pred_cot is not None:
        ax["C"].axis("off")
        ax["C"].set_title("Predicted CoT", fontsize=10)
        ax["C"].text(
            0.02,
            0.98,
            pred_cot,
            transform=ax["C"].transAxes,
            fontsize=7,
            verticalalignment="top",
            wrap=True,
            family="monospace",
            bbox=dict(boxstyle="round,pad=0.3", facecolor="lightyellow", alpha=0.8),
        )
    if gt_cot is not None:
        panel = "D" if "D" in ax else "C"
        ax[panel].axis("off")
        ax[panel].set_title("Annotation CoT", fontsize=10)
        ax[panel].text(
            0.02,
            0.98,
            gt_cot,
            transform=ax[panel].transAxes,
            fontsize=7,
            verticalalignment="top",
            wrap=True,
            family="monospace",
            bbox=dict(boxstyle="round,pad=0.3", facecolor="lightyellow", alpha=0.8),
        )

    return fig


def plot_pref_trajs(
    pref_trajs: list[tuple[torch.Tensor, float]], fig: plt.Figure | None = None
):
    ax = fig.axes[1]  # Get BEV plot

    for pref_traj, pref_score in pref_trajs:
        pref_traj = pref_traj.cpu().numpy()
        ax.plot(
            -pref_traj[:, 1],
            pref_traj[:, 0],
            "x-",
            label="pref",
            alpha=0.25,
            color="green",
            markersize=4.5,
        )
        ax.annotate(
            str(pref_score),
            (-pref_traj[-1, 1] + 0.3, pref_traj[-1, 0]),
            alpha=0.5,
            color="green",
        )

    return fig


def _find_camera_data(sample: E2EDataSample, camera_name: str) -> tuple | None:
    """
    Helper function to find camera data by name across all camera dicts.

    Args:
        sample: E2EDataSample containing the sample data
        camera_name: Name of the camera to find (e.g., "FRONT", "FRONT_LEFT")

    Returns:
        Tuple of (image, cal_dict) if found, None otherwise
    """
    camera_dict = sample.cameras[-1]  # get the most recent timestamp's camera dict
    if camera_name in camera_dict:
        camera_data = camera_dict[camera_name]
        if hasattr(camera_data, "image") and hasattr(camera_data, "cal_dict"):
            return camera_data.image, camera_data.cal_dict
    return None


def visualize_e2e_sample(
    sample: E2EDataSample,
    pred_traj: torch.Tensor | None = None,
    cot: str | None = None,
):
    """
    Visualizes an E2E sample with past trajectory, predicted trajectory (if provided),
    and future trajectory (if available).

    Uses project_trajectory_to_image and plot_pred_in_cam internally.
    Only uses the FRONT camera for visualization.

    Args:
        sample: E2EDataSample containing the sample data
        pred_traj: Optional predicted trajectory tensor (N, 2) or (N, 3)
        cot: Optional chain-of-thought / chain-of-causation reasoning string

    Returns:
        matplotlib figure with visualization (camera view + BEV view)
    """
    if len(sample.cameras) == 0:
        raise ValueError("Sample has no cameras")

    # Find FRONT camera only
    image, cal_dict = _find_camera_data(sample, "FRONT")

    if image is None or cal_dict is None:
        raise ValueError("FRONT camera not found")

    # Extract past trajectory
    if "past_traj" not in sample.agent_input:
        raise ValueError("Sample agent_input missing 'past_traj'")
    past_traj = sample.agent_input["past_traj"]

    fut_traj_tensor = sample.fut_traj  # Already a plain tensor [T, 2] or None

    # Reuse plot_pred_in_cam which handles everything
    return plot_pred_in_cam(
        image=image,
        cal_dict=cal_dict,
        past_traj=past_traj,
        pred_traj=pred_traj,  # Can be None - will only show if provided
        fut_traj=fut_traj_tensor,  # Always show GT if available
        intent=sample.intent,
        pred_cot=cot,
    )
