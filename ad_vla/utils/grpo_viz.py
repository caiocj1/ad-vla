"""Bird's-eye views of Flow-GRPO groups on AlpaSim states (``SimPDMReward`` reward contexts).

``plot_group`` draws one state in its plan frame, forward to the right and left up: the drivable raster (lanes and
road areas) with the road edges, the recorded human path, its lane centreline and lane edges (v8 states), the logged
actors (boxes
at 0.25 s, their motion over the plan horizon, dashed outlines at the plan end), the ego box, and the group's plans
coloured by reward (red 0 -> green 1; poly-smoothed as the reward scores them; plans failing a gate dashed), with the
deterministic deployment plan when given. The side panel shows the front camera and the group's sorted rewards with
the gate each failing plan breaks (col collision, off off-road, cor corridor, ttc time to collision).

Used by ``GRPOModule`` (``viz_cfg``: groups of training batches and fixed probe states, logged as images) and by
``scripts/utils/grpo_snapshot_viz.py`` (the same probe views from saved checkpoints).
"""

from __future__ import annotations

import io
import os

import numpy as np
import torch
from matplotlib import colormaps
from matplotlib.backends.backend_agg import FigureCanvasAgg
from matplotlib.cm import ScalarMappable
from matplotlib.colors import Normalize
from matplotlib.figure import Figure
from matplotlib.lines import Line2D
from matplotlib.patches import Patch, Polygon

from ad_vla.rewards.sim_pdm_reward import poly_smooth

CMAP = colormaps["RdYlGn"]
NORM = Normalize(0.0, 1.0)
GATES = (("collision", "col"), ("offroad", "off"), ("corridor", "cor"), ("ttc", "ttc"))


def plan_terms(
    reward_fn, plans: torch.Tensor, ctx: dict
) -> tuple[np.ndarray, list[dict]]:
    """Reward and gate failures of each plan ``[G, T, 2]``, scored one by one."""
    rewards, flags = [], []
    for g in range(plans.shape[0]):
        r, t = reward_fn.score(plans[g : g + 1].float(), ctx)
        collision = bool(t["collision_rate"] > 0)
        flags.append(
            {
                "collision": collision,
                "offroad": bool(t["dac"] < 1.0),
                "corridor": bool(t["corridor"] < 1.0),
                "ttc": bool(t["ttc"] < 1.0) and not collision,
            }
        )
        rewards.append(float(r[0]))
    return np.asarray(rewards), flags


def _raster_image(ctx: dict) -> tuple[np.ndarray, list[float]]:
    """Drivable raster (lanes + road areas) with road edges as an RGB image in plan-frame axes (x right, y up)."""
    drivable = np.asarray(ctx["drivable"], dtype=bool)  # [x, y]
    img = np.full(drivable.shape + (3,), 0.965)
    img[drivable] = (0.85, 0.86, 0.87)
    if "road_edge" in ctx:
        img[np.asarray(ctx["road_edge"], dtype=bool)] = (0.30, 0.30, 0.32)
    x0, y0 = (float(v) for v in ctx["drivable_origin"])
    res = float(ctx["drivable_res"])
    hx, wy = drivable.shape
    return img.transpose(1, 0, 2), [
        x0,
        x0 + hx * res,
        y0,
        y0 + wy * res,
    ]  # rows = y (origin="lower"), cols = x


def _box(center, yaw, length, width):
    c, s = np.cos(yaw), np.sin(yaw)
    local = np.array(
        [
            [length / 2, width / 2],
            [length / 2, -width / 2],
            [-length / 2, -width / 2],
            [-length / 2, width / 2],
        ]
    )
    return np.asarray(center) + local @ np.array([[c, s], [-s, c]])


def _lane_edges(center: np.ndarray, half: np.ndarray):
    """Left / right edges of the lane under the human path: the centreline offset by the lane half width."""
    tangent = np.gradient(center, axis=0)
    tangent /= np.maximum(np.linalg.norm(tangent, axis=1, keepdims=True), 1e-6)
    normal = np.stack([-tangent[:, 1], tangent[:, 0]], axis=1)  # to the left
    half = np.where(np.isfinite(half), half, np.nan)[:, None]
    return center + normal * half, center - normal * half


def plot_group(
    ctx: dict,
    plans: np.ndarray | torch.Tensor,
    rewards: np.ndarray,
    flags: list[dict],
    deploy: np.ndarray | torch.Tensor | None = None,
    camera: np.ndarray | None = None,
    title: str = "",
    dt: float = 0.25,
    smooth: bool = True,
    deploy_reward: float | None = None,
) -> Figure:
    """One state and its group of plans; see the module docstring. ``plans`` [G, T, 2], ``rewards`` [G].
    Bird's-eye view in the plan frame: forward to the right, left of the ego up."""
    plans = torch.as_tensor(plans, dtype=torch.float32).cpu()
    if smooth:
        plans = poly_smooth(plans, dt)
    plans = plans.numpy()
    if deploy is not None:
        deploy = torch.as_tensor(deploy, dtype=torch.float32).cpu().reshape(1, -1, 2)
        deploy = (poly_smooth(deploy, dt) if smooth else deploy).numpy()[0]
    rewards = np.asarray(rewards, dtype=np.float64)
    num = len(rewards)

    fig = Figure(figsize=(14, 9.5), dpi=100)
    FigureCanvasAgg(fig)
    grid = fig.add_gridspec(
        2,
        2,
        height_ratios=[1.0, 0.85],
        width_ratios=[1.0, 1.0],
        hspace=0.28,
        wspace=0.12,
    )
    ax = fig.add_subplot(grid[0, :])
    img, extent = _raster_image(ctx)
    ax.imshow(img, origin="lower", extent=extent, interpolation="nearest", zorder=0)

    # view: the plans and the near part of the human path
    reach = max(float(plans[..., 0].max()), 12.0)
    path = np.asarray(ctx["path_xy"], dtype=np.float64)
    near = (
        path[(path[:, 0] > -8) & (path[:, 0] < reach + 8)]
        if len(path)
        else np.zeros((0, 2))
    )
    lat = np.concatenate([plans[..., 1].ravel(), near[:, 1], [0.0]])
    x_lo, x_hi = -8.0, min(reach + 8.0, 80.0)
    y_mid = 0.5 * (lat.max() + lat.min())
    half_h = min(
        max(0.5 * (lat.max() - lat.min()) + 5.0, 8.0, 0.16 * (x_hi - x_lo)), 32.0
    )
    ax.set_xlim(x_lo, x_hi)
    ax.set_ylim(y_mid - half_h, y_mid + half_h)

    corners = np.asarray(ctx["agent_corners"], dtype=np.float64)  # [Ta, A, 4, 2]
    valid = np.asarray(ctx["agent_valid"], dtype=bool)
    end_idx = min(
        int(round(plans.shape[1] * dt / float(ctx["agent_times_s"][0]))) - 1,
        corners.shape[0] - 1,
    )
    for a in range(corners.shape[1]):
        if not valid[:, a].any():
            continue
        track = corners[valid[:, a], a].mean(axis=1)
        ax.plot(track[:, 0], track[:, 1], color="#6b6f76", lw=0.8, alpha=0.7, zorder=3)
        if valid[0, a]:
            ax.add_patch(
                Polygon(
                    corners[0, a],
                    closed=True,
                    fc="#a9adb4",
                    ec="#3c4048",
                    lw=0.9,
                    zorder=4,
                )
            )
        if valid[end_idx, a]:
            ax.add_patch(
                Polygon(
                    corners[end_idx, a],
                    closed=True,
                    fill=False,
                    ec="#3c4048",
                    lw=0.7,
                    ls="--",
                    alpha=0.6,
                    zorder=4,
                )
            )
    ego = _box(
        np.asarray(ctx["rig_to_center_xy"], dtype=np.float64),
        0.0,
        float(ctx["ego_length"]),
        float(ctx["ego_width"]),
    )
    ax.add_patch(
        Polygon(ego, closed=True, fc="#2f5fd0", ec="#14307a", lw=1.0, zorder=6)
    )

    failing = [any(f.values()) for f in flags]
    for g in np.argsort(rewards):  # best on top
        line = np.concatenate([np.zeros((1, 2)), plans[g]])
        ax.plot(
            line[:, 0],
            line[:, 1],
            color=CMAP(NORM(rewards[g])),
            lw=1.5,
            alpha=0.9,
            ls=(0, (4, 2)) if failing[g] else "-",
            zorder=5,
        )
        ax.plot(
            line[-1, 0], line[-1, 1], "o", ms=3, color=CMAP(NORM(rewards[g])), zorder=5
        )
    if deploy is not None:  # under the samples, so that they stay visible
        line = np.concatenate([np.zeros((1, 2)), deploy])
        ax.plot(
            line[:, 0],
            line[:, 1],
            color="#1238a8",
            lw=4.5,
            alpha=0.75,
            solid_capstyle="round",
            zorder=4.5,
        )

    # references on top of the samples
    center = ctx.get("centerline_xy")
    has_center = center is not None and len(center) >= 2
    if has_center:
        center = np.asarray(center, dtype=np.float64)
        left, right = _lane_edges(
            center, np.asarray(ctx["center_halfwidth"], dtype=np.float64)
        )
        for edge in (left, right):
            ax.plot(
                edge[:, 0], edge[:, 1], color="#1b7f3b", lw=0.9, alpha=0.75, zorder=9
            )
        ax.plot(
            center[:, 0],
            center[:, 1],
            color="#1b7f3b",
            lw=1.4,
            ls=(0, (1, 2)),
            zorder=9,
        )
    if len(path):
        ax.plot(
            path[:, 0], path[:, 1], color="black", lw=1.0, ls=(0, (6, 3)), zorder=10
        )

    ax.set_aspect("equal")
    ax.set_xlabel("forward (m)")
    ax.set_ylabel("left (m)")
    handles = [Line2D([], [], color="black", ls=(0, (6, 3)), label="human path")]
    if has_center:
        handles += [
            Line2D(
                [],
                [],
                color="#1b7f3b",
                ls=(0, (1, 2)),
                lw=1.4,
                label="its lane centreline",
            ),
            Line2D([], [], color="#1b7f3b", lw=0.9, label="its lane edges"),
        ]
    handles += [
        Patch(fc="#a9adb4", ec="#3c4048", label="actors at 0.25 s (dashed: plan end)"),
        Patch(fc="#2f5fd0", label="ego"),
        Patch(fc="#d9dbde", ec="#999", label="drivable"),
    ]
    if "road_edge" in ctx:
        handles.append(Patch(fc="#4d4d52", label="road edge"))
    if deploy is not None:
        handles.append(
            Line2D(
                [],
                [],
                color="#1238a8",
                lw=4.5,
                alpha=0.75,
                label="deployment plan (model)",
            )
        )
    handles.append(
        Line2D([], [], color=CMAP(0.1), ls=(0, (4, 2)), label="sample failing a gate")
    )
    ax.legend(
        handles=handles,
        loc="lower left",
        bbox_to_anchor=(0.0, 1.01),
        ncol=5,
        fontsize=8,
        frameon=False,
    )
    cbar = fig.colorbar(
        ScalarMappable(norm=NORM, cmap=CMAP), ax=ax, fraction=0.02, pad=0.01
    )
    cbar.set_label("sample reward")

    cam_ax = fig.add_subplot(grid[1, 0])
    if camera is not None:
        cam_ax.imshow(np.asarray(camera))
        cam_ax.set_title("front camera (model input)", fontsize=9)
    cam_ax.axis("off")

    bar_ax = fig.add_subplot(grid[1, 1])
    order = np.argsort(rewards)
    bar_ax.bar(
        np.arange(num),
        rewards[order],
        color=[CMAP(NORM(r)) for r in rewards[order]],
        edgecolor="#333",
        lw=0.4,
    )
    for i, g in enumerate(order):
        tag = "\n".join(label for key, label in GATES if flags[g][key])
        if tag:
            bar_ax.text(
                i,
                rewards[g] + 0.02,
                tag,
                ha="center",
                va="bottom",
                fontsize=7,
                color="#7a1010",
                linespacing=0.9,
            )
    if deploy_reward is not None:
        bar_ax.axhline(deploy_reward, color="#1238a8", lw=1.2, ls="--")
    bar_ax.set_ylim(0, 1.25)
    bar_ax.set_xticks([])
    bar_ax.set_ylabel("reward")
    counts = {key: sum(f[key] for f in flags) for key, _ in GATES}
    bar_ax.set_title(
        f"{num} samples, sorted: mean {rewards.mean():.2f}, std {rewards.std(ddof=1) if num > 1 else 0.0:.2f}"
        + (
            f"; deployment plan {deploy_reward:.2f} (dashed)"
            if deploy_reward is not None
            else ""
        )
        + f"\nfailing samples: col collision {counts['collision']}, off off-road {counts['offroad']}, "
        f"cor corridor (4 m) {counts['corridor']}, ttc time to collision {counts['ttc']}",
        fontsize=9,
    )
    fig.suptitle(title, fontsize=11, x=0.01, ha="left")
    return fig


def save_figure(fig: Figure, path: str) -> str:
    os.makedirs(os.path.dirname(path), exist_ok=True)
    buf = io.BytesIO()
    fig.savefig(buf, format="png", bbox_inches="tight")
    with open(path, "wb") as handle:
        handle.write(buf.getvalue())
    return path


def pick_probe_states(dataset, num: int) -> list[int]:
    """``num`` fixed probe states (``AlpaSimStateDataset`` with a state list), spread over the probe groups and
    their scenes: the same states at every probe pass and in every run on the same probe list."""
    by_group: dict[str, list[int]] = {}
    for i, path in enumerate(dataset.files):
        by_group.setdefault(dataset.groups.get(path, "all"), []).append(i)
    names = sorted(by_group)
    picked: list[int] = []
    seen: set[str] = set()  # one state per scene over all groups
    for k, name in enumerate(names):
        idx = by_group[name]
        share = num // len(names) + (1 if k < num % len(names) else 0)
        chosen = []
        for j in np.linspace(0, len(idx) - 1, num=max(8 * share, 1)).astype(
            int
        ):  # spread over the group
            scene = os.path.basename(dataset.files[idx[j]]).split("__")[0]
            if scene not in seen and len(chosen) < share:
                seen.add(scene)
                chosen.append(idx[j])
        picked += chosen
    return picked


def state_title(sample, extra: str = "") -> str:
    name = sample.metadata.get("scenario_id", "")
    parts = name.split("__")
    short = (
        f"{parts[0].replace('clipgt-', '')[:6]} t={int(parts[-1]) / 1e6:.1f}s"
        if len(parts) >= 3
        else name
    )
    group = sample.metadata.get("probe_group")
    return f"{short}{f' [{group}]' if group else ''}  {sample.intent}{('  ' + extra) if extra else ''}"


def front_camera(sample) -> np.ndarray | None:
    try:
        image = sample.cameras[0]["FRONT"].image
    except (IndexError, KeyError, AttributeError, TypeError):
        return None
    image = (
        image.detach().cpu().numpy() if torch.is_tensor(image) else np.asarray(image)
    )
    return image if image.ndim == 3 else None


__all__ = [
    "plan_terms",
    "plot_group",
    "save_figure",
    "pick_probe_states",
    "state_title",
    "front_camera",
]
