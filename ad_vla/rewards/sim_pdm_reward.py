"""GT-free PDM-style reward for closed-loop AlpaSim states (``AlpaSimStateDataset``).

Scores each candidate plan (ego frame of the state, x forward, y left, positions of the rig)
against the state's reward context, in the spirit of NAVSIM's PDM score:

    R = NC * DAC * COR * (w_ep * EP + w_ttc * TTC + w_pk * PK + w_c * C) / (w_ep + w_ttc + w_pk + w_c)

- NC (no at-fault collision): 0 if the ego box overlaps a logged actor box at any plan pose
  (separating-axis test), else 1; binary as in PDM. Overlaps with an actor behind the ego, or
  while the ego is stationary, are not the ego's fault and are ignored. (A graded "fraction of
  the horizon before the collision" made late collisions nearly free; with EP rewarding the
  fastest candidate, FlowGRPO-v1-r1 learned faster plans that rear-ended more.)
- TTC (time to collision): 0 if, at some pose, the ego box moved ahead along its heading at its
  current plan speed by up to ``ttc_horizon_s`` overlaps an actor's logged box at that later
  time (same at-fault rules), else 1; PDM's TTC term, which penalizes closing in without contact.
- DAC (drivable area): the ego footprint (corners and edge midpoints) on the drivable raster
  (lanes and road areas, the AlpaSim off-road scorer's road) at every plan pose.
  ``dac_mode="binary"``: 1 if it is on the raster at all poses, else 0, as in PDM and as AlpaSim
  fails a scene on its first off-road moment. ``"graded"``: the fraction of poses on the raster
  (FlowGRPO-v3-r1: a plan with one off-road pose kept 95 % of its reward, and the probe's
  off-road rate only went 22 -> 20 %).
  Maps with road edges and no road areas (PhysicalAI NuRec; the state has a ``road_edge`` raster):
  as AlpaSim's scorer there, a pose is off-road when the footprint is not inside the lanes and it
  touches a road edge at the ego's level; the footprint outline is sampled every ``outline_step_m``
  so that an edge crossing a side between two corners is seen.
- COR (corridor): with ``corridor_m`` set, 0 if the box centre is ``corridor_m`` or more from the
  recorded human path at any plan pose, else 1; AlpaSim's ``left_corridor_laterally`` (lateral
  distance to the recorded path >= ``max_dist_to_gt_trajectory``, 4 m) fails the scene. ``None``:
  off (COR = 1).
- EP (ego progress): progress along the recorded human path divided by a reference, clipped
  to [0, 1]. ``ep_reference="human"`` (default, when the state has ``human_progress_m``): the
  distance the human drove in the same horizon, so driving faster than the human earns nothing
  (AlpaSim's progress score is relative to the human as well). Beyond
  ``max(ep_over_ratio * human, human + ep_over_margin_m)`` EP falls linearly, to 0 at
  ``ep_over_ramp * human`` further (``ep_over_ratio=None``: no penalty); without it, v3-r1's plans
  kept lengthening (probe progress / human 1.21 -> 1.29), as nothing pushed back. With the PDM-style
  ``ep_reference="group"`` (also the fallback for states without it) the reference is the best
  candidate of the group; that rewarded the fastest sample, and FlowGRPO-v2-r1 learned
  ~20 % longer plans (PhAI open-loop ADE 1.31 -> 1.87). Below ``ep_min_progress_m`` of
  reference distance, EP is 1 for everyone.
- PK (path keeping): ``exp(-0.5 (mean lateral distance to the path / pk_sigma_m)^2)``; this
  is the drift term (staying on the route's lane instead of continuing a self-made turn).
  ``pk_path="centerline"``: the distance is to ``centerline_xy`` instead, the human path snapped to
  the centreline of its lane (lane changes follow the
  human), so that driving between two lanes costs reward: v6b drove on lane lines, and PK against the
  human path with sigma 1 m barely noticed (states without a centreline keep the human path).
  When the state has a centreline, the terms also report ``center_lateral_m`` (mean distance of the
  plan to it) and ``straddle`` (share of plans whose box crosses a lane line at some pose:
  centreline distance + half the ego width beyond the lane's half width).
- C (comfort): 1 if PDM's six comfort limits hold (longitudinal / lateral acceleration, jerk magnitude,
  longitudinal jerk, yaw rate, yaw acceleration; ``COMFORT_LIMITS``), else 0.
- DM (drivable margin, ``w_dm`` > 0, FlowGRPO v11): clip(d / ``dm_full_m``, 0, 1), d = the smallest distance from the
  ego outline (sampled every ``outline_step_m``) to the nearest non-drivable cell over the poses from ``dm_start_s`` on
  (the part the plan controls). DAC is all-or-nothing: a plan 5 cm from the road edge scored like one 1 m away, and PK
  pulls towards the human path even where the human hugged the edge, so nothing taught a margin before a sample
  crossed (3 of v4-r1's 10 avoidable NavTest-test off-road failures clipped the edge within 0.35 m of the human path).
  The distance comes from a Euclidean distance transform of the drivable raster (cached in the context); with road
  edges (PhysicalAI) it is the distance to the nearest road-edge cell instead. Added to the weighted sum of the PDM
  mode (``R = NC * DAC * COR * (... + w_dm * DM) / (... + w_dm)``); ``w_dm`` = 0 leaves the reward unchanged.

Plans are smoothed first (``smooth="poly"``: the degree-4 polynomial the driver applies before
execution), so the reward scores what would be driven, not the SDE sampling noise.

``mode="tiered"`` (FlowGRPO v10) replaces the gate product above. A plan that
violates a gate gets a negative reward from the first violation time t (s) of each violated gate,
``min_g(-severity_g + fail_time_slope * t_g / horizon)``: at-fault collision (``sev_collision``), corridor exit
(``sev_corridor``), off-road (``sev_offroad``), so that every failing plan ranks below every passing one, an earlier
failure below a later one, and (with the default severities) any collision below any other failure. Groups where all
16 samples fail still rank them (the binary gates gave them all 0). A passing plan gets

    R = M_ttc * (w_ep * EP + w_cl * CL + w_ce * CE + w_cm * CM + w_c * C) / (w_ep + w_cl + w_ce + w_cm + w_c)

- M_ttc: ``ttc_factor`` when the TTC check (``ttc_horizon_s``) fires, else 1 (a soft gate instead of a term);
- EP and C as above;
- CL (clearance): min over poses and actors in the ego's path band (lateral overlap with the box widened by
  ``cl_lateral_margin_m`` per side, ahead of the front bumper) of clip(gap / (``cl_min_gap_m`` + ``cl_headway_s`` *
  plan speed), 0, 1): closing in on a slower or stopped vehicle ahead costs reward before contact;
- CM (corridor margin): clip((``cm_zero_m`` - D) / (``cm_zero_m`` - ``cm_full_m``), 0, 1), D = max lateral distance
  of the box centre to the human path (the corridor gate itself stays at ``corridor_m``);
- CE (centring), on the poses after ``ce_late_s`` (the part the plan controls; it starts at the ego), against the lane
  centreline where it is trusted (snapped, and within ``ce_trust_m`` of the human path), else the human path:
  0.4 * clip(1 - (mean distance - ``ce_full_m``) / (``ce_zero_m`` - ``ce_full_m``)) (linear: v8's Gaussian with
  sigma 0.5 m saturated at the usual 0.6-2.7 m offsets)
  + 0.2 * clip(1 - (end distance - max(``ce_full_m``, ``ce_conv_ratio`` * start distance)) / ``ce_conv_tol_m``)
  + 0.2 * clip(1 - |heading error to the path at the plan end| / ``ce_head_zero_deg``)
  + 0.2 * (1 - share of trusted late poses whose box crosses a lane line).
"""

from __future__ import annotations

import math

import torch

from ad_vla.dataset.data_types import E2EDataBatch, TrajectorySampling

# PDM comfort limits (NAVSIM / nuPlan ego_is_comfortable): longitudinal acceleration (min / max),
# |lateral acceleration|, |jerk| (magnitude of the jerk vector), |longitudinal jerk|, |yaw rate|,
# |yaw acceleration|. The last two of the six checks were added on 03-11 (before: yaw acceleration and the
# jerk magnitude were not checked, so C was slightly more lenient than PDM's for FlowGRPO v1-v12).
COMFORT_LIMITS = {
    "min_lon_accel": -4.05,
    "max_lon_accel": 2.40,
    "max_abs_lat_accel": 4.89,
    "max_abs_mag_jerk": 8.37,
    "max_abs_lon_jerk": 4.13,
    "max_abs_yaw_rate": 0.95,
    "max_abs_yaw_accel": 1.93,
}


def poly_smooth(trajs: torch.Tensor, dt: float, degree: int = 4) -> torch.Tensor:
    """Least-squares polynomial (no constant: starts at the ego) per plan; ``trajs`` [N, T, 2]."""
    t = (
        torch.arange(1, trajs.shape[1] + 1, device=trajs.device, dtype=torch.float64)
        * dt
    )
    basis = torch.stack([t**k for k in range(1, degree + 1)], dim=1)  # [T, degree]
    proj = basis @ torch.linalg.pinv(basis)  # [T, T]
    return (proj.to(trajs.dtype) @ trajs).to(trajs.dtype)


def headings(positions: torch.Tensor, min_step_m: float = 0.25) -> torch.Tensor:
    """Heading of each pose from its displacement; ``positions`` [N, T + 1, 2] with the ego at
    index 0 (heading 0). Steps shorter than ``min_step_m`` (1 m/s at 0.25 s) keep the previous
    heading: at walking pace a few centimetres of plan noise would swing it arbitrarily."""
    steps = positions[:, 1:] - positions[:, :-1]
    yaw = torch.atan2(steps[..., 1], steps[..., 0])
    moving = steps.norm(dim=-1) > min_step_m
    out = torch.zeros_like(yaw)
    prev = torch.zeros_like(yaw[:, 0])
    for k in range(yaw.shape[1]):
        prev = torch.where(moving[:, k], yaw[:, k], prev)
        out[:, k] = prev
    return out


def box_points(
    center: torch.Tensor, yaw: torch.Tensor, length: float, width: float, mids: bool
):
    """Corners (and edge midpoints) of boxes: ``center`` [..., 2], ``yaw`` [...] -> [..., K, 2]."""
    hl, hw = length / 2.0, width / 2.0
    local = [[hl, hw], [hl, -hw], [-hl, -hw], [-hl, hw]]
    if mids:
        local += [[hl, 0.0], [0.0, -hw], [-hl, 0.0], [0.0, hw]]
    local = torch.tensor(local, device=center.device, dtype=center.dtype)
    c, s = torch.cos(yaw)[..., None], torch.sin(yaw)[..., None]
    x = center[..., None, 0] + c * local[:, 0] - s * local[:, 1]
    y = center[..., None, 1] + s * local[:, 0] + c * local[:, 1]
    return torch.stack([x, y], dim=-1)


def box_outline(
    center: torch.Tensor, yaw: torch.Tensor, length: float, width: float, step: float
):
    """Points every ~``step`` m along the outline of boxes (corners included): -> [..., K, 2]."""
    corners = box_points(center, yaw, length, width, mids=False)  # [..., 4, 2]
    out = []
    for k, side in enumerate(
        (width, length, width, length)
    ):  # box_points order: front, right, rear, left
        n = max(1, int(math.ceil(side / step)))
        a, b = corners[..., k, :], corners[..., (k + 1) % 4, :]
        for i in range(n):
            out.append(a + (b - a) * (i / n))
    return torch.stack(out, dim=-2)


def boxes_overlap(a: torch.Tensor, b: torch.Tensor) -> torch.Tensor:
    """Separating-axis test for rectangles given by 4 corners each; broadcasts ``[..., 4, 2]``."""
    a, b = torch.broadcast_tensors(a, b)
    axes = []
    for box in (a, b):
        e1 = box[..., 1, :] - box[..., 0, :]
        e2 = box[..., 3, :] - box[..., 0, :]
        axes += [e1, e2]
    overlap = torch.ones(a.shape[:-2], dtype=torch.bool, device=a.device)
    for axis in axes:
        pa = (a * axis[..., None, :]).sum(-1)
        pb = (b * axis[..., None, :]).sum(-1)
        separated = (pa.max(-1).values < pb.min(-1).values) | (
            pb.max(-1).values < pa.min(-1).values
        )
        overlap &= ~separated
    return overlap


def project_on_path(points: torch.Tensor, path: torch.Tensor):
    """Lateral distance to and arc length along a polyline; ``points`` [..., 2], ``path`` [P, 2]."""
    a, b = path[:-1], path[1:]
    seg = b - a
    seg_len2 = (seg**2).sum(-1).clamp_min(1e-9)
    arc0 = torch.cat([seg.new_zeros(1), seg_len2.sqrt().cumsum(0)[:-1]])
    d = points[..., None, :] - a  # [..., S, 2]
    u = ((d * seg).sum(-1) / seg_len2).clamp(0.0, 1.0)
    closest = a + u[..., None] * seg
    dist = (points[..., None, :] - closest).norm(dim=-1)
    best = dist.argmin(dim=-1, keepdim=True)
    lateral = dist.gather(-1, best).squeeze(-1)
    arc = (arc0 + u * seg_len2.sqrt()).gather(-1, best).squeeze(-1)
    return lateral, arc


class SimPDMReward:
    """See the module docstring. ``target_sampling`` tells GRPOModule the plan grid to score."""

    def __init__(
        self,
        num_poses: int = 20,
        interval_length: float = 0.25,
        smooth: str = "poly",
        w_ep: float = 5.0,
        w_ttc: float = 5.0,
        w_pk: float = 5.0,
        w_c: float = 2.0,
        pk_sigma_m: float = 1.0,
        pk_path: str = "human",
        ep_min_progress_m: float = 5.0,
        ttc_horizon_s: float = 1.0,
        ep_reference: str = "human",
        dac_mode: str = "graded",
        ep_over_ratio: float | None = None,
        ep_over_margin_m: float = 2.0,
        ep_over_ramp: float = 0.4,
        corridor_m: float | None = None,
        outline_step_m: float = 0.25,
        comfort_limits: dict | None = None,
        mode: str = "pdm",
        sev_collision: float = 1.0,
        sev_corridor: float = 0.75,
        sev_offroad: float = 0.7,
        fail_time_slope: float = 0.2,
        ttc_factor: float = 0.5,
        w_cl: float = 4.0,
        w_ce: float = 3.0,
        w_cm: float = 3.0,
        cl_min_gap_m: float = 2.0,
        cl_headway_s: float = 0.6,
        cl_lateral_margin_m: float = 0.3,
        cm_full_m: float = 1.0,
        cm_zero_m: float = 3.0,
        ce_late_s: float = 2.0,
        ce_full_m: float = 0.3,
        ce_zero_m: float = 1.5,
        ce_conv_ratio: float = 0.5,
        ce_conv_tol_m: float = 0.6,
        ce_head_zero_deg: float = 15.0,
        ce_trust_m: float = 0.75,
        w_dm: float = 0.0,
        dm_full_m: float = 0.5,
        dm_start_s: float = 1.0,
    ) -> None:
        if mode not in ("pdm", "tiered"):
            raise ValueError("mode must be 'pdm' or 'tiered'")
        self.mode = mode
        self.tiered = {
            "sev_collision": float(sev_collision),
            "sev_corridor": float(sev_corridor),
            "sev_offroad": float(sev_offroad),
            "fail_time_slope": float(fail_time_slope),
            "ttc_factor": float(ttc_factor),
            "w_ep": float(w_ep),
            "w_cl": float(w_cl),
            "w_ce": float(w_ce),
            "w_cm": float(w_cm),
            "w_c": float(w_c),
            "cm_full_m": float(cm_full_m),
            "cm_zero_m": float(cm_zero_m),
        }
        self.cl = (float(cl_min_gap_m), float(cl_headway_s), float(cl_lateral_margin_m))
        self.ce = {
            "late_s": float(ce_late_s),
            "full_m": float(ce_full_m),
            "zero_m": float(ce_zero_m),
            "conv_ratio": float(ce_conv_ratio),
            "conv_tol_m": float(ce_conv_tol_m),
            "head_zero_rad": math.radians(float(ce_head_zero_deg)),
            "trust_m": float(ce_trust_m),
        }
        if smooth not in ("none", "poly"):
            raise ValueError("smooth must be 'none' or 'poly'")
        if ep_reference not in ("human", "group"):
            raise ValueError("ep_reference must be 'human' or 'group'")
        if dac_mode not in ("binary", "graded"):
            raise ValueError("dac_mode must be 'binary' or 'graded'")
        if pk_path not in ("human", "centerline"):
            raise ValueError("pk_path must be 'human' or 'centerline'")
        self.pk_path = pk_path
        self.dac_mode = dac_mode
        self.corridor_m = None if corridor_m is None else float(corridor_m)
        self.outline_step_m = float(outline_step_m)
        self.ep_over_ratio = None if ep_over_ratio is None else float(ep_over_ratio)
        self.ep_over_margin_m = float(ep_over_margin_m)
        self.ep_over_ramp = float(ep_over_ramp)
        self.ep_reference = ep_reference
        self.target_sampling = TrajectorySampling(
            num_poses=num_poses, interval_length=interval_length
        )
        self.dt = float(interval_length)
        self.smooth = smooth
        self.weights = (float(w_ep), float(w_ttc), float(w_pk), float(w_c))
        if w_dm < 0 or dm_full_m <= 0:
            raise ValueError("w_dm must be >= 0 and dm_full_m > 0")
        if w_dm > 0 and mode == "tiered":
            raise ValueError(
                "the drivable margin (w_dm) is only implemented for mode='pdm'"
            )
        self.w_dm, self.dm_full_m, self.dm_start_s = (
            float(w_dm),
            float(dm_full_m),
            float(dm_start_s),
        )
        # Look-ahead times of the TTC check, on the plan grid (0.25 s ... ttc_horizon_s).
        self.ttc_leads = [
            self.dt * k for k in range(1, int(round(ttc_horizon_s / self.dt)) + 1)
        ]
        self.pk_sigma_m = float(pk_sigma_m)
        self.ep_min_progress_m = float(ep_min_progress_m)
        self.comfort_limits = {**COMFORT_LIMITS, **(comfort_limits or {})}
        self.last_terms: dict[str, float] = {}
        self.last_components: dict | None = (
            None  # tiered mode: per-plan components of the last score() call
        )

    def __call__(self, pred_trajs: torch.Tensor, batch: E2EDataBatch) -> torch.Tensor:
        """Rewards ``[B, G]`` for plans ``pred_trajs`` ``[B, G, T, 2]`` on ``target_sampling``."""
        rewards, terms = [], []
        for b, sample in enumerate(batch.samples):
            r, t = self.score(pred_trajs[b].float(), sample.metadata["rl_ctx"])
            rewards.append(r)
            terms.append(t)
        # Terms a state cannot measure are NaN (centreline terms without a centreline): averaged over the others,
        # left out when no state has them.
        stacked = {k: torch.stack([t[k] for t in terms]) for k in terms[0]}
        self.last_terms = {
            k: float(v.nanmean())
            for k, v in stacked.items()
            if not torch.isnan(v).all()
        }
        return torch.stack(rewards).to(pred_trajs.dtype)

    def score(self, plans: torch.Tensor, ctx: dict) -> tuple[torch.Tensor, dict]:
        """Reward ``[G]`` and mean terms for the ``G`` plans ``[G, T, 2]`` of one state."""
        device = plans.device
        as_t = lambda v: torch.as_tensor(v, device=device, dtype=torch.float32)  # noqa: E731
        num_plans, num_poses = plans.shape[:2]
        if self.smooth == "poly":
            plans = poly_smooth(plans, self.dt)
        positions = torch.cat([plans.new_zeros(num_plans, 1, 2), plans], dim=1)
        yaw = headings(positions)  # [G, T]
        length, width = float(ctx["ego_length"]), float(ctx["ego_width"])
        offset = as_t(ctx["rig_to_center_xy"])
        cos, sin = torch.cos(yaw), torch.sin(yaw)
        centers = plans + torch.stack(
            [cos * offset[0] - sin * offset[1], sin * offset[0] + cos * offset[1]],
            dim=-1,
        )

        # DAC: footprint on the drivable raster (and, with road edges, off it only where it touches one).
        origin, res = as_t(ctx["drivable_origin"]), float(ctx["drivable_res"])

        def lookup(raster, pts):
            raster = torch.as_tensor(raster, device=device)
            idx = torch.floor((pts - origin) / res).long()
            inside = (
                (idx[..., 0] >= 0)
                & (idx[..., 0] < raster.shape[0])
                & (idx[..., 1] >= 0)
                & (idx[..., 1] < raster.shape[1])
            )
            hit = torch.zeros_like(inside)
            hit[inside] = raster[idx[..., 0][inside], idx[..., 1][inside]]
            return hit

        if "road_edge" in ctx:
            pts = box_outline(
                centers, yaw, length, width, self.outline_step_m
            )  # [G, T, K, 2]
            in_lanes = lookup(ctx["drivable"], pts).all(dim=-1)
            touches_edge = lookup(ctx["road_edge"], pts).any(dim=-1)
            pose_ok = in_lanes | ~touches_edge  # [G, T]
        else:
            pts = box_points(centers, yaw, length, width, mids=True)  # [G, T, 8, 2]
            pose_ok = lookup(ctx["drivable"], pts).all(dim=-1)  # [G, T]
        dac_frac = pose_ok.float().mean(dim=1)
        dac = pose_ok.all(dim=1).float() if self.dac_mode == "binary" else dac_frac
        margin_m = dm = None
        if self.w_dm > 0:
            margin_m = self._drivable_margin(
                centers, yaw, length, width, ctx, origin, res
            )  # [G]
            dm = (margin_m / self.dm_full_m).clamp(0.0, 1.0)

        # NC and TTC against the logged actors; lead 0 is the plan itself (collision), the
        # positive leads move each pose ahead at its own speed (TTC).
        corners = as_t(ctx["agent_corners"])  # [Ta, A, 4, 2]
        collided_any = torch.zeros(num_plans, dtype=torch.bool, device=device)
        ttc_violation = torch.zeros(num_plans, dtype=torch.bool, device=device)
        # Tiered mode: per-pose collisions, the shortest TTC look-ahead that fires, the actors at the plan times.
        collide_pose = torch.zeros(
            num_plans, num_poses, dtype=torch.bool, device=device
        )
        ttc_lead = torch.full((num_plans,), float("inf"), device=device)
        agents0 = valid0 = None
        if corners.shape[1] > 0:
            agent_dt = float(ctx["agent_times_s"][0])
            valid_all = torch.as_tensor(ctx["agent_valid"], device=device)
            times = torch.arange(1, num_poses + 1, device=device) * self.dt
            step = (positions[:, 1:] - positions[:, :-1]).norm(dim=-1)  # [G, T]
            speed = step / self.dt
            heading = torch.stack([cos, sin], dim=-1)  # [G, T, 2]
            # Not at fault: the actor is behind the ego (rear-ended), or the ego stands still.
            moving = step > 0.05
            for lead in [0.0] + self.ttc_leads:
                a_idx = (torch.round((times + lead) / agent_dt).long() - 1).clamp(
                    0, corners.shape[0] - 1
                )
                agents = corners[a_idx]  # [T, A, 4, 2]
                valid = valid_all[a_idx]  # [T, A]
                ahead = centers + heading * (speed * lead)[..., None]
                ego_box = box_points(
                    ahead, yaw, length, width, mids=False
                )  # [G, T, 4, 2]
                hit = boxes_overlap(ego_box[:, :, None], agents[None])  # [G, T, A]
                rel = agents.mean(dim=2)[None] - ahead[:, :, None]  # [G, T, A, 2]
                rel_x = rel[..., 0] * cos[..., None] + rel[..., 1] * sin[..., None]
                at_fault = (rel_x > -length / 4.0) & moving[..., None]
                hit_pose = (hit & valid[None] & at_fault).any(dim=-1)  # [G, T]
                bad = hit_pose.any(dim=-1)  # [G]
                if lead == 0.0:
                    collided_any = bad
                    collide_pose = hit_pose
                    agents0, valid0 = agents, valid
                else:
                    ttc_violation |= bad
                    ttc_lead = torch.where(
                        bad & torch.isinf(ttc_lead),
                        torch.full_like(ttc_lead, lead),
                        ttc_lead,
                    )
        nc = (~collided_any).float()
        ttc = (~(collided_any | ttc_violation)).float()

        # EP / PK: progress along and distance to the recorded human path.
        path = as_t(ctx["path_xy"])
        if path.shape[0] >= 2:
            lateral, arc = project_on_path(plans, path)  # [G, T]
            _, arc0 = project_on_path(plans.new_zeros(1, 2), path)
            progress = arc[:, -1] - arc0[0]
            # Poses beyond the end of the stored path (fast plans outrun its 120 m window) only measure the distance
            # to its last point: they are left out of PK and COR.
            path_len = (path[1:] - path[:-1]).norm(dim=-1).sum()
            within = arc < path_len - 0.5  # [G, T]
            mean_lateral = (lateral * within).sum(dim=1) / within.sum(dim=1).clamp_min(
                1
            )
            pk = torch.exp(-0.5 * (mean_lateral / self.pk_sigma_m) ** 2)
            if self.corridor_m is not None:
                center_lateral, center_arc = project_on_path(centers, path)
                center_lateral = center_lateral * (center_arc < path_len - 0.5)
                corridor = (center_lateral.max(dim=1).values < self.corridor_m).float()
            else:
                corridor = torch.ones_like(progress)
        else:
            progress = plans[:, -1, 0]
            mean_lateral = torch.zeros_like(progress)
            pk = torch.ones_like(progress)
            corridor = torch.ones_like(progress)
        center_lateral, straddle = self._lane_centring(plans, centers, width, ctx)
        if self.pk_path == "centerline" and center_lateral is not None:
            pk = torch.exp(-0.5 * (center_lateral / self.pk_sigma_m) ** 2)
        human = ctx.get("human_progress_m")
        use_human = (
            self.ep_reference == "human"
            and human is not None
            and math.isfinite(float(human))
        )
        reference = float(human) if use_human else float(progress.max())
        if reference > self.ep_min_progress_m:
            progress_rel = progress / reference
            ep = progress_rel.clamp(0.0, 1.0)
            if use_human and self.ep_over_ratio is not None:
                threshold = max(
                    self.ep_over_ratio * reference, reference + self.ep_over_margin_m
                )
                over = ((progress - threshold) / (self.ep_over_ramp * reference)).clamp(
                    0.0, 1.0
                )
                ep = ep * (1.0 - over)
        else:
            progress_rel = torch.ones_like(progress)
            ep = torch.ones_like(progress)

        # C: comfort on the smoothed plan, with the current speed from the past trajectory.
        comfort = self._comfort(positions, yaw, ctx)

        if self.mode == "tiered":
            comp = self._tiered_components(
                positions,
                centers,
                yaw,
                length,
                width,
                offset,
                pose_ok,
                collide_pose,
                ttc_lead,
                agents0,
                valid0,
                ctx,
            )
            comp.update(ep=ep, progress_rel=progress_rel, comfort=comfort)
            reward, extra = tiered_reward(comp, self.tiered_params())
            nan = plans.new_tensor(float("nan"))
            has_center = bool(comp["n_trusted"].sum() > 0)
            terms = {
                "nc": (~torch.isfinite(comp["t_collision"])).float().mean(),
                "collision_rate": torch.isfinite(comp["t_collision"]).float().mean(),
                "ttc": (
                    ~(
                        torch.isfinite(comp["t_collision"])
                        | torch.isfinite(comp["ttc_lead"])
                    )
                )
                .float()
                .mean(),
                "dac": (~torch.isfinite(comp["t_offroad"])).float().mean(),
                "dac_frac": dac_frac.mean(),
                "ep": ep.mean(),
                "progress_m": progress.mean(),
                "progress_rel": progress_rel.mean(),
                "pk": extra["centring"].mean(),
                "corridor": (~torch.isfinite(comp["t_corridor"])).float().mean(),
                "lateral_m": comp["late_dist_m"].mean(),
                "comfort": comfort.mean(),
                "center_lateral_m": comp["late_dist_m"].mean() if has_center else nan,
                "straddle": (comp["straddle_frac"] > 0).float().mean()
                if has_center
                else nan,
                "fail_rate": extra["failed"].float().mean(),
                "clearance": comp["clearance"].mean(),
                "corridor_margin": extra["corridor_margin"].mean(),
                "corridor_max_m": comp["corridor_max_m"].mean(),
                "centring": extra["centring"].mean(),
                "heading_err_deg": torch.rad2deg(comp["heading_err"]).mean(),
                "ttc_fire": torch.isfinite(comp["ttc_lead"]).float().mean(),
                "quality": extra["quality"].mean(),
            }
            self.last_components = comp
            return reward, terms

        w_ep, w_ttc, w_pk, w_c = self.weights
        quality = w_ep * ep + w_ttc * ttc + w_pk * pk + w_c * comfort
        total_w = w_ep + w_ttc + w_pk + w_c
        if dm is not None:
            quality = quality + self.w_dm * dm
            total_w = total_w + self.w_dm
        reward = nc * dac * corridor * quality / total_w
        nan = plans.new_tensor(float("nan"))
        terms = {
            "nc": nc.mean(),
            "collision_rate": collided_any.float().mean(),
            "ttc": ttc.mean(),
            "dac": dac.mean(),
            "dac_frac": dac_frac.mean(),  # fraction of poses on the raster (= dac when graded)
            "ep": ep.mean(),
            "progress_m": progress.mean(),
            "progress_rel": progress_rel.mean(),  # progress / EP reference, unclipped (> 1: faster than it)
            "pk": pk.mean(),
            "corridor": corridor.mean(),  # share of plans within corridor_m of the human path (1 when off)
            "lateral_m": mean_lateral.mean(),
            "comfort": comfort.mean(),
            "center_lateral_m": center_lateral.mean()
            if center_lateral is not None
            else nan,
            "straddle": straddle.mean() if straddle is not None else nan,
        }
        if dm is not None:
            terms["dm"] = dm.mean()
            terms["margin_m"] = margin_m.clamp(
                max=2.0
            ).mean()  # capped: open road would dominate the mean
        return reward, terms

    def _drivable_margin(
        self, centers, yaw, length, width, ctx, origin, res
    ) -> torch.Tensor:
        """Smallest distance (m) from the ego outline to a non-drivable (or road-edge) cell over the poses from
        ``dm_start_s`` on, per plan ``[G]``; 0 for a plan that leaves the raster or is off-road there."""
        key = "_dm_edt_edge" if "road_edge" in ctx else "_dm_edt"
        edt = ctx.get(key)
        if edt is None:
            import numpy as np
            from scipy.ndimage import distance_transform_edt

            if "road_edge" in ctx:
                free = ~np.asarray(ctx["road_edge"], dtype=bool)
            else:
                free = np.asarray(ctx["drivable"], dtype=bool)
            # Distance from each free cell centre to the nearest blocked cell centre, minus half a cell: the distance
            # to that cell's near side (0 inside a blocked cell).
            edt = np.clip(distance_transform_edt(free) - 0.5, 0.0, None) * float(res)
            edt = edt.astype(np.float32)
            ctx[key] = edt
        edt = torch.as_tensor(edt, device=centers.device)
        num_poses = centers.shape[1]
        first = min(
            max(int(math.ceil(self.dm_start_s / self.dt - 1e-6)) - 1, 0), num_poses - 1
        )  # pose k at (k+1) dt
        pts = box_outline(
            centers[:, first:], yaw[:, first:], length, width, self.outline_step_m
        )  # [G, T', K, 2]
        idx = torch.floor((pts - origin) / res).long()
        inside = (
            (idx[..., 0] >= 0)
            & (idx[..., 0] < edt.shape[0])
            & (idx[..., 1] >= 0)
            & (idx[..., 1] < edt.shape[1])
        )
        dist = torch.zeros(inside.shape, device=centers.device)
        dist[inside] = edt[idx[..., 0][inside], idx[..., 1][inside]]
        return dist.flatten(1).min(dim=1).values

    def tiered_params(self) -> dict:
        """Constants of ``tiered_reward`` (the weights, thresholds and severities of this reward)."""
        return {
            **self.tiered,
            "horizon_s": self.dt * self.target_sampling.num_poses,
            "ttc_horizon_s": self.ttc_leads[-1] if self.ttc_leads else 0.0,
            "ce_full_m": self.ce["full_m"],
            "ce_zero_m": self.ce["zero_m"],
            "ce_conv_ratio": self.ce["conv_ratio"],
            "ce_conv_tol_m": self.ce["conv_tol_m"],
            "ce_head_zero_rad": self.ce["head_zero_rad"],
        }

    def _tiered_components(
        self,
        positions,
        centers,
        yaw,
        length,
        width,
        offset,
        pose_ok,
        collide_pose,
        ttc_lead,
        agents0,
        valid0,
        ctx,
    ) -> dict:
        """Per-plan raw quantities ``[G]`` of the tiered reward (``tiered_reward`` combines them): first violation
        times (s, inf when none) of the gates, the TTC look-ahead that fires, clearance, distances to the human path
        and to the trusted centreline, heading error and lane-line straddling."""
        device = centers.device
        num_plans, num_poses = centers.shape[:2]
        times = (
            torch.arange(1, num_poses + 1, device=device, dtype=torch.float32) * self.dt
        )
        inf = torch.full((num_plans,), float("inf"), device=device)

        def first_time(flags):  # [G, T] -> [G]
            return torch.where(flags.any(dim=1), times[flags.int().argmax(dim=1)], inf)

        def tangent_angle(
            polyline, idx
        ):  # direction of the polyline at point indices idx
            k = idx.clamp(0, polyline.shape[0] - 2)
            d = polyline[k + 1] - polyline[k]
            return torch.atan2(d[..., 1], d[..., 0])

        path = torch.as_tensor(ctx["path_xy"], device=device, dtype=torch.float32)
        box0 = offset[
            None
        ]  # box centre at t = 0 (plan frame: rig at the origin, heading 0)
        if path.shape[0] >= 2:
            lat_h, arc_h = project_on_path(centers, path)
            lat_h0, _ = project_on_path(box0, path)
            path_cum = torch.cat(
                [path.new_zeros(1), (path[1:] - path[:-1]).norm(dim=-1).cumsum(0)]
            )
            within = (
                arc_h < path_cum[-1] - 0.5
            )  # poses beyond the stored 120 m window measure only its last point
            # When the recording ends inside the horizon, the stored path continues along the route (it can cut
            # through non-drivable areas; the human path itself was off-road under the raster in 19 % of NavTest chunk-5
            # states, mostly the last ones of a rollout), and AlpaSim's corridor check stops at the recording end:
            # poses past the recorded part don't count for the corridor and the human-path distances.
            prog, covered = ctx.get("human_progress_m"), ctx.get("human_covered_s")
            horizon = self.dt * num_poses
            if (
                prog is not None
                and covered is not None
                and math.isfinite(float(prog))
                and float(covered) < horizon - 1e-3
            ):
                _, arc0 = project_on_path(path.new_zeros(1, 2), path)
                rec_end = (
                    float(arc0[0])
                    + float(prog) * max(float(covered), 0.0) / horizon
                    + 1.0
                )
                within = within & (arc_h < rec_end)
            end_idx = torch.searchsorted(path_cum, arc_h[:, -1].contiguous())
            end_dir = tangent_angle(path, end_idx)
        else:
            lat_h = torch.zeros(num_plans, num_poses, device=device)
            lat_h0 = torch.zeros(1, device=device)
            within = torch.zeros(num_plans, num_poses, dtype=torch.bool, device=device)
            end_dir = torch.zeros(num_plans, device=device)
        lat_h_in = lat_h * within
        t_corridor = (
            first_time(lat_h_in >= self.corridor_m)
            if self.corridor_m is not None
            else inf
        )

        # Clearance to actors in the ego's path band, at the plan times.
        clearance = torch.ones(num_plans, device=device)
        speed = (positions[:, 1:] - positions[:, :-1]).norm(dim=-1) / self.dt  # [G, T]
        if agents0 is not None:
            g0, headway, margin = self.cl
            rel = agents0[None] - centers[:, :, None, None, :]  # [G, T, A, 4, 2]
            c, s = torch.cos(yaw)[..., None, None], torch.sin(yaw)[..., None, None]
            x = rel[..., 0] * c + rel[..., 1] * s
            y = -rel[..., 0] * s + rel[..., 1] * c
            band = (y.min(dim=-1).values < width / 2.0 + margin) & (
                y.max(dim=-1).values > -(width / 2.0 + margin)
            )
            relevant = (
                band & (x.mean(dim=-1) > 0.0) & valid0[None]
            )  # [G, T, A]: in the band, centre ahead
            gap = (x.min(dim=-1).values - length / 2.0).clamp_min(0.0)
            ratio = (gap / (g0 + headway * speed)[..., None]).clamp(0.0, 1.0)
            ratio = torch.where(relevant, ratio, torch.ones_like(ratio))
            clearance = ratio.flatten(1).min(dim=1).values

        # Centring: the trusted centreline where there is one, else the human path.
        dist = lat_h.clone()
        dist0 = lat_h0.expand(num_plans).clone()
        trusted = torch.zeros(num_plans, num_poses, dtype=torch.bool, device=device)
        straddle = torch.zeros(num_plans, num_poses, dtype=torch.bool, device=device)
        center = ctx.get("centerline_xy")
        if center is not None and len(center) >= 2:
            cl_xy = torch.as_tensor(center, device=device, dtype=torch.float32)
            half = torch.as_tensor(
                ctx["center_halfwidth"], device=device, dtype=torch.float32
            )
            cum = torch.cat(
                [cl_xy.new_zeros(1), (cl_xy[1:] - cl_xy[:-1]).norm(dim=-1).cumsum(0)]
            )
            if (
                path.shape == cl_xy.shape
            ):  # the centreline is the human path with each point snapped to its lane
                trusted_pt = torch.isfinite(half) & (
                    (cl_xy - path).norm(dim=-1) < self.ce["trust_m"]
                )
            else:
                trusted_pt = torch.isfinite(half)
            lat_c, arc_c = project_on_path(centers, cl_xy)
            idx = torch.searchsorted(cum, arc_c.contiguous()).clamp(
                max=cl_xy.shape[0] - 1
            )
            trusted = trusted_pt[idx] & (arc_c < cum[-1] - 0.5)
            dist = torch.where(trusted, lat_c, lat_h)
            straddle = trusted & (
                lat_c + width / 2.0 > torch.nan_to_num(half[idx], nan=float("inf"))
            )
            lat_c0, arc_c0 = project_on_path(box0, cl_xy)
            idx0 = torch.searchsorted(cum, arc_c0.contiguous()).clamp(
                max=cl_xy.shape[0] - 1
            )
            if bool(trusted_pt[idx0]):
                dist0 = lat_c0.expand(num_plans).clone()
            end_c = tangent_angle(cl_xy, idx[:, -1])
            end_dir = torch.where(trusted[:, -1], end_c, end_dir)
        valid = (
            trusted | within
        )  # a trusted centreline, or the recorded human path within the stored window
        late = (times >= self.ce["late_s"])[None] & valid
        late_dist = (dist * late).sum(dim=1) / late.sum(dim=1).clamp_min(1)
        end_valid = valid[:, -4:]
        end_dist = torch.where(
            end_valid.any(dim=1),
            (dist[:, -4:] * end_valid).sum(dim=1) / end_valid.sum(dim=1).clamp_min(1),
            late_dist,
        )
        heading_err = torch.atan2(
            torch.sin(yaw[:, -1] - end_dir), torch.cos(yaw[:, -1] - end_dir)
        ).abs()
        n_trusted = (trusted & late).sum(dim=1)
        straddle_frac = (straddle & late).sum(dim=1) / n_trusted.clamp_min(1)
        return {
            "t_collision": first_time(collide_pose),
            "t_offroad": first_time(~pose_ok),
            "t_corridor": t_corridor,
            "ttc_lead": ttc_lead,
            "clearance": clearance,
            "corridor_max_m": lat_h_in.max(dim=1).values,
            "late_dist_m": late_dist,
            "end_dist_m": end_dist,
            "start_dist_m": dist0,
            "heading_err": heading_err,
            "straddle_frac": straddle_frac.float(),
            "n_trusted": n_trusted.float(),
        }

    def _lane_centring(self, plans, centers, width, ctx):
        """Mean distance ``[G]`` of the plan poses to ``centerline_xy`` and whether the box crosses a lane line
        ``[G]``; ``(None, None)`` without a centreline. Poses beyond its end are left out, as for PK."""
        if ctx.get("centerline_xy") is None or len(ctx["centerline_xy"]) < 2:
            return None, None
        center = torch.as_tensor(
            ctx["centerline_xy"], device=plans.device, dtype=torch.float32
        )
        cum = torch.cat(
            [center.new_zeros(1), (center[1:] - center[:-1]).norm(dim=-1).cumsum(0)]
        )
        lateral, arc = project_on_path(plans, center)
        within = arc < cum[-1] - 0.5
        mean_lateral = (lateral * within).sum(dim=1) / within.sum(dim=1).clamp_min(1)
        box_lateral, box_arc = project_on_path(centers, center)
        half = torch.as_tensor(
            ctx["center_halfwidth"], device=plans.device, dtype=torch.float32
        )
        half = half[
            torch.searchsorted(cum, box_arc).clamp(max=half.shape[0] - 1)
        ]  # lane half width there
        crosses = (
            (box_lateral + width / 2.0 > half)
            & torch.isfinite(half)
            & (box_arc < cum[-1] - 0.5)
        )
        return mean_lateral, crosses.any(dim=1).float()

    def _comfort(self, positions, yaw, ctx) -> torch.Tensor:
        dt = self.dt
        vel = (positions[:, 1:] - positions[:, :-1]) / dt  # [G, T, 2]
        # Initial velocity in the plan frame: the last past step, when the past is long enough.
        past = torch.as_tensor(
            ctx.get("past_traj_xy", [[0.0, 0.0]]), device=vel.device
        ).float()
        past_dt = float(ctx.get("past_interval_s", dt))
        v0 = (
            (past[-1] - past[-2]) / past_dt if past.shape[0] >= 2 else vel[:, 0].mean(0)
        )
        vel = torch.cat([v0.expand(vel.shape[0], 1, 2), vel], dim=1)
        acc = (vel[:, 1:] - vel[:, :-1]) / dt  # [G, T, 2]
        cos, sin = torch.cos(yaw), torch.sin(yaw)
        lon = acc[..., 0] * cos + acc[..., 1] * sin
        lat = -acc[..., 0] * sin + acc[..., 1] * cos
        jerk = (lon[:, 1:] - lon[:, :-1]) / dt
        mag_jerk = ((acc[:, 1:] - acc[:, :-1]) / dt).norm(dim=-1)  # [G, T - 1]
        yaw_rate = torch.diff(
            torch.cat([yaw.new_zeros(yaw.shape[0], 1), yaw], dim=1), dim=1
        )
        yaw_rate = torch.atan2(torch.sin(yaw_rate), torch.cos(yaw_rate)) / dt
        yaw_accel = (yaw_rate[:, 1:] - yaw_rate[:, :-1]) / dt  # [G, T - 1]
        lim = self.comfort_limits
        ok = (
            (lon.min(dim=1).values >= lim["min_lon_accel"])
            & (lon.max(dim=1).values <= lim["max_lon_accel"])
            & (lat.abs().max(dim=1).values <= lim["max_abs_lat_accel"])
            & (mag_jerk.max(dim=1).values <= lim["max_abs_mag_jerk"])
            & (jerk.abs().max(dim=1).values <= lim["max_abs_lon_jerk"])
            & (yaw_rate.abs().max(dim=1).values <= lim["max_abs_yaw_rate"])
            & (yaw_accel.abs().max(dim=1).values <= lim["max_abs_yaw_accel"])
        )
        return ok.float()


def tiered_reward(c: dict, p: dict) -> tuple[torch.Tensor, dict]:
    """Tiered reward ``[G]`` from the per-plan components ``c`` (``SimPDMReward._tiered_components`` plus ``ep``,
    ``comfort``) and the constants ``p`` (``SimPDMReward.tiered_params``); see the module docstring. A pure function of
    stored components, so a state selection can store them once and re-weight offline."""
    horizon = p["horizon_s"]
    values = []
    for key, sev in (
        ("t_collision", "sev_collision"),
        ("t_corridor", "sev_corridor"),
        ("t_offroad", "sev_offroad"),
    ):
        t = c[key]
        values.append(
            torch.where(
                torch.isfinite(t),
                -p[sev] + p["fail_time_slope"] * t / horizon,
                torch.full_like(t, float("inf")),
            )
        )
    fail_value = torch.stack(values).min(dim=0).values
    failed = torch.isfinite(fail_value)

    cm_span = max(p["cm_zero_m"] - p["cm_full_m"], 1e-6)
    corridor_margin = ((p["cm_zero_m"] - c["corridor_max_m"]) / cm_span).clamp(0.0, 1.0)
    ce_span = max(p["ce_zero_m"] - p["ce_full_m"], 1e-6)
    q_dist = (1.0 - (c["late_dist_m"] - p["ce_full_m"]) / ce_span).clamp(0.0, 1.0)
    target = (p["ce_conv_ratio"] * c["start_dist_m"]).clamp_min(p["ce_full_m"])
    q_conv = (1.0 - (c["end_dist_m"] - target) / p["ce_conv_tol_m"]).clamp(0.0, 1.0)
    q_head = (1.0 - c["heading_err"] / p["ce_head_zero_rad"]).clamp(0.0, 1.0)
    q_str = 1.0 - c["straddle_frac"]
    centring = 0.4 * q_dist + 0.2 * q_conv + 0.2 * q_head + 0.2 * q_str

    weights = (p["w_ep"], p["w_cl"], p["w_ce"], p["w_cm"], p["w_c"])
    quality = (
        weights[0] * c["ep"]
        + weights[1] * c["clearance"]
        + weights[2] * centring
        + weights[3] * corridor_margin
        + weights[4] * c["comfort"]
    ) / max(sum(weights), 1e-6)
    ttc_fire = c["ttc_lead"] <= p["ttc_horizon_s"] + 1e-6
    m_ttc = torch.where(
        ttc_fire, torch.full_like(quality, p["ttc_factor"]), torch.ones_like(quality)
    )
    reward = torch.where(failed, fail_value, m_ttc * quality)
    return reward, {
        "failed": failed,
        "fail_value": fail_value,
        "quality": quality,
        "m_ttc": m_ttc,
        "corridor_margin": corridor_margin,
        "centring": centring,
        "q_dist": q_dist,
        "q_conv": q_conv,
        "q_head": q_head,
        "q_str": q_str,
    }


__all__ = [
    "SimPDMReward",
    "poly_smooth",
    "boxes_overlap",
    "project_on_path",
    "headings",
    "tiered_reward",
]
