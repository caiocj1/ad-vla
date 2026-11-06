"""AgentInfo boxes, the box store, the tracker and the collision reward, on synthetic data with known answers."""

import numpy as np
import pytest
import torch

from ad_vla.dataset.agent_boxes import (
    AgentBoxStore,
    box_corners_xyz,
    boxes_to_bev,
    camera_boxes_to_ego,
    coarse_type,
)
from ad_vla.dataset.agent_tracking import (
    align_2d,
    anchors_dense,
    anchors_from_tracks,
    dedupe_boxes,
    from_common,
    to_common,
    smooth_dense,
    track_chain,
    track_dense,
    waymo_relative_pose,
)
from ad_vla.dataset.data_types import AgentInfo
from ad_vla.rewards.agent_collision_reward import AgentCollisionReward, agent_ctx


def info(boxes, future=None, mask=None, types=None):
    boxes = torch.tensor(boxes, dtype=torch.float32).reshape(-1, 7)
    n = len(boxes)
    fut = (
        torch.tensor(future, dtype=torch.float32)
        if future is not None
        else boxes[:, None].repeat(1, 10, 1)
    )
    m = (
        torch.tensor(mask, dtype=torch.bool)
        if mask is not None
        else torch.ones(n, fut.shape[1], dtype=torch.bool)
    )
    return AgentInfo(
        boxes=boxes,
        coarse_types=types or ["vehicle"] * n,
        fut_boxes=fut,
        fut_mask=m,
        fut_interval_s=0.5,
        source="gt",
    )


def test_none_and_empty_are_different():
    assert AgentInfo(boxes=None).boxes is None
    assert len(AgentInfo(boxes=torch.zeros(0, 7)).boxes) == 0


def test_box_shape_is_validated():
    with pytest.raises(ValueError):
        AgentInfo(boxes=torch.zeros(3, 6))
    with pytest.raises(ValueError):
        AgentInfo(boxes=torch.zeros(3, 7), coarse_types=["vehicle"])
    with pytest.raises(ValueError):
        AgentInfo(boxes=torch.zeros(3, 7), fut_boxes=torch.zeros(3, 4, 7))


def test_coarse_types():
    assert coarse_type("vehicle.car") == "vehicle"
    assert coarse_type("vehicle.bicycle") == "cyclist"
    assert coarse_type("human.pedestrian.adult") == "person"
    assert coarse_type("movable_object.barrier") == "other"
    assert coarse_type("automobile") == "vehicle"
    assert coarse_type("generic_object") == "other"


def test_corners_follow_yaw_and_size():
    corners = box_corners_xyz(
        torch.tensor([[10.0, 0.0, 1.0, 4.0, 2.0, 2.0, np.pi / 2]])
    )[0]
    assert corners[:, 0].min() == pytest.approx(
        9.0, abs=1e-5
    )  # width along x after a quarter turn
    assert corners[:, 1].max() == pytest.approx(2.0, abs=1e-5)  # length along y
    assert corners[:, 2].max() == pytest.approx(2.0, abs=1e-5)


def test_camera_box_to_ego():
    # Camera at (1, 0, 1.5) looking forward: optical z = ego x, optical x = -ego y, optical y = -ego z.
    rotation = np.array([[0.0, 0.0, 1.0], [-1.0, 0.0, 0.0], [0.0, -1.0, 0.0]])
    box = np.array([[2.0, 1.0, 20.0, 1.8, 4.5, 1.5, 1.0, 0.0, 0.0, 0.0]])
    out = camera_boxes_to_ego(box, rotation, np.array([1.0, 0.0, 1.5]))[0]
    assert out[:3] == pytest.approx([21.0, -2.0, 0.5])
    assert out[3:6] == pytest.approx([4.5, 1.8, 1.5])


def test_agent_ctx_interpolates_and_drops_unknown():
    future = np.zeros((1, 10, 7), dtype=np.float32)
    future[0, :, 3:6] = [4, 2, 1.5]
    future[0, :, 0] = 10 + np.arange(1, 11) * 5.0  # 10 m/s
    mask = np.ones((1, 10), dtype=bool)
    mask[0, 4:] = False
    ctx = agent_ctx(info([[10, 0, 0, 4, 2, 1.5, 0]], future, mask), 20, 0.25)
    centre = ctx["agent_corners"].mean(dim=2)
    assert centre[0, 0, 0] == pytest.approx(
        12.5
    )  # 0.25 s, between now and the first step
    assert centre[1, 0, 0] == pytest.approx(15.0)
    assert ctx["agent_valid"][7, 0]  # 2.0 s
    assert not ctx["agent_valid"][8, 0]  # 2.25 s runs into the lost track
    assert not ctx["agent_valid"][-1, 0]


def plan(speed, n=20, dt=0.25):
    t = torch.arange(1, n + 1) * dt
    return torch.stack([t * speed, 0 * t], dim=-1)


def test_reward_detects_collision_with_a_stopped_car_and_clears_the_rest():
    reward = AgentCollisionReward()
    car = info([[30, 0, 0, 4.5, 2, 1.5, 0]])
    scores = reward.score(torch.stack([plan(15.0), plan(2.0)]), car)
    assert scores.tolist() == [0.0, 1.0]


def test_reward_follows_a_moving_lead_that_a_static_box_would_flag():
    future = np.zeros((1, 10, 7), dtype=np.float32)
    future[0, :, 3:6] = [4.5, 2, 1.5]
    future[0, :, 0] = 15 + np.arange(1, 11) * 0.5 * 10.0
    moving = info([[15, 0, 0, 4.5, 2, 1.5, 0]], future)
    still = info([[15, 0, 0, 4.5, 2, 1.5, 0]])
    follow = plan(10.0)[None]
    reward = AgentCollisionReward()
    assert reward.score(follow, moving).item() == 1.0
    assert reward.score(follow, still).item() == 0.0


def test_reward_ignores_unknown_future_other_classes_and_missing_annotations():
    reward = AgentCollisionReward()
    fast = plan(15.0)[None]
    box = [[30, 0, 0, 4.5, 2, 1.5, 0]]
    lost = info(box, mask=np.zeros((1, 10), dtype=bool))
    assert reward.score(fast, lost).item() == 1.0
    assert reward.score(fast, info(box, types=["other"])).item() == 1.0
    assert reward.score(fast, AgentInfo(boxes=None)).item() == 1.0
    assert reward.score(fast, None).item() == 1.0


def test_reward_car_behind_is_not_at_fault():
    assert (
        AgentCollisionReward()
        .score(plan(0.0)[None], info([[0.5, 0, 0, 4.5, 2, 1.5, 0]]))
        .item()
        == 1.0
    )


def test_align_2d_recovers_a_rigid_motion():
    yaw, xy = 0.3, np.array([5.0, -2.0])
    src = np.array([[0.0, 0.0], [5.0, 0.2], [10.0, 1.0], [15.0, 2.5]])
    c, s = np.cos(yaw), np.sin(yaw)
    dst = src @ np.array([[c, s], [-s, c]]) + xy
    got_yaw, got_xy = align_2d(src, dst)
    assert got_yaw == pytest.approx(yaw)
    assert got_xy == pytest.approx(xy)


def test_align_2d_stationary_keeps_rotation_zero():
    src = np.zeros((5, 2)) + 0.01
    assert align_2d(src, src + 1.0)[0] == 0.0


def test_relative_pose_from_overlapping_states():
    # Ego drives along a curve; two records hold the same world path in their own frames.
    t = np.arange(-4.0, 5.01, 0.25)
    heading = 0.05 * t
    world = np.stack(
        [
            np.cumsum(10.0 * np.cos(heading)) * 0.25,
            np.cumsum(10.0 * np.sin(heading)) * 0.25,
        ],
        -1,
    )
    now_a = 16
    now_b = 18

    def local(now):
        c, s = np.cos(heading[now]), np.sin(heading[now])
        return (world - world[now]) @ np.array([[c, -s], [s, c]])

    a, b = local(now_a), local(now_b)
    # As in the records: past ends at the frame itself.
    yaw, xy = waymo_relative_pose(
        a[now_a - 15 : now_a + 1],
        a[now_a + 1 : now_a + 21],
        b[now_b - 15 : now_b + 1],
        2,
    )
    assert yaw == pytest.approx(heading[now_b] - heading[now_a], abs=1e-6)
    assert xy == pytest.approx(a[now_b], abs=1e-6)
    # Without a future (test split), the shared past is enough.
    yaw, xy = waymo_relative_pose(
        a[now_a - 15 : now_a + 1], None, b[now_b - 15 : now_b + 1], 2
    )
    assert yaw == pytest.approx(heading[now_b] - heading[now_a], abs=1e-6)


def test_common_frame_round_trip():
    boxes = np.array([[3.0, 4.0, 1.0, 4.5, 2.0, 1.5, 0.2]], dtype=np.float32)
    back = from_common(
        to_common(boxes, 0.4, np.array([1.0, -2.0])), 0.4, np.array([1.0, -2.0])
    )
    assert back == pytest.approx(boxes, abs=1e-5)


def scene(n_frames, ego_speed=0.0, lead_speed=10.0):
    """A lead car, a parked car and a person; boxes in each frame's ego frame, ego driving straight."""
    boxes, classes, poses = [], [], []
    for f in range(n_frames):
        ego_x = ego_speed * 0.5 * f
        boxes.append(
            np.array(
                [
                    [20 + lead_speed * 0.5 * f - ego_x, 0, 0.8, 4.5, 2, 1.5, 0],
                    [40 - ego_x, 6, 0.8, 4.5, 2, 1.5, 0],
                    [15 - ego_x, -8 + 0.7 * f, 0.9, 0.7, 0.7, 1.7, 1.5],
                ],
                dtype=np.float32,
            )
        )
        classes.append(np.array([0, 0, 1]))
        poses.append((0.0, np.array([ego_x, 0.0])))
    return boxes, classes, poses


def test_tracker_keeps_identity_and_futures_are_in_the_anchor_frame():
    boxes, classes, poses = scene(12, ego_speed=8.0)
    tracks = track_chain(boxes, classes, poses, 0.5)
    assert len(tracks) == 3
    assert all(len(t.frames) == 12 for t in tracks)
    out = anchors_from_tracks(tracks, 12, poses, 10, [3] * 12)
    assert out[0]["fut_mask"][0].all()
    # Lead at frame 3, seen from frame 0: 20 + 10 * 1.5 = 35 ahead of where the ego was.
    assert out[0]["fut_boxes"][0, 2, 0] == pytest.approx(35.0, abs=1e-3)
    # Parked car stays at 40 in frame 0's coordinates.
    assert out[0]["fut_boxes"][1, :, 0] == pytest.approx(np.full(10, 40.0), abs=1e-3)


def test_tracker_bridges_a_missed_frame_and_marks_the_end_unknown():
    boxes, classes, poses = scene(8)
    boxes[3], classes[3] = boxes[3][1:], classes[3][1:]  # the lead is missed once
    tracks = track_chain(boxes, classes, poses, 0.5)
    lead = next(t for t in tracks if t.frames[0] == 0 and t.detections[0] == 0)
    assert lead.frames == [0, 1, 2, 4, 5, 6, 7]
    out = anchors_from_tracks(tracks, 8, poses, 10, [len(b) for b in boxes])
    assert out[0]["fut_mask"][0, :7].all() and not out[0]["fut_mask"][0, 7:].any()
    assert out[0]["fut_boxes"][0, 2, 0] == pytest.approx(
        35.0, abs=0.01
    )  # interpolated frame


def test_tracker_does_not_swap_classes():
    boxes = [
        np.array([[10, 0, 0, 4, 2, 1.5, 0]], dtype=np.float32),
        np.array([[10.2, 0, 0, 0.7, 0.7, 1.7, 0]], dtype=np.float32),
    ]
    tracks = track_chain(
        boxes, [np.array([0]), np.array([1])], [(0.0, np.zeros(2))] * 2, 0.5
    )
    assert len(tracks) == 2


def test_first_step_does_not_jump_sideways_between_parked_cars():
    yaw0 = lambda x, y: np.array([[x, y, 0, 4.5, 2, 1.5, 0]], dtype=np.float32)  # noqa: E731
    poses = [(0.0, np.zeros(2))] * 2
    classes = [np.array([0]), np.array([0])]
    sideways = track_chain([yaw0(10, 0), yaw0(10, 8)], classes, poses, 0.5)
    assert len(sideways) == 2
    along = track_chain(
        [yaw0(10, 0), yaw0(22, 0.5)], classes, poses, 0.5
    )  # 24 m/s, along the car
    assert len(along) == 1


def test_single_detection_has_unknown_future():
    boxes = [
        np.array([[10, 0, 0, 4, 2, 1.5, 0]], dtype=np.float32),
        np.zeros((0, 7), dtype=np.float32),
    ]
    tracks = track_chain(
        boxes, [np.array([0]), np.zeros(0, dtype=int)], [(0.0, np.zeros(2))] * 2, 0.5
    )
    out = anchors_from_tracks(tracks, 2, [(0.0, np.zeros(2))] * 2, 10, [1, 0])
    assert not out[0]["fut_mask"].any()
    assert out[0]["track_ids"][0] >= 0


def test_dedupe_keeps_the_best_of_each_class():
    boxes = np.array(
        [
            [10, 0, 0, 4, 2, 1, 0],
            [10.5, 0, 0, 4, 2, 1, 0],
            [10.5, 0, 0, 1, 1, 2, 0],
            [30, 0, 0, 4, 2, 1, 0],
        ],
        dtype=np.float32,
    )
    keep = dedupe_boxes(boxes, np.array([0.5, 0.9, 0.8, 0.6]), np.array([0, 0, 1, 0]))
    assert sorted(keep.tolist()) == [1, 2, 3]


def write_store(root):
    directory = root / "val"
    directory.mkdir()
    ids = np.array(["aaa-001", "bbb-002", "ccc-003"])
    np.save(directory / "scenario_ids.npy", ids)
    np.save(directory / "offsets.npy", np.array([0, 2, 2, 3], dtype=np.int64))
    boxes = np.arange(21, dtype=np.float32).reshape(3, 7)
    np.save(directory / "boxes.npy", boxes)
    np.save(directory / "scores.npy", np.array([0.9, 0.4, 1.2], dtype=np.float32))
    np.save(directory / "class_ids.npy", np.array([0, 0, 1], dtype=np.uint8))
    np.save(directory / "track_ids.npy", np.array([5, 6, 7], dtype=np.int32))
    np.save(directory / "fut_boxes.npy", np.zeros((3, 10, 7), dtype=np.float16))
    np.save(directory / "fut_mask.npy", np.zeros((3, 10), dtype=bool))
    visible = np.zeros((3, 2), dtype=np.uint8)
    visible[0, 0] = 0b10000000
    np.save(directory / "visible_bev.npy", visible)
    (directory / "meta.json").write_text(
        '{"source": "wilddet3d", "class_names": ["vehicle", "person"], "fut_interval_s": 0.5, "bev_shape": [4, 4]}'
    )


def test_store_finds_filters_and_separates_unknown_from_empty(tmp_path):
    write_store(tmp_path)
    store = AgentBoxStore(str(tmp_path), "val", min_score=0.5)
    first = AgentInfo(**store.agent_info("aaa-001"))
    assert first.boxes.tolist() == [list(range(7))]  # the 0.4 score is filtered out
    assert first.coarse_types == ["vehicle"] and first.source == "wilddet3d"
    assert (
        first.visible_bev.shape == (4, 4)
        and first.visible_bev[0, 0]
        and first.visible_bev.sum() == 1
    )
    empty = AgentInfo(**store.agent_info("bbb-002"))
    assert empty.boxes.shape == (0, 7)
    assert AgentInfo(**store.agent_info("ccc-003")).coarse_types == ["person"]
    assert store.agent_info("zzz-999")["boxes"] is None
    assert store.agent_info("aaa-000")["boxes"] is None


def test_bev_axes_and_yaw():
    bev = boxes_to_bev(torch.tensor([[10.0, 5.0, 1.0, 4.0, 2.0, 1.5, 0.0]]))
    cells = torch.nonzero(bev)
    assert bev.shape == (120, 120)
    assert cells[:, 0].min() == 16 and cells[:, 0].max() == 23  # x 8 to 12
    assert cells[:, 1].min() == 68 and cells[:, 1].max() == 71  # y 4 to 6, j grows left
    turned = torch.nonzero(
        boxes_to_bev(torch.tensor([[10.0, 5.0, 1.0, 4.0, 2.0, 1.5, np.pi / 2]]))
    )
    assert (
        turned[:, 0].max() - turned[:, 0].min() == 3
        and turned[:, 1].max() - turned[:, 1].min() == 7
    )
    assert not boxes_to_bev(torch.zeros(0, 7)).any()


def dense_scene(n=60, ego_speed=8.0, noise=0.0, seed=0):
    """10 Hz: a lead at 10 m/s, parked cars in a row along the road, a person crossing; ego drives straight."""
    rng = np.random.default_rng(seed)
    times = np.arange(n) * 0.1
    boxes, classes, poses = [], [], []
    for f, t in enumerate(times):
        ego_x = ego_speed * t
        rows = [[20 + 10 * t, 0, 0.8, 4.5, 2, 1.5, 0]]
        rows += [[15 + 7.0 * i, 6, 0.8, 4.5, 2, 1.5, 0] for i in range(4)]
        rows += [[30, -8 + 1.4 * t, 0.9, 0.7, 0.7, 1.7, 1.57]]
        world = np.array(rows, dtype=np.float32)
        world[:, :2] += rng.normal(0, noise, (len(world), 2))
        own = world.copy()
        own[:, 0] -= ego_x
        boxes.append(own)
        classes.append(np.array([0, 0, 0, 0, 0, 1]))
        poses.append((0.0, np.array([ego_x, 0.0])))
    return boxes, classes, poses, times


def test_dense_keeps_identities_under_noise_and_holds_parked_cars_still():
    boxes, classes, poses, times = dense_scene(noise=0.5)
    tracks = track_dense(boxes, classes, poses, times)
    # Vehicles only: 0.5 m of noise is large for a person, whose gate is tighter, so that track may fragment.
    long_tracks = [t for t in tracks if len(t.frames) >= 55 and t.cls == 0]
    assert len(long_tracks) == 5
    parked = [
        t for t in long_tracks if abs(float(np.median([b[1] for b in t.boxes])) - 6) < 1
    ]
    assert len(parked) == 4
    for track in parked:
        smooth = smooth_dense(track, times)
        assert np.ptp(smooth[:, 0]) < 1e-3 and np.ptp(smooth[:, 1]) < 1e-3


def test_dense_futures_land_on_the_true_positions():
    boxes, classes, poses, times = dense_scene(noise=0.2)
    tracks = track_dense(boxes, classes, poses, times)
    step_times = [None] * len(times)
    step_times[10] = times[10] + 0.5 * np.arange(
        1, 11
    )  # to 6.0 s, past the last frame at 5.9 s
    out = anchors_dense(tracks, times, poses, [len(b) for b in boxes], step_times)
    assert out[0] is None
    lead = int(np.argmin(np.hypot(boxes[10][:, 0] - (30 - 8), boxes[10][:, 1])))
    assert out[10]["fut_mask"][lead, :9].all() and not out[10]["fut_mask"][lead, 9]
    # Lead at 1 + 2 s: world 20 + 30 = 50; seen from frame 10, where the ego was at 8.
    assert out[10]["fut_boxes"][lead, 3, 0] == pytest.approx(50 - 8, abs=0.5)


def test_dense_bridges_a_gap_within_the_limit_and_stops_at_it():
    boxes, classes, poses, times = dense_scene(n=40, noise=0.0)
    for f in range(10, 15):
        boxes[f], classes[f] = boxes[f][1:], classes[f][1:]  # lead missed for 0.5 s
    tracks = track_dense(boxes, classes, poses, times)
    lead = max(
        (t for t in tracks if t.cls == 0 and abs(t.boxes[0][1]) < 1),
        key=lambda t: len(t.frames),
    )
    assert len(lead.frames) == 35
    for f in range(10, 30):
        boxes[f], classes[f] = boxes[f][:0], classes[f][:0]
    assert not any(
        9 in t.frames and 30 in t.frames
        for t in track_dense(boxes, classes, poses, times)
    )


def test_dense_does_not_jump_sideways():
    box = lambda x, y: np.array([[x, y, 0, 4.5, 2, 1.5, 0]], dtype=np.float32)  # noqa: E731
    times = np.array([0.0, 0.1])
    poses = [(0.0, np.zeros(2))] * 2
    classes = [np.array([0]), np.array([0])]
    assert len(track_dense([box(10, 0), box(10, 6)], classes, poses, times)) == 2
    assert len(track_dense([box(10, 0), box(13, 0.2)], classes, poses, times)) == 1


def test_fractional_relative_pose_matches_the_true_motion():
    t = np.arange(-4.0, 5.01, 0.05)
    heading = 0.08 * t
    world = np.stack(
        [
            np.cumsum(10.0 * np.cos(heading)) * 0.05,
            np.cumsum(10.0 * np.sin(heading)) * 0.05,
        ],
        -1,
    )
    now_a = int(np.argmin(np.abs(t)))
    now_b = now_a + 2  # 0.1 s later, 0.4 state steps of 0.25 s

    def states(now, first, last):
        c, s = np.cos(heading[now]), np.sin(heading[now])
        local = (world - world[now]) @ np.array([[c, -s], [s, c]])
        return local[now + first * 5 : now + last * 5 + 1 : 5]

    yaw, xy = waymo_relative_pose(
        states(now_a, -15, 0), states(now_a, 1, 20), states(now_b, -15, 0), 0.4
    )
    assert yaw == pytest.approx(heading[now_b] - heading[now_a], abs=1e-3)
    c, s = np.cos(heading[now_a]), np.sin(heading[now_a])
    truth = (world[now_b] - world[now_a]) @ np.array([[c, -s], [s, c]])
    assert xy == pytest.approx(truth, abs=0.01)


def test_ftheta_undistortion_sends_rays_where_the_lens_does():
    from ad_vla.dataset.physical_ai.physical_ai_dataset import ftheta_to_pinhole_maps

    fw_poly = np.array([0.0, 1000.0, -30.0, 8.0, 0.0])
    intrinsics = np.array([[1000.0, 0, 960.0], [0, 1000.0, 500.0], [0, 0, 1]])
    map_x, map_y, k = ftheta_to_pinhole_maps(fw_poly, intrinsics, 1080, 35.0, 19.5)
    height, width = map_x.shape
    # Centred principal point, same focal, +-35 deg across.
    assert k[0, 2] == pytest.approx(width / 2) and k[1, 2] == pytest.approx(height / 2)
    assert k[0, 0] == 1000.0 and width == pytest.approx(
        2 * 1000 * np.tan(np.radians(35)), abs=1
    )
    # The image centre looks along the axis, so it lands on the real principal point.
    assert map_x[height // 2, width // 2] == pytest.approx(960.0, abs=1.0)
    assert map_y[height // 2, width // 2] == pytest.approx(500.0, abs=1.0)
    # A point 30 deg off the axis: pinhole pixel by tan, source pixel by the lens polynomial.
    u = int(round(k[0, 2] + 1000 * np.tan(np.radians(30))))
    expected = 960.0 + np.polynomial.polynomial.polyval(np.radians(30), fw_poly)
    assert map_x[height // 2, u] == pytest.approx(expected, abs=2.0)
    assert map_y[height // 2, u] == pytest.approx(500.0, abs=1.0)
    # The requested vertical field sets the framing wherever the principal point is.
    assert height == pytest.approx(2 * 1000 * np.tan(np.radians(19.5)), abs=1)
    other_kit = np.array([[1000.0, 0, 960.0], [0, 1000.0, 700.0], [0, 0, 1]])
    assert (
        ftheta_to_pinhole_maps(fw_poly, other_kit, 1080, 35.0, 19.5)[0].shape
        == map_x.shape
    )
    # Asking for more than the lens has below the axis clamps it, and stays inside the frame.
    tall_y = ftheta_to_pinhole_maps(fw_poly, other_kit, 1080, 35.0, 40.0)[1]
    assert 0 <= tall_y.min() and tall_y.max() < 1080
