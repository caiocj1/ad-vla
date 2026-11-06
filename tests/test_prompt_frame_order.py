"""Image order and frame labels with separate cameras, in the prompt and in the token geometry."""

from types import SimpleNamespace

import torch

from ad_vla.utils import camera_geometry
from ad_vla.utils.prompts import PromptBuilder

CAMERAS = ["FRONT_LEFT", "FRONT"]


def _frames(num_frames: int) -> list[dict]:
    return [
        {
            c: SimpleNamespace(image=torch.full((1, 1, 3), t * 10 + i))
            for i, c in enumerate(CAMERAS)
        }
        for t in range(num_frames)
    ]


def _layout(num_frames: int, num_past_image_frames: int) -> tuple[str, list[int]]:
    content = PromptBuilder.build_user_content(
        None,
        _frames(num_frames),
        torch.zeros(1, 2),
        "GO_STRAIGHT",
        frozenset(),
        CAMERAS,
        num_past_image_frames,
        concat_cameras=False,
    )[1:]
    text = "".join(c["text"] if c["type"] == "text" else "<image>" for c in content)
    return text, [int(c["image"].flatten()[0]) for c in content if c["type"] == "image"]


def test_multi_frame_groups_by_camera_with_labels() -> None:
    text, images = _layout(2, 2)
    assert (
        text
        == "FRONT_LEFT:<frame -1><image><frame 0><image>FRONT:<frame -1><image><frame 0><image>"
    )
    assert images == [0, 10, 1, 11]


def test_clip_start_keeps_labels_and_single_frame_config_is_unchanged() -> None:
    assert _layout(1, 3)[0] == "FRONT_LEFT:<frame 0><image>FRONT:<frame 0><image>"
    assert _layout(1, 1)[0] == "FRONT_LEFT:<image>FRONT:<image>"


def test_token_geometry_follows_prompt_order(monkeypatch) -> None:
    calls = []

    def fake_features(image_cameras, grid, merge_size, frame_id, require_calibration):
        calls.append((image_cameras[0], frame_id))
        return (
            torch.zeros(1, camera_geometry.NUM_GEOMETRY_FEATURES),
            torch.ones(1, dtype=torch.bool),
        )

    monkeypatch.setattr(camera_geometry, "image_token_features", fake_features)
    frames = [{c: f"{c}@{t}" for c in CAMERAS} for t in range(2)]
    camera_geometry.token_geometry(
        [frames],
        CAMERAS,
        2,
        False,
        torch.ones(1, 4, dtype=torch.long),
        torch.ones(4, 3, dtype=torch.long),
        1,
        2,
    )
    assert calls == [
        ("FRONT_LEFT@0", -1),
        ("FRONT_LEFT@1", 0),
        ("FRONT@0", -1),
        ("FRONT@1", 0),
    ]
