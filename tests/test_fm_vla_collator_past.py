"""FlowMatchingVLACollator past trajectory: separate prompt and expert grids (CPU, no backbone)."""

from types import SimpleNamespace

import torch

from ad_vla.dataset.data_types import TrajectorySampling
from ad_vla.models.vlas.fm_vla import FlowMatchingVLACollator
from ad_vla.utils.prompts import ALL_INPUTS


class _PromptBuilder:
    """Writes the past trajectory as the real prompt does ("Past ego trajectory:[[x, y], ...]")."""

    def __init__(self, active_inputs=ALL_INPUTS):
        self.active_inputs = active_inputs

    def sample_active_inputs(self, drop_cfg):
        return self.active_inputs

    def build_message(self, *, past_traj, active_inputs, **kwargs):
        content = []
        if "past_traj" in active_inputs:
            poses = ", ".join(
                f"[{x:.1f}, {y:.1f}]" for x, y in past_traj[:, :2].tolist()
            )
            content.append({"type": "text", "text": f"Past ego trajectory:[{poses}]"})
        return [{"role": "user", "content": content}]


def _collator(prompt_builder=None, drop_cfg=None, **kwargs) -> FlowMatchingVLACollator:
    return FlowMatchingVLACollator(
        processor=None,
        prompt_builder=prompt_builder or _PromptBuilder(),
        past_traj_sampling=TrajectorySampling(num_poses=16, interval_length=0.25),
        pred_traj_sampling=TrajectorySampling(num_poses=20, interval_length=0.25),
        camera_sequence=[],
        num_past_image_frames=1,
        drop_cfg=drop_cfg,
        **kwargs,
    )


def _sample(num_poses: int, freq: float = 2.0) -> SimpleNamespace:
    # Constant 10 m/s along +x, present at the origin.
    t = torch.arange(-num_poses + 1, 1, dtype=torch.float32) / freq
    past = torch.stack((10.0 * t, torch.zeros_like(t)), dim=-1)
    return SimpleNamespace(
        agent_input={"past_traj": past},
        metadata={"sampling_freq": freq},
        cameras=[],
        intent=None,
    )


def _past_text(message) -> str:
    return message[0]["content"][0]["text"]


def test_separate_prompt_and_expert_grids_with_missing_history():
    collator = _collator(
        prompt_past_traj_sampling=TrajectorySampling(num_poses=4, interval_length=0.5),
        pad_missing_past_traj=True,
    )
    # 1 s of history at 2 Hz (3 poses: -1.0, -0.5, 0.0 s).
    message, past, pose_valid, active = collator._get_sample_message(_sample(3))

    assert _past_text(message) == (
        "Past ego trajectory:[[X.X, X.X], [-10.0, 0.0], [-5.0, 0.0], [0.0, 0.0]]"
    )
    assert active
    # Expert grid: 16 x 0.25 s; poses back to -1.0 s (5 poses) are real.
    assert past.shape == (16, 2)
    assert pose_valid.tolist() == [False] * 11 + [True] * 5
    torch.testing.assert_close(
        past[11:, 0], torch.tensor([-10.0, -7.5, -5.0, -2.5, 0.0])
    )


def test_full_history_has_no_padding():
    collator = _collator(
        prompt_past_traj_sampling=TrajectorySampling(num_poses=4, interval_length=0.5),
        pad_missing_past_traj=True,
    )
    message, past, pose_valid, _ = collator._get_sample_message(_sample(9))
    assert "X.X" not in _past_text(message)
    assert _past_text(message).count("[") == 5  # 4 poses
    assert pose_valid.all()
    torch.testing.assert_close(past[0, 0], torch.tensor(-37.5))


def test_dropped_past_is_not_written_and_marks_row_invalid():
    collator = _collator(
        prompt_builder=_PromptBuilder(active_inputs=ALL_INPUTS - {"past_traj"}),
        drop_cfg={"past_traj": 1.0},
        prompt_past_traj_sampling=TrajectorySampling(num_poses=4, interval_length=0.5),
        pad_missing_past_traj=True,
    )
    message, _, _, active = collator._get_sample_message(_sample(3))
    assert message[0]["content"] == []
    assert not active


def test_default_shares_the_grid_and_does_not_write_placeholders():
    collator = _collator()
    message, past, pose_valid, _ = collator._get_sample_message(_sample(3))
    assert "X.X" not in _past_text(message)
    assert _past_text(message).count("[") == 6  # 5 poses on the 0.25 s grid
    assert pose_valid.sum() == 5
    # Legacy left padding with the oldest pose.
    torch.testing.assert_close(past[:11], past[11:12].expand(11, -1))


def test_initial_speed_of_single_pose_history_is_zero():
    from ad_vla.models.vlas.fm_vla import initial_speed

    assert initial_speed(_sample(1)).item() == 0.0
    torch.testing.assert_close(initial_speed(_sample(3)), torch.tensor(10.0))


def test_single_pose_history_marks_all_older_expert_poses_missing():
    collator = _collator(
        prompt_past_traj_sampling=TrajectorySampling(num_poses=4, interval_length=0.5),
        pad_missing_past_traj=True,
    )
    message, past, pose_valid, _ = collator._get_sample_message(_sample(1))
    assert _past_text(message) == (
        "Past ego trajectory:[[X.X, X.X], [X.X, X.X], [X.X, X.X], [0.0, 0.0]]"
    )
    assert pose_valid.tolist() == [False] * 15 + [True]
    assert past.abs().max() == 0


class _Sample(SimpleNamespace):
    def model_copy(self, update):
        return _Sample(**{**vars(self), **update})


def test_past_truncation_shortens_some_samples_and_keeps_the_present():
    import random

    random.seed(0)
    collator = _collator(past_truncation_prob=0.25)
    sample = _Sample(**vars(_sample(9)))
    lengths = [
        collator._maybe_truncate_past(sample).agent_input["past_traj"].shape[0]
        for _ in range(4000)
    ]
    truncated = [n for n in lengths if n < 9]
    assert 0.22 < len(truncated) / len(lengths) < 0.28
    assert set(truncated) == set(range(1, 9))
    assert sample.agent_input["past_traj"].shape[0] == 9  # dataset sample untouched
    out = collator._maybe_truncate_past(_Sample(**vars(_sample(9))))
    torch.testing.assert_close(out.agent_input["past_traj"][-1], torch.zeros(2))


def test_past_truncation_is_off_at_inference_and_by_default():
    sample = _Sample(**vars(_sample(9)))
    for collator in (
        _collator(),
        _collator(past_truncation_prob=1.0, inference_mode=True),
    ):
        assert collator._maybe_truncate_past(sample) is sample
    collator = _collator(past_truncation_prob=1.0)
    assert collator._maybe_truncate_past(sample).agent_input["past_traj"].shape[0] < 9
    single = _Sample(**vars(_sample(1)))
    assert collator._maybe_truncate_past(single) is single


def test_independent_expert_past_dropout():
    # Prompt keeps the past, expert drops it.
    collator = _collator(drop_cfg={"past_traj": 0.0, "expert_past_traj": 1.0})
    message, _, _, expert_valid = collator._get_sample_message(_sample(9))
    assert "Past ego trajectory" in _past_text(message)
    assert not expert_valid
    # Prompt drops the past, expert keeps it.
    collator = _collator(
        prompt_builder=_PromptBuilder(active_inputs=ALL_INPUTS - {"past_traj"}),
        drop_cfg={"past_traj": 1.0, "expert_past_traj": 0.0},
    )
    message, _, _, expert_valid = collator._get_sample_message(_sample(9))
    assert message[0]["content"] == []
    assert expert_valid


def test_independent_expert_past_dropout_rate():
    import random

    random.seed(0)
    collator = _collator(drop_cfg={"expert_past_traj": 0.3})
    valid = [collator._get_sample_message(_sample(9))[3] for _ in range(2000)]
    assert abs(1 - sum(valid) / len(valid) - 0.3) < 0.04
