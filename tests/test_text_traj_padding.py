from copy import deepcopy

from ad_vla.dataset.data_types import TrajectorySampling
from ad_vla.models.vlas.text_traj_vlm import TextTrajCollator, TextTrajVLMConfig


def _message() -> list[dict]:
    return [
        {"role": "system", "content": []},
        {
            "role": "user",
            "content": [
                {
                    "type": "text",
                    "text": "Past ego trajectory:[[-1.0, 0.0], [0.0, 0.0]]",
                }
            ],
        },
        {"role": "assistant", "content": []},
    ]


def _collator(pad_missing_past_traj: bool) -> TextTrajCollator:
    collator = object.__new__(TextTrajCollator)
    collator.past_traj_sampling = TrajectorySampling(num_poses=4, interval_length=0.25)
    collator.pad_missing_past_traj = pad_missing_past_traj
    return collator


def test_missing_past_text_padding_is_disabled_by_default():
    assert TextTrajVLMConfig().pad_missing_past_traj is False

    message = _message()
    expected = deepcopy(message)
    _collator(False)._maybe_pad_past_traj_text(message, num_available_poses=2)
    assert message == expected


def test_missing_past_text_padding_prepends_oldest_slots():
    message = _message()
    _collator(True)._maybe_pad_past_traj_text(message, num_available_poses=2)

    assert message[1]["content"][0]["text"] == (
        "Past ego trajectory:[[X.X, X.X], [X.X, X.X], [-1.0, 0.0], [0.0, 0.0]]"
    )
