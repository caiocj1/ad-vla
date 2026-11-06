from types import SimpleNamespace

import torch

from ad_vla.dataset.data_types import TrajectorySampling
from ad_vla.models.vlas.text_traj_vlm import TextTrajVLM


def _bare_model() -> TextTrajVLM:
    model = object.__new__(TextTrajVLM)
    torch.nn.Module.__init__(model)
    return model


def test_generation_eos_includes_tokenizer_and_nested_model_tokens() -> None:
    model = _bare_model()
    model.processor = SimpleNamespace(tokenizer=SimpleNamespace(eos_token_id=248046))
    model.vlm = SimpleNamespace(
        config=SimpleNamespace(
            text_config=SimpleNamespace(eos_token_id=248044),
            eos_token_id=None,
        )
    )

    assert model._generation_eos_token_id() == [248046, 248044]


def test_parser_uses_valid_first_turn_when_generation_continues() -> None:
    model = _bare_model()
    model.pred_traj_sampling = TrajectorySampling(
        num_poses=5,
        interval_length=1.0,
    )
    generated_text = (
        "Future ego trajectory:[[1, 0], [2, 0], [3, 0], [4, 0], [5, 0]]"
        "<|im_end|>\n<|im_start|>user\nPast ego trajectory:[[0, 0]]"
        "<|im_end|>\n<|im_start|>assistant\nFuture ego trajectory:[[9, 9]]"
    )

    trajectory_text = model._trajectory_text_from_generation(generated_text)
    trajectory, success = model.text_to_traj(trajectory_text)

    assert success
    assert torch.equal(
        trajectory,
        torch.tensor([[1.0, 0.0], [2.0, 0.0], [3.0, 0.0], [4.0, 0.0], [5.0, 0.0]]),
    )


class _LeftPadTokenizer:
    pad_token_id = 0
    eos_token_id = 9

    def pad(self, batch, padding=True, return_tensors="pt"):
        ids = batch["input_ids"]
        length = max(len(x) for x in ids)
        input_ids = torch.zeros(len(ids), length, dtype=torch.long)
        mask = torch.zeros(len(ids), length, dtype=torch.long)
        for i, x in enumerate(ids):
            input_ids[i, length - len(x) :] = x
            mask[i, length - len(x) :] = 1
        return {"input_ids": input_ids, "attention_mask": mask}


def test_prediction_passes_mm_token_type_ids_for_multimodal_rope() -> None:
    """Qwen3.5 (transformers >= 5.18) needs mm_token_type_ids to give image tokens 3D M-RoPE
    positions; without them they silently get 1D text positions."""
    image_token_id = 7
    captured = {}

    def generate(**kwargs):
        captured.update(kwargs)
        n = kwargs["input_ids"].shape[0]
        new = torch.full((n, 3), 5)
        return SimpleNamespace(sequences=torch.cat([kwargs["input_ids"], new], dim=1))

    model = _bare_model()
    model.dummy = torch.nn.Parameter(torch.zeros(1))  # gives the bare model a device
    model.pred_traj_sampling = TrajectorySampling(num_poses=5, interval_length=1.0)
    tokenizer = _LeftPadTokenizer()
    model.processor = SimpleNamespace(
        tokenizer=tokenizer,
        batch_decode=lambda ids: [
            "Future ego trajectory:[[1, 0], [2, 0], [3, 0], [4, 0], [5, 0]]"
        ]
        * len(ids),
    )
    model.vlm = SimpleNamespace(
        config=SimpleNamespace(
            image_token_id=image_token_id, text_config=None, eos_token_id=None
        ),
        generate=generate,
    )
    input_ids = torch.tensor([[1, 7, 7, 7, 2, 3], [0, 1, 7, 7, 7, 4]])
    model.predict_from_processed_inputs(
        input_ids=input_ids,
        pixel_values=torch.zeros(6, 4),
        image_grid_thw=torch.tensor([[1, 2, 6], [1, 2, 6]]),
        labels=torch.full_like(input_ids, -100),
    )

    assert "mm_token_type_ids" in captured
    ids, mm = captured["input_ids"], captured["mm_token_type_ids"]
    assert mm.shape == ids.shape
    assert torch.equal(mm, (ids == image_token_id).to(mm.dtype))
    assert int(mm.sum()) == 6
