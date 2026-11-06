from types import SimpleNamespace

import torch
from torch import nn

from ad_vla.dataset.data_types import E2EDataBatch
from ad_vla.dataset.nuscenes.nuscenes_types import _NUSCENES_QA_CAMERAS
from ad_vla.models.aux_heads.spatial_forcing import SpatialForcingAuxHead


class _FakeAggregator(nn.Module):
    def __init__(self):
        super().__init__()
        self.patch_size = 2
        self.camera_token = nn.Parameter(torch.zeros(1, 2, 1, 4))
        self.calls = []

    def forward(self, images):
        self.calls.append(images.detach().cpu())
        batch_size, num_images, _, height, width = images.shape
        num_patches = height // self.patch_size * (width // self.patch_size)
        image_means = images.mean(dim=(2, 3, 4))
        features = image_means[:, :, None, None].expand(
            batch_size,
            num_images,
            num_patches + 1,
            8,
        )
        channel_offsets = torch.arange(8, device=images.device) / 100
        return [(features + channel_offsets).to(torch.bfloat16)], 1


class _FakeVGGT(nn.Module):
    def __init__(self):
        super().__init__()
        self.aggregator = _FakeAggregator()


class _Tokenizer:
    @staticmethod
    def convert_tokens_to_ids(token):
        return {"<|vision_start|>": 10, "<|vision_end|>": 11}[token]


class _FakeVLM(nn.Module):
    def __init__(self):
        super().__init__()
        layer = nn.Identity()
        self.model = SimpleNamespace(
            language_model=SimpleNamespace(layers=nn.ModuleList([layer]))
        )
        self.config = SimpleNamespace(
            text_config=SimpleNamespace(hidden_size=6),
            vision_config=SimpleNamespace(spatial_merge_size=2),
        )


def _model():
    return SimpleNamespace(
        processor=SimpleNamespace(tokenizer=_Tokenizer()),
        camera_sequence=["FRONT_LEFT", "FRONT", "FRONT_RIGHT"],
        num_past_image_frames=1,
        concat_cameras=True,
        vlm=_FakeVLM(),
    )


def _camera_frame(camera_names, start_value):
    return {
        camera_name: SimpleNamespace(
            image=torch.full(
                (8, 6, 3),
                start_value + camera_idx,
                dtype=torch.uint8,
            )
        )
        for camera_idx, camera_name in enumerate(camera_names)
    }


def _image_token_row(span_lengths, sequence_length):
    tokens = []
    for span_length in span_lengths:
        tokens.extend([10, *([1] * span_length), 11])
    return tokens + [0] * (sequence_length - len(tokens))


def test_mixed_driving_and_qa_rows_use_consistent_multiview_vggt(monkeypatch):
    from vggt.models.vggt import VGGT

    fake_vggt = _FakeVGGT()
    monkeypatch.setattr(
        VGGT,
        "from_pretrained",
        classmethod(lambda cls, *_args, **_kwargs: fake_vggt),
    )
    head = SpatialForcingAuxHead(
        _model(),
        target_layer=0,
        grid_size=16,
        add_pos_embedding=True,
    )

    qa_sample = SimpleNamespace(
        cameras=[_camera_frame(_NUSCENES_QA_CAMERAS, start_value=1)]
    )
    driving_samples = [
        SimpleNamespace(
            cameras=[_camera_frame(_model().camera_sequence, start_value=value)]
        )
        for value in (20, 40)
    ]

    # QA has six separate 2x2 post-merge grids. Each driving panorama has one
    # 2x6 grid produced from three source camera views.
    sequence_length = 36
    input_ids = torch.tensor(
        [
            _image_token_row([4] * 6, sequence_length),
            _image_token_row([12], sequence_length),
            _image_token_row([12], sequence_length),
        ]
    )
    image_grid_thw = torch.tensor([[1, 4, 4]] * 6 + [[1, 4, 12], [1, 4, 12]])
    batch = SimpleNamespace(
        samples=[qa_sample, *driving_samples],
        model_inputs={
            "input_ids": input_ids,
            "image_grid_thw": image_grid_thw,
        },
        qa_mask=torch.tensor([True, False, False]),
    )
    hidden_states = torch.randn(
        3,
        sequence_length,
        6,
        dtype=torch.bfloat16,
        requires_grad=True,
    )
    head.captured_states[0] = hidden_states

    head.train()
    loss, _ = head.compute_aux_loss(batch)
    loss.backward()

    calls_by_view_count = {
        images.shape[1]: images for images in fake_vggt.aggregator.calls
    }
    assert set(calls_by_view_count) == {3, 6}
    assert calls_by_view_count[3].shape[:2] == (2, 3)
    assert calls_by_view_count[6].shape[:2] == (1, 6)

    # Six QA views retain the exact prompt camera ordering when sent to VGGT.
    qa_view_means = calls_by_view_count[6].mean(dim=(2, 3, 4))[0]
    torch.testing.assert_close(
        qa_view_means,
        torch.arange(1, 7, dtype=torch.float32) / 255,
    )
    assert torch.isfinite(loss)
    assert hidden_states.grad is not None
    assert all(parameter.grad is None for parameter in head.vggt.parameters())
    assert not head.vggt.training
    assert 0 not in head.captured_states


def test_batch_qa_mask_stays_outside_model_inputs_during_mapping_rebuild():
    batch = E2EDataBatch(
        samples=[],
        model_inputs={"input_ids": torch.empty(0, 1, dtype=torch.long)},
        qa_mask=torch.empty(0, dtype=torch.bool),
    )

    rebuilt = E2EDataBatch(dict(batch.items()))

    assert "qa_mask" not in rebuilt.model_inputs
    assert rebuilt.qa_mask is not None
    assert rebuilt.qa_mask.dtype == torch.bool


def test_missing_batch_qa_mask_defaults_to_planning():
    batch = SimpleNamespace(samples=[object(), object()], qa_mask=None)

    assert torch.equal(
        SpatialForcingAuxHead._get_qa_mask(batch),
        torch.tensor([False, False]),
    )
