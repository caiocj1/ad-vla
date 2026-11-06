import copy
import json
import pickle
from types import MethodType
from types import SimpleNamespace

import pytest
import pytorch_lightning as pl
import torch
import torch.nn.functional as F

from ad_vla.dataset.data_types import (
    CalibrationDict,
    CameraData,
    E2EDataSample,
    E2EDataBatch,
    QuestionAnswerPair,
    TrajectorySampling,
)
from ad_vla.dataset.nuscenes.nuscenes_dataset import NuScenesDataset
from ad_vla.dataset.nuscenes.nuscenes_types import (
    _NUSCENES_QA_ANSWER_HINT,
    _NUSCENES_QA_ANSWERS,
    _NUSCENES_QA_CAMERAS,
)
from ad_vla.dataset.qa_datasets import QAPairDataset, QATrainDataset
from ad_vla.models.vlas.qwen_fm_vla import (
    QwenFlowMatchingVLA,
    QwenFlowMatchingVLACollator,
)
from ad_vla.models.vlas.text_traj_vlm import TextTrajCollator, TextTrajVLM
from ad_vla.dataset.nuscenes.nuscenes_types import _NUSCENES_QA_SYSTEM_PROMPT
from ad_vla.dataset.waymo.waymo_qa import _pair as _waymo_pair
from ad_vla.dataset.waymo.waymo_qa import load_waymo_qa
from ad_vla.train_stages.base_module import BaseTrainingModule
from ad_vla.utils.prompts import PromptBuilder, normalize_qa_prediction

_LEGACY_PROMPT = PromptBuilder({"text": "Driving prompt."})


def _shared_prompt() -> PromptBuilder:
    return PromptBuilder(
        {
            "qa_uses_system_prompt": True,
            "intro": "You are an expert driver.\nInput:",
            "input_sections": {"images": "- images"},
            "task_text": "Task: drive or answer.",
        }
    )


def _camera(translation_x: float = 0.0) -> CameraData:
    calibration = CalibrationDict(
        intrinsics=torch.tensor([[8.0, 0.0, 4.0], [0.0, 8.0, 4.0], [0.0, 0.0, 1.0]]),
        translation=torch.tensor([translation_x, 0.0, 1.0]),
        rotation=torch.eye(3),
        original_img_size=(8, 8),
        is_flu=False,
    )
    return CameraData(image=torch.zeros(8, 8, 3), cal_dict=calibration)


def _sample(*, with_qa: bool = True, selected: bool = False) -> E2EDataSample:
    cameras = {
        name: _camera(float(index)) for index, name in enumerate(_NUSCENES_QA_CAMERAS)
    }
    qa_pairs = (
        QuestionAnswerPair(
            question="Is there a car in front of me?",
            answer="yes",
            num_hop=0,
            template_type="exist",
            allowed_answers=_NUSCENES_QA_ANSWERS,
            answer_hint=_NUSCENES_QA_ANSWER_HINT,
        ),
    )
    return E2EDataSample(
        metadata={"scenario_id": "sample-token", "sampling_freq": 2},
        agent_input={"past_traj": torch.zeros(4, 2)},
        cameras=[cameras],
        intent="UNKNOWN",
        fut_traj=torch.zeros(2, 2),
        fut_traj_sampling=TrajectorySampling(num_poses=2, interval_length=0.5),
        qa_pairs=qa_pairs if with_qa else None,
        qa_pair=qa_pairs[0] if selected else None,
    )


def test_nuscenes_qa_loader_groups_by_existing_sample_token(tmp_path):
    assert NuScenesDataset._load_qa_annotations(None, expected_split="train") == {}

    annotations_path = tmp_path / "NuScenes_train_questions.json"
    annotations_path.write_text(
        json.dumps(
            {
                "info": {"split": "train"},
                "questions": [
                    {
                        "sample_token": "token-a",
                        "question": "How many cars?",
                        "answer": "2",
                        "num_hop": 0,
                        "template_type": "count",
                    },
                    {
                        "sample_token": "token-a",
                        "question": "Is a car moving?",
                        "answer": "yes",
                        "num_hop": 1,
                        "template_type": "status",
                    },
                ],
            }
        ),
        encoding="utf-8",
    )

    grouped = NuScenesDataset._load_qa_annotations(
        str(annotations_path), expected_split="train"
    )

    assert tuple(grouped) == ("token-a",)
    assert [pair.answer for pair in grouped["token-a"]] == ["2", "yes"]
    with pytest.raises(ValueError, match="does not match dataset split"):
        NuScenesDataset._load_qa_annotations(
            str(annotations_path), expected_split="val"
        )


def test_qa_pair_dataset_has_one_deterministic_item_per_question():
    class FakeNuScenesDataset(NuScenesDataset):
        def __init__(self):
            self.samples = [{"token": "token-b"}, {"token": "token-a"}]
            self.qa_by_sample_token = {
                "token-a": (
                    QuestionAnswerPair(question="Question one?", answer="yes"),
                    QuestionAnswerPair(question="Question two?", answer="2"),
                ),
                "token-b": (
                    QuestionAnswerPair(question="Question three?", answer="no"),
                ),
            }
            self.loads = []

        def __getitem__(self, index):
            self.loads.append(index)
            sample = _sample()
            sample.metadata["scenario_id"] = self.samples[index]["token"]
            return sample

    scene_dataset = FakeNuScenesDataset()
    qa_dataset = QAPairDataset(scene_dataset)

    assert len(qa_dataset) == 3
    assert qa_dataset[0][0].metadata["scenario_id"] == "token-a"
    assert qa_dataset[0][1].question == "Question one?"
    assert qa_dataset[1][1].question == "Question two?"
    # Adjacent questions tied to one scene reuse the loaded camera sample.
    assert scene_dataset.loads == [1]
    assert qa_dataset[2][0].metadata["scenario_id"] == "token-b"
    assert scene_dataset.loads == [1, 0]


@pytest.mark.parametrize(
    ("require_full_past", "require_full_future", "expected_tokens"),
    [
        (True, True, ["b", "c"]),
        (False, True, ["a", "b", "c"]),
        (True, False, ["b", "c", "d"]),
        (False, False, ["a", "b", "c", "d"]),
    ],
)
def test_nuscenes_context_requirements_filter_independently(
    monkeypatch,
    tmp_path,
    require_full_past,
    require_full_future,
    expected_tokens,
):
    samples = {
        "a": {"token": "a", "prev": "", "next": "b"},
        "b": {"token": "b", "prev": "a", "next": "c"},
        "c": {"token": "c", "prev": "b", "next": "d"},
        "d": {"token": "d", "prev": "c", "next": ""},
    }

    class FakeNuScenes:
        def __init__(self, **_):
            self.scene = [
                {"name": "selected-scene", "first_sample_token": "a"},
                {"name": "other-scene", "first_sample_token": "a"},
            ]

        @staticmethod
        def get(table, token):
            assert table == "sample"
            return samples[token]

    monkeypatch.setattr(
        "ad_vla.dataset.nuscenes.nuscenes_dataset.NuScenes", FakeNuScenes
    )
    monkeypatch.setattr(
        "ad_vla.dataset.nuscenes.nuscenes_dataset.create_splits_scenes",
        lambda: {"train": ["selected-scene"]},
    )

    dataset = NuScenesDataset(
        root=str(tmp_path),
        split="train",
        num_past_steps=1,
        num_future_steps=1,
        require_full_past=require_full_past,
        require_full_future=require_full_future,
    )

    assert [sample["token"] for sample in dataset.samples] == expected_tokens


@pytest.mark.parametrize(
    "collator_class", [QwenFlowMatchingVLACollator, TextTrajCollator]
)
def test_qa_message_uses_six_separate_labeled_images_and_exact_answer(
    collator_class,
):
    sample = _sample()
    collator = object.__new__(collator_class)
    collator.prompt_builder = _LEGACY_PROMPT
    message = collator.prompt_builder.build_qa_message(sample, sample.qa_pairs[0])

    assert [turn["role"] for turn in message] == ["system", "user", "assistant"]
    user_content = message[1]["content"]
    assert sum(item["type"] == "image" for item in user_content) == 6
    user_text = " ".join(
        item["text"] for item in user_content if item["type"] == "text"
    )
    assert all(f"{name} camera:" in user_text for name in _NUSCENES_QA_CAMERAS)
    assert "Question: Is there a car in front of me?" in user_text
    assert "trajectory" not in user_text.lower()
    assert message[2]["content"] == [{"type": "text", "text": "yes"}]
    assert "Output only the answer" in message[0]["content"][0]["text"]


def test_older_prompts_keep_the_nuscenes_system_turn_and_refuse_other_questions():
    sample = _sample()
    message = _LEGACY_PROMPT.build_qa_message(sample, sample.qa_pairs[0])
    assert message[0]["content"][0]["text"] == _NUSCENES_QA_SYSTEM_PROMPT
    assert message[1]["content"][-1]["text"] == (
        "Question: Is there a car in front of me?"
    )
    free_text = QuestionAnswerPair(question="Why?", answer="Because.")
    with pytest.raises(ValueError, match="drive_vqa"):
        _LEGACY_PROMPT.build_qa_message(sample, free_text)
    del sample.cameras[-1]["REAR"]
    with pytest.raises(ValueError, match="all six"):
        _LEGACY_PROMPT.build_qa_message(sample, sample.qa_pairs[0])


def test_shared_prompt_is_the_same_for_driving_and_qa_and_questions_carry_their_form():
    sample = _sample()
    builder = _shared_prompt()
    nuscenes = builder.build_qa_message(sample, sample.qa_pairs[0])
    assert nuscenes[0]["content"][0]["text"] == builder.build_system_prompt()
    assert nuscenes[1]["content"][-1]["text"].endswith(_NUSCENES_QA_ANSWER_HINT)

    choice = QuestionAnswerPair(
        question="Which?",
        answer="B",
        options=("A. Left", "B. Right"),
        allowed_answers=frozenset("ab"),
    )
    message = builder.build_qa_message(sample, choice)
    assert message[1]["content"][-1]["text"] == (
        "Question: Which?\nOptions:\nA. Left\nB. Right"
    )
    assert message[2]["content"][0]["text"] == "B"


def test_qa_shows_every_camera_a_sample_has_in_a_fixed_order():
    sample = _sample()
    sample.cameras[-1]["SIDE_RIGHT"] = _camera()
    sample.cameras[-1]["SIDE_LEFT"] = _camera()
    message = _shared_prompt().build_qa_message(sample, sample.qa_pairs[0])
    labels = [
        item["text"].removesuffix(" camera:")
        for item in message[1]["content"]
        if item["text" if item["type"] == "text" else "type"].endswith("camera:")
    ]
    assert labels == [
        "FRONT_LEFT",
        "FRONT",
        "FRONT_RIGHT",
        "SIDE_LEFT",
        "SIDE_RIGHT",
        "REAR_LEFT",
        "REAR",
        "REAR_RIGHT",
    ]


def test_qa_with_several_frames_groups_them_by_camera_oldest_first():
    sample = _sample()
    old, new = {"FRONT": _camera(1.0)}, {"FRONT": _camera(2.0)}
    sample.cameras = [old, new]
    message = _shared_prompt().build_qa_message(sample, sample.qa_pairs[0], 3)
    content = message[1]["content"]
    texts = [item["text"] for item in content if item["type"] == "text"]
    assert texts[:4] == [
        "Surround-view images:",
        "FRONT camera:",
        "<frame -1>",
        "<frame 0>",
    ]
    assert sum(item["type"] == "image" for item in content) == 2
    assert content[3]["image"] is sample.cameras[0]["FRONT"].image


def test_waymo_qa_finds_frames_through_the_clip_index(tmp_path):
    clips = tmp_path / "clips.pkl"
    clips.write_bytes(pickle.dumps({"hash": [(9, 100), (10, 101), (11, 102)]}))
    rows = [
        {
            "token": "hash",
            "frame_index": 10,
            "type": "image",
            "question": "A?",
            "answer": "None.",
        },
        {
            "token": "hash",
            "frame_index": None,
            "type": "video",
            "question": "B?",
            "answer": "Stop",
        },
        {
            "token": "hash",
            "frame_index": 99,
            "type": "image",
            "question": "C?",
            "answer": "x",
        },
        {
            "token": "other",
            "frame_index": 1,
            "type": "image",
            "question": "D?",
            "answer": "x",
        },
    ]
    annotations = tmp_path / "train.jsonl"
    annotations.write_text(
        "".join(json.dumps(row | {"waymo_split": "train"}) + "\n" for row in rows)
    )

    by_index, by_scenario, dropped = load_waymo_qa(
        str(annotations), "train", str(clips)
    )

    assert by_index[101][0].answer == "None."
    assert by_index[102][0].scope == "video"
    assert by_scenario == {"hash-010": 101, "hash-011": 102}
    assert dropped == 2
    with pytest.raises(ValueError, match="dataset split is val"):
        load_waymo_qa(str(annotations), "val", str(clips))
    with pytest.raises(ValueError, match="clip_index_path"):
        load_waymo_qa(str(annotations), "train", None)


def test_waymo_multiple_choice_answer_is_the_letter():
    pair = _waymo_pair(
        {
            "question": "Which?",
            "type": "image",
            "category": "Normal Planning",
            "options": ["A. Stop", "B. Go", "C. Turn", "D. None"],
            "answer_index": 2,
        }
    )
    assert pair.answer == "C"
    assert pair.allowed_answers == frozenset("abcd")
    assert normalize_qa_prediction("C.", pair.allowed_answers) == "c"
    assert normalize_qa_prediction("C. Turn", pair.allowed_answers) is None


def test_qa_train_dataset_with_null_questions_per_frame_has_every_question():
    scene = _qa_train_scene({"a": ["q1", "q2", "q3"], "b": ["q4"]}, ["a", "b"])
    dataset = QATrainDataset(scene, None)
    assert len(dataset) == 4
    assert all(len(pairs) == 1 for _, pairs in dataset.items)


def _qa_train_scene(pairs_by_token: dict[str, list[str]], tokens: list[str]):
    scene = object.__new__(NuScenesDataset)
    scene.samples = [{"token": token} for token in tokens]
    scene.qa_by_sample_token = {
        token: tuple(
            QuestionAnswerPair(
                question=q, answer="yes", num_hop=0, template_type="exist"
            )
            for q in questions
        )
        for token, questions in pairs_by_token.items()
    }
    scene.__class__ = type(
        "Scene", (NuScenesDataset,), {"__getitem__": lambda self, index: _sample()}
    )
    return scene


def test_qa_train_dataset_leaves_out_frames_without_questions():
    scene = _qa_train_scene({"a": ["q1"]}, ["a", "no_questions"])
    assert [item[0] for item in QATrainDataset(scene).items] == [0]


def test_qa_train_dataset_never_repeats_a_question_of_a_frame_in_one_epoch():
    questions = [f"q{i}" for i in range(7)]
    dataset = QATrainDataset(_qa_train_scene({"a": questions}, ["a"]), 3)
    assert len(dataset) == 3
    groups = [{pair.question for pair in pairs} for _, pairs in dataset.items]
    assert set.union(*groups) == set(questions)
    assert sum(len(group) for group in groups) == len(questions)
    for _ in range(20):
        drawn = [dataset[index].qa_pair.question for index in range(3)]
        assert len(set(drawn)) == 3


def test_qa_train_dataset_has_fewer_items_on_a_frame_with_few_questions():
    scene = _qa_train_scene({"a": ["q1", "q2"]}, ["a"])
    assert len(QATrainDataset(scene, 3)) == 2


def test_text_traj_collator_only_follows_the_drawn_question():
    collator = object.__new__(TextTrajCollator)
    expected_messages = [[{"role": "assistant", "content": "trajectory"}]]
    collator._get_sample_message = MethodType(
        lambda _self, _sample: expected_messages[0], collator
    )
    collator._collate_messages = MethodType(
        lambda _self, _batch, messages, qa_mask=None: (messages, qa_mask), collator
    )

    messages, qa_mask = collator([_sample()])
    assert messages == expected_messages
    assert torch.equal(qa_mask, torch.tensor([False]))


def test_text_traj_collator_routes_annotated_rows_to_qa_prompt():
    collator = object.__new__(TextTrajCollator)
    collator.prompt_builder = _LEGACY_PROMPT
    collator.num_past_image_frames = 1
    collator._collate_messages = MethodType(
        lambda _self, _batch, messages, qa_mask=None: (messages, qa_mask), collator
    )

    messages, qa_mask = collator([_sample(selected=True)])

    assert [turn["role"] for turn in messages[0]] == ["system", "user", "assistant"]
    assert messages[0][-1]["content"] == [{"type": "text", "text": "yes"}]
    assert torch.equal(qa_mask, torch.tensor([True]))


@pytest.mark.parametrize(
    "collator_class",
    [TextTrajCollator, QwenFlowMatchingVLACollator],
)
@pytest.mark.parametrize("selected", [False, True])
def test_collators_store_qa_mask_only_on_batch(collator_class, selected):
    class Processor:
        tokenizer = SimpleNamespace(pad_token_id=0)

        @staticmethod
        def apply_chat_template(messages, add_generation_prompt, **_kwargs):
            length = 2 if add_generation_prompt else 3
            input_ids = torch.ones(len(messages), length, dtype=torch.long)
            return {
                "input_ids": input_ids,
                "attention_mask": torch.ones_like(input_ids),
            }

    collator = object.__new__(collator_class)
    collator.prompt_builder = _LEGACY_PROMPT
    collator.num_past_image_frames = 1
    collator.processor = Processor()
    collator._get_sample_message = lambda sample: [
        {"role": "user", "content": []},
        {"role": "assistant", "content": []},
    ]
    # Isolate task metadata from model-specific calibration/state construction.
    collator.calib_image_patch_injection = False
    collator._get_sample_calibration_features = lambda sample: torch.zeros(1)
    collator.use_cape = False
    collator._resample_past_traj = lambda sample: torch.zeros(4, 2)
    collator._initial_velocity = lambda sample: torch.tensor(0.0)

    batch = collator([_sample(selected=selected), _sample(with_qa=False)])

    assert torch.equal(batch.qa_mask, torch.tensor([selected, False]))
    assert "qa_mask" not in batch.model_inputs
    assert "__qa_mask__" not in batch.model_inputs
    rebuilt = E2EDataBatch(dict(batch.items()))
    assert torch.equal(rebuilt.qa_mask, batch.qa_mask)
    assert "qa_mask" not in rebuilt.model_inputs


@pytest.mark.parametrize("dtype", [torch.float32, torch.bfloat16])
def test_text_traj_selected_logits_match_full_hf_loss_and_gradients(dtype):
    class SelectiveVLM(torch.nn.Module):
        def __init__(self):
            super().__init__()
            torch.manual_seed(17)
            self.hidden = torch.nn.Parameter(torch.randn(3, 8, 7, dtype=dtype))
            self.lm_head = torch.nn.Linear(7, 31, bias=False, dtype=dtype)
            self.logits_to_keep_requests = []
            self.last_logits = None

        def forward(self, input_ids, labels=None, logits_to_keep=0, **_kwargs):
            self.logits_to_keep_requests.append(logits_to_keep)
            positions = (
                slice(-logits_to_keep, None)
                if isinstance(logits_to_keep, int)
                else logits_to_keep
            )
            logits = self.lm_head(self.hidden[: input_ids.shape[0], positions])
            self.last_logits = logits.detach()
            loss = None
            if labels is not None:
                shifted_labels = F.pad(labels, (0, 1), value=-100)[:, 1:]
                loss = F.cross_entropy(
                    logits.float().reshape(-1, logits.shape[-1]),
                    shifted_labels.reshape(-1),
                    ignore_index=-100,
                )
            return SimpleNamespace(logits=logits, loss=loss)

    labels = torch.full((3, 8), -100, dtype=torch.long)
    labels[0, 5:] = torch.tensor([3, 5, 7])
    labels[1, 6:] = torch.tensor([2, 4])
    labels[2, [4, 6, 7]] = torch.tensor([1, 6, 8])
    model_inputs = {
        "input_ids": torch.zeros_like(labels),
        "attention_mask": torch.ones_like(labels),
        "labels": labels,
    }

    full_vlm = SelectiveVLM()
    optimized_vlm = copy.deepcopy(full_vlm)
    model = TextTrajVLM.__new__(TextTrajVLM)
    torch.nn.Module.__init__(model)
    model.vlm = optimized_vlm

    full_loss = full_vlm(**model_inputs).loss
    optimized_loss = model.compute_loss(SimpleNamespace(model_inputs=model_inputs))[
        "loss"
    ]

    torch.testing.assert_close(optimized_loss, full_loss, rtol=0.0, atol=0.0)
    assert full_vlm.logits_to_keep_requests == [0]
    assert torch.equal(
        optimized_vlm.logits_to_keep_requests[0], torch.tensor([3, 4, 5, 6])
    )
    torch.testing.assert_close(
        optimized_vlm.last_logits,
        full_vlm.last_logits[:, [3, 4, 5, 6]],
        rtol=0.0,
        atol=0.0,
    )
    # compute_loss must not consume or mutate the collated labels.
    assert model_inputs["labels"] is labels

    full_loss.backward()
    optimized_loss.backward()
    for (full_name, full_param), (optimized_name, optimized_param) in zip(
        full_vlm.named_parameters(), optimized_vlm.named_parameters()
    ):
        assert full_name == optimized_name
        torch.testing.assert_close(
            optimized_param.grad,
            full_param.grad,
            rtol=1e-6,
            atol=2e-9,
        )


@pytest.mark.parametrize(
    ("raw_prediction", "expected"),
    [
        ("yes", "yes"),
        ("YES.", "yes"),
        ("</think>\ntraffic cone", "traffic cone"),
        ("no\nuser\nAre there any cones?", None),
        ("I can see a car", None),
    ],
)
def test_qa_prediction_normalization_rejects_appended_text(raw_prediction, expected):
    assert normalize_qa_prediction(raw_prediction, _NUSCENES_QA_ANSWERS) == expected


@pytest.mark.parametrize("model_class", [QwenFlowMatchingVLA, TextTrajVLM])
def test_qa_generation_uses_training_prompt_and_explicit_tokenizer_eos(model_class):
    class Batch(dict):
        def to(self, device):
            return Batch(
                {
                    key: value.to(device) if isinstance(value, torch.Tensor) else value
                    for key, value in self.items()
                }
            )

    class Processor:
        def __init__(self):
            self.tokenizer = SimpleNamespace(
                eos_token_id=248046,
                pad_token_id=0,
                padding_side="right",
            )
            self.messages = None

        def apply_chat_template(self, messages, **_kwargs):
            self.messages = messages
            return Batch(
                {
                    "input_ids": torch.tensor([[1, 2], [1, 2]]),
                    "attention_mask": torch.ones(2, 2, dtype=torch.long),
                }
            )

        @staticmethod
        def batch_decode(_token_ids, skip_special_tokens):
            assert skip_special_tokens
            return ["yes", "NO."]

    class VLM(torch.nn.Module):
        def __init__(self):
            super().__init__()
            self.anchor = torch.nn.Parameter(torch.zeros(()))
            self.config = SimpleNamespace(
                text_config=SimpleNamespace(eos_token_id=248044),
                eos_token_id=None,
            )
            self.generation_kwargs = None

        def generate(self, **kwargs):
            self.generation_kwargs = kwargs
            batch_size = kwargs["input_ids"].shape[0]
            return torch.cat(
                [kwargs["input_ids"], torch.ones(batch_size, 1, dtype=torch.long)],
                dim=1,
            )

    model = model_class.__new__(model_class)
    torch.nn.Module.__init__(model)
    model.processor = Processor()
    model.prompt_builder = _LEGACY_PROMPT
    model.num_past_image_frames = 1
    model.vlm = VLM()
    qa_batch = [(_sample(), _sample().qa_pairs[0])] * 2

    assert model.predict_qa_answers(qa_batch) == ["yes", "no"]
    assert model.processor.tokenizer.padding_side == "left"
    assert all(
        [turn["role"] for turn in messages] == ["system", "user"]
        for messages in model.processor.messages
    )
    assert model.vlm.generation_kwargs["eos_token_id"] == [248046, 248044]
    assert model.vlm.generation_kwargs["pad_token_id"] == 0


def test_second_validation_loader_accumulates_question_weighted_qa_counts():
    class QAModel(torch.nn.Module):
        def __init__(self):
            super().__init__()
            self.anchor = torch.nn.Parameter(torch.zeros(()))

        @staticmethod
        def predict_qa_answers(_batch):
            return ["yes", None, "no"]

    module = BaseTrainingModule.__new__(BaseTrainingModule)
    pl.LightningModule.__init__(module)
    module.model = QAModel()
    module.qa_val_metrics = torch.nn.ModuleDict({"qa": module._qa_metrics()})
    module._qa_val_seen = set()
    qa_batch = [
        (_sample(), QuestionAnswerPair(question="One?", answer="yes")),
        (_sample(), QuestionAnswerPair(question="Two?", answer="no")),
        (_sample(), QuestionAnswerPair(question="Three?", answer="no")),
    ]

    assert module.transfer_batch_to_device(qa_batch, torch.device("cpu"), 1) is qa_batch
    module.validation_step(qa_batch, batch_idx=0, dataloader_idx=1)

    metrics = module.qa_val_metrics["qa"]
    assert metrics["correct"].compute().item() == 2
    assert metrics["valid"].compute().item() == 2
    assert metrics["total"].compute().item() == 3


def test_mixed_geometry_consumes_six_qa_images_and_driving_concat_image():
    class _Tokenizer:
        @staticmethod
        def convert_tokens_to_ids(_token):
            return 99

    collator = object.__new__(QwenFlowMatchingVLACollator)
    collator.processor = SimpleNamespace(
        tokenizer=_Tokenizer(),
        image_processor=SimpleNamespace(merge_size=2),
        image_token_id=99,
    )
    collator.camera_sequence = ["FRONT_LEFT", "FRONT", "FRONT_RIGHT"]
    collator.num_past_image_frames = 1
    collator.concat_cameras = True

    # Each tiny grid produces one merged visual token: six for QA, one for drive.
    input_ids = torch.tensor([[99, 99, 99, 99, 99, 99], [0, 0, 0, 0, 0, 99]])
    grids = torch.tensor([[1, 2, 2]] * 7)
    poses, coords, times, references = collator._expert_geometry(
        [_sample(), _sample(with_qa=False)],
        input_ids,
        grids,
        torch.tensor([True, False]),
    )

    assert poses.shape == (2, 6, 4, 4)
    assert coords.shape == (2, 6, 2)
    assert times.shape == (2, 6)
    assert references.shape == (2, 4, 4)
    front_index = _NUSCENES_QA_CAMERAS.index("FRONT")
    assert references[:, 0, 3].tolist() == [float(front_index)] * 2


def test_group_ce_uses_only_selected_rows():
    logits = torch.tensor(
        [
            [[2.0, 0.0], [0.0, 2.0]],
            [[0.0, 2.0], [2.0, 0.0]],
        ],
        requires_grad=True,
    )
    labels = torch.tensor([[1, -100], [0, -100]])

    loss = QwenFlowMatchingVLA._token_ce_for_group(
        logits, labels, torch.tensor([True, False])
    )
    expected = F.cross_entropy(logits[0, :1], torch.tensor([1]))
    assert torch.allclose(loss, expected)


def test_bf16_group_ce_approximates_previous_fp32_full_row_loss_and_gradients():
    torch.manual_seed(7)
    labels = torch.full((3, 9), -100, dtype=torch.long)
    labels[0, 6:] = torch.tensor([3, 5, 7])
    labels[1, 7:] = torch.tensor([2, 4])
    labels[2, 5:] = torch.tensor([1, 6, 8, 9])
    sample_mask = torch.tensor([True, False, True])

    old_logits = torch.randn(3, 9, 32, dtype=torch.bfloat16, requires_grad=True)
    new_logits = old_logits.detach().clone().requires_grad_(True)

    # Previous implementation: gather complete rows, materialize fp32, then
    # let ignore_index discard prompt positions inside cross entropy.
    old_loss = F.cross_entropy(
        old_logits[sample_mask, :-1].float().reshape(-1, old_logits.shape[-1]),
        labels[sample_mask, 1:].reshape(-1),
        ignore_index=-100,
    )
    old_loss.backward()

    shifted_labels = labels[:, 1:]
    positions = shifted_labels.ne(-100).any(dim=0).nonzero(as_tuple=True)[0]
    new_loss = QwenFlowMatchingVLA._token_ce_for_group(
        new_logits[:, positions],
        shifted_labels[:, positions],
        sample_mask,
    )
    new_loss.backward()

    assert new_loss.dtype == torch.bfloat16
    assert torch.allclose(new_loss.float(), old_loss, rtol=1e-2, atol=2e-2)
    assert torch.allclose(
        new_logits.grad.float(), old_logits.grad.float(), rtol=2e-2, atol=1e-4
    )


class _FakeVLM(torch.nn.Module):
    """Small causal LM exposing per-row logits for loss-routing assertions."""

    def __init__(self):
        super().__init__()
        self.logits = torch.nn.Parameter(torch.randn(2, 4, 4))
        self.config = SimpleNamespace(image_token_id=2)
        self.output_hidden_states_requests: list[bool] = []
        self.labels_requests = []
        self.logits_to_keep_requests = []

    def forward(
        self,
        input_ids,
        labels=None,
        output_hidden_states=False,
        logits_to_keep=0,
        **_kwargs,
    ):
        self.output_hidden_states_requests.append(output_hidden_states)
        self.labels_requests.append(labels)
        self.logits_to_keep_requests.append(logits_to_keep)
        logits = self.logits[: input_ids.shape[0], logits_to_keep]
        hidden_states = (
            ((torch.zeros(*input_ids.shape, 3) + self.logits.mean()),)
            if output_hidden_states
            else None
        )
        return SimpleNamespace(loss=None, logits=logits, hidden_states=hidden_states)


class _FakeDiffusion(torch.nn.Module):
    def __init__(self):
        super().__init__()
        self.target_batch_sizes: list[int] = []

    def construct_training_data(self, target):
        self.target_batch_sizes.append(target.shape[0])
        return {
            # Exercise the repeat-and-realign path used by FlowMatching.
            "noisy_x": target.repeat_interleave(3, dim=0),
            "timesteps": target.new_zeros(target.shape[0] * 3),
        }

    @staticmethod
    def compute_loss_from_pred(_data, pred):
        return pred.square().mean() + 2.0


def _loss_test_model():
    model = QwenFlowMatchingVLA.__new__(QwenFlowMatchingVLA)
    torch.nn.Module.__init__(model)
    model.vlm = _FakeVLM()
    model.diffusion = _FakeDiffusion()
    model.pred_traj_sampling = TrajectorySampling(num_poses=2, interval_length=0.5)
    model.action_representation = "trajs"
    model.traj_normalization = 1.0
    model.ce_loss_weight = 1.0
    model.qa_loss_weight = 1.0
    model.diff_loss_weight = 1.0
    action_calls = []

    def _predict_vector_field(self, x, _t, *conditioning):
        action_calls.append(
            {
                "action_batch": x.shape[0],
                "conditioning_batches": [value.shape[0] for value in conditioning],
            }
        )
        return x * 0.0

    model._predict_vector_field = MethodType(_predict_vector_field, model)
    return model, action_calls


def _loss_test_batch(qa_mask: list[bool]) -> E2EDataBatch:
    samples = []
    for is_qa in qa_mask:
        sample = _sample(with_qa=is_qa)
        if is_qa:
            # This makes accidental action supervision of a QA row fail loudly.
            sample.fut_traj = None
            sample.fut_traj_sampling = None
        samples.append(sample)

    batch_size, sequence_length = len(samples), 4
    input_ids = torch.tensor([[0, 2, 0, 0]] * batch_size)
    labels = torch.full_like(input_ids, -100)
    labels[:, -1] = torch.arange(batch_size) + 1
    identity = torch.eye(4).view(1, 1, 4, 4)
    return E2EDataBatch(
        samples=samples,
        model_inputs={
            "input_ids": input_ids,
            "attention_mask": torch.ones_like(input_ids),
            "labels": labels,
            "pixel_values": torch.zeros(batch_size, 1),
            "image_grid_thw": torch.ones(batch_size, 3, dtype=torch.long),
            "proprio_state": torch.zeros(batch_size, 8),
            "init_vel": torch.zeros(batch_size),
            "visual_camera_poses": identity.expand(
                batch_size, sequence_length, -1, -1
            ).clone(),
            "visual_patch_coords": torch.zeros(batch_size, sequence_length, 2),
            "visual_time_ids": torch.zeros(
                batch_size, sequence_length, dtype=torch.long
            ),
            "reference_camera_pose": torch.eye(4)
            .view(1, 4, 4)
            .expand(batch_size, -1, -1)
            .clone(),
        },
        qa_mask=torch.tensor(qa_mask),
    )


def test_joint_loss_requires_batch_qa_mask():
    model, _ = _loss_test_model()
    batch = _loss_test_batch([True, False])
    batch.qa_mask = None

    with pytest.raises(ValueError, match=r"requires batch\.qa_mask"):
        model.compute_loss(batch)


def test_mixed_loss_routes_only_driving_row_through_action_expert():
    model, action_calls = _loss_test_model()
    # Put QA first to test boolean selection instead of simple truncation.
    losses = model.compute_loss(_loss_test_batch([True, False]))

    # Diffusion saw one target, then repeated it and every conditioning row 3x.
    assert model.diffusion.target_batch_sizes == [1]
    assert action_calls == [{"action_batch": 3, "conditioning_batches": [3] * 9}]
    assert losses["qa_fraction"].item() == 0.5
    assert losses["diff_loss"].item() == 2.0
    assert model.vlm.labels_requests == [None]
    assert torch.equal(model.vlm.logits_to_keep_requests[0], torch.tensor([2]))

    losses["loss"].backward()
    # Both rows still receive VLM supervision despite only row one using actions.
    assert model.vlm.logits.grad[0].abs().sum() > 0
    assert model.vlm.logits.grad[1].abs().sum() > 0


def test_all_qa_loss_skips_hidden_states_and_action_expert():
    model, action_calls = _loss_test_model()
    losses = model.compute_loss(_loss_test_batch([True, True]))

    assert model.vlm.output_hidden_states_requests == [False]
    assert model.diffusion.target_batch_sizes == []
    assert action_calls == []
    assert losses["qa_fraction"].item() == 1.0
    assert losses["diff_loss"].item() == 0.0


def test_missing_future_row_is_skipped_only_by_action_loss():
    model, action_calls = _loss_test_model()
    batch = _loss_test_batch([False, False])
    batch.samples[0].fut_traj = None
    batch.samples[0].fut_traj_sampling = None

    losses = model.compute_loss(batch)

    assert model.diffusion.target_batch_sizes == [1]
    assert action_calls == [{"action_batch": 3, "conditioning_batches": [3] * 9}]
    assert losses["diff_loss"].item() == 2.0

    losses["loss"].backward()
    assert model.vlm.logits.grad[0].abs().sum() > 0
    assert model.vlm.logits.grad[1].abs().sum() > 0


def test_driving_only_loss_preserves_full_action_batch_and_backbone_ce():
    model, action_calls = _loss_test_model()
    batch = _loss_test_batch([False, False])
    losses = model.compute_loss(batch)

    expected_ce = F.cross_entropy(
        model.vlm.logits[:, :-1].reshape(-1, model.vlm.logits.shape[-1]),
        batch.model_inputs["labels"][:, 1:].reshape(-1),
        ignore_index=-100,
    )
    assert torch.allclose(losses["ce_loss"], expected_ce)
    assert losses["qa_fraction"].item() == 0.0
    assert losses["qa_ce_loss"].item() == 0.0
    assert model.diffusion.target_batch_sizes == [2]
    assert action_calls == [{"action_batch": 6, "conditioning_batches": [6] * 9}]
