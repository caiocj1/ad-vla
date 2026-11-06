import json
from pathlib import Path

import numpy as np
import pytest
import torch
from omegaconf import OmegaConf
from PIL import Image

from ad_vla.dataset.builder import build_train_dataset
from ad_vla.dataset.data_types import TrajectorySampling
from ad_vla.dataset.nuplan_qa.nuplan_qa_dataset import NuPlanQADataset, parse_question
from ad_vla.dataset.qa_datasets import QATrainDataset
from ad_vla.models.vlas.qwen_fm_vla import QwenFlowMatchingVLACollator
from ad_vla.utils.prompts import PromptBuilder, normalize_qa_prediction

CAMERAS = ("CAM_F0", "CAM_L0", "CAM_R0", "CAM_L2", "CAM_R2", "CAM_B0")
PREAMBLE_TRAIN = (
    "You are driving on the road. Based on given images of the surroudings of your vehicle and "
    "information from your vehicle extracted past 1.5 seconds, answer the following question."
)
PREAMBLE_EVAL = (
    "You are driving on the road. Based on given images of the surroudings of your vehicle and "
    "information from your vehicle extracted past 1.5 seconds, choose right answer among (a) to "
    "(d) for the given question."
)
TRAFFIC_LIGHT_OPTIONS = (
    "(a) Red light is visible.\n(b) Yellow light is visible.\n"
    "(c) Green light is visible.\n(d) No traffic light is visible."
)


def test_free_text_question_drops_the_preamble_and_keeps_the_ego_state_after_it():
    text = (
        f"{PREAMBLE_TRAIN}\n        - Steering angles: [0.002, -0.967]\n"
        "Is the vehicle making any adjustments to its path?"
    )
    pair = parse_question(
        text, " Ego-vehicle slightly steers to the right. ", "Ego-Vehicle Maneuver"
    )
    assert pair.question == (
        "Is the vehicle making any adjustments to its path?\n"
        "Ego vehicle over the past 1.5 s:\n- Steering angles: [0.002, -0.967]"
    )
    assert pair.answer == "Ego-vehicle slightly steers to the right."
    assert pair.category == "Ego-Vehicle Maneuver"
    assert pair.options is None and pair.allowed_answers is None

    without = parse_question(text, "x", ego_state_text=False)
    assert without.question == "Is the vehicle making any adjustments to its path?"


def test_multiple_choice_answer_is_the_letter():
    text = (
        f"{PREAMBLE_EVAL}\n- Velocity(m/s): [5.3, 4.6]\n- Steering angles: [-0.02, 0.1]\n\n"
        f" Are there any traffic lights ahead that the ego-vehicle must follow?\n\n"
        f"{TRAFFIC_LIGHT_OPTIONS}"
    )
    pair = parse_question(text, "(c) Green light is visible.")
    assert pair.answer == "C"
    assert pair.options == (
        "A. Red light is visible.",
        "B. Yellow light is visible.",
        "C. Green light is visible.",
        "D. No traffic light is visible.",
    )
    assert pair.allowed_answers == frozenset("abcd")
    assert pair.question.startswith(
        "Are there any traffic lights ahead that the ego-vehicle must follow?\n"
        "Ego vehicle over the past 1.5 s:\n- Velocity(m/s)"
    )
    assert normalize_qa_prediction("C.", pair.allowed_answers) == "c"


def test_options_listed_twice_keep_the_second_list_which_the_answer_is_from():
    text = (
        f"{PREAMBLE_EVAL}\n- Velocity(m/s): [0.5]\n\n"
        " Is there a traffic signal the ego-vehicle should give priority to\n        \n"
        "(a) No traffic signals are visible.\n(b) Green lights are visible.\n"
        "(c) Red lights are visible.\n(d) Yellow lights are visible.?\n\n"
        "(a) No traffic lights are present.\n(b) Green lights are visible.\n"
        "(c) Yellow lights are visible.\n(d) Red lights are visible."
    )
    pair = parse_question(text, "(d) Red lights are visible.")
    assert pair.answer == "D"
    assert pair.options[0] == "A. No traffic lights are present."
    assert pair.options[3] == "D. Red lights are visible."
    assert pair.question.startswith(
        "Is there a traffic signal the ego-vehicle should give priority to?\n"
    )


def test_an_answer_that_is_not_an_option_is_refused():
    text = f"{PREAMBLE_EVAL}\nAre there traffic lights?\n{TRAFFIC_LIGHT_OPTIONS}"
    with pytest.raises(ValueError, match="not one of the options"):
        parse_question(text, "(c) Red light is visible.")


@pytest.fixture
def nuplan_qa(tmp_path):
    """Scene f0 <- f1 <- f2 in part_7 (extracted), g0 in part_8 (not extracted)."""
    frames = [
        ("f0", "", "part_7"),
        ("f1", "f0", "part_7"),
        ("f2", "f1", "part_7"),
        ("g0", "", "part_8"),
    ]
    n = len(frames)
    index_path = tmp_path / "index.npz"
    np.savez(
        index_path,
        cameras=np.array(CAMERAS),
        tokens=np.array([token for token, _, _ in frames]),
        prev=np.array([prev for _, prev, _ in frames]),
        scene_tokens=np.array(["s", "s", "s", "t"]),
        scenario_types=np.array(["on_intersection"] * n),
        timestamps=np.arange(n, dtype=np.int64) * 600_000,
        image_paths=np.array(
            [
                [f"{part}/{token}_{camera}.jpg" for camera in CAMERAS]
                for token, _, part in frames
            ]
        ),
        intrinsics=np.tile(np.eye(3, dtype=np.float32), (n, 6, 1, 1)),
        cam_translation=np.zeros((n, 6, 3), np.float32)
        + np.arange(6, dtype=np.float32)[None, :, None],
        cam_rotation=np.tile(np.eye(3, dtype=np.float32), (n, 6, 1, 1)),
    )
    images = tmp_path / "images"
    (images / "part_7").mkdir(parents=True)
    (
        images / "part_9.partial"
    ).mkdir()  # an extraction in progress isn't an archive folder
    for token, _, part in frames[:3]:
        for c, camera in enumerate(CAMERAS):
            Image.new("RGB", (16, 8), (40 * c, 0, 0)).save(
                images / part / f"{token}_{camera}.jpg"
            )
    question = f"{PREAMBLE_TRAIN}\n- Steering angles: [0.1]\nWhat is the vehicle's current maneuver?"
    rows = [
        {
            "sample_token": "f1",
            "task": "Ego-Vehicle Maneuver",
            "QA": [{"q": question, "a": "Turning left."}],
        },
        {
            "sample_token": "f1",
            "task": "Road Type",
            "QA": [{"q": f"{PREAMBLE_TRAIN}\nRoad?", "a": "Dry."}],
        },
        {
            "sample_token": "f2",
            "task": "Road Type",
            "QA": [{"q": f"{PREAMBLE_TRAIN}\nRoad?", "a": "Wet."}],
        },
        {
            "sample_token": "g0",
            "task": "Road Type",
            "QA": [{"q": f"{PREAMBLE_TRAIN}\nRoad?", "a": "Dry."}],
        },
        {
            "sample_token": "zz",
            "task": "Road Type",
            "QA": [{"q": f"{PREAMBLE_TRAIN}\nRoad?", "a": "Dry."}],
        },
    ]
    annotations = tmp_path / "nuplanqa_train.json"
    annotations.write_text(json.dumps(rows))
    return {
        "root": str(images),
        "index_path": str(index_path),
        "qa_annotations_path": str(annotations),
    }


def test_dataset_has_the_annotated_frames_of_extracted_archives(nuplan_qa):
    dataset = NuPlanQADataset(**nuplan_qa)
    assert dataset.available_parts == ["part_7"]
    assert len(dataset) == 2
    assert dataset.qa_dropped == 2
    assert dataset.missing_parts == {"part_8", "not in the index"}
    pairs = dataset.qa_pairs_by_index()
    assert [len(pairs[i]) for i in range(2)] == [2, 1]
    with pytest.raises(ValueError, match="part_8"):
        dataset.qa_pairs_by_index(strict=True)


def test_samples_have_six_cameras_with_calibration_and_no_driving_targets(nuplan_qa):
    sample = NuPlanQADataset(**nuplan_qa)[0]
    assert sample.metadata["scenario_id"] == "f1"
    assert sample.agent_input == {}
    assert (
        sample.intent is None and sample.fut_traj is None and sample.agent_info is None
    )
    assert len(sample.qa_pairs) == 2
    assert len(sample.cameras) == 1
    frame = sample.cameras[0]
    assert set(frame) == {
        "FRONT",
        "FRONT_LEFT",
        "FRONT_RIGHT",
        "REAR_LEFT",
        "REAR_RIGHT",
        "REAR",
    }
    # FRONT_LEFT is CAM_L0, the index's second camera.
    left = frame["FRONT_LEFT"]
    assert left.image.shape == (8, 16, 3)
    assert abs(int(left.image[..., 0].float().mean()) - 40) <= 3
    assert left.cal_dict.translation.tolist() == [1.0, 1.0, 1.0]
    assert left.cal_dict.original_img_size == (8, 16)
    assert not left.cal_dict.is_flu


def test_past_frames_follow_the_previous_frame_and_stop_at_the_scene_start(nuplan_qa):
    dataset = NuPlanQADataset(**nuplan_qa, num_past_frames=2)
    assert [len(dataset[i].cameras) for i in range(2)] == [2, 3]
    strided = NuPlanQADataset(**nuplan_qa, num_past_frames=1, past_frame_stride=2)
    assert [len(strided[i].cameras) for i in range(2)] == [1, 2]


def test_builder_makes_a_qa_training_entry_at_the_entry_size(nuplan_qa):
    entry = {
        "_target_": "ad_vla.dataset.nuplan_qa.nuplan_qa_dataset.NuPlanQADataset",
        **nuplan_qa,
        "task": "qa",
        "sensors": "all_cams",
        "size": [16, 32],
        "questions_per_frame": 1,
        "num_past_frames": 2,
    }
    dataset = build_train_dataset(OmegaConf.create({"train": {"nuplan_qa": entry}}))
    assert isinstance(dataset, QATrainDataset)
    assert len(dataset) == 2
    item = dataset[1]
    assert item.qa_pair is not None and item.qa_pair.answer == "Wet."
    assert item.cameras[-1]["FRONT"].image.shape == (16, 32, 3)
    assert item.cameras[-1]["FRONT"].cal_dict.intrinsics[0, 0] == pytest.approx(2.0)


def test_qa_prompt_shows_every_camera_of_every_frame(nuplan_qa):
    sample = NuPlanQADataset(**nuplan_qa, num_past_frames=2)[1]
    sample = sample.model_copy(update={"qa_pair": sample.qa_pairs[0]})
    prompt = PromptBuilder(
        {
            "qa_uses_system_prompt": True,
            "intro": "You are an expert driver.\nInput:",
            "input_sections": {"images": "- images"},
            "task_text": "Task: drive or answer.",
        }
    )
    message = prompt.build_qa_message(sample, sample.qa_pair, num_past_image_frames=3)
    content = message[1]["content"]
    assert sum(part["type"] == "image" for part in content) == 18
    labels = [
        part["text"] for part in content if part.get("text", "").endswith(" camera:")
    ]
    assert labels == [
        f"{name} camera:"
        for name in (
            "FRONT_LEFT",
            "FRONT",
            "FRONT_RIGHT",
            "REAR_LEFT",
            "REAR",
            "REAR_RIGHT",
        )
    ]
    assert content[-1]["text"] == "Question: Road?"
    assert message[2]["content"][0]["text"] == "Wet."


def test_fm_collator_gives_question_only_rows_a_zero_state(nuplan_qa):
    collator = object.__new__(QwenFlowMatchingVLACollator)
    collator.past_traj_sampling = TrajectorySampling(num_poses=4, interval_length=0.5)
    sample = NuPlanQADataset(**nuplan_qa)[0]
    question = sample.model_copy(update={"qa_pair": sample.qa_pairs[0]})
    assert torch.equal(collator._resample_past_traj(question), torch.zeros(4, 2))
    assert collator._initial_velocity(question) == 0
    with pytest.raises(KeyError):  # a driving row still needs its past
        collator._resample_past_traj(sample)


def _drive_vqa() -> PromptBuilder:
    """The real drive_vqa prompt, resolved for a 3-frame model."""
    prompt = OmegaConf.load(
        Path(__file__).parents[1] / "ad_vla/conf/prompt/drive_vqa.yaml"
    )
    sampling = {"num_poses": 4, "interval_length": 0.5}
    cfg = OmegaConf.create(
        {
            "model": {
                "model_args": {
                    "past_traj_sampling": sampling,
                    "pred_traj_sampling": sampling,
                    "num_past_image_frames": 3,
                }
            },
            "prompt": prompt,
        }
    )
    return PromptBuilder(OmegaConf.to_container(cfg.prompt, resolve=True))


def test_qa_turn_states_the_samples_own_frame_spacing(nuplan_qa):
    prompt = _drive_vqa()
    sample = NuPlanQADataset(**nuplan_qa, num_past_frames=1)[1]
    sample = sample.model_copy(update={"qa_pair": sample.qa_pairs[0]})
    message = prompt.build_qa_message(sample, sample.qa_pair, num_past_image_frames=3)
    system = message[0]["content"][0]["text"]
    driving_system = prompt.build_system_prompt()
    assert "0.5s intervals" in driving_system
    # Only the images line changes, not the trajectories' 0.5 s here.
    assert system.count("0.5s intervals") == driving_system.count("0.5s intervals") - 1
    assert system == driving_system.replace("frames, 0.5s", "frames, 0.6s")
    content = message[1]["content"]
    assert sum(part["type"] == "image" for part in content) == 12
    assert [
        part["text"] for part in content if part.get("text", "").startswith("<frame")
    ][:2] == [
        "<frame -1>",
        "<frame 0>",
    ]

    # Other datasets' samples don't state a spacing and keep the prompt's.
    other = sample.model_copy(
        update={"metadata": {"scenario_id": "x", "sampling_freq": 2.0}}
    )
    other_message = prompt.build_qa_message(
        other, other.qa_pair, num_past_image_frames=3
    )
    assert other_message[0]["content"][0]["text"] == driving_system

    strided = NuPlanQADataset(**nuplan_qa, num_past_frames=1, past_frame_stride=2)[1]
    assert strided.metadata["image_interval_s"] == pytest.approx(1.2)


def test_a_prompt_without_the_spacing_text_refuses_another_spacing(nuplan_qa):
    prompt = PromptBuilder(
        {
            "qa_uses_system_prompt": True,
            "image_interval": 0.5,
            "intro": "You are an expert driver.\nInput:",
            "input_sections": {"images": "- images"},
            "task_text": "Task: drive or answer.",
        }
    )
    sample = NuPlanQADataset(**nuplan_qa)[0]
    with pytest.raises(ValueError, match="0.6s apart"):
        prompt.build_qa_message(sample, sample.qa_pairs[0])
