"""NuPlanQA (github.com/sungyeonparkk/NuPlanQA): questions about nuPlan frames with six cameras.

It is a question-answering dataset only: its samples have images and calibration but no past or
future trajectory, intent or agents, so it is a ``task: qa`` entry. Questions come with the ego
vehicle's speed and steering over the past 1.5 s as text, which is kept, after the question.

Training questions have free-text answers. The evaluation questions are multiple choice: they show
lettered options, and the answer is the letter, as for Waymo QA.

Frames are 0.6 s apart, not 0.5 s like the other datasets', so samples state their spacing in
``metadata['image_interval_s']`` and the QA system prompt says it instead of the prompt's.

Images are read from ``root/part_<archive>/<image token>.jpg``, one folder per camera archive, the
layout NuPlanQA expects. Frames whose archive has no folder yet are left out of training, and
validation refuses them. The index of frames and cameras is made once from NuPlanQA's metadata
pickle by ``scripts/utils/build_nuplanqa_index.py``.
"""

import json
import os
import re
import sys
from pathlib import Path

import numpy as np
import torch
from PIL import Image

from ad_vla.dataset.base_dataset import BaseDataset
from ad_vla.dataset.data_types import E2EDataSample, QuestionAnswerPair
from ad_vla.dataset.navsim.navsim_types import NAVSIM_CAM_LABELS

# Frames of one scene are 0.6 s apart (nuPlan's camera frames, subsampled by NuPlanQA).
FRAME_INTERVAL_S = 0.6

_PREAMBLE = "You are driving on the road."
_OPTION = re.compile(r"\(([a-z])\)\s*(.+)")
_PART = re.compile(r"part_\d+")


def parse_question(
    text: str, answer: str, task: str | None = None, ego_state_text: bool = True
) -> QuestionAnswerPair:
    """One NuPlanQA question, without its preamble, which our system prompt replaces.

    The lines starting with ``- `` are the ego vehicle's past speed and steering; they follow the
    question when ``ego_state_text``. Option lines ``(a) ...`` become options ``A. ...`` and the
    answer ``(a) ...`` becomes ``A``. A few questions list their options twice, once inside the
    question; their answer is from the second list, so a repeated letter replaces the first.
    """
    question, ego_lines, options = [], [], {}
    for line in text.split("\n"):
        line = line.strip()
        if not line or line.startswith(_PREAMBLE):
            continue
        option = _OPTION.fullmatch(line)
        if option:
            options[option[1]] = option[2].strip()
        elif options:
            raise ValueError(f"Question text after its options: {text!r}")
        elif line.startswith("- "):
            ego_lines.append(line)
        else:
            question.append(line)
    if not question:
        raise ValueError(f"No question in {text!r}")
    question_text = " ".join(question)
    if question_text[-1] not in "?.":
        question_text += "?"
    if ego_state_text and ego_lines:
        question_text += "\nEgo vehicle over the past 1.5 s:\n" + "\n".join(ego_lines)

    answer = answer.strip()
    if not options:
        return QuestionAnswerPair(
            question=question_text, answer=answer, category=task, scope="image"
        )
    letters = sorted(options)
    if letters != [chr(ord("a") + i) for i in range(len(letters))]:
        raise ValueError(f"Options are not lettered from (a): {text!r}")
    chosen = _OPTION.fullmatch(answer)
    if chosen is None or options.get(chosen[1]) != chosen[2].strip():
        raise ValueError(f"Answer {answer!r} is not one of the options of {text!r}")
    return QuestionAnswerPair(
        question=question_text,
        answer=chosen[1].upper(),
        category=task,
        scope="image",
        options=tuple(f"{letter.upper()}. {options[letter]}" for letter in letters),
        allowed_answers=frozenset(letters),
    )


def load_nuplan_qa(
    annotations_path: str, ego_state_text: bool = True
) -> dict[str, list[QuestionAnswerPair]]:
    """Questions of each frame token, from a NuPlanQA JSON file (train or eval)."""
    with Path(annotations_path).expanduser().open(encoding="utf-8") as f:
        rows = json.load(f)
    by_token: dict[str, list[QuestionAnswerPair]] = {}
    for row in rows:
        task = sys.intern(row["task"])
        pairs = by_token.setdefault(row["sample_token"], [])
        for qa in row["QA"]:
            pairs.append(parse_question(qa["q"], qa["a"], task, ego_state_text))
    return by_token


class NuPlanQADataset(BaseDataset):
    """NuPlanQA frames that have questions, with their six cameras and optionally past frames.

    Samples carry ``qa_pairs`` and no driving targets: ``agent_input`` is empty and ``intent``,
    ``fut_traj`` and ``agent_info`` are None.
    """

    def __init__(
        self,
        root: str,
        index_path: str,
        qa_annotations_path: str,
        split: str = "train",
        sensors: str = "all_cams",
        img_transform=None,
        reasoning_traces_path: str | None = None,
        num_past_frames: int = 0,
        past_frame_stride: int = 1,
        ego_state_text: bool = True,
    ) -> None:
        """
        Args:
            root: Folder of the extracted camera archives, ``part_<n>/<image token>.jpg``.
            index_path: ``.npz`` from ``scripts/utils/build_nuplanqa_index.py``.
            qa_annotations_path: ``nuplanqa_train.json`` or ``nuplanqa_eval_test.json``.
            split: Dataset split; the annotations file decides which questions there are.
            sensors: Camera configuration; QA entries use ``all_cams``.
            img_transform: Image transform applied to each camera image.
            reasoning_traces_path: Unused here; accepted like every training entry's.
            num_past_frames: Past frames added to ``sample.cameras``, oldest first. Near the start
                of a scene the list is shorter; frames are never repeated.
            past_frame_stride: Spacing of past frames, in frames of 0.6 s.
            ego_state_text: Keep the past speed and steering that come with each question.
        """
        super().__init__(root, split, sensors, img_transform, reasoning_traces_path)
        if num_past_frames < 0 or past_frame_stride < 1:
            raise ValueError("num_past_frames must be >= 0 and past_frame_stride >= 1.")
        self.num_past_frames = num_past_frames
        self.past_frame_stride = past_frame_stride

        with np.load(index_path, allow_pickle=False) as index:
            raw_cameras = [str(name) for name in index["cameras"]]
            self.tokens = index["tokens"]
            self.scene_tokens = index["scene_tokens"]
            self.scenario_types = index["scenario_types"]
            self.timestamps = index["timestamps"]
            self.image_paths = index["image_paths"]
            self.intrinsics = index["intrinsics"]
            self.cam_translation = index["cam_translation"]
            self.cam_rotation = index["cam_rotation"]
            prev = index["prev"]
        position = {str(token): i for i, token in enumerate(self.tokens)}
        self.prev_position = np.array(
            [position.get(str(token), -1) for token in prev], dtype=np.int64
        )

        standard = {
            NAVSIM_CAM_LABELS[name.lower()]: c for c, name in enumerate(raw_cameras)
        }
        self.camera_columns = [
            (name, standard[name]) for name in self._resolve_cameras(set(standard))
        ]

        root_path = Path(root)
        self.available_parts = sorted(
            p.name
            for p in root_path.iterdir()
            if p.is_dir() and _PART.fullmatch(p.name)
        )
        available = set(self.available_parts)
        questions = load_nuplan_qa(qa_annotations_path, ego_state_text)
        kept, self.missing_parts, self.qa_dropped = [], set(), 0
        for token, pairs in questions.items():
            i = position.get(token, -1)
            part = self.image_paths[i, 0].split("/")[0] if i >= 0 else None
            if part in available:
                kept.append((i, tuple(pairs)))
            else:
                self.missing_parts.add(part or "not in the index")
                self.qa_dropped += len(pairs)
        kept.sort(
            key=lambda item: (self.scene_tokens[item[0]], self.timestamps[item[0]])
        )
        self.positions = np.array([i for i, _ in kept], dtype=np.int64)
        self.qa_by_index = {k: pairs for k, (_, pairs) in enumerate(kept)}
        print(
            f"NuPlanQA {split}: {sum(map(len, self.qa_by_index.values())):,} questions on "
            f"{len(self.positions):,} frames of {self.available_parts}; "
            f"{self.qa_dropped:,} questions left out ({sorted(self.missing_parts)})."
        )

    def qa_pairs_by_index(
        self, strict: bool = False
    ) -> dict[int, tuple[QuestionAnswerPair, ...]]:
        """Questions of each sample. See the base class."""
        if strict and self.qa_dropped:
            raise ValueError(
                f"{self.qa_dropped:,} NuPlanQA questions have no images in {self.root}: "
                f"extract {sorted(self.missing_parts)} first."
            )
        return self.qa_by_index

    def __len__(self) -> int:
        return len(self.positions)

    def _past_positions(self, position: int) -> list[int]:
        """Index positions of the past frames, nearest first, stopping at the scene's start."""
        past = []
        for _ in range(self.num_past_frames):
            for _ in range(self.past_frame_stride):
                position = self.prev_position[position]
                if position < 0:
                    return past
            past.append(int(position))
        return past

    def _load_frame(self, position: int) -> dict[str, dict]:
        frame = {}
        for name, column in self.camera_columns:
            path = os.path.join(self.root, self.image_paths[position, column])
            with Image.open(path) as image:
                pixels = torch.from_numpy(np.array(image.convert("RGB")))
            cal_dict = {
                "intrinsics": torch.from_numpy(
                    self.intrinsics[position, column].copy()
                ),
                "translation": torch.from_numpy(
                    self.cam_translation[position, column].copy()
                ),
                "rotation": torch.from_numpy(
                    self.cam_rotation[position, column].copy()
                ),
                "original_img_size": tuple(pixels.shape[:2]),
                "is_flu": False,
            }
            image, cal_dict = self._apply_img_transform(pixels, cal_dict, name)
            frame[name] = {"image": image, "cal_dict": cal_dict}
        return frame

    def __getitem__(self, idx: int) -> E2EDataSample:
        position = int(self.positions[idx])
        frames = [position] + self._past_positions(position)
        return E2EDataSample(
            metadata={
                "scenario_id": str(self.tokens[position]),
                "sampling_freq": 1.0 / FRAME_INTERVAL_S,
                # Not the prompt's 0.5 s: QA turns state this spacing instead.
                "image_interval_s": FRAME_INTERVAL_S * self.past_frame_stride,
                "dataset": "nuplan_qa",
                "scene_token": str(self.scene_tokens[position]),
                "scenario_type": str(self.scenario_types[position]),
                "timestamp_us": int(self.timestamps[position]),
            },
            agent_input={},
            cameras=[self._load_frame(p) for p in reversed(frames)],
            intent=None,
            qa_pairs=self.qa_by_index[idx],
        )
