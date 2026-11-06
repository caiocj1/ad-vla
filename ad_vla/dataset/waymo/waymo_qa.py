"""Waymo QA annotations: JSONL rows matched to dataset samples without decoding any record."""

import json
import pickle
from pathlib import Path

from ad_vla.dataset.data_types import QuestionAnswerPair

_ANNOTATION_SPLITS = {"train": "train", "val": "validation", "test": "test"}


def _pair(row: dict) -> QuestionAnswerPair:
    """One annotation row. Multiple choice rows have the option's letter as their answer."""
    options = tuple(row["options"]) if row.get("options") else None
    if options is None:
        answer = str(row["answer"]).strip()
        allowed = None
    else:
        answer = chr(ord("A") + int(row["answer_index"]))
        allowed = frozenset(chr(ord("a") + i) for i in range(len(options)))
    return QuestionAnswerPair(
        question=str(row["question"]).strip(),
        answer=answer,
        category=row.get("category"),
        scope=row["type"],
        options=options,
        allowed_answers=allowed,
    )


def load_waymo_qa(
    annotations_path: str, split: str, clip_index_path: str | None
) -> tuple[dict[int, tuple[QuestionAnswerPair, ...]], dict[str, int], int]:
    """Attach each question to its sample, through the clip index: clip -> [(frame, dataset idx)].

    A row names its clip by ``token``. Image questions name a frame too. Video questions have
    none and go on the clip's last frame, which has the most past behind it.

    Returns:
        Questions by dataset index, the dataset index of each ``<token>-<frame>`` scenario id with
        questions, and how many questions had no frame in this dataset.
    """
    if clip_index_path is None:
        raise ValueError("Waymo QA needs clip_index_path to find the annotated frames.")
    with open(clip_index_path, "rb") as clip_file:
        clips = pickle.load(clip_file)

    grouped: dict[int, list[QuestionAnswerPair]] = {}
    index_by_scenario: dict[str, int] = {}
    dropped = 0
    path = Path(annotations_path).expanduser()
    with path.open(encoding="utf-8") as annotations_file:
        for line_number, line in enumerate(annotations_file, 1):
            row = json.loads(line)
            if row["waymo_split"] != _ANNOTATION_SPLITS[split]:
                raise ValueError(
                    f"{path}:{line_number} is from {row['waymo_split']}, "
                    f"but the dataset split is {split}."
                )
            frames = dict(clips.get(row["token"], ()))
            frame = (
                max(frames, default=None)
                if row["type"] == "video"
                else row["frame_index"]
            )
            if frame not in frames:
                dropped += 1
                continue
            index = frames[frame]
            grouped.setdefault(index, []).append(_pair(row))
            index_by_scenario[f"{row['token']}-{frame:03d}"] = index
    return {i: tuple(p) for i, p in grouped.items()}, index_by_scenario, dropped
