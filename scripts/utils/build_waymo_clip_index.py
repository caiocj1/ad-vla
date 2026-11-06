"""Build temporal clip-to-dataset-index maps for Waymo E2E splits.

Each output pickle contains this mapping::

    dict[str, list[tuple[int, int]]]
    # clip_id -> [(frame_number, dataset_index), ...]

The list for every clip is sorted by ``frame_number``. ``dataset_index`` refers
to the corresponding position in ``wod_e2e_{split}_index.pkl`` and can be used
directly with ``WaymoE2EDataset`` configured for that split.

The script reads only the protobuf fields leading to ``frame.context.name``;
it skips image payloads and does not decode camera images.

Example:

    python scripts/utils/build_waymo_clip_index.py

By default, the script reads ``WAYMO_INDEX_PATH`` (including from ``.env``),
processes train/val/test, and writes the maps beside the source index files.
"""

from __future__ import annotations

import argparse
from collections import defaultdict
from collections.abc import Iterable
import os
from pathlib import Path
import pickle
import tempfile
import time
from typing import BinaryIO

from dotenv import load_dotenv


SPLITS = ("train", "val", "test")
OUTPUT_NAME = "wod_e2e_{split}_clip_to_dataset_indices.pkl"


class ProtobufWireError(ValueError):
    """Raised when the expected protobuf fields cannot be read safely."""


def _read_varint(stream: BinaryIO, end_offset: int) -> int:
    """Read one protobuf varint without crossing ``end_offset``."""
    value = 0
    shift = 0
    while stream.tell() < end_offset and shift < 70:
        byte = stream.read(1)
        if not byte:
            break
        value |= (byte[0] & 0x7F) << shift
        if not byte[0] & 0x80:
            return value
        shift += 7
    raise ProtobufWireError("Truncated or invalid protobuf varint")


def _skip_field(stream: BinaryIO, wire_type: int, end_offset: int) -> None:
    """Skip a protobuf value while staying inside its enclosing message."""
    if wire_type == 0:  # varint
        _read_varint(stream, end_offset)
        return
    if wire_type == 1:  # fixed64
        new_offset = stream.tell() + 8
    elif wire_type == 2:  # length-delimited
        new_offset = stream.tell() + _read_varint(stream, end_offset)
    elif wire_type == 5:  # fixed32
        new_offset = stream.tell() + 4
    else:
        raise ProtobufWireError(f"Unsupported protobuf wire type {wire_type}")

    if new_offset > end_offset:
        raise ProtobufWireError("Protobuf field crosses its enclosing message")
    stream.seek(new_offset)


def _find_message_field(
    stream: BinaryIO,
    *,
    end_offset: int,
    field_number: int,
) -> tuple[int, int]:
    """Find a length-delimited field and return its payload bounds."""
    while stream.tell() < end_offset:
        key = _read_varint(stream, end_offset)
        current_field = key >> 3
        wire_type = key & 0x07
        if current_field == 0:
            raise ProtobufWireError("Encountered invalid protobuf field number 0")

        if current_field == field_number:
            if wire_type != 2:
                raise ProtobufWireError(
                    f"Field {field_number} has wire type {wire_type}, expected 2"
                )
            size = _read_varint(stream, end_offset)
            start_offset = stream.tell()
            field_end = start_offset + size
            if field_end > end_offset:
                raise ProtobufWireError(
                    f"Field {field_number} crosses its enclosing message"
                )
            return start_offset, field_end

        _skip_field(stream, wire_type, end_offset)

    raise ProtobufWireError(f"Protobuf field {field_number} was not found")


def read_scenario_id(
    stream: BinaryIO,
    *,
    record_offset: int,
    record_length: int,
) -> str:
    """Read ``E2EDFrame.frame.context.name`` from an indexed TFRecord entry."""
    # A TFRecord has an 8-byte length, a 4-byte length CRC, then its payload.
    record_start = record_offset + 12
    record_end = record_start + record_length
    stream.seek(record_start)

    # E2EDFrame.frame (field 1) -> Frame.context (field 1)
    # -> Context.name (field 1).
    _, frame_end = _find_message_field(
        stream,
        end_offset=record_end,
        field_number=1,
    )
    _, context_end = _find_message_field(
        stream,
        end_offset=frame_end,
        field_number=1,
    )
    name_start, name_end = _find_message_field(
        stream,
        end_offset=context_end,
        field_number=1,
    )

    stream.seek(name_start)
    raw_name = stream.read(name_end - name_start)
    if len(raw_name) != name_end - name_start:
        raise ProtobufWireError("Truncated frame.context.name")
    try:
        return raw_name.decode("utf-8")
    except UnicodeDecodeError as exc:
        raise ProtobufWireError("frame.context.name is not valid UTF-8") from exc


def split_scenario_id(scenario_id: str) -> tuple[str, int]:
    """Split ``<clip_id>-<frame_number>`` at its final hyphen."""
    try:
        clip_id, frame_text = scenario_id.rsplit("-", 1)
    except ValueError as exc:
        raise ValueError(
            f"Scenario ID {scenario_id!r} does not end in '-<frame_number>'"
        ) from exc
    if not clip_id or not frame_text.isdigit():
        raise ValueError(
            f"Scenario ID {scenario_id!r} does not end in '-<frame_number>'"
        )
    return clip_id, int(frame_text)


def build_clip_index(
    record_index: Iterable[dict[str, object]],
    *,
    progress_every: int = 10_000,
) -> dict[str, list[tuple[int, int]]]:
    """Build a temporally sorted clip map from a Waymo TFRecord index."""
    by_clip: defaultdict[str, list[tuple[int, int]]] = defaultdict(list)
    seen_frames: defaultdict[str, dict[int, int]] = defaultdict(dict)
    current_path: Path | None = None
    stream: BinaryIO | None = None

    try:
        for dataset_index, entry in enumerate(record_index):
            path = Path(str(entry["path"]))
            record_offset = int(entry["offset"])
            record_length = int(entry["length"])

            if path != current_path:
                if stream is not None:
                    stream.close()
                stream = path.open("rb")
                current_path = path

            try:
                scenario_id = read_scenario_id(
                    stream,
                    record_offset=record_offset,
                    record_length=record_length,
                )
                clip_id, frame_number = split_scenario_id(scenario_id)
            except (OSError, ProtobufWireError, ValueError) as exc:
                raise RuntimeError(
                    "Failed to extract the scenario ID for "
                    f"dataset index {dataset_index} at {path}:{record_offset}"
                ) from exc

            previous_index = seen_frames[clip_id].get(frame_number)
            if previous_index is not None:
                raise ValueError(
                    f"Duplicate frame {frame_number} in clip {clip_id!r}: "
                    f"dataset indices {previous_index} and {dataset_index}"
                )
            seen_frames[clip_id][frame_number] = dataset_index
            by_clip[clip_id].append((frame_number, dataset_index))

            if progress_every and (dataset_index + 1) % progress_every == 0:
                print(f"  processed {dataset_index + 1:,} records", flush=True)
    finally:
        if stream is not None:
            stream.close()

    result = dict(by_clip)
    for entries in result.values():
        entries.sort(key=lambda item: item[0])
    return result


def _count_missing_frames(clip_index: dict[str, list[tuple[int, int]]]) -> int:
    """Count absent frame numbers between each clip's first and last frame."""
    missing = 0
    for entries in clip_index.values():
        if entries:
            missing += entries[-1][0] - entries[0][0] + 1 - len(entries)
    return missing


def _write_pickle_atomic(value: object, output_path: Path) -> None:
    output_path.parent.mkdir(parents=True, exist_ok=True)
    with tempfile.NamedTemporaryFile(
        mode="wb",
        dir=output_path.parent,
        prefix=f".{output_path.name}.",
        suffix=".tmp",
        delete=False,
    ) as stream:
        temporary_path = Path(stream.name)
        pickle.dump(value, stream, protocol=pickle.HIGHEST_PROTOCOL)
    try:
        temporary_path.replace(output_path)
    except BaseException:
        temporary_path.unlink(missing_ok=True)
        raise


def process_split(index_root: Path, output_dir: Path, split: str) -> Path:
    index_path = index_root / f"wod_e2e_{split}_index.pkl"
    output_path = output_dir / OUTPUT_NAME.format(split=split)

    print(f"[{split}] Loading {index_path}", flush=True)
    with index_path.open("rb") as stream:
        record_index = pickle.load(stream)
    if not isinstance(record_index, list):
        raise TypeError(f"Expected a list in {index_path}, got {type(record_index)}")

    started = time.monotonic()
    clip_index = build_clip_index(record_index)
    elapsed = time.monotonic() - started
    missing_frames = _count_missing_frames(clip_index)
    _write_pickle_atomic(clip_index, output_path)

    print(
        f"[{split}] Wrote {len(clip_index):,} clips / "
        f"{len(record_index):,} frames to {output_path} in {elapsed:.1f}s "
        f"({missing_frames:,} missing frame numbers within clip ranges)",
        flush=True,
    )
    return output_path


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument(
        "--index-root",
        type=Path,
        default=None,
        help="Directory containing wod_e2e_<split>_index.pkl. "
        "Defaults to WAYMO_INDEX_PATH.",
    )
    parser.add_argument(
        "--output-dir",
        type=Path,
        default=None,
        help="Output directory. Defaults to --index-root.",
    )
    parser.add_argument(
        "--splits",
        nargs="+",
        choices=SPLITS,
        default=list(SPLITS),
        help="Splits to process (default: train val test).",
    )
    return parser.parse_args()


def main() -> None:
    load_dotenv()
    args = parse_args()
    index_root = args.index_root
    if index_root is None:
        configured_root = os.environ.get("WAYMO_INDEX_PATH")
        if not configured_root:
            raise ValueError(
                "Set WAYMO_INDEX_PATH or pass --index-root explicitly"
            )
        index_root = Path(configured_root)
    output_dir = args.output_dir or index_root

    for split in args.splits:
        process_split(index_root, output_dir, split)


if __name__ == "__main__":
    main()
