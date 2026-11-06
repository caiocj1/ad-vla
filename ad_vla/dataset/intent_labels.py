"""Precomputed 7-class intents that a dataset uses instead of its own labels (``intent_labels_path``).

The tables are built offline with the rule of ``intent7_from_path`` on whole logs: one JSON
per dataset and split, ``{"classes": [...], "labels": {key: [intent, source]}, ...}``, plus
``by_index: {dataset index: intent}`` for Waymo. Keys: NavSim frame token (the scene token),
nuScenes sample token, Waymo dataset index, PhysicalAI ``<clip_id>__<t0_us>``. Labels use the whole
logged future, which a sample doesn't hold; a sample missing from the table gets the same rule on
its own future (``intent7_from_path``), so a table only needs to cover the samples a run uses.
"""

import json
import sys
from collections.abc import Hashable, Iterable

from ad_vla.utils.intent_utils import INTENTS_7


class IntentTable:
    """Intent per sample key, from one table file."""

    def __init__(self, path: str, by_index: bool = False) -> None:
        with open(path) as f:
            raw = json.load(f)
        if set(raw["classes"]) != set(INTENTS_7):
            raise ValueError(f"{path}: classes {raw['classes']} are not {INTENTS_7}")
        names = {c: sys.intern(c) for c in INTENTS_7}
        if by_index:
            if "by_index" not in raw:
                raise ValueError(f"{path} has no by_index table")
            self.labels: dict[Hashable, str] = {
                int(k): names[v] for k, v in raw["by_index"].items()
            }
        else:
            self.labels = {k: names[v[0]] for k, v in raw["labels"].items()}
        self.path = path

    def __len__(self) -> int:
        return len(self.labels)

    def get(self, key: Hashable) -> str | None:
        return self.labels.get(key)

    def report(self, name: str, keys: Iterable[Hashable]) -> str:
        """One line on how many of a dataset's samples the table covers."""
        keys = list(keys)
        found = sum(1 for k in keys if k in self.labels)
        return (
            f"{name}: intents from {self.path}: {found:,} of {len(keys):,} samples "
            f"(the rest use the same rule on their own future)"
        )


def load_intent_table(path: str | None, by_index: bool = False) -> IntentTable | None:
    return None if path is None else IntentTable(path, by_index=by_index)
