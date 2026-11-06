"""Intent tables read by the datasets (ad_vla/dataset/intent_labels.py)."""

import json

import pytest

from ad_vla.dataset.intent_labels import IntentTable, load_intent_table
from ad_vla.utils.intent_utils import INTENTS_7


def _write(tmp_path, **extra):
    path = tmp_path / "intents.json"
    raw = {
        "classes": list(INTENTS_7),
        "labels": {"a": ["TURN_LEFT", "path"], "b": ["UNKNOWN", "nothing known"]},
        **extra,
    }
    path.write_text(json.dumps(raw))
    return str(path)


def test_lookup_by_key(tmp_path):
    table = IntentTable(_write(tmp_path))
    assert len(table) == 2
    assert table.get("a") == "TURN_LEFT"
    assert table.get("missing") is None
    assert "1 of 2 samples" in table.report("test", ["a", "c"])


def test_lookup_by_index(tmp_path):
    table = IntentTable(
        _write(tmp_path, by_index={"7": "LANE_CHANGE_RIGHT"}), by_index=True
    )
    assert table.get(7) == "LANE_CHANGE_RIGHT"
    assert table.get("7") is None
    with pytest.raises(ValueError, match="by_index"):
        IntentTable(_write(tmp_path), by_index=True)


def test_rejects_other_classes(tmp_path):
    path = tmp_path / "bad.json"
    path.write_text(json.dumps({"classes": ["GO_LEFT"], "labels": {}}))
    with pytest.raises(ValueError, match="classes"):
        IntentTable(str(path))


def test_no_path_no_table():
    assert load_intent_table(None) is None
