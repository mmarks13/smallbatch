"""Pins the crash-safe write contract before the six ad-hoc writers merge.

Every persisted JSON/JSONL file must be staged beside its target under the
full name plus `.tmp` (so `result.local.json` stages as
`result.local.json.tmp`, never `result.local.tmp`), written with
`ensure_ascii=False`, and moved into place atomically.
"""

import json

import pytest

from smallbatch.atomic import atomic_json, atomic_jsonl, atomic_text


def test_json_replaces_existing_content_and_keeps_unicode_raw(tmp_path):
    path = tmp_path / "meta.json"
    path.write_text("old")
    atomic_json(path, {"reason": "café ☕"})
    assert json.loads(path.read_text()) == {"reason": "café ☕"}
    assert "café" in path.read_text()  # ensure_ascii=False, no \u escapes


def test_no_tmp_file_survives_a_successful_write(tmp_path):
    path = tmp_path / "meta.json"
    atomic_json(path, {"ok": True})
    assert [entry.name for entry in tmp_path.iterdir()] == ["meta.json"]


def test_staging_name_keeps_every_suffix(tmp_path):
    # replace() onto a directory fails after staging, exposing the tmp name;
    # a multi-dot target must stage under its full name, not lose ".json"
    target = tmp_path / "result.local.json"
    target.mkdir()
    with pytest.raises(OSError):
        atomic_json(target, {"ok": True})
    leftovers = [entry.name for entry in tmp_path.iterdir() if entry.is_file()]
    assert leftovers == ["result.local.json.tmp"]


def test_failed_serialization_leaves_the_target_untouched(tmp_path):
    path = tmp_path / "meta.json"
    path.write_text('{"kept": true}')
    with pytest.raises(TypeError):
        atomic_json(path, {"bad": object()})
    assert json.loads(path.read_text()) == {"kept": True}


def test_missing_parent_fails_unless_mkdir_requested(tmp_path):
    path = tmp_path / "builds" / "b1" / "manifest.json"
    with pytest.raises(FileNotFoundError):
        atomic_json(path, {})
    atomic_json(path, {"made": True}, mkdir=True)
    assert json.loads(path.read_text()) == {"made": True}


def test_jsonl_writes_one_compact_line_per_row(tmp_path):
    path = tmp_path / "train.jsonl"
    atomic_jsonl(path, [{"id": "a", "text": "héllo"}, {"id": "b"}])
    lines = path.read_text().splitlines()
    assert [json.loads(line) for line in lines] == [
        {"id": "a", "text": "héllo"},
        {"id": "b"},
    ]
    assert "héllo" in lines[0]


def test_text_writes_verbatim(tmp_path):
    path = tmp_path / "notes.txt"
    atomic_text(path, "line\n")
    assert path.read_text() == "line\n"
