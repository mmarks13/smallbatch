import json
import re

import pytest

from conftest import imported_records, make_spec
from smallbatch.labeling import (
    build_dataset,
    dataset_hash,
    normalize_item_records,
    read_jsonl,
    row_id,
)


class FakeTeacher:
    def complete(self, prompt):
        count = len(re.findall(r'"id":\s*\d+', prompt.split("Reply with", 1)[0]))
        return json.dumps(
            [
                {"id": index, "output": "urgent" if index % 2 else "normal", "reason": "test"}
                for index in range(count)
            ]
        )


class AugmentationTeacher:
    def __init__(self, crash_on_label=False):
        self.crash_on_label = crash_on_label
        self.generation_calls = 0

    def complete(self, prompt):
        if prompt.startswith("Generate realistic"):
            self.generation_calls += 1
            return json.dumps(
                [
                    {
                        "title": f"generated {self.generation_calls}",
                        "body": "production outage",
                    }
                ]
            )
        if self.crash_on_label:
            raise KeyboardInterrupt
        count = len(re.findall(r'"id":\s*\d+', prompt.split("Reply with", 1)[0]))
        return json.dumps(
            [
                {"id": index, "output": "urgent", "reason": "test"}
                for index in range(count)
            ]
        )


def test_one_envelope_and_all_or_none_decisions():
    spec = make_spec()
    inputs, outputs = normalize_item_records(spec, imported_records(2))
    assert len(inputs) == 2 and outputs == ["normal", "urgent"]
    with pytest.raises(ValueError, match="cannot mix"):
        normalize_item_records(
            spec,
            [imported_records(1)[0], {"input": {"title": "x", "body": "y"}}],
        )
    with pytest.raises(ValueError, match="must use"):
        normalize_item_records(spec, [{"title": "flat", "body": "record"}])


def test_imported_decisions_split_and_metadata(tmp_path):
    spec = make_spec()
    meta = build_dataset(spec, imported_records(30), tmp_path)
    assert meta["decision_source"] == "imported"
    assert meta["counts"] == {"train": 21, "dev": 3, "eval": 6}
    assert meta["teacher"] is None
    rows = read_jsonl(tmp_path / "labeled.jsonl")
    assert len(rows) == 30
    assert dataset_hash(rows) == meta["dataset_hash"]
    assert all(set(row) >= {"id", "input", "output", "origin", "split"} for row in rows)


def test_teacher_decisions_use_same_dataset_shape(tmp_path):
    spec = make_spec(teacher={"backend": "codex-cli", "model": "test"})
    records = [{"input": record["input"]} for record in imported_records(20)]
    meta = build_dataset(spec, records, tmp_path, teacher=FakeTeacher())
    assert meta["decision_source"] == "teacher"
    assert meta["counts"] == {"train": 14, "dev": 2, "eval": 4}


def test_append_keeps_existing_evaluation_membership(tmp_path):
    spec = make_spec()
    build_dataset(spec, imported_records(30), tmp_path)
    before = {row["id"] for row in read_jsonl(tmp_path / "eval.jsonl")}
    records = imported_records(30)
    records.extend(
        {
            "input": {"title": f"new {index}", "body": "routine question account"},
            "output": "normal",
        }
        for index in range(10)
    )
    build_dataset(spec, records, tmp_path, append=True)
    after = {row["id"] for row in read_jsonl(tmp_path / "eval.jsonl")}
    assert before <= after


def test_calibration_ids_can_be_forced_to_train(tmp_path):
    spec = make_spec()
    records = imported_records(30)
    forced = {row_id(record["input"]) for record in records[:10]}
    build_dataset(spec, records, tmp_path, force_train_ids=forced)
    train = {row["id"] for row in read_jsonl(tmp_path / "train.jsonl")}
    assert forced <= train


def test_completed_augmentation_generation_replays_after_labeling_crash(tmp_path):
    spec = make_spec(
        teacher={"backend": "codex-cli", "model": "test"},
        augmentation={"paraphrase": {"cap": 2}},
    )
    with pytest.raises(KeyboardInterrupt):
        build_dataset(
            spec,
            imported_records(30),
            tmp_path,
            teacher=AugmentationTeacher(crash_on_label=True),
            max_variants=2,
        )

    resumed = AugmentationTeacher()
    meta = build_dataset(
        spec,
        imported_records(30),
        tmp_path,
        teacher=resumed,
        max_variants=2,
    )
    assert resumed.generation_calls == 0
    assert meta["variants"] == 2
