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


def test_teacher_decisions_use_same_dataset_shape(tmp_path, capsys):
    spec = make_spec(teacher={"backend": "codex-cli", "model": "test"})
    records = [{"input": record["input"]} for record in imported_records(20)]
    meta = build_dataset(spec, records, tmp_path, teacher=FakeTeacher())
    assert meta["decision_source"] == "teacher"
    assert meta["counts"] == {"train": 14, "dev": 2, "eval": 4}
    progress = capsys.readouterr().err
    assert "teacher real batch=1/1 rows=20 attempt=1/2" in progress
    assert "teacher real complete rows=20" in progress


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


def test_split_gives_every_split_a_proportional_share_of_rare_classes(tmp_path):
    """A skewed distribution must not concentrate rare decisions in eval:
    slicing the class-interleaved head starved dev (breaking early stopping)
    and train (hiding classes from candidates) in the CFPB case study."""
    spec = make_spec(output={"type": "int", "range": [0, 4]})
    records = []
    for index in range(300):
        # skewed like real data: two dominant classes, three rare ones
        output = [0, 4, 1, 2, 3][index % 5] if index < 45 else (2 if index % 2 else 3)
        records.append(
            {
                "input": {"title": f"case {index}", "body": f"details {index}"},
                "output": output,
            }
        )
    meta = build_dataset(spec, records, tmp_path)
    assert meta["counts"] == {"train": 210, "dev": 30, "eval": 60}
    histograms = meta["split_label_histograms"]
    totals = {}
    for histogram in histograms.values():
        for label, count in histogram.items():
            totals[label] = totals.get(label, 0) + count
    for label, total in totals.items():
        eval_share = histograms["eval"].get(label, 0) / total
        train_share = histograms["train"].get(label, 0) / total
        assert 0.1 <= eval_share <= 0.4, f"label {label} eval share {eval_share}"
        assert train_share >= 0.5, f"label {label} train share {train_share}"
        if total >= 9:
            assert histograms["dev"].get(label, 0) >= 1, f"label {label} absent from dev"


class DropLastTeacher(AugmentationTeacher):
    """Labels every row in a batch except the last one, on every attempt."""

    def complete(self, prompt):
        if prompt.startswith("Generate realistic"):
            return super().complete(prompt)
        count = len(re.findall(r'"id":\s*\d+', prompt.split("Reply with", 1)[0]))
        return json.dumps(
            [
                {"id": index, "output": "urgent", "reason": "test"}
                for index in range(count - 1)
            ]
        )


def test_augmentation_drops_unlabelable_variants_instead_of_aborting(tmp_path, capsys):
    spec = make_spec(
        teacher={"backend": "codex-cli", "model": "test"},
        augmentation={"paraphrase": {"cap": 2}},
    )
    meta = build_dataset(
        spec,
        imported_records(30),
        tmp_path,
        teacher=DropLastTeacher(),
        max_variants=2,
    )
    assert meta["variants"] == 1
    assert "teacher paraphrase dropped 1 of 2 rows" in capsys.readouterr().err


def test_real_decisions_stay_all_or_nothing(tmp_path):
    spec = make_spec(teacher={"backend": "codex-cli", "model": "test"})
    records = [{"input": record["input"]} for record in imported_records(4)]
    with pytest.raises(ValueError, match="failed to return valid decisions"):
        build_dataset(spec, records, tmp_path, teacher=DropLastTeacher())


class ScriptedTeacher:
    """Answers by item title. A title's scripted list is consumed one output
    per draw, repeating its last entry once exhausted; unscripted titles get
    the default. `crash_at_call` simulates dying mid-run (e.g. between
    passes) to exercise journal resume."""

    def __init__(self, script=None, default="normal", crash_at_call=None):
        self.script = {title: list(outputs) for title, outputs in (script or {}).items()}
        self.default = default
        self.crash_at_call = crash_at_call
        self.completions = 0
        self.draws_by_title = {}

    def complete(self, prompt):
        self.completions += 1
        if self.crash_at_call is not None and self.completions >= self.crash_at_call:
            raise KeyboardInterrupt
        items = json.loads(prompt.split("Items:\n", 1)[1].split("\n\nReply", 1)[0])
        entries = []
        for entry in items:
            title = entry["title"]
            self.draws_by_title[title] = self.draws_by_title.get(title, 0) + 1
            sequence = self.script.get(title)
            if sequence:
                output = sequence.pop(0) if len(sequence) > 1 else sequence[0]
            else:
                output = self.default
            entries.append({"id": entry["id"], "output": output, "reason": "test"})
        return json.dumps(entries)


def unlabeled_records(count=10):
    return [
        {"input": {"title": f"item {index}", "body": "text"}} for index in range(count)
    ]


def passes2_spec(**overrides):
    return make_spec(
        teacher={"backend": "codex-cli", "model": "test", "passes": 2}, **overrides
    )


def test_two_passes_agreeing_rows_are_unanimous_full_weight(tmp_path):
    spec = passes2_spec()
    teacher = ScriptedTeacher()
    meta = build_dataset(spec, unlabeled_records(10), tmp_path, teacher=teacher)
    rows = read_jsonl(tmp_path / "labeled.jsonl")
    assert len(rows) == 10
    assert all(row["weight"] == 1.0 and row["agreement"] == "unanimous" for row in rows)
    assert all(row["origin"] == "real" for row in rows)
    # every item drawn exactly twice, no tiebreak spend
    assert set(teacher.draws_by_title.values()) == {2}
    noise = meta["teacher_noise"]
    assert noise["passes"] == 2
    assert noise["measured_rows"] == 10
    assert noise["self_agreement"]["overall"] == 1.0
    assert noise["unresolved"] == 0
    assert not (tmp_path / "unresolved.jsonl").exists()


def test_flip_gets_tiebreak_draw_and_majority_wins(tmp_path):
    spec = passes2_spec()
    teacher = ScriptedTeacher(script={"item 3": ["urgent", "normal", "urgent"]})
    meta = build_dataset(spec, unlabeled_records(10), tmp_path, teacher=teacher)
    flipped = next(
        row for row in read_jsonl(tmp_path / "labeled.jsonl")
        if row["input"]["title"] == "item 3"
    )
    assert flipped["output"] == "urgent"
    assert flipped["agreement"] == "majority"
    assert flipped["weight"] == round(2 / 3, 4)
    assert teacher.draws_by_title["item 3"] == 3
    assert teacher.draws_by_title["item 0"] == 2  # tiebreak is targeted
    noise = meta["teacher_noise"]
    assert noise["agreement_counts"] == {"unanimous": 9, "majority": 1}
    assert noise["self_agreement"]["overall"] == 0.9


def test_three_way_int_scale_resolves_to_median(tmp_path):
    spec = passes2_spec(output={"type": "int", "range": [0, 4]})
    teacher = ScriptedTeacher(default=2, script={"item 1": [1, 4, 3]})
    build_dataset(spec, unlabeled_records(6), tmp_path, teacher=teacher)
    row = next(
        row for row in read_jsonl(tmp_path / "labeled.jsonl")
        if row["input"]["title"] == "item 1"
    )
    assert row["output"] == 3  # median of the three draws
    assert row["agreement"] == "median"
    assert row["weight"] == round(1 / 3, 4)
    assert not (tmp_path / "unresolved.jsonl").exists()


def test_three_way_enum_split_is_set_aside_not_labeled(tmp_path, capsys):
    spec = passes2_spec(output={"type": "enum", "labels": ["urgent", "normal", "low"]})
    teacher = ScriptedTeacher(script={"item 2": ["urgent", "normal", "low"]})
    meta = build_dataset(spec, unlabeled_records(10), tmp_path, teacher=teacher)
    rows = read_jsonl(tmp_path / "labeled.jsonl")
    assert len(rows) == 9  # the split item holds no decision
    assert all(row["input"]["title"] != "item 2" for row in rows)
    assert meta["teacher_noise"]["unresolved"] == 1
    unresolved = read_jsonl(tmp_path / "unresolved.jsonl")
    assert len(unresolved) == 1
    assert unresolved[0]["input"]["title"] == "item 2"
    assert unresolved[0]["provenance"]["unresolved"] is True
    assert [draw["output"] for draw in unresolved[0]["provenance"]["draws"]] == [
        "urgent", "normal", "low",
    ]
    err = capsys.readouterr().err
    assert "smallbatch label" in err and "--append" in err
    assert "under-determines" in err  # 1/10 exceeds the rubric-rate threshold


def test_user_resolution_rides_import_path_and_clears_unresolved(tmp_path):
    spec = passes2_spec(output={"type": "enum", "labels": ["urgent", "normal", "low"]})
    teacher = ScriptedTeacher(script={"item 2": ["urgent", "normal", "low"]})
    build_dataset(spec, unlabeled_records(10), tmp_path, teacher=teacher)
    resolution = read_jsonl(tmp_path / "unresolved.jsonl")
    for record in resolution:
        record["output"] = "low"
    meta = build_dataset(spec, resolution, tmp_path, append=True)
    resolved = next(
        row for row in read_jsonl(tmp_path / "labeled.jsonl")
        if row["input"]["title"] == "item 2"
    )
    assert resolved["origin"] == "user-resolved"
    assert resolved["output"] == "low"
    assert resolved["weight"] == 1.0
    assert not (tmp_path / "unresolved.jsonl").exists()
    noise = meta["teacher_noise"]
    assert noise["user_resolved"] == 1
    assert noise["unresolved"] == 0
    assert meta["real"] == 10


def test_crash_between_passes_resumes_without_respending_pass_one(tmp_path):
    spec = passes2_spec()
    crashing = ScriptedTeacher(crash_at_call=2)
    with pytest.raises(KeyboardInterrupt):
        build_dataset(spec, unlabeled_records(10), tmp_path, teacher=crashing)
    assert crashing.completions == 2  # pass 1 landed, pass 2 died
    resumed = ScriptedTeacher()
    build_dataset(spec, unlabeled_records(10), tmp_path, teacher=resumed)
    # pass 1 replays from the journal: the resumed teacher only paid pass 2
    assert set(resumed.draws_by_title.values()) == {1}
    rows = read_jsonl(tmp_path / "labeled.jsonl")
    assert len(rows) == 10 and all(row["agreement"] == "unanimous" for row in rows)


def test_passes_1_rows_and_meta_are_unchanged(tmp_path):
    spec = make_spec(teacher={"backend": "codex-cli", "model": "test"})
    teacher = ScriptedTeacher()
    meta = build_dataset(spec, unlabeled_records(10), tmp_path, teacher=teacher)
    rows = read_jsonl(tmp_path / "labeled.jsonl")
    assert all("weight" not in row and "agreement" not in row for row in rows)
    assert set(teacher.draws_by_title.values()) == {1}
    noise = meta["teacher_noise"]
    assert noise["passes"] == 1
    assert noise["measured_rows"] == 0
    assert noise["self_agreement"]["overall"] is None
