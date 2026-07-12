"""Gold labels: reserved annotation, gate-only routing, leakage guarantees,
three-way report metrics. Gold judges; it is never trained on and never
reaches the teacher."""

import json

import pytest

from smallbatch.labeling import build_dataset
from smallbatch.spec import FunctionSpec

SENTINEL = 3  # a value we can grep prompts for is not enough for ints; see
# test_gold_never_reaches_teacher_prompt which uses a capturing teacher

SPEC = FunctionSpec(
    name="toy",
    description="Score.",
    input_schema={"title": "str"},
    output={"type": "int", "range": [0, 4]},
    rubric="-",
    teacher={
        "backend": "claude-cli",
        "model": "sonnet",
        "examples": 10,
        "holdout": 0.2,
        "batch_size": 50,
    },
)


class FakeTeacher:
    """Labels everything score=1; records every prompt it ever sees."""

    def __init__(self):
        self.prompts: list[str] = []

    def complete(self, prompt: str) -> str:
        self.prompts.append(prompt)
        if "data labeler" in prompt:
            n = prompt.count('"title"')
            return json.dumps([{"id": i, "score": 1, "reason": "r"} for i in range(n)])
        return "[]"


def items(n, gold_for=()):
    out = []
    for i in range(n):
        it = {"title": f"t{i}"}
        if i in gold_for:
            it["gold"] = 2
        out.append(it)
    return out


def read_split(tmp_path, split):
    return [
        json.loads(line)
        for line in (tmp_path / f"{split}.jsonl").read_text().splitlines()
    ]


def test_gold_rows_routed_to_gate_never_train_or_dev(tmp_path):
    build_dataset(FakeTeacher(), SPEC, items(10, gold_for={0, 1, 2}), tmp_path)
    gate_ids = {r["input"]["title"] for r in read_split(tmp_path, "gate")}
    assert {"t0", "t1", "t2"} <= gate_ids
    for split in ("train", "dev"):
        assert not any(r.get("gold") is not None for r in read_split(tmp_path, split))


def test_gate_grows_to_hold_excess_gold(tmp_path):
    # planned gate is 2 (0.2 of 10) but 4 items carry gold
    meta = build_dataset(FakeTeacher(), SPEC, items(10, gold_for={0, 1, 2, 3}), tmp_path)
    assert meta["gate"] >= 4
    assert meta["gold"] == 4


def test_invalid_gold_fails_before_any_teacher_call(tmp_path):
    teacher = FakeTeacher()
    bad = items(5)
    bad[0]["gold"] = 99  # outside the [0, 4] contract
    with pytest.raises(ValueError, match="gold"):
        build_dataset(teacher, SPEC, bad, tmp_path)
    assert teacher.prompts == []  # zero paid calls


def test_all_gold_starves_train_and_fails(tmp_path):
    with pytest.raises(ValueError, match="no train rows"):
        build_dataset(FakeTeacher(), SPEC, items(5, gold_for={0, 1, 2, 3, 4}), tmp_path)


def test_gold_never_reaches_teacher_prompt_or_row_input(tmp_path):
    teacher = FakeTeacher()
    data = items(10, gold_for={0})
    data[0]["title"] = "UNIQUE-SENTINEL-TITLE"
    build_dataset(teacher, SPEC, data, tmp_path)
    # the sentinel title must appear (inputs do go to the teacher)...
    assert any("UNIQUE-SENTINEL-TITLE" in p for p in teacher.prompts)
    # ...but no prompt may carry the gold annotation
    assert not any('"gold"' in p for p in teacher.prompts)
    for r in read_split(tmp_path, "gate"):
        assert "gold" not in r["input"]


def test_retroactive_gold_moves_row_and_evicts_descendants(tmp_path):
    teacher = FakeTeacher()
    build_dataset(teacher, SPEC, items(10), tmp_path)
    # forge a synthetic descendant of t0 (as if augmentation derived from it)
    rows = [json.loads(line) for line in (tmp_path / "labeled.jsonl").read_text().splitlines()]
    t0 = next(r for r in rows if r["input"]["title"] == "t0")
    t0["split"] = "train"  # ensure it's a train row for the move
    rows.append({
        "id": "synthetic-1", "input": {"title": "t0 paraphrased"}, "score": 1,
        "reason": "", "origin": "variant", "split": "train", "source_ids": [t0["id"]],
    })
    (tmp_path / "labeled.jsonl").write_text("\n".join(json.dumps(r) for r in rows))

    build_dataset(teacher, SPEC, items(10, gold_for={0}), tmp_path, append=True)
    after = [json.loads(line) for line in (tmp_path / "labeled.jsonl").read_text().splitlines()]
    t0_after = next(r for r in after if r["input"]["title"] == "t0")
    assert t0_after["split"] == "gate" and t0_after["gold"] == 2
    assert not any(r["id"] == "synthetic-1" for r in after)  # descendant evicted


def test_input_field_named_gold_rejected():
    with pytest.raises(ValueError, match="reserved"):
        FunctionSpec(
            name="toy",
            description="-",
            input_schema={"gold": "str"},
            output={"type": "int", "range": [0, 4]},
            rubric="-",
            teacher={"backend": "claude-cli", "model": "sonnet"},
        )


def test_report_three_way_gold_section():
    from smallbatch.report import build_report

    gate_rows = [
        # teacher said 1 everywhere; gold disagrees on two of three gold rows
        {"input": {"title": "a"}, "score": 1, "origin": "real", "gold": 1},
        {"input": {"title": "b"}, "score": 1, "origin": "real", "gold": 4},
        {"input": {"title": "c"}, "score": 1, "origin": "real", "gold": 4},
        {"input": {"title": "d"}, "score": 1, "origin": "real"},  # no gold
    ]
    adapter = {
        "n": 4, "agreement": 1.0, "agreement_ci": [0.5, 1.0], "exact": 1.0,
        "invalid_rate": 0.0, "preds": [1, 1, 1, 1],
    }
    report = build_report(SPEC, gate_rows, adapter, None, {"passed": True, "reasons": []}, {})
    g = report["gold"]
    assert g["n"] == 3
    assert g["student_vs_teacher"]["agreement"] == 1.0  # perfect imitation...
    assert g["student_vs_gold"]["agreement"] == round(1 / 3, 4)  # ...of a wrong teacher
    assert g["teacher_vs_gold"]["agreement"] == round(1 / 3, 4)
    assert any("NOT independent quality validation" in w for w in report["warnings"])
    md = __import__("smallbatch.report", fromlist=["render_markdown"]).render_markdown(report)
    assert "Gold labels (n=3)" in md and "teacher vs gold" in md
