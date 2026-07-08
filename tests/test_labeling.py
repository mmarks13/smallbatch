import json

from smallbatch.labeling import build_dataset, plan_variant_bands, split_holdout
from smallbatch.spec import FunctionSpec

SPEC = FunctionSpec(
    name="toy",
    description="Score.",
    input_schema={"title": "str"},
    output={"type": "int", "range": [0, 4]},
    rubric="-",
    teacher={
        "backend": "claude-cli",
        "model": "sonnet",
        "examples": 40,
        "holdout": 0.2,
        "batch_size": 50,
    },
)


def rows(scores, origin="real"):
    return [
        {"input": {"title": f"t{i}"}, "score": s, "reason": "", "origin": origin}
        for i, s in enumerate(scores)
    ]


def test_plan_targets_sparse_bands():
    real = rows([2] * 20 + [3] * 10)
    plan = plan_variant_bands(SPEC, real, target_total=40)
    assert sum(plan.values()) in range(9, 12)  # ~10 needed, rounding tolerated
    assert 2 not in plan  # already over uniform target
    assert plan.get(0) and plan.get(4)  # empty bands get coverage


def test_holdout_real_only_and_stratified():
    data = rows([0] * 10 + [4] * 10) + rows([2] * 30, origin="variant")
    train, holdout = split_holdout(data, frac=0.2)
    assert all(r["origin"] == "real" for r in holdout)
    assert len(holdout) == 4 and {r["score"] for r in holdout} == {0, 4}
    assert len(train) + len(holdout) == 50


class FakeTeacher:
    """Labels everything score=1; generates items named v<i>."""

    def __init__(self):
        self.calls = 0

    def complete(self, prompt: str) -> str:
        self.calls += 1
        if "data labeler" in prompt:
            items = json.loads(prompt[prompt.index("[") :][: self._arr_len(prompt)])
            return json.dumps([{"id": it["id"], "score": 1, "reason": "r"} for it in items])
        return json.dumps([{"title": f"v{i}"} for i in range(5)])

    @staticmethod
    def _arr_len(prompt):
        text = prompt[prompt.index("[") :]
        depth = 0
        for i, c in enumerate(text):
            depth += c == "["
            depth -= c == "]"
            if depth == 0:
                return i + 1
        raise ValueError


def test_build_dataset_end_to_end(tmp_path):
    teacher = FakeTeacher()
    items = [{"title": f"real{i}"} for i in range(10)]
    meta = build_dataset(teacher, SPEC, items, tmp_path)
    assert meta["real"] == 10
    assert meta["variants"] > 0
    for split in ("train", "dev", "gate"):
        assert (tmp_path / f"{split}.jsonl").exists()
    labeled = (tmp_path / "labeled.jsonl").read_text().splitlines()
    assert len(labeled) == meta["real"] + meta["variants"]
    first = json.loads(labeled[0])
    assert first["teacher_model"] and first["origin"] == "real"
    assert first["id"] and first["split"] in ("train", "dev", "gate")
    # variants are train-only and carry the ids of the reals that seeded them
    rows = [json.loads(l) for l in labeled]
    real_ids = {r["id"] for r in rows if r["origin"] == "real"}
    train_real_ids = {r["id"] for r in rows if r["origin"] == "real" and r["split"] == "train"}
    for v in (r for r in rows if r["origin"] == "variant"):
        assert v["split"] == "train"
        assert v["source_ids"] and set(v["source_ids"]) <= train_real_ids, (
            "variant seeded from a non-train real"
        )
    assert real_ids  # sanity


def test_append_keeps_gate_sticky_and_dedupes(tmp_path):
    teacher = FakeTeacher()
    items = [{"title": f"real{i}"} for i in range(10)]
    build_dataset(teacher, SPEC, items, tmp_path)
    gate_before = {
        json.loads(l)["id"] for l in (tmp_path / "gate.jsonl").read_text().splitlines()
    }

    more = items[:5] + [{"title": f"new{i}"} for i in range(20)]  # 5 dupes + 20 new
    meta = build_dataset(teacher, SPEC, more, tmp_path, append=True, max_variants=0)
    assert meta["real"] == 30  # dupes skipped, not relabeled
    gate_after = {
        json.loads(l)["id"] for l in (tmp_path / "gate.jsonl").read_text().splitlines()
    }
    assert gate_before <= gate_after  # sticky: nothing ever leaves the gate
    assert len(gate_after) == 6  # 0.2 of 30 reals


def test_legacy_dataset_migrates_holdout_to_gate(tmp_path):
    # simulate a pre-v0.2 layout: labeled/train/holdout, no ids or splits
    real = rows([0, 1, 2, 3, 4] * 4)
    holdout = real[:4]
    for path, rs in (("labeled.jsonl", real), ("holdout.jsonl", holdout)):
        (tmp_path / path).write_text("\n".join(json.dumps(r) for r in rs))

    meta = build_dataset(
        FakeTeacher(), SPEC, [{"title": "brand-new"}], tmp_path,
        append=True, max_variants=0,
    )
    gate = [json.loads(l) for l in (tmp_path / "gate.jsonl").read_text().splitlines()]
    old_titles = {r["input"]["title"] for r in holdout}
    assert old_titles <= {r["input"]["title"] for r in gate}
    assert meta["real"] == 21
