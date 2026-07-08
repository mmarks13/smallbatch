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
    assert (tmp_path / "train.jsonl").exists() and (tmp_path / "holdout.jsonl").exists()
    labeled = (tmp_path / "labeled.jsonl").read_text().splitlines()
    assert len(labeled) == meta["real"] + meta["variants"]
    first = json.loads(labeled[0])
    assert first["teacher_model"] and first["origin"] == "real"
