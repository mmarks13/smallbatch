"""Public Python API: label() round-trip with a stub teacher (torch-free)."""

import json

import smallbatch
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
        "examples": 10,
        "holdout": 0.2,
        "batch_size": 50,
    },
)


class FakeTeacher:
    """Labels every item with score 2; returns no variants."""

    def complete(self, prompt: str) -> str:
        if "id" in prompt and "score" in prompt:  # label prompt
            n = prompt.count('"title"') or 10
            return json.dumps([{"id": i, "score": 2, "reason": "r"} for i in range(n)])
        return "[]"  # variant prompt


def test_public_surface():
    assert callable(smallbatch.label)
    assert callable(smallbatch.compile)
    assert callable(smallbatch.load_fn)
    assert callable(smallbatch.load_spec)


def test_label_roundtrip(tmp_path, monkeypatch):
    import smallbatch.api as api

    monkeypatch.setattr("smallbatch.teacher.make_teacher", lambda cfg: FakeTeacher())
    items = [{"title": f"t{i}"} for i in range(10)]
    result = api.label(SPEC, items, out_dir=tmp_path / "data")
    assert (tmp_path / "data" / "train.jsonl").exists()
    assert (tmp_path / "data" / "holdout.jsonl").exists()
    assert result.meta["real"] == 10
    assert result.compressed  # all labels in one bin by construction


def test_compile_requires_labeled_data(tmp_path):
    import pytest

    import smallbatch.api as api

    with pytest.raises((ValueError, FileNotFoundError)):
        api.compile(SPEC, data_dir=tmp_path / "nope", artifacts_root=tmp_path / "a")
