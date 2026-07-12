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
    for split in ("train", "dev", "gate"):
        assert (tmp_path / "data" / f"{split}.jsonl").exists()
    assert result.meta["real"] == 10
    assert result.meta["gate"] == 2  # holdout: 0.2 of 10 reals
    assert result.compressed  # all labels in one bin by construction


def test_compile_requires_labeled_data(tmp_path):
    import pytest

    import smallbatch.api as api

    with pytest.raises((ValueError, FileNotFoundError)):
        api.compile(SPEC, data_dir=tmp_path / "nope", artifacts_root=tmp_path / "a")


def test_compile_fails_closed_on_labeling_mismatch(tmp_path, monkeypatch):
    """Changing the rubric after labeling must refuse to train on old labels."""
    import pytest

    import smallbatch.api as api

    monkeypatch.setattr("smallbatch.teacher.make_teacher", lambda cfg: FakeTeacher())
    api.label(SPEC, [{"title": f"t{i}"} for i in range(10)], out_dir=tmp_path / "data")
    changed = SPEC.model_copy(deep=True)
    changed.rubric = "completely different rubric"
    with pytest.raises(ValueError, match="labeling identity"):
        api.compile(changed, data_dir=tmp_path / "data", artifacts_root=tmp_path / "a")


def test_check_labeling_identity_paths(tmp_path):
    """Direct checks: match → None; override → recorded; legacy meta → None."""
    from smallbatch.api import _check_labeling_identity

    spec = SPEC.model_copy(deep=True)
    lh = spec.labeling_hash()
    assert _check_labeling_identity(spec, tmp_path, {"labeling_hash": lh}, False) is None
    override = _check_labeling_identity(
        spec, tmp_path, {"labeling_hash": "0" * 64}, True
    )
    assert override["spec_labeling_hash"] == lh
    assert override["dataset_labeling_hash"] == "0" * 64
    # legacy dataset without a labeling identity: warn-only, never a hard stop
    assert _check_labeling_identity(spec, tmp_path, {"spec_hash": "x"}, False) is None


def test_build_overrides_do_not_trip_labeling_check(tmp_path, monkeypatch):
    """--base/--precision style changes must not invalidate the dataset."""
    from smallbatch.api import _check_labeling_identity

    monkeypatch.setattr("smallbatch.teacher.make_teacher", lambda cfg: FakeTeacher())
    result = api_label_helper(tmp_path)
    meta = json.loads((result.out_dir / "meta.json").read_text())
    overridden = SPEC.model_copy(deep=True)
    overridden.train.base = "some/other-model"
    overridden.train.precision = "fp32"
    assert _check_labeling_identity(overridden, result.out_dir, meta, False) is None


def api_label_helper(tmp_path):
    import smallbatch.api as api

    return api.label(SPEC, [{"title": f"t{i}"} for i in range(10)], out_dir=tmp_path / "data")
