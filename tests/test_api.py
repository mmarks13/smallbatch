import json

import pytest

from conftest import imported_records, make_spec
from smallbatch.api import compile as compile_fn
from smallbatch.api import label
from smallbatch.runtime import load_fn


def test_import_label_compile_and_explicit_run(tmp_path):
    spec = make_spec()
    data = tmp_path / "data"
    root = tmp_path / "artifacts"
    labeled = label(spec, imported_records(30), out_dir=data)
    assert labeled.meta["decision_source"] == "imported"
    result = compile_fn(spec, data_dir=data, artifacts_root=root, cpu_threads=1)
    assert result.candidates["tfidf"]["status"] == "completed"
    assert result.candidates["tfidf"]["metrics"]["decision_agreement"] >= 0.5
    assert not (root / spec.name / "active.json").exists()
    function = load_fn(
        spec.name,
        artifacts_root=root,
        version=result.build_id,
        candidate="tfidf",
    )
    assert function({"title": "urgent outage", "body": "server down"}) in {"urgent", "normal"}
    with pytest.raises(FileNotFoundError, match="no active candidate"):
        load_fn(spec.name, artifacts_root=root)


def test_compile_refuses_changed_decision_identity(tmp_path):
    data = tmp_path / "data"
    original = make_spec()
    label(original, imported_records(30), out_dir=data)
    changed = make_spec(prompt="different decision instructions")
    with pytest.raises(ValueError, match="different prompt"):
        compile_fn(changed, data_dir=data, artifacts_root=tmp_path / "artifacts")


def test_compile_resumes_complete_matching_build(tmp_path):
    spec = make_spec()
    data = tmp_path / "data"
    root = tmp_path / "artifacts"
    label(spec, imported_records(30), out_dir=data)
    first = compile_fn(spec, data_dir=data, artifacts_root=root, cpu_threads=1)
    second = compile_fn(spec, data_dir=data, artifacts_root=root, cpu_threads=1)
    assert first.build_id == second.build_id
    assert json.loads((second.version_dir / "build_state.json").read_text())["status"] == "complete"
