import json

import yaml

from conftest import make_spec
from smallbatch import artifacts


def completed_build(tmp_path):
    root = tmp_path / "artifacts"
    spec = make_spec()
    build = artifacts.build_dir(root, spec.name, spec.build_hash(), "dataset")
    (build / "spec.yaml").write_text(yaml.safe_dump(spec.model_dump(mode="json"), sort_keys=False))
    (build / "report.json").write_text("{}")
    manifest = {
        "manifest_schema_version": 3,
        "function": spec.name,
        "version": build.name,
        "decision_hash": spec.decision_hash(),
        "build_hash": spec.build_hash(),
        "dataset_hash": "dataset",
        "candidates": {"tfidf": {"status": "completed", "backend": "tfidf"}},
    }
    manifest["artifact_files"] = artifacts.file_hashes(build)
    artifacts.write_manifest(build, manifest)
    state = artifacts.read_build_state(build)
    state["status"] = "complete"
    artifacts.write_build_state(build, state)
    return root, build, manifest


def test_matching_build_resumes_and_versions_are_nested(tmp_path):
    root = tmp_path / "artifacts"
    first = artifacts.build_dir(root, "ticket-priority", "build", "data")
    second = artifacts.build_dir(root, "ticket-priority", "build", "data")
    assert first == second
    assert first.parent.name == "builds"


def test_integrity_detects_tampering(tmp_path):
    _root, build, _manifest = completed_build(tmp_path)
    assert artifacts.artifact_integrity(build) is None
    (build / "report.json").write_text('{"changed": true}')
    assert "changed" in artifacts.artifact_integrity(build)


def test_damaged_complete_build_allocates_new_revision(tmp_path):
    root, build, manifest = completed_build(tmp_path)
    (build / "report.json").write_text('{"changed": true}')
    replacement = artifacts.build_dir(
        root,
        "ticket-priority",
        manifest["build_hash"],
        manifest["dataset_hash"],
    )
    assert replacement != build
    assert replacement.name.endswith("-r2")


def test_selection_is_separate_and_clearable(tmp_path):
    root, build, _manifest = completed_build(tmp_path)
    selection = {
        "function": "ticket-priority",
        "build": build.name,
        "candidate": "tfidf",
        "package": "packages/test",
    }
    artifacts.activate(root, "ticket-priority", selection)
    assert artifacts.read_active(root, "ticket-priority")["candidate"] == "tfidf"
    artifacts.clear_active(root, "ticket-priority")
    assert artifacts.read_active(root, "ticket-priority") is None
    history = (root / "ticket-priority" / "selection-history.jsonl").read_text().splitlines()
    assert len(history) == 2 and json.loads(history[-1])["candidate"] is None


def test_explicit_runtime_requires_completed_candidate(tmp_path):
    root, build, _manifest = completed_build(tmp_path)
    resolved, candidate = artifacts.resolve_runtime(
        root, "ticket-priority", build.name, "tfidf"
    )
    assert resolved == build and candidate == "tfidf"


def test_pre_v3_manifest_rejected(tmp_path):
    path = tmp_path / "old"
    path.mkdir()
    (path / "manifest.json").write_text('{"manifest_schema_version": 2}')
    try:
        artifacts.read_manifest(path)
    except ValueError as exc:
        assert "unsupported pre-v0.2 artifact" in str(exc)
    else:
        raise AssertionError("old manifest was accepted")
