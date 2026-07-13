import pytest

from smallbatch import artifacts
from smallbatch.api import select


def test_packaging_failure_does_not_change_active_selection(tmp_path, monkeypatch):
    root = tmp_path / "artifacts"
    previous = {
        "function": "ticket-priority",
        "build": "old-build",
        "candidate": "tfidf",
        "package": "packages/old",
    }
    artifacts.activate(root, "ticket-priority", previous)

    def fail(*args, **kwargs):
        raise RuntimeError("package failed")

    monkeypatch.setattr("smallbatch.standalone.package_selection", fail)
    with pytest.raises(RuntimeError, match="package failed"):
        select("ticket-priority", "setfit", artifacts_root=root)
    assert artifacts.read_active(root, "ticket-priority") == previous


def test_clear_records_none_without_deleting_builds(tmp_path):
    root = tmp_path / "artifacts"
    build = root / "ticket-priority" / "builds" / "kept"
    build.mkdir(parents=True)
    artifacts.activate(
        root,
        "ticket-priority",
        {"function": "ticket-priority", "build": "kept", "candidate": "tfidf"},
    )
    event = artifacts.clear_active(root, "ticket-priority")
    assert event["candidate"] is None
    assert build.exists()
