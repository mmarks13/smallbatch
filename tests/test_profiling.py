import json
import subprocess

import pytest

from conftest import make_spec
from smallbatch.profiling import (
    _load_zeroshot,
    _percentile,
    _run_profile,
    _runtime_dependencies,
    _wait_for_profile,
)


def test_profile_percentiles_are_nearest_rank():
    assert _percentile([5, 1, 3, 2, 4], 0.5) == 3
    assert _percentile([5, 1, 3, 2, 4], 0.95) == 5


def test_runtime_dependencies_are_candidate_specific():
    tfidf = _runtime_dependencies("tfidf")
    assert "scikit-learn" in tfidf and "torch" not in tfidf
    assert "torch" in _runtime_dependencies("lora")


def test_zero_shot_profile_preserves_invalid_output_for_metrics(tmp_path, monkeypatch):
    spec = make_spec(output={"type": "int", "range": [0, 4]})

    class Model:
        @staticmethod
        def eval():
            return None

    monkeypatch.setattr("smallbatch.spec.load_spec", lambda path: spec)
    monkeypatch.setattr(
        "smallbatch.training.load_base_model", lambda model, precision: (object(), Model())
    )
    monkeypatch.setattr(
        "smallbatch.evaluate.generate_batch",
        lambda *args, **kwargs: (["not an integer"], 1),
    )

    function = _load_zeroshot(tmp_path, "example/model")

    assert function({"title": "outage", "body": "server down"}) is None


def test_profile_wait_emits_bounded_row_progress(tmp_path, monkeypatch, capsys):
    progress = tmp_path / "progress.json"

    class Process:
        returncode = 0

        def __init__(self):
            self.calls = 0

        def communicate(self, timeout):
            assert timeout == 0
            self.calls += 1
            if self.calls == 1:
                progress.write_text(json.dumps({"completed": 7, "total": 20}))
                raise subprocess.TimeoutExpired("profile", timeout)
            return "", ""

    monkeypatch.setattr("smallbatch.profiling.PROFILE_HEARTBEAT_SECONDS", 0)

    assert _wait_for_profile(Process(), progress, "student") == ("", "")
    emitted = capsys.readouterr().err
    assert "CPU evaluation progress student" in emitted
    assert "rows=7/20" in emitted


def test_profile_runner_preserves_failure_detail_and_cleans_temporary_files(
    tmp_path, monkeypatch
):
    class Process:
        returncode = 1

        @staticmethod
        def communicate(timeout):
            return "", "worker traceback"

    monkeypatch.setattr("smallbatch.profiling.subprocess.Popen", lambda *args, **kwargs: Process())

    with pytest.raises(RuntimeError, match="CPU profile failed: worker traceback"):
        _run_profile(
            tmp_path,
            "student",
            {"items": []},
            threads=1,
            error_prefix="CPU profile failed",
        )

    assert not list(tmp_path.glob(".profile-student*"))
