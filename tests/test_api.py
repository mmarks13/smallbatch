import json

import pytest

from conftest import imported_records, make_spec
from smallbatch.api import compile as compile_fn
from smallbatch.api import label
from smallbatch.runtime import load_fn


def test_import_label_compile_and_explicit_run(tmp_path, capsys):
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
    progress = capsys.readouterr().err
    assert "[smallbatch] label start" in progress
    assert "candidate 1/1 tfidf (tfidf) training start" in progress
    assert "CPU evaluation start rows=6 threads=1" in progress
    assert "agreement=" in progress
    assert "p50_ms=" in progress
    assert "compile complete" in progress


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


def test_compile_retries_only_failed_candidates_in_new_revision(tmp_path, monkeypatch):
    from smallbatch import api, artifacts

    spec = make_spec(
        candidates={
            "first": {"type": "tfidf"},
            "retry": {"type": "tfidf"},
        }
    )
    data = tmp_path / "data"
    root = tmp_path / "artifacts"
    label(spec, imported_records(30), out_dir=data)
    calls = []

    def train_candidate(_spec, candidate_id, config, _train, _dev, candidate_dir):
        calls.append(candidate_id)
        if candidate_id == "retry" and calls.count("retry") == 1:
            raise RuntimeError("temporary incompatibility")
        model_dir = candidate_dir / "model"
        model_dir.mkdir(parents=True, exist_ok=True)
        (model_dir / "model.bin").write_bytes(candidate_id.encode())
        return {
            "candidate": candidate_id,
            "backend": config.type,
            "status": "completed",
            "artifact_path": f"candidates/{candidate_id}/model",
            "train_seconds": 1.0,
            "format": "test",
            "error": None,
        }

    profile = {
        "runtime": "test",
        "batch_one_latency_ms": {"p50": 1.0, "p95": 2.0, "n": 6},
        "peak_rss_bytes": 100,
        "candidate_owned_bytes": 10,
        "required_shared_bytes": 0,
    }
    monkeypatch.setattr(api, "_train_candidate", train_candidate)
    monkeypatch.setattr(
        "smallbatch.profiling.profile_candidate",
        lambda *args, **kwargs: {
            "predictions": ["normal", "urgent", "normal", "urgent", "normal", "urgent"],
            "profile": profile,
        },
    )

    first = compile_fn(spec, data_dir=data, artifacts_root=root, cpu_threads=1)
    original_hashes = dict(first.manifest["artifact_files"])
    assert first.candidates["retry"]["status"] == "error"

    second = compile_fn(spec, data_dir=data, artifacts_root=root, cpu_threads=1)

    assert second.build_id == f"{first.build_id}-r2"
    assert second.manifest["retry_of"] == first.build_id
    assert second.candidates["retry"]["status"] == "completed"
    assert calls == ["first", "retry", "retry"]
    assert artifacts.file_hashes(first.version_dir) == original_hashes


def test_compile_resumes_completed_zero_shot_diagnostic(tmp_path, monkeypatch, capsys):
    from smallbatch import api, report

    spec = make_spec(candidates={"student": {"type": "lora", "model": "example/base"}})
    data = tmp_path / "data"
    root = tmp_path / "artifacts"
    label(spec, imported_records(30), out_dir=data)

    monkeypatch.setattr(
        api,
        "_train_candidate",
        lambda *args, **kwargs: {
            "candidate": "student",
            "backend": "lora",
            "status": "completed",
            "artifact_path": "candidates/student/model",
            "train_seconds": 1.0,
            "base_model": "example/base",
            "inference_precision": "fp32",
            "eval_batch_size": 1,
            "error": None,
        },
    )
    profile = {
        "batch_one_latency_ms": {"p50": 1.0, "p95": 2.0, "n": 6},
        "peak_rss_bytes": 100,
        "candidate_owned_bytes": 10,
        "required_shared_bytes": 0,
    }
    monkeypatch.setattr(
        "smallbatch.profiling.profile_candidate",
        lambda *args, **kwargs: {
            "predictions": ["normal", "urgent", "normal", "urgent", "normal", "urgent"],
            "profile": profile,
        },
    )
    zero_calls = 0

    def zero_profile(*args, **kwargs):
        nonlocal zero_calls
        zero_calls += 1
        return {
            "predictions": ["normal", "urgent", "normal", "urgent", "normal", "urgent"],
            "profile": profile,
        }

    monkeypatch.setattr("smallbatch.profiling.profile_zeroshot", zero_profile)
    original_build_report = report.build_report
    monkeypatch.setattr(report, "build_report", lambda *args, **kwargs: (_ for _ in ()).throw(KeyboardInterrupt()))
    with pytest.raises(KeyboardInterrupt):
        compile_fn(spec, data_dir=data, artifacts_root=root, cpu_threads=1)
    assert zero_calls == 1

    monkeypatch.setattr(report, "build_report", original_build_report)
    compile_fn(spec, data_dir=data, artifacts_root=root, cpu_threads=1)
    assert zero_calls == 1
    progress = capsys.readouterr().err
    assert "diagnostic 1/1" in progress
    assert "resumed complete" in progress


def test_compile_retries_failed_zero_shot_diagnostic_in_new_revision(tmp_path, monkeypatch):
    from smallbatch import api

    spec = make_spec(candidates={"student": {"type": "lora", "model": "example/base"}})
    data = tmp_path / "data"
    root = tmp_path / "artifacts"
    label(spec, imported_records(30), out_dir=data)
    train_calls = 0

    def train_candidate(*args, **kwargs):
        nonlocal train_calls
        train_calls += 1
        return {
            "candidate": "student",
            "backend": "lora",
            "status": "completed",
            "artifact_path": "candidates/student/model",
            "train_seconds": 1.0,
            "base_model": "example/base",
            "inference_precision": "fp32",
            "eval_batch_size": 1,
            "error": None,
        }

    monkeypatch.setattr(api, "_train_candidate", train_candidate)
    profile = {
        "batch_one_latency_ms": {"p50": 1.0, "p95": 2.0, "n": 6},
        "peak_rss_bytes": 100,
        "candidate_owned_bytes": 10,
        "required_shared_bytes": 0,
    }
    predictions = ["normal", "urgent", "normal", "urgent", "normal", "urgent"]
    monkeypatch.setattr(
        "smallbatch.profiling.profile_candidate",
        lambda *args, **kwargs: {"predictions": predictions, "profile": profile},
    )
    zero_calls = 0

    def zero_profile(*args, **kwargs):
        nonlocal zero_calls
        zero_calls += 1
        if zero_calls == 1:
            raise RuntimeError("invalid generated output")
        return {"predictions": predictions, "profile": profile}

    monkeypatch.setattr("smallbatch.profiling.profile_zeroshot", zero_profile)

    first = compile_fn(spec, data_dir=data, artifacts_root=root, cpu_threads=1)
    failed = [
        record
        for name, record in first.report["diagnostics"].items()
        if name.startswith("zero-shot-")
    ]
    assert len(failed) == 1 and failed[0]["status"] == "error"

    second = compile_fn(spec, data_dir=data, artifacts_root=root, cpu_threads=1)

    assert second.build_id == f"{first.build_id}-r2"
    assert second.manifest["retry_of"] == first.build_id
    assert train_calls == 1
    assert zero_calls == 2
    assert all(
        record.get("status") != "error"
        for record in second.report["diagnostics"].values()
    )


def test_lora_record_carries_the_dev_selected_decoder(tmp_path, monkeypatch):
    """runtime.py and the standalone package read record["decode"]; dropping it
    silently serves argmax no matter what the dev comparison selected."""
    import smallbatch.training
    from smallbatch import api
    from smallbatch.spec import LoraCandidateSpec

    spec = make_spec(output={"type": "int", "range": [0, 4]})
    comparison = {"argmax": {"within_one": 0.5}, "median": {"within_one": 0.9}}
    info = {
        "precision": "fp32",
        "objective": "ordinal",
        "decode": "median",
        "dev_decode_comparison": comparison,
        "train_rows": 4,
        "train_loss": 0.1,
        "adapter_dir": str(tmp_path / "model"),
        "curve": [],
        "best_epoch": 1,
        "best_dev_agreement": 0.9,
        "epochs_run": 1,
        "stopped_reason": "max_epochs",
        "dev_rows": 2,
    }
    monkeypatch.setattr(smallbatch.training, "train", lambda *args, **kwargs: info)
    record = api._train_candidate(
        spec,
        "lora",
        LoraCandidateSpec(type="lora"),
        [],
        [],
        tmp_path / "candidates" / "lora",
    )
    assert record["decode"] == "median"
    assert record["training"]["dev_decode_comparison"] == comparison


def test_release_accelerator_memory_empties_cuda_and_survives_no_torch(monkeypatch):
    import sys
    import types

    from smallbatch import api

    calls = []
    fake_torch = types.SimpleNamespace(
        cuda=types.SimpleNamespace(
            is_available=lambda: True, empty_cache=lambda: calls.append("empty")
        )
    )
    monkeypatch.setitem(sys.modules, "torch", fake_torch)
    api._release_accelerator_memory()
    assert calls == ["empty"]

    monkeypatch.delitem(sys.modules, "torch")
    api._release_accelerator_memory()
    assert calls == ["empty"]
