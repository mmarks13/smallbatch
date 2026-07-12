"""All-fail acceptance: decision text snapshots + candidate-scoped state."""

import pytest

from smallbatch import artifacts
from smallbatch.decision import all_completed_failed, build_decision_text
from smallbatch.spec import FunctionSpec


def spec_for(output):
    return FunctionSpec(
        name="toy", description="-", input_schema={"title": "str"},
        output=output, rubric="-",
        teacher={"backend": "claude-cli", "model": "sonnet"},
    )


def int_metrics(agreement, mae=0.9, invalid=0.0):
    return {
        "n": 50, "valid_n": 50, "agreement": agreement,
        "agreement_ci": [agreement - 0.1, agreement + 0.05],
        "exact": agreement - 0.15, "mae": mae, "p90_absolute_error": 2,
        "max_absolute_error": 4, "mean_signed_error": 0.1, "pearson_r": 0.8,
        "spearman_rho": 0.79, "invalid_rate": invalid,
        "severe": {"threshold": 3, "count": 2, "rate": 0.04},
        "constant_baseline": {"value": 5, "agreement": 0.36},
    }


def manifest_all_fail(kind="int"):
    metrics = int_metrics
    return {
        "manifest_schema_version": 2,
        "function": "toy",
        "candidates": {
            "tfidf": {
                "backend": "tfidf", "status": "completed", "artifact_path": "tfidf",
                "artifact_size_bytes": 400_000,
                "metrics": metrics(0.82),
                "gate": {"passed": False,
                         "reasons": ["agreement 82.00% < required 85%"]},
            },
            "lora": {
                "backend": "lora", "status": "completed", "artifact_path": "adapter",
                "artifact_size_bytes": 18_000_000, "base_model": "some/base",
                "metrics": metrics(0.78, mae=1.1),
                "gate": {"passed": False,
                         "reasons": ["agreement 78.00% < required 85%"]},
            },
        },
        "selection": {"winner": "tfidf",
                      "reason": "highest gate agreement among completed candidates (none passed)"},
        "deployment": None,
        "metrics": {"adapter": metrics(0.78, mae=1.1),
                    "zeroshot": metrics(0.64, mae=1.4, invalid=0.02)},
        "gate": {"passed": False, "reasons": ["agreement 82.00% < required 85%"]},
    }


def test_all_completed_failed_predicate():
    m = manifest_all_fail()
    assert all_completed_failed(m)
    m["candidates"]["tfidf"]["gate"]["passed"] = True
    assert not all_completed_failed(m)  # a passing candidate: no offer
    m["candidates"]["tfidf"]["status"] = "error"
    m["candidates"]["lora"]["status"] = "error"
    assert not all_completed_failed(m)  # all-error is an operational failure


def test_decision_text_integer_columns_never_omitted():
    spec = spec_for({"type": "int", "range": [0, 10]})
    text = build_decision_text(spec, manifest_all_fail(), report=None)
    for header in ("agree ±1", "MAE", "p90 err", "max err", "pearson", "invalid"):
        assert header in text  # decision metrics must not be dropped
    assert "tfidf *" in text  # winner marked
    assert "FAIL -3pt" in text and "FAIL -7pt" in text  # miss margins
    assert "zero-shot" in text and "baseline" in text
    assert "oracle" in text  # constant named as oracle, not a trained model
    assert "0.4 MB" in text and "18.0 MB" in text


def test_decision_text_errored_candidate_row_and_displacement():
    spec = spec_for({"type": "int", "range": [0, 10]})
    m = manifest_all_fail()
    m["candidates"]["lora"] = {"backend": "lora", "status": "error",
                               "error": "OutOfMemoryError: CUDA OOM after training"}
    text = build_decision_text(spec, m, report=None, displaced="2026-07-01")
    assert "ERROR — OutOfMemoryError: CUDA OOM after training" in text
    assert "CURRENTLY DEPLOYED: 2026-07-01" in text
    assert "declining keeps 2026-07-01 active" in text


def test_decision_text_gold_conflict_exposed():
    spec = spec_for({"type": "int", "range": [0, 10]})
    report = {
        "gold": {
            "n": 25,
            "teacher_vs_gold": int_metrics(0.92),
            "student_vs_gold": int_metrics(0.70),
            "student_vs_teacher": int_metrics(0.82),
        },
        "candidates": {
            "tfidf": {"gold_agreement": 0.70},
            "lora": {"gold_agreement": 0.88},
        },
    }
    text = build_decision_text(spec, manifest_all_fail(), report)
    assert "Gold subset (n=25)" in text
    assert "gold prefers 'lora'" in text  # ranking conflict, not hidden


def test_decision_text_constant_not_beaten_called_out():
    spec = spec_for({"type": "int", "range": [0, 10]})
    m = manifest_all_fail()
    m["candidates"]["tfidf"]["metrics"]["constant_baseline"] = {
        "value": 5, "agreement": 0.90,
    }
    text = build_decision_text(spec, m, report=None)
    assert "does NOT beat the constant" in text


def test_accept_candidate_is_scoped_and_persistent(tmp_path):
    v = artifacts.new_version_dir(tmp_path / "artifacts", "toy")
    artifacts.write_manifest(v, manifest_all_fail())
    # before acceptance: nothing usable, latest() skips it
    assert artifacts.latest(tmp_path / "artifacts", "toy") is None
    m = artifacts.accept_candidate(v, "tfidf", via="interactive_compile")
    assert m["deployment"]["accepted_candidate"] == "tfidf"
    assert m["gate"]["passed"] is False  # acceptance never rewrites the gate
    # persistent + scoped
    reloaded = artifacts.read_manifest(v)
    assert artifacts.candidate_is_usable(reloaded, "tfidf")
    assert not artifacts.candidate_is_usable(reloaded, "lora")
    assert artifacts.latest(tmp_path / "artifacts", "toy") == v
    with pytest.raises(ValueError, match="lora"):
        artifacts.resolve_version(tmp_path / "artifacts", "toy", None, False,
                                  candidate="lora")


def test_enum_decision_columns():
    spec = spec_for({"type": "enum", "labels": ["urgent", "normal", "low"]})
    m = manifest_all_fail()
    enum_metrics = {
        "n": 50, "valid_n": 49, "agreement": 0.8, "agreement_ci": [0.66, 0.89],
        "exact": 0.8, "macro_f1": 0.71, "weighted_f1": 0.78,
        "balanced_accuracy": 0.7, "invalid_rate": 0.02,
        "worst_class_recall": {"label": "low", "recall": 0.4},
        "per_class": {}, "classes_absent": [],
        "constant_baseline": {"value": "normal", "agreement": 0.5},
    }
    for rec in m["candidates"].values():
        rec["metrics"] = dict(enum_metrics)
    m["metrics"]["zeroshot"] = dict(enum_metrics)
    text = build_decision_text(spec, m, report=None)
    for header in ("accuracy", "macro F1", "bal acc", "worst recall", "invalid"):
        assert header in text
    assert "40% (low)" in text  # worst-class recall with its label
    assert "MAE" not in text  # numeric columns never rendered for enums
