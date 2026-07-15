import json

from conftest import make_spec
from smallbatch.report import build_report, evidence_summary, observed_dominance, write_report


def candidate(predictions, latency=1, memory=100, size=10):
    return {
        "backend": "tfidf",
        "status": "completed",
        "predictions": predictions,
        "metrics": {"decision_agreement": 1.0},
        "profile": {
            "batch_one_latency_ms": {"p50": latency, "p95": latency},
            "peak_rss_bytes": memory,
            "candidate_owned_bytes": size,
            "required_shared_bytes": 0,
        },
    }


def test_observed_dominance_is_row_level_and_operational():
    spec = make_spec()
    refs = ["urgent", "normal"]
    completed = {
        "a": candidate(refs, 1, 100, 10),
        "b": candidate(["urgent", "urgent"], 2, 200, 20),
    }
    assert observed_dominance(spec, completed, refs) == [{"dominates": "b", "candidate": "a"}]


def test_public_report_redacts_inputs_and_local_details_keep_them(tmp_path):
    spec = make_spec()
    rows = [
        {"input": {"title": "SECRET", "body": "PRIVATE"}, "output": "urgent"},
        {"input": {"title": "other", "body": "routine"}, "output": "normal"},
    ]
    candidates = {"tfidf": candidate(["normal", "normal"])}
    report, details = build_report(spec, rows, candidates, {})
    path = write_report(tmp_path, report, details)
    assert "SECRET" not in path.read_text()
    assert "SECRET" in (tmp_path / "report_details.local.json").read_text()
    assert report["claims"]["decision_correctness_validated"] is False
    assert report["claims"]["energy_measured"] is False
    assert json.loads(path.read_text())["selection_bias_note"]


def test_integer_evidence_summary_includes_quality_and_operating_metrics():
    record = candidate([1, 2])
    record["metrics"] = {
        "exact": 0.4,
        "within_one": 0.8,
        "mae": 0.9,
        "p90_absolute_error": 2,
        "max_absolute_error": 3,
        "mean_signed_error": -0.2,
        "pearson_r": 0.7,
        "spearman_rho": 0.6,
        "invalid_rate": 0.0,
    }
    summary = evidence_summary(record)
    for expected in (
        "exact=0.4000",
        "within_one=0.8000",
        "mae=0.9000",
        "p90_error=2",
        "max_error=3",
        "signed_error=-0.2000",
        "pearson=0.7000",
        "spearman=0.6000",
        "p50_ms=1",
        "peak_rss=0.0MiB",
    ):
        assert expected in summary
