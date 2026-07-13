import json

from conftest import make_spec
from smallbatch.report import build_report, observed_dominance, write_report


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
