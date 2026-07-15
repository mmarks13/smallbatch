import json

from conftest import make_spec
from smallbatch.report import build_report


def test_public_evidence_has_no_inputs_or_reasons():
    spec = make_spec()
    rows = [
        {
            "input": {"title": "PRIVATE-TITLE", "body": "PRIVATE-BODY"},
            "output": "urgent",
            "reason": "PRIVATE-REASON",
        }
    ]
    candidates = {
        "tfidf": {
            "backend": "tfidf",
            "status": "completed",
            "predictions": ["urgent"],
            "metrics": {"decision_agreement": 1.0},
            "profile": {
                "batch_one_latency_ms": {"p50": 1, "p95": 1},
                "peak_rss_bytes": 1,
                "candidate_owned_bytes": 1,
                "required_shared_bytes": 0,
            },
        }
    }
    report, details = build_report(spec, rows, candidates, {})
    public = json.dumps(report)
    assert "PRIVATE" not in public
    assert "PRIVATE-TITLE" in json.dumps(details)


def test_product_contract_explicitly_disclaims_correctness_and_energy():
    spec = make_spec()
    report, _ = build_report(spec, [], {}, {})
    assert report["claims"] == {
        "decision_correctness_validated": False,
        "energy_measured": False,
        "operating_proxies": ["CPU latency", "peak RSS", "artifact bytes"],
    }
