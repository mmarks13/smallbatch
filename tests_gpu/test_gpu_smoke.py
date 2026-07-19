"""gpu-smoke: the PR gate. Three function shapes end to end on a real GPU.

Each test trains a real student, evaluates on CPU, selects, and calls the
packaged function. ~15 minutes total; run with

    SMALLBATCH_GPU_TESTS=1 pytest tests_gpu -m gpu_smoke -q
"""

from __future__ import annotations

import pytest
from gpu_cases import build_spec, compile_and_select, lora_candidate, triage_records

pytestmark = pytest.mark.gpu_smoke


def test_bounded_scalar_end_to_end(tmp_path):
    spec = build_spec(
        "gpu-scale",
        {"type": "int", "range": [0, 4]},
        candidate=lora_candidate(max_epochs=6, patience=2),
    )
    compiled, selected, function = compile_and_select(
        spec, triage_records(text=False), tmp_path
    )
    record = compiled.candidates["student"]
    # the ordinal path: constrained scoring keeps every output valid
    assert record["metrics"]["invalid_rate"] == 0.0
    assert record["metrics"]["within_one"] >= 0.8
    assert selected.evidence["parity"]["exact"] == 1.0
    assert function({"title": "outage", "body": "everything is down"}) in range(5)


def test_mixed_decision_plus_text_end_to_end(tmp_path):
    spec = build_spec(
        "gpu-note",
        {
            "priority": {"range": [0, 4]},
            "explanation": {"type": "text", "max_chars": 80},
        },
        training={"loss_weights": {"priority": 2.0}},
    )
    compiled, selected, function = compile_and_select(
        spec, triage_records(text=True), tmp_path
    )
    record = compiled.candidates["student"]
    training = record["training"]
    scores = {e["epoch"]: e["checkpoint_score"] for e in training["curve"]}
    assert training["best_epoch"] == min(scores, key=scores.get)
    # free-running JSON must be reliably valid at this data volume, or the
    # v0.3 text path has regressed
    assert record["metrics"]["invalid_rate"] <= 0.1
    assert record["metrics"]["joint_decision_agreement"] >= 0.8
    assert record["text_fidelity"]["bits_per_byte"] < 1.0
    assert selected.evidence["parity"]["exact"] == 1.0
    output = function({"title": "checkout down", "body": "no customer can pay"})
    assert set(output) == {"priority", "explanation"}
    assert len(output["explanation"]) <= 80


def test_text_only_end_to_end(tmp_path):
    spec = build_spec("gpu-rewrite", {"type": "text", "max_chars": 80})
    compiled, selected, function = compile_and_select(
        spec, triage_records(text=True, decision=False), tmp_path
    )
    record = compiled.candidates["student"]
    assert record["metrics"]["invalid_rate"] <= 0.1
    assert record["text_fidelity"]["bits_per_byte"] < 1.0
    output = function({"title": "site down", "body": "all requests fail"})
    assert isinstance(output, str) and 0 < len(output) <= 80
