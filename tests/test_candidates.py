"""The tfidf candidate: train/predict round-trip, skops trust policy,
selection order, torch-free runtime dispatch, serve dispatch."""

import json
import sys

import pytest

from smallbatch import artifacts
from smallbatch.candidates import (
    TFIDF_DIR,
    _load_pipelines,
    predict_tfidf,
    train_tfidf,
)
from smallbatch.spec import FunctionSpec

SPEC = FunctionSpec(
    name="toy",
    description="Priority.",
    input_schema={"title": "str", "body": "str"},
    output={"type": "enum", "labels": ["urgent", "normal"]},
    rubric="-",
    teacher={"backend": "claude-cli", "model": "sonnet"},
)

URGENT = ["server down", "outage now", "cannot login urgent", "prod broken outage"]
NORMAL = ["question about invoice", "feature request", "docs typo", "minor question"]


def train_rows():
    rows = []
    for i, t in enumerate(URGENT):
        rows.append({"id": f"u{i}", "input": {"title": t, "body": t}, "score": "urgent",
                     "origin": "real", "split": "train"})
    for i, t in enumerate(NORMAL):
        rows.append({"id": f"n{i}", "input": {"title": t, "body": t}, "score": "normal",
                     "origin": "real", "split": "train"})
    return rows


def test_train_predict_roundtrip(tmp_path):
    fmt = train_tfidf(SPEC, train_rows(), tmp_path / TFIDF_DIR)
    assert fmt["format"] == "skops" and fmt["sklearn_version"]
    preds = predict_tfidf(tmp_path / TFIDF_DIR, SPEC,
                          [{"title": "big outage", "body": "server down now"},
                           {"title": "invoice question", "body": "question"}])
    assert preds[0] == "urgent" and preds[1] == "normal"
    assert all(isinstance(p, str) for p in preds)  # JSON-native, not numpy
    json.dumps(preds)


def test_int_range_treated_as_classes(tmp_path):
    spec = SPEC.model_copy(deep=True)
    spec.output = type(spec.output)(fields={"score": {"range": [0, 4]}})
    rows = [
        {"id": f"r{i}", "input": {"title": t, "body": t}, "score": s,
         "origin": "real", "split": "train"}
        for i, (t, s) in enumerate([("bad thing", 0), ("awful", 0),
                                    ("great stuff", 4), ("excellent", 4)])
    ]
    train_tfidf(spec, rows, tmp_path / TFIDF_DIR)
    (pred,) = predict_tfidf(tmp_path / TFIDF_DIR, spec, [{"title": "awful bad", "body": ""}])
    assert pred in (0, 4) and isinstance(pred, int)


def test_single_class_train_is_a_data_error(tmp_path):
    rows = [r for r in train_rows() if r["score"] == "urgent"]
    with pytest.raises(ValueError, match="single observed class"):
        train_tfidf(SPEC, rows, tmp_path / TFIDF_DIR)


def test_skops_unexpected_type_refused(tmp_path):
    import skops.io as sio

    class Sneaky:
        pass

    out = tmp_path / TFIDF_DIR
    out.mkdir(parents=True)
    sio.dump({"score": Sneaky()}, out / "model.skops")
    with pytest.raises(ValueError, match="refusing to load"):
        _load_pipelines(out)


def test_select_winner_order():
    def rec(status="completed", passed=True, agreement=0.9, size=100):
        return {"status": status, "gate": {"passed": passed},
                "metrics": {"agreement": agreement}, "artifact_size_bytes": size}

    # passing candidate beats a higher-scoring failing one
    sel = artifacts.select_winner(
        {"tfidf": rec(passed=True, agreement=0.86, size=1),
         "lora": rec(passed=False, agreement=0.95, size=1000)}, 0.02)
    assert sel["winner"] == "tfidf"
    # best score among passing wins
    sel = artifacts.select_winner(
        {"tfidf": rec(agreement=0.86, size=1), "lora": rec(agreement=0.93, size=1000)}, 0.02)
    assert sel["winner"] == "lora"
    # tie within margin -> smaller artifact
    sel = artifacts.select_winner(
        {"tfidf": rec(agreement=0.91, size=1), "lora": rec(agreement=0.92, size=1000)}, 0.02)
    assert sel["winner"] == "tfidf" and "tie" in sel["reason"]
    # all failed: best completed offered (feeds the acceptance flow)
    sel = artifacts.select_winner(
        {"tfidf": rec(passed=False, agreement=0.7, size=1),
         "lora": rec(passed=False, agreement=0.8, size=1000)}, 0.02)
    assert sel["winner"] == "lora" and "none passed" in sel["reason"]
    # errored candidates never compete
    sel = artifacts.select_winner(
        {"tfidf": rec(passed=False, agreement=0.7),
         "lora": {"status": "error", "error": "OOM"}}, 0.02)
    assert sel["winner"] == "tfidf"
    with pytest.raises(ValueError, match="no completed"):
        artifacts.select_winner({"lora": {"status": "error"}}, 0.02)


def _tfidf_artifact(tmp_path, passed=True):
    """A version dir whose winner is a trained tfidf candidate."""
    v = artifacts.new_version_dir(tmp_path / "artifacts", "toy")
    train_tfidf(SPEC, train_rows(), v / TFIDF_DIR)
    (v / "spec.yaml").write_text(
        "name: toy\ndescription: Priority.\n"
        "input_schema: {title: str, body: str}\n"
        "output: {type: enum, labels: [urgent, normal]}\nrubric: '-'\n"
        "teacher: {backend: claude-cli, model: sonnet}\n"
    )
    artifacts.write_manifest(v, {
        "manifest_schema_version": 2,
        "function": "toy",
        "candidates": {
            "tfidf": {"backend": "tfidf", "status": "completed",
                      "artifact_path": TFIDF_DIR,
                      "gate": {"passed": passed, "reasons": []}},
        },
        "selection": {"winner": "tfidf", "reason": "only candidate"},
        "deployment": None,
        "gate": {"passed": passed, "reasons": []},
    })
    return v


def test_load_fn_tfidf_winner_never_imports_torch(tmp_path):
    from smallbatch.runtime import load_fn

    _tfidf_artifact(tmp_path)
    before = "torch" in sys.modules
    fn = load_fn("toy", artifacts_root=tmp_path / "artifacts")
    assert fn({"title": "outage", "body": "server down"}) == "urgent"
    outs = fn.batch([{"title": "invoice", "body": "question"}])
    assert outs == ["normal"]
    if not before:
        assert "torch" not in sys.modules  # the tfidf path stays torch-free


def test_serve_dispatches_tfidf_via_python_runtime(tmp_path):
    from smallbatch.serve import handle_call_direct

    v = _tfidf_artifact(tmp_path)
    from smallbatch import candidates as cand

    predict = lambda item: cand.predict_tfidf(v / TFIDF_DIR, SPEC, [item])[0]  # noqa: E731
    status, payload = handle_call_direct(SPEC, {"title": "outage", "body": "down"}, predict)
    assert status == 200 and payload["output"] == "urgent"
    status, payload = handle_call_direct(SPEC, {"title": "x"}, predict)
    assert status == 400  # missing input field
