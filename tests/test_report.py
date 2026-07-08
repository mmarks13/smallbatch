"""Eval report builder + markdown rendering (torch-free)."""

import json

from smallbatch.evaluate import compute_metrics
from smallbatch.report import build_report, render_markdown, write_report
from smallbatch.spec import FunctionSpec

SPEC = FunctionSpec(
    name="toy",
    description="Score.",
    input_schema={"title": "str"},
    output={"type": "int", "range": [0, 4]},
    rubric="-",
    teacher={"backend": "claude-cli", "model": "sonnet"},
)

GATE_ROWS = [
    {"input": {"title": f"t{i}"}, "score": s, "reason": f"because {i}", "origin": "real"}
    for i, s in enumerate([0, 1, 2, 3, 4, 2])
]
PREDS = [0, 1, 2, 0, 4, 2]  # one severe miss: gold 3 -> pred 0

TRAINING = {
    "precision": "fp32",
    "train_rows": 100,
    "dev_rows": 10,
    "epochs_run": 5,
    "best_epoch": 3,
    "best_dev_agreement": 0.9,
    "stopped_reason": "early_stop(patience=2)",
    "train_loss": 0.42,
    "curve": [{"epoch": e, "train_loss": 1.0 / e, "dev_agreement": 0.5 + e / 10} for e in (1, 2, 3, 4, 5)],
}


def make_report():
    adapter = compute_metrics(SPEC, PREDS, [r["score"] for r in GATE_ROWS])
    adapter["preds"] = PREDS
    gate = {"passed": False, "reasons": ["agreement 83.33% < required 85%"]}
    zeroshot = {"agreement": 0.3}
    return build_report(SPEC, GATE_ROWS, adapter, zeroshot, gate, TRAINING)


def test_build_report_shape():
    r = make_report()
    assert r["headline"]["agreement_ci"] is not None
    assert r["headline"]["zeroshot_agreement"] == 0.3
    assert r["small_gate_warning"]  # 6 items << 50
    assert r["training"]["best_epoch"] == 3
    # per-value: gold 3 got pred 0 -> 0% agreement in that band
    assert r["per_value"]["3"]["agreement"] == 0.0
    assert r["per_value"]["2"]["agreement"] == 1.0
    # confusion: row for gold 3 has its count in the "0" column
    row3 = r["confusion"]["matrix"][r["confusion"]["gold_order"].index("3")]
    assert row3[r["confusion"]["labels"].index("0")] == 1
    assert r["severe"]["count"] == 1  # |0-3| >= 3
    assert r["failures"][0]["teacher"] == 3 and r["failures"][0]["adapter"] == 0


def test_markdown_renders_key_sections():
    md = render_markdown(make_report())
    for needle in ("95% CI", "FAIL", "best epoch 3", "noise-dominated",
                   "Confusion", "Largest disagreements", "because 3"):
        assert needle in md, f"missing {needle!r}"


def test_write_report_files(tmp_path):
    path = write_report(tmp_path, make_report())
    assert path.name == "report.md" and path.exists()
    data = json.loads((tmp_path / "report.json").read_text())
    assert data["function"] == "toy"
