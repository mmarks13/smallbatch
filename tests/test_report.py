"""Eval report builder + markdown rendering (torch-free)."""

import json

from smallbatch.evaluate import compute_metrics
from smallbatch.report import (
    build_report,
    render_markdown,
    shortcut_audit,
    write_report,
)
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


# --- shortcut audit -------------------------------------------------------


AUDIT_SPEC = FunctionSpec(
    name="toy",
    description="Score.",
    input_schema={"title": "str", "signals": "str"},
    output={"type": "int", "range": [0, 10]},
    rubric="-",
    teacher={"backend": "claude-cli", "model": "sonnet"},
)


def _audit_rows(n=24):
    """Half the rows carry `upvotes: <k>`; teacher labels ignore upvotes
    (flat 5s), student predictions track them hard."""
    rows, golds, preds = [], [], []
    for i in range(n):
        has_signal = i % 2 == 0
        signals = f"upvotes: {i * 3}" if has_signal else ""
        rows.append({
            "input": {"title": f"story {i} " + "x" * (i * 5), "signals": signals},
            "score": 5,
            "origin": "real",
        })
        golds.append(5)
        preds.append(min(10, i // 3) if has_signal else 5)
    return rows, preds, golds


def test_audit_presence_slices_and_numeric_token_feature():
    rows, preds, golds = _audit_rows()
    # slight teacher variance so spearman is defined on the teacher side
    golds = [5 if i % 4 else 4 for i in range(len(golds))]
    for r, g in zip(rows, golds):
        r["score"] = g
    audit = shortcut_audit(AUDIT_SPEC, rows, preds, golds)
    names = [s["slice"] for s in audit["slices"]]
    assert "signals: present" in names and "signals: empty" in names
    empty = next(s for s in audit["slices"] if s["slice"] == "signals: empty")
    assert empty["agreement"] == 1.0  # student is perfect where the shortcut is absent
    feats = {s["feature"]: s for s in audit["surface"]}
    assert "signals:upvotes" in feats
    assert feats["signals:upvotes"]["student_rho"] > 0.8


def test_audit_skips_surface_when_teacher_labels_constant():
    rows, preds, golds = _audit_rows()  # golds all 5 -> teacher rho undefined
    audit = shortcut_audit(AUDIT_SPEC, rows, preds, golds)
    assert audit["surface"] == [] and audit["warnings"] == []


def test_audit_flags_student_only_correlation():
    rows, preds, golds = _audit_rows()
    # give the teacher labels slight variance so rho is defined but ~0
    golds = [5 if i % 4 else 4 for i in range(len(golds))]
    for r, g in zip(rows, golds):
        r["score"] = g
    audit = shortcut_audit(AUDIT_SPEC, rows, preds, golds)
    feats = {s["feature"]: s for s in audit["surface"]}
    assert feats["signals:upvotes"]["flag"]
    assert any("signals:upvotes" in w for w in audit["warnings"])


def test_audit_length_terciles():
    rows, preds, golds = _audit_rows()
    audit = shortcut_audit(AUDIT_SPEC, rows, preds, golds)
    names = [s["slice"] for s in audit["slices"]]
    assert "title: longest third" in names


def test_report_carries_audit_and_headline_baselines():
    r = make_report()
    assert "shortcut_audit" in r and "warnings" in r
    assert r["headline"]["mae"] is not None
    assert r["headline"]["constant_baseline"]["agreement"] > 0
    md = render_markdown(r)
    assert "best constant baseline" in md and "MAE" in md


def test_audit_markdown_section():
    rows, preds, golds = _audit_rows()
    golds = [5 if i % 4 else 4 for i in range(len(golds))]
    for r, g in zip(rows, golds):
        r["score"] = g
    adapter = compute_metrics(AUDIT_SPEC, preds, golds)
    adapter["preds"] = preds
    report = build_report(
        AUDIT_SPEC, rows, adapter, None, {"passed": True, "reasons": []}, TRAINING
    )
    md = render_markdown(report)
    assert "## Shortcut audit" in md
    assert "signals:upvotes" in md and "⚠" in md
