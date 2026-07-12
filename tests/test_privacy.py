"""Privacy: shipped files carry no user input text or teacher rationales.

Sentinel strings are planted in gate inputs/rationales; every distributable
file is then scanned recursively. report_details.json and
provenance.local.json hold the sensitive detail and must never ship.
"""

import json

from smallbatch.hub import SHIP_PATTERNS, ship_list
from smallbatch.report import build_report, redact_report, render_markdown, write_report
from smallbatch.spec import FunctionSpec

SENTINEL_INPUT = "SSN-987-65-4320-CUSTOMER-ACME"
SENTINEL_REASON = "RATIONALE-MENTIONS-SECRET-PROJECT-X"

SPEC = FunctionSpec(
    name="toy",
    description="Score.",
    input_schema={"title": "str", "tier": "str"},
    output={"type": "int", "range": [0, 4]},
    rubric="-",
    teacher={"backend": "claude-cli", "model": "sonnet"},
)


def gate_rows():
    rows = []
    for i in range(16):
        rows.append({
            "input": {"title": f"{SENTINEL_INPUT} ticket {i}", "tier": "enterprise" if i % 2 else "free"},
            "score": i % 5,
            "reason": SENTINEL_REASON,
            "origin": "real",
        })
    return rows


def full_report():
    rows = gate_rows()
    adapter = {
        "n": len(rows), "agreement": 0.5, "agreement_ci": [0.3, 0.7],
        "exact": 0.4, "invalid_rate": 0.0, "mae": 1.2,
        "preds": [(r["score"] + 2) % 5 for r in rows],  # everything misses
    }
    return build_report(SPEC, rows, adapter, None, {"passed": False, "reasons": ["x"]}, {})


def test_redacted_report_has_no_sentinels():
    report = full_report()
    # the full (local) report DOES carry the diagnostics
    assert SENTINEL_INPUT in json.dumps(report)
    red = redact_report(report)
    text = json.dumps(red) + render_markdown(red)
    assert SENTINEL_INPUT not in text
    assert SENTINEL_REASON not in text
    assert "enterprise" not in text  # slice values anonymized
    # ...while the decision-relevant facts survive
    assert red["failures"] and red["failures"][0]["gate_row"] is not None
    assert red["redacted"] is True


def test_write_report_splits_private_and_shippable(tmp_path):
    write_report(tmp_path, full_report())
    details = (tmp_path / "report_details.json").read_text()
    assert SENTINEL_INPUT in details and SENTINEL_REASON in details
    for shipped in ("report.json", "report.md"):
        text = (tmp_path / shipped).read_text()
        assert SENTINEL_INPUT not in text, shipped
        assert SENTINEL_REASON not in text, shipped


def test_ship_list_is_a_positive_whitelist(tmp_path):
    # a realistic artifact dir with both shippable and local-only files
    files = {
        "manifest.json": "{}",
        "spec.yaml": "name: toy",
        "report.json": "{}",
        "report.md": "# r",
        "report_details.json": SENTINEL_INPUT,
        "provenance.local.json": "/home/michael/secret-project/spec.yaml",
        "adapter/adapter_model.safetensors": "w",
        "tfidf/model.skops": "m",
        "spec_files/abc123-rubric.md": "rubric",
        "export/toy.gbnf": "g",
        "export/merged/model.gguf": "huge",
        "trainer/state.json": "t",
        "journal/events.jsonl": "j",
    }
    for rel, content in files.items():
        p = tmp_path / rel
        p.parent.mkdir(parents=True, exist_ok=True)
        p.write_text(content)
    shipped = ship_list(tmp_path)
    assert "report_details.json" not in shipped
    assert "provenance.local.json" not in shipped
    assert "trainer/state.json" not in shipped
    assert "journal/events.jsonl" not in shipped
    assert "export/merged/model.gguf" not in shipped
    assert {"manifest.json", "spec.yaml", "report.json", "report.md",
            "adapter/adapter_model.safetensors", "tfidf/model.skops",
            "spec_files/abc123-rubric.md", "export/toy.gbnf"} == set(shipped)
    # every shipped text file scanned against the sentinel
    for rel in shipped:
        assert SENTINEL_INPUT not in (tmp_path / rel).read_text()


def test_ship_patterns_never_include_local_only_names():
    for pat in SHIP_PATTERNS:
        assert "details" not in pat and "provenance" not in pat and "journal" not in pat
