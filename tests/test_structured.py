"""Structured (multi-field) output contracts, end to end on the torch-free
surface: spec parsing, prompts/completions, teacher labeling, metrics/gate,
grammar, and report."""

import json

import pytest

from smallbatch import prompts
from smallbatch.evaluate import compute_metrics, run_gate
from smallbatch.export import gbnf_grammar
from smallbatch.labeling import build_dataset, label_items, row_output
from smallbatch.report import build_report, render_markdown
from smallbatch.spec import FunctionSpec

TEACHER = {"backend": "claude-cli", "model": "sonnet", "examples": 0, "holdout": 0.2}


def make_spec(**over):
    base = dict(
        name="tri",
        description="Classify.",
        input_schema={"title": "str"},
        output={
            "priority": {"labels": ["urgent", "normal", "low"]},
            "reason": {"labels": ["outage", "billing", "question"]},
            "confidence": {"range": [1, 3]},
        },
        rubric="-",
        teacher=TEACHER,
    )
    base.update(over)
    return FunctionSpec(**base)


SPEC = make_spec()


# --- spec ---------------------------------------------------------------


def test_flat_fields_and_inferred_types():
    f = SPEC.output.fields
    assert list(f) == ["priority", "reason", "confidence"]  # declaration order
    assert f["priority"].type == "enum" and f["confidence"].type == "int"
    assert f["confidence"].values() == [1, 2, 3]
    assert not SPEC.output.is_scalar


def test_legacy_scalar_form_maps_to_score_field():
    spec = make_spec(output={"type": "int", "range": [0, 4]})
    assert spec.output.is_scalar
    assert spec.output.type == "int" and spec.output.range == (0, 4)
    assert list(spec.output.fields) == ["score"]


def test_reserved_and_bad_field_names_rejected():
    with pytest.raises(ValueError, match="reserved"):
        make_spec(output={"labels": {"labels": ["a"]}, "x": {"range": [0, 1]}})
    with pytest.raises(ValueError, match="identifier"):
        make_spec(output={"has space": {"labels": ["a"]}})
    with pytest.raises(ValueError):
        make_spec(output={"x": {"labels": ["a"], "range": [0, 1]}})  # both kinds
    with pytest.raises(ValueError):
        make_spec(output={})


def test_rationale_field_name_collision_rejected():
    with pytest.raises(ValueError, match="rationale"):
        make_spec(
            output={"rationale": {"labels": ["a"]}, "x": {"range": [0, 1]}},
            train={"rationale_distillation": True},
        )


# --- prompts ------------------------------------------------------------


def test_student_completion_fixed_order_lines():
    out = {"priority": "urgent", "reason": "outage", "confidence": 2}
    assert prompts.student_completion(SPEC, out) == (
        " priority: urgent\nreason: outage\nconfidence: 2"
    )


def test_allowed_completions_cross_product():
    cs = prompts.allowed_completions(SPEC)
    assert len(cs) == 3 * 3 * 3
    assert " priority: urgent\nreason: outage\nconfidence: 1" in cs


def test_allowed_completions_capped():
    big = make_spec(
        output={"a": {"range": [0, 99]}, "b": {"range": [0, 99]}}
    )
    assert prompts.allowed_completions(big) is None  # 10k > cap -> unconstrained


def test_parse_output_multi():
    text = " priority: urgent\nreason: outage\nconfidence: 2"
    assert prompts.parse_output(SPEC, text) == {
        "priority": "urgent", "reason": "outage", "confidence": 2,
    }
    partial = prompts.parse_output(SPEC, "priority: normal\nconfidence: 9")
    assert partial == {"priority": "normal", "reason": None, "confidence": None}
    assert prompts.parse_output(SPEC, "gibberish") is None


def test_completion_budget_scales_with_fields():
    assert prompts.completion_budget(make_spec(output={"type": "int", "range": [0, 4]})) == 8
    assert prompts.completion_budget(SPEC) > 16


# --- labeling -----------------------------------------------------------


class MultiTeacher:
    def complete(self, prompt: str) -> str:
        start = prompt.index("[")
        depth = 0
        for i, c in enumerate(prompt[start:]):
            depth += c == "["
            depth -= c == "]"
            if depth == 0:
                items = json.loads(prompt[start : start + i + 1])
                break
        return json.dumps(
            [
                {
                    "id": it["id"],
                    "output": {"priority": "normal", "reason": "billing", "confidence": "2"},
                    "reason": "because",
                }
                for it in items
            ]
        )


def test_label_items_multi_validates_and_stores_output_dict():
    rows = label_items(MultiTeacher(), SPEC, [{"title": "t"}], origin="real")
    assert len(rows) == 1
    out = row_output(SPEC, rows[0])
    assert out == {"priority": "normal", "reason": "billing", "confidence": 2}  # coerced int
    assert "score" not in rows[0]


def test_build_dataset_multi_end_to_end(tmp_path):
    items = [{"title": f"t{i}"} for i in range(10)]
    meta = build_dataset(MultiTeacher(), SPEC, items, tmp_path, max_variants=0)
    assert meta["real"] == 10 and meta["gate"] == 2
    row = json.loads((tmp_path / "train.jsonl").read_text().splitlines()[0])
    assert row["output"]["priority"] == "normal"


# --- metrics + gate -----------------------------------------------------


GOLDS = [
    {"priority": "urgent", "reason": "outage", "confidence": 3},
    {"priority": "normal", "reason": "billing", "confidence": 2},
    {"priority": "low", "reason": "question", "confidence": 1},
    {"priority": "low", "reason": "question", "confidence": 1},
]
PREDS = [
    {"priority": "urgent", "reason": "outage", "confidence": 2},   # int within 1 -> joint ok
    {"priority": "normal", "reason": "outage", "confidence": 2},   # reason wrong
    {"priority": "low", "reason": "question", "confidence": 1},    # perfect
    None,                                                          # invalid
]


def test_compute_metrics_per_field_and_joint():
    m = compute_metrics(SPEC, PREDS, GOLDS)
    assert m["agreement"] == 0.5  # joint: rows 0 and 2
    assert m["fields"]["priority"]["agreement"] == 0.75
    assert m["fields"]["reason"]["agreement"] == 0.5
    assert m["fields"]["confidence"]["agreement"] == 0.75  # ±1 semantics
    assert m["invalid_rate"] == 0.25
    assert m["agreement_ci"] is not None


def test_run_gate_per_field_with_overrides():
    m = compute_metrics(SPEC, PREDS, GOLDS)
    spec = make_spec(gate={"agreement": 0.7, "fields": {"reason": 0.4}, "must_beat_zeroshot": False, "must_beat_constant": False})
    gate = run_gate(spec, m, None)
    assert gate["passed"]  # priority 0.75, confidence 0.75 >= 0.7; reason 0.5 >= 0.4
    strict = make_spec(gate={"agreement": 0.7, "must_beat_zeroshot": False, "must_beat_constant": False})
    gate = run_gate(strict, m, None)
    assert not gate["passed"] and any("reason" in r for r in gate["reasons"])


# --- grammar ------------------------------------------------------------


def test_gbnf_multi_field_sequence():
    g = gbnf_grammar(SPEC)
    assert 'root ::= " "? "priority: " f0 "\\n" "reason: " f1 "\\n" "confidence: " f2' in g
    assert 'f0 ::= "urgent" | "normal" | "low"' in g
    assert 'f2 ::= "1" | "2" | "3"' in g


def test_gbnf_multi_field_rationale():
    spec = make_spec(train={"rationale_distillation": True})
    g = gbnf_grammar(spec)
    assert '"rationale: " rationale "\\n"' in g and "rationale ::= [^\\n]+" in g


# --- report -------------------------------------------------------------


def test_report_multi_field_sections():
    rows = [{"input": {"title": f"t{i}"}, "output": g, "reason": "", "origin": "real"}
            for i, g in enumerate(GOLDS)]
    adapter = compute_metrics(SPEC, PREDS, GOLDS)
    adapter["preds"] = PREDS
    gate = {"passed": False, "reasons": ["reason: agreement 50.00% < required 85%"]}
    report = build_report(SPEC, rows, adapter, None, gate, {"precision": "fp32"})
    assert set(report["fields"]) == {"priority", "reason", "confidence"}
    assert report["fields"]["reason"]["metrics"]["agreement"] == 0.5
    md = render_markdown(report)
    assert "Field `reason`" in md and "Field `confidence`" in md
    # worst failure listed first: the all-invalid row misses all 3 fields
    assert report["failures"][0]["adapter"] is None
