"""Doctor preflight checks (the pure, torch-free findings)."""

import json

from smallbatch.doctor import contract_findings, data_findings, items_findings
from smallbatch.spec import FunctionSpec

TEACHER = {"backend": "claude-cli", "model": "sonnet", "examples": 100, "holdout": 0.2}


def make_spec(**over):
    base = dict(
        name="toy",
        description="-",
        input_schema={"title": "str", "body": "str"},
        output={"type": "int", "range": [0, 4]},
        rubric="-",
        teacher=TEACHER,
    )
    base.update(over)
    return FunctionSpec(**base)


def levels(findings):
    return [lvl for lvl, _ in findings]


def test_contract_ok_and_constrained():
    f = contract_findings(make_spec())
    assert "fail" not in levels(f)
    assert any("5 legal completions" in msg for _, msg in f)


def test_contract_warns_on_unconstrainable_cross_product():
    spec = make_spec(output={"a": {"range": [0, 99]}, "b": {"range": [0, 99]}})
    f = contract_findings(spec)
    assert any(lvl == "warn" and "cross product" in msg for lvl, msg in f)


def test_contract_fails_on_huge_int_range():
    spec = make_spec(output={"type": "int", "range": [0, 100000]})
    assert "fail" in levels(contract_findings(spec))


def test_items_flag_missing_fields_and_small_gate():
    spec = make_spec()
    items = [{"title": f"t{i}"} for i in range(10)]  # body missing everywhere
    f = items_findings(spec, items)
    assert any(lvl == "fail" and "'body'" in msg for lvl, msg in f)
    assert any("gate split would be 2" in msg for _, msg in f)


def test_items_budget_estimate():
    f = items_findings(make_spec(), [{"title": "t", "body": "b"}] * 90)
    assert any("labeling call" in msg for _, msg in f)


def test_data_findings_read_new_layout(tmp_path):
    spec = make_spec()
    rows = [{"input": {"title": "t", "body": "b"}, "score": 1, "origin": "real"}]
    for name, n in (("train", 8), ("dev", 1), ("gate", 1)):
        (tmp_path / f"{name}.jsonl").write_text("\n".join([json.dumps(rows[0])] * n))
    (tmp_path / "labeled.jsonl").write_text(json.dumps(rows[0]))
    (tmp_path / "meta.json").write_text(json.dumps({
        "label_histogram": {"0": 0, "1": 10, "2": 3, "3": 2, "4": 1},
        "spec_hash": spec.spec_hash(),
    }))
    f = data_findings(spec, tmp_path)
    assert any("8 train / 1 dev / 1 gate" in msg for _, msg in f)
    assert any(lvl == "warn" and "zero examples: 0" in msg for lvl, msg in f)
    assert not any(lvl == "warn" and "different spec" in msg for lvl, msg in f)
