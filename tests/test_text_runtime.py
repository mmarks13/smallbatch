"""Text functions end to end: metrics, splits, packaging, and selection."""

from __future__ import annotations

import importlib.util
import json
import sys
from pathlib import Path

import pytest

import smallbatch
from conftest import imported_records, make_spec
from smallbatch.labeling import assign_splits, build_dataset
from smallbatch.metrics import compare


def text_spec(**overrides):
    defaults = {
        "output": {"type": "text", "max_chars": 100},
        "candidates": {"granite": {"type": "lora"}},
    }
    defaults.update(overrides)
    return make_spec(**defaults)


def mixed_spec():
    return make_spec(
        output={
            "priority": {"range": [0, 4]},
            "explanation": {"type": "text", "max_chars": 100},
        },
        candidates={"granite": {"type": "lora"}},
    )


def text_rows(count=30):
    return [
        {
            "id": str(index),
            "input": {"title": f"t{index}", "body": f"body {index}"},
            "output": f"unique rewrite number {index}",
            "origin": "real",
        }
        for index in range(count)
    ]


# ---------------------------------------------------------------- metrics


def test_text_only_metrics_are_structural_never_semantic():
    spec = text_spec()
    references = ["a rewrite", "another rewrite", "third"]
    predictions = ["some text", None, "  "]
    metrics = compare(spec, predictions, references)
    assert metrics["n"] == 3
    assert metrics["valid_n"] == 1
    assert metrics["invalid_rate"] == pytest.approx(2 / 3, abs=1e-4)
    assert "decision_agreement" not in metrics
    assert "fidelity" in metrics["note"] or "not scored" in metrics["note"]


def test_mixed_metrics_stay_atomic_and_joint_requires_a_complete_record():
    spec = mixed_spec()
    references = [
        {"priority": 3, "explanation": "ref one"},
        {"priority": 1, "explanation": "ref two"},
    ]
    # row 2 was an invalid record: the whole prediction is None — its valid
    # priority must not resurface anywhere
    predictions = [{"priority": 3, "explanation": "worded differently"}, None]
    metrics = compare(spec, predictions, references)
    assert metrics["joint_decision_agreement"] == 0.5
    assert metrics["invalid_rate"] == 0.5
    assert metrics["fields"]["priority"]["exact"] == 0.5
    assert metrics["fields"]["explanation"]["note"]


# ----------------------------------------------------------------- splits


def test_text_only_splitting_is_deterministic_and_never_stratifies_text():
    spec = text_spec()
    rows = text_rows()
    assign_splits(spec, rows)
    counts = {split: sum(r["split"] == split for r in rows) for split in ("train", "dev", "eval")}
    assert counts == {"train": 21, "dev": 3, "eval": 6}
    # deterministic: a fresh identical run assigns identically
    again = text_rows()
    assign_splits(spec, again)
    assert [r["split"] for r in again] == [r["split"] for r in rows]


def test_eval_membership_stays_sticky_when_text_data_is_appended(tmp_path):
    spec = text_spec()
    records = [
        {"input": row["input"], "output": row["output"]} for row in text_rows(30)
    ]
    build_dataset(spec, records, tmp_path)
    eval_before = {
        json.loads(line)["id"]
        for line in (tmp_path / "eval.jsonl").read_text().splitlines()
    }
    more = [
        {"input": {"title": f"new{i}", "body": f"nb {i}"}, "output": f"new rewrite {i}"}
        for i in range(10)
    ]
    build_dataset(spec, records + more, tmp_path, append=True)
    eval_after = {
        json.loads(line)["id"]
        for line in (tmp_path / "eval.jsonl").read_text().splitlines()
    }
    assert eval_before <= eval_after


def test_mixed_splitting_stratifies_on_the_bounded_field_not_the_text():
    from smallbatch.labeling import stratum_value

    spec = mixed_spec()
    row = {
        "input": {"title": "t", "body": "b"},
        "output": {"priority": 2, "explanation": "unique text"},
    }
    assert stratum_value(spec, row) == 2
    assert stratum_value(text_spec(), {"input": {}, "output": "unique"}) is None


def test_append_under_a_different_decision_identity_fails_closed(tmp_path):
    spec = make_spec()
    build_dataset(spec, imported_records(30), tmp_path)
    changed = make_spec(prompt="a different rubric")
    with pytest.raises(ValueError, match="different prompt"):
        build_dataset(changed, imported_records(30), tmp_path, append=True)


# ------------------------------------------------- standalone template


def load_common_template(tmp_path, fields, runtime=None):
    package_dir = tmp_path / "commontpl"
    package_dir.mkdir()
    template = (
        Path(smallbatch.__file__).parent / "standalone_templates" / "common.py.tmpl"
    )
    (package_dir / "common.py").write_text(template.read_text())
    (package_dir / "function.json").write_text(
        json.dumps(
            {
                "name": "fn",
                "input_schema": {"query": "string"},
                "output": {"fields": fields},
                "runtime": runtime or {},
            }
        )
    )
    spec = importlib.util.spec_from_file_location(
        "commontpl_common", package_dir / "common.py"
    )
    module = importlib.util.module_from_spec(spec)
    sys.modules["commontpl_common"] = module
    try:
        spec.loader.exec_module(module)
    finally:
        sys.modules.pop("commontpl_common", None)
    return module


TEXT_FIELDS = {
    "priority": {"range": [0, 4], "labels": None, "max_chars": None},
    "explanation": {"range": None, "labels": None, "max_chars": 20},
}


def test_packaged_validation_fails_atomically_with_categories(tmp_path):
    module = load_common_template(tmp_path, TEXT_FIELDS)
    good = module.parse_strict_json('{"priority": 3, "explanation": "fine"}')
    assert good == {"priority": 3, "explanation": "fine"}

    cases = {
        "```json\n{}\n```": "malformed_json",
        '{"explanation": "x", "priority": 3}': "contract",  # wrong key order
        '{"priority": 3, "explanation": "  "}': "empty_text",
        '{"priority": 3, "explanation": "' + "x" * 21 + '"}': "char_limit",
        '{"priority": 9, "explanation": "ok"}': "contract",
    }
    for text, category in cases.items():
        with pytest.raises(module.InvalidOutputError) as caught:
            module.parse_strict_json(text)
        assert caught.value.category == category, text

    with pytest.raises(module.InvalidOutputError) as caught:
        module.parse_strict_json('"cut off', exhausted=True)
    assert caught.value.category == "budget_exhausted"
    # InvalidOutputError stays a ValueError so existing handlers still work
    assert issubclass(module.InvalidOutputError, ValueError)


def test_packaged_scalar_text_returns_a_string(tmp_path):
    module = load_common_template(
        tmp_path, {"score": {"range": None, "labels": None, "max_chars": 50}}
    )
    assert module.parse_strict_json('"a compact rewrite"') == "a compact rewrite"
    with pytest.raises(module.InvalidOutputError):
        module.parse_strict_json("[1]")


# ------------------------------------------------------------- selection


def test_invalid_package_outputs_block_selection(tmp_path, monkeypatch):
    """Any invalid output on the complete held-out package evaluation blocks
    selection; the candidate stays inspectable but cannot become active."""
    from smallbatch import artifacts, standalone
    from smallbatch.api import compile as compile_fn
    from smallbatch.api import label, select

    spec = make_spec()
    data = tmp_path / "data"
    root = tmp_path / "artifacts"
    label(spec, imported_records(30), out_dir=data)
    compiled = compile_fn(spec, data_dir=data, artifacts_root=root, cpu_threads=1)

    evaluate_wheel = standalone._evaluate_wheel

    def with_invalid(*args, **kwargs):
        result = evaluate_wheel(*args, **kwargs)
        result["predictions"][0] = None  # one atomic failure on held-out rows
        return result

    monkeypatch.setattr(standalone, "_evaluate_wheel", with_invalid)
    with pytest.raises(ValueError, match="selection is blocked"):
        select(
            spec.name,
            "tfidf",
            version=compiled.build_id,
            artifacts_root=root,
            interactive=False,
        )
    assert artifacts.read_active(root, spec.name) is None
    # no automatic retry happened: the single patched evaluation was final
    manifest = artifacts.read_manifest(
        artifacts.resolve_build(root, spec.name, compiled.build_id)
    )
    assert manifest["candidates"]["tfidf"]["status"] == "completed"

    # the package source (written before evaluation) carries the resolved
    # contract and the exact deterministic decoding settings
    function_json = next((root / spec.name / "packages").rglob("function.json"))
    function = json.loads(function_json.read_text())
    assert function["output"]["fields"] == {
        "score": {"range": None, "labels": ["urgent", "normal"], "max_chars": None}
    }
    decoding = function["runtime"]["decoding"]
    assert decoding["strategy"] == "greedy"
    assert decoding["do_sample"] is False
    assert isinstance(decoding["max_new_tokens"], int)
