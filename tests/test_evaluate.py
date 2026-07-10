from smallbatch.evaluate import (
    compute_metrics,
    constant_baseline,
    pearson_r,
    run_gate,
    wilson_ci,
)
from smallbatch.spec import FunctionSpec

SPEC = FunctionSpec(
    name="toy",
    description="Score.",
    input_schema={"title": "str"},
    output={"type": "int", "range": [0, 10]},
    rubric="-",
    teacher={"backend": "claude-cli", "model": "sonnet"},
)


def test_pearson_perfect_and_inverse():
    import pytest

    assert pearson_r([1, 2, 3], [2, 4, 6]) == pytest.approx(1.0)
    assert pearson_r([1, 2, 3], [6, 4, 2]) == pytest.approx(-1.0)


def test_pearson_degenerate():
    assert pearson_r([1], [2]) is None
    assert pearson_r([3, 3, 3], [1, 2, 3]) is None


def test_metrics_include_pearson_and_skip_invalid():
    m = compute_metrics(SPEC, [5, 4, None, 10], [5, 5, 5, 0])
    assert m["agreement"] == 0.5  # 5==5 and |4-5|<=1; None and 10-vs-0 miss
    assert m["invalid_rate"] == 0.25
    assert m["pearson_r"] is not None and m["pearson_r"] < 0  # 10-vs-0 flips it


def test_enum_metrics_have_no_pearson():
    enum_spec = FunctionSpec(
        name="kind",
        description="-",
        input_schema={"title": "str"},
        output={"type": "enum", "labels": ["a", "b"]},
        rubric="-",
        teacher={"backend": "claude-cli", "model": "sonnet"},
    )
    m = compute_metrics(enum_spec, ["a", "b"], ["a", "a"])
    assert m["agreement"] == 0.5 and "pearson_r" not in m


def test_wilson_ci_bounds_and_width():
    assert wilson_ci(0, 0) is None
    lo, hi = wilson_ci(19, 22)  # sweep-3 scale: 86% on 22 items
    assert 0 <= lo < 0.86 < hi <= 1
    assert hi - lo > 0.2  # n=22 really is that noisy
    lo60, hi60 = wilson_ci(52, 60)
    assert hi60 - lo60 < hi - lo  # bigger gate, tighter interval


def test_metrics_carry_agreement_ci():
    m = compute_metrics(SPEC, [5, 5, 5, 5], [5, 5, 5, 0])
    lo, hi = m["agreement_ci"]
    assert lo < m["agreement"] < hi


def test_mae_excludes_invalid_preds():
    m = compute_metrics(SPEC, [5, 4, None, 10], [5, 5, 5, 0])
    assert m["mae"] == round((0 + 1 + 10) / 3, 4)
    only_invalid = compute_metrics(SPEC, [None, None], [5, 5])
    assert only_invalid["mae"] is None


def test_constant_baseline_picks_strongest_constant():
    # labels massed at 3-5: constant 4 covers all of 3/4/5 under ±1
    golds = [3, 3, 4, 4, 5, 5, 7, 0]
    b = constant_baseline(SPEC.output.scalar, golds)
    assert b["value"] == 4 and b["agreement"] == 0.75
    assert constant_baseline(SPEC.output.scalar, []) is None


def test_constant_baseline_enum_is_majority_class():
    enum_spec = FunctionSpec(
        name="kind",
        description="-",
        input_schema={"title": "str"},
        output={"type": "enum", "labels": ["a", "b"]},
        rubric="-",
        teacher={"backend": "claude-cli", "model": "sonnet"},
    )
    b = constant_baseline(enum_spec.output.scalar, ["a", "a", "a", "b"])
    assert b == {"value": "a", "agreement": 0.75}


def test_gate_must_beat_constant():
    # model at 0.75 agreement ties the constant baseline (0.75) -> FAIL
    golds = [3, 3, 4, 4, 5, 5, 7, 0]
    preds = [3, 3, 4, 4, 5, 5, 0, 7]  # 6/8 within ±1
    m = compute_metrics(SPEC, preds, golds)
    spec = SPEC.model_copy(
        update={"gate": SPEC.gate.model_copy(update={"agreement": 0.7,
                                                     "must_beat_zeroshot": False})}
    )
    gate = run_gate(spec, m, None)
    assert not gate["passed"]
    assert any("constant" in r for r in gate["reasons"])
    # disabling the check restores the old behavior
    off = spec.model_copy(
        update={"gate": spec.gate.model_copy(update={"must_beat_constant": False})}
    )
    assert run_gate(off, m, None)["passed"]
