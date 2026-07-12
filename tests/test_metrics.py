"""Hand-computed fixtures for the shared metric layer (metrics.py)."""

import json
import math

from smallbatch import metrics as m
from smallbatch.spec import FieldSpec, FunctionSpec

INT_FIELD = FieldSpec(range=(0, 10))
ENUM_FIELD = FieldSpec(labels=["urgent", "normal", "low"])


def spec_for(output):
    return FunctionSpec(
        name="toy",
        description="-",
        input_schema={"t": "str"},
        output=output,
        rubric="-",
        teacher={"backend": "claude-cli", "model": "sonnet"},
    )


# ---------------------------------------------------------------- integers

def test_int_metrics_hand_computed():
    #        pred  ref  |err|  signed
    # r0:      5    5     0      0
    # r1:      7    5     2      2
    # r2:      2    5     3     -3     (severe at delta=3)
    # r3:   None    5     -      -     (invalid: fails agreement, exact, severe)
    # r4:      6    5     1      1
    preds = [5, 7, 2, None, 6]
    refs = [5, 5, 5, 5, 5]
    out = m.int_field_metrics(INT_FIELD, preds, refs, severe_delta=3)
    assert out["n"] == 5 and out["valid_n"] == 4
    assert out["invalid_rate"] == 0.2
    assert out["agreement"] == 0.4  # r0 (0) and r4 (1) within ±1
    assert out["exact"] == 0.2  # r0 only
    assert out["mae"] == round((0 + 2 + 3 + 1) / 4, 4)  # invalid excluded
    assert out["mean_signed_error"] == 0.0  # (0+2-3+1)/4
    assert out["max_absolute_error"] == 3
    # nearest-rank p90 of [0,1,2,3]: ceil(0.9*4)=4th smallest = 3
    assert out["p90_absolute_error"] == 3
    assert out["severe"] == {"threshold": 3, "count": 2, "rate": 0.4}
    # refs constant -> correlations undefined -> None (JSON null, never NaN)
    assert out["pearson_r"] is None and out["spearman_rho"] is None
    assert "macro_f1" not in out  # enum metrics never rendered for ints


def test_int_correlations_and_percentile():
    preds = [1, 2, 3, 4, 10]
    refs = [1, 2, 3, 4, 5]
    out = m.int_field_metrics(INT_FIELD, preds, refs)
    assert out["spearman_rho"] == 1.0  # monotone despite the outlier
    assert 0 < out["pearson_r"] < 1
    # |errs| = [0,0,0,0,5]; ceil(.9*5)=5th smallest = 5
    assert out["p90_absolute_error"] == 5


def test_nearest_rank_p90_definition():
    assert m.nearest_rank_p90([]) is None
    assert m.nearest_rank_p90([7]) == 7
    assert m.nearest_rank_p90(list(range(1, 11))) == 9  # ceil(9.0)=9th of 1..10
    assert m.nearest_rank_p90(list(range(1, 12))) == 10  # ceil(9.9)=10th of 1..11


def test_metrics_json_serializable_no_nan():
    out = m.int_field_metrics(INT_FIELD, [None, None], [5, 5])
    text = json.dumps(out)  # would raise on NaN with allow_nan=False
    assert "NaN" not in text
    assert out["mae"] is None and out["pearson_r"] is None
    parsed = json.loads(json.dumps(out))
    assert not any(isinstance(v, float) and math.isnan(v) for v in parsed.values() if v)


# ------------------------------------------------------------------- enums

def test_enum_metrics_hand_computed():
    #  refs:   urgent urgent urgent normal normal low
    #  preds:  urgent urgent normal normal urgent None
    preds = ["urgent", "urgent", "normal", "normal", "urgent", None]
    refs = ["urgent", "urgent", "urgent", "normal", "normal", "low"]
    out = m.enum_field_metrics(ENUM_FIELD, preds, refs)
    assert out["n"] == 6 and out["valid_n"] == 5
    assert out["agreement"] == out["exact"] == 0.5
    # urgent: tp=2, pred=3, support=3 -> P=2/3, R=2/3, F1=2/3
    u = out["per_class"]["urgent"]
    assert (u["precision"], u["recall"], u["support"]) == (round(2 / 3, 4), round(2 / 3, 4), 3)
    # normal: tp=1, pred=2, support=2 -> P=1/2, R=1/2, F1=1/2
    assert out["per_class"]["normal"]["f1"] == 0.5
    # low: support=1, never predicted -> P=0 (zero-division policy), R=0, F1=0
    assert out["per_class"]["low"] == {"precision": 0.0, "recall": 0.0, "f1": 0.0, "support": 1}
    assert out["macro_f1"] == round((2 / 3 + 0.5 + 0.0) / 3, 4)
    assert out["weighted_f1"] == round((2 / 3 * 3 + 0.5 * 2 + 0.0 * 1) / 6, 4)
    assert out["balanced_accuracy"] == round((2 / 3 + 0.5 + 0.0) / 3, 4)
    assert out["worst_class_recall"] == {"label": "low", "recall": 0.0}
    assert out["classes_absent"] == []
    assert "mae" not in out and "pearson_r" not in out  # numeric metrics omitted


def test_enum_absent_contract_class_excluded_from_averages():
    # 'low' never appears in refs or preds: recorded, not averaged
    preds = ["urgent", "normal"]
    refs = ["urgent", "normal"]
    out = m.enum_field_metrics(ENUM_FIELD, preds, refs)
    assert out["classes_absent"] == ["low"]
    assert out["per_class"]["low"]["support"] == 0
    assert out["macro_f1"] == 1.0  # not halved by the unexercised label
    assert out["balanced_accuracy"] == 1.0


def test_enum_all_invalid():
    out = m.enum_field_metrics(ENUM_FIELD, [None, None], ["low", "low"])
    assert out["valid_n"] == 0 and out["agreement"] == 0.0
    assert out["macro_f1"] == 0.0  # 'low' observed in refs, F1 0
    assert out["worst_class_recall"] == {"label": "low", "recall": 0.0}


# -------------------------------------------------------------- structured

def test_structured_compare_joint_and_worst_field():
    spec = spec_for({"priority": {"labels": ["hi", "lo"]}, "score": {"range": [0, 5]}})
    spec.gate.fields = {"priority": 0.9}
    preds = [
        {"priority": "hi", "score": 3},   # both agree
        {"priority": "lo", "score": 0},   # priority wrong, score off by 3
        {"priority": "hi", "score": None},  # score invalid
    ]
    refs = [
        {"priority": "hi", "score": 3},
        {"priority": "hi", "score": 3},
        {"priority": "hi", "score": 4},
    ]
    out = m.compare(spec, preds, refs)
    assert out["n"] == 3
    assert out["agreement"] == round(1 / 3, 4)  # only row 0 agrees jointly
    assert out["invalid_rate"] == round(1 / 3, 4)  # row 2 has a None field
    assert set(out["fields"]) == {"priority", "score"}
    # priority: 2/3 agreement vs 0.9 threshold -> margin -0.2337;
    # score: 1/3 vs default 0.85 -> margin -0.5167 -> worst
    assert out["worst_field"]["name"] == "score"
    assert out["fields"]["score"]["severe"]["count"] == 2  # |0-3|=3 and invalid


def test_scalar_compare_uses_gate_severe_delta():
    spec = spec_for({"type": "int", "range": [0, 10]})
    spec.gate.severe_delta = 2
    out = m.compare(spec, [4], [6])
    assert out["severe"] == {"threshold": 2, "count": 1, "rate": 1.0}


# ------------------------------------------------------- shared primitives

def test_spearman_ties_and_short_series():
    import pytest

    assert m.spearman_rho([1], [1]) is None
    assert m.spearman_rho([1, 2, 2, 3], [10, 20, 20, 30]) == pytest.approx(1.0)
    assert m.spearman_rho([1, 2, 3], [3, 2, 1]) == pytest.approx(-1.0)


def test_wilson_ci_bounds():
    lo, hi = m.wilson_ci(18, 22)
    assert 0.60 < lo < 0.63 and 0.92 < hi < 0.94  # the audit's 18/22 example
    assert m.wilson_ci(0, 0) is None
