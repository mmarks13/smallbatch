from conftest import make_spec
from smallbatch.metrics import (
    _enum_scalar_scores,
    compare,
    nearest_rank_p90,
    pearson_r,
    wilson_ci,
)


def test_bootstrap_cis_are_opt_in_and_cover_non_proportion_metrics():
    spec = make_spec(output={"type": "int", "range": [0, 4]})
    preds = [0, 2, 4, None, 1]
    refs = [0, 1, 2, 3, 4]
    # the hot per-epoch path takes the default and pays nothing
    assert "mae_ci" not in compare(spec, preds, refs)
    enriched = compare(spec, preds, refs, bootstrap=True)
    for key in ("mae_ci", "mean_signed_error_ci", "pearson_r_ci", "spearman_rho_ci"):
        assert key in enriched
    low, high = enriched["mae_ci"]
    assert low <= enriched["mae"] <= high


def test_bootstrap_cis_are_deterministic():
    spec = make_spec(output={"type": "int", "range": [0, 4]})
    preds, refs = [0, 2, 4, 1, 1], [0, 1, 2, 3, 4]
    first = compare(spec, preds, refs, bootstrap=True)
    second = compare(spec, preds, refs, bootstrap=True)
    assert first["mae_ci"] == second["mae_ci"]
    assert first["spearman_rho_ci"] == second["spearman_rho_ci"]


def test_enum_bootstrap_cis_agree_with_their_point_estimates():
    spec = make_spec(output={"type": "enum", "labels": ["a", "b", "c"]})
    preds = ["a", "a", "b", "c", None]
    refs = ["a", "b", "b", "c", "a"]
    metrics = compare(spec, preds, refs, bootstrap=True)
    macro, weighted, balanced = _enum_scalar_scores(["a", "b", "c"], preds, refs)
    assert round(macro, 4) == metrics["macro_f1"]
    assert round(weighted, 4) == metrics["weighted_f1"]
    assert round(balanced, 4) == metrics["balanced_accuracy"]
    for key in ("macro_f1_ci", "weighted_f1_ci", "balanced_accuracy_ci"):
        low, high = metrics[key]
        assert low <= high


def test_integer_metrics_are_descriptive_and_named():
    spec = make_spec(output={"type": "int", "range": [0, 4]})
    metrics = compare(spec, [0, 2, 4, None, 1], [0, 1, 2, 3, 4])
    assert metrics["exact"] == 0.2
    assert metrics["within_one"] == 0.4
    assert metrics["mae"] == 1.5
    assert metrics["absolute_error_histogram"] == {"0": 1, "1": 1, "2": 1, "3": 1}
    assert metrics["p90_absolute_error"] == 3
    assert metrics["max_absolute_error"] == 3
    assert metrics["mean_signed_error"] == 0.0
    assert metrics["invalid_rate"] == 0.2
    assert "agreement" not in metrics


def test_integer_metrics_break_out_each_rubric_level():
    spec = make_spec(output={"type": "int", "range": [0, 2]})
    metrics = compare(spec, [0, 0, 1, 2, None], [0, 1, 1, 1, 2])

    level_0 = metrics["per_level"]["0"]
    assert level_0 == {
        "support": 1,
        "predicted": 2,
        "exact": 1.0,
        "within_one": 1.0,
        "mean_signed_error": 0.0,
    }
    level_1 = metrics["per_level"]["1"]
    assert level_1["support"] == 3
    assert level_1["predicted"] == 1
    assert level_1["exact"] == 0.3333
    assert level_1["within_one"] == 1.0
    assert level_1["mean_signed_error"] == 0.0
    # the invalid prediction leaves level 2 with no valid pairs to score
    level_2 = metrics["per_level"]["2"]
    assert level_2["support"] == 1
    assert level_2["exact"] is None


def test_enum_metrics_cover_all_classes():
    spec = make_spec(output={"type": "enum", "labels": ["a", "b", "c"]})
    metrics = compare(spec, ["a", "a", None], ["a", "b", "b"])
    assert metrics["decision_agreement"] == 0.3333
    assert metrics["per_class"]["b"]["recall"] == 0
    assert metrics["classes_absent"] == ["c"]
    assert metrics["confusion"]["labels"] == ["a", "b", "c"]


def test_structured_reports_joint_and_each_field():
    spec = make_spec(
        output={"priority": {"labels": ["a", "b"]}, "score": {"range": [0, 4]}}
    )
    refs = [{"priority": "a", "score": 1}, {"priority": "b", "score": 3}]
    preds = [{"priority": "a", "score": 2}, {"priority": "a", "score": 0}]
    metrics = compare(spec, preds, refs)
    assert metrics["joint_decision_agreement"] == 0.5
    assert metrics["joint_exact"] == 0
    assert metrics["fields"]["score"]["within_one"] == 0.5
    assert metrics["fields"]["priority"]["decision_agreement"] == 0.5


def test_statistical_helpers():
    assert wilson_ci(5, 10) == [0.2366, 0.7634]
    assert nearest_rank_p90(list(range(1, 11))) == 9
    assert pearson_r([1, 1], [1, 2]) is None


def test_macro_scores_count_contract_classes_absent_from_the_split():
    """A hard class dropping out of a resampled eval split must not inflate
    macro_f1/balanced_accuracy: absent contract classes count as zero."""
    spec = make_spec(output={"type": "enum", "labels": ["a", "b", "c"]})
    metrics = compare(spec, ["a", "a", None], ["a", "b", "b"])
    assert metrics["classes_absent"] == ["c"]
    assert metrics["macro_f1"] == 0.2222  # (0.6667 + 0 + 0) / 3
    assert metrics["balanced_accuracy"] == 0.3333  # (1 + 0 + 0) / 3
