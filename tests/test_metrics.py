from conftest import make_spec
from smallbatch.metrics import compare, nearest_rank_p90, pearson_r, wilson_ci


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
