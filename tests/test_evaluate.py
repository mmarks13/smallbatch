from smallbatch.evaluate import compute_metrics, pearson_r
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
