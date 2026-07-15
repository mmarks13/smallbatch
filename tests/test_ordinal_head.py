from __future__ import annotations

import numpy as np
import pytest
from sklearn.linear_model import LogisticRegression

from conftest import make_spec
from smallbatch import ordinal


def fit_binary(features, targets):
    return LogisticRegression(max_iter=1000).fit(features, targets)


def test_applies_only_to_integer_scales():
    assert ordinal.applies(make_spec(output={"type": "int", "range": [0, 4]}), "score")
    assert not ordinal.applies(make_spec(), "score")  # enum labels have no order


def test_chain_recovers_the_ordered_levels():
    """One feature that increases with the level: the ordered head must map it
    back to the right level, which is the ordering a multinomial head ignores."""
    values = [0, 1, 2, 3, 4]
    features = np.array([[float(level)] for level in values for _ in range(12)])
    labels = [level for level in values for _ in range(12)]
    head = ordinal.build(values, features, labels, fit_binary)

    predicted = ordinal.predict(head, np.array([[0.0], [2.0], [4.0]]))
    assert predicted == [0, 2, 4]


def test_a_level_with_no_rows_is_a_constant_step_not_a_crash():
    """CFPB produced a scale where one level had almost no rows; a boundary with
    only one observed side has nothing to learn and must not fail training."""
    values = [0, 1, 2]
    features = np.array([[0.0]] * 10 + [[1.0]] * 10)
    labels = [1] * 10 + [2] * 10  # level 0 never occurs
    head = ordinal.build(values, features, labels, fit_binary)

    assert head["steps"][0]["model"] is None
    assert head["steps"][0]["above"] == 1.0
    assert ordinal.predict(head, np.array([[0.0], [1.0]])) == [1, 2]


def test_predictions_stay_monotonic_when_binary_models_disagree():
    """The per-boundary models are fit independently, so P(y>1) can exceed
    P(y>0) on some row; differencing that directly would yield a negative
    probability, so the chain is forced monotonic first."""
    head = {
        "kind": ordinal.KIND,
        "values": [0, 1, 2],
        "steps": [{"above": 0.30, "model": None}, {"above": 0.80, "model": None}],
    }
    assert ordinal.predict(head, np.zeros((1, 1))) == [0]


def test_tfidf_trains_an_ordered_head_and_round_trips(tmp_path):
    from smallbatch.candidates import predict_tfidf, train_tfidf

    spec = make_spec(output={"type": "int", "range": [0, 2]})
    rows = []
    for level, text in ((0, "routine question"), (1, "delayed response"), (2, "money withheld")):
        for index in range(8):
            rows.append(
                {
                    "id": f"{level}-{index}",
                    "input": {"title": f"case {index}", "body": text},
                    "output": level,
                    "origin": "real",
                }
            )
    record = train_tfidf(spec, rows, tmp_path)
    assert record["objective"] == {"score": "ordinal"}

    predictions = predict_tfidf(
        tmp_path, spec, [{"title": "case", "body": "money withheld"}]
    )
    assert predictions == [2]


def test_enum_output_keeps_the_multinomial_head(tmp_path):
    from smallbatch.candidates import train_tfidf

    spec = make_spec()  # urgent / normal labels
    rows = [
        {
            "id": str(index),
            "input": {"title": f"t{index}", "body": "outage" if index % 2 else "question"},
            "output": "urgent" if index % 2 else "normal",
            "origin": "real",
        }
        for index in range(10)
    ]
    record = train_tfidf(spec, rows, tmp_path)
    assert record["objective"] == {"score": "multinomial"}


def test_ordered_head_persists_as_stock_sklearn_only(tmp_path):
    """Artifacts must load under skops' strict policy and carry no Smallbatch
    class, or generated packages could not unpickle them without smallbatch."""
    import skops.io as sio

    from smallbatch.candidates import train_tfidf

    spec = make_spec(output={"type": "int", "range": [0, 1]})
    rows = [
        {
            "id": str(index),
            "input": {"title": f"t{index}", "body": "withheld" if index % 2 else "asked"},
            "output": index % 2,
            "origin": "real",
        }
        for index in range(10)
    ]
    train_tfidf(spec, rows, tmp_path)
    path = tmp_path / "model.skops"
    assert sio.get_untrusted_types(file=path) == []
    loaded = sio.load(path, trusted=[])
    assert loaded["score"]["head"]["kind"] == ordinal.KIND


def test_predict_matches_the_standalone_template_chain():
    """The generated package reimplements the chain; the two must agree."""
    template = (
        __import__("pathlib")
        .Path("src/smallbatch/standalone_templates/common.py.tmpl")
        .read_text()
    )
    namespace: dict = {"json": __import__("json")}
    body = template.split("def ordinal_predict", 1)[1]
    exec("def ordinal_predict" + body.split("\ndef metadata", 1)[0], namespace)

    head = {
        "kind": ordinal.KIND,
        "values": [0, 1, 2, 3],
        "steps": [
            {"above": 0.9, "model": None},
            {"above": 0.7, "model": None},
            {"above": 0.1, "model": None},
        ],
    }
    features = np.zeros((2, 1))
    assert namespace["ordinal_predict"](head, features) == ordinal.predict(head, features)


@pytest.mark.parametrize("levels", [2, 5, 11])
def test_class_probabilities_are_a_distribution(levels):
    values = list(range(levels))
    head = {
        "kind": ordinal.KIND,
        "values": values,
        "steps": [
            {"above": 1.0 - (index + 1) / levels, "model": None}
            for index in range(levels - 1)
        ],
    }
    predicted = ordinal.predict(head, np.zeros((3, 1)))
    assert all(value in values for value in predicted)
