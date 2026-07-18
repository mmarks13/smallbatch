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


@pytest.mark.parametrize("decoder", [None, "argmax", "median", "within_one"])
def test_predict_matches_the_standalone_template_chain(decoder):
    """The generated package reimplements the chain; the two must agree for
    every decode rule and for heads persisted before the decoder key existed."""
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
    if decoder is not None:
        head["decoder"] = decoder
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

    distribution = ordinal.class_distribution(head, np.zeros((3, 1)))
    assert distribution.shape == (3, levels)
    assert np.allclose(distribution.sum(axis=1), 1.0)
    assert (distribution >= 0).all()


def test_class_distribution_matches_the_differenced_chain():
    head = {
        "kind": ordinal.KIND,
        "values": [0, 1, 2],
        "steps": [{"above": 0.9, "model": None}, {"above": 0.4, "model": None}],
    }
    distribution = ordinal.class_distribution(head, np.zeros((1, 1)))
    assert np.allclose(distribution, [[0.1, 0.5, 0.4]])


# class distribution [0.40, 0.05, 0.25, 0.30, 0.00]: the three decode rules
# each read a different level (argmax 0, within_one 1, median 2)
SPLIT_MASS_HEAD = {
    "kind": ordinal.KIND,
    "values": [0, 1, 2, 3, 4],
    "steps": [
        {"above": 0.60, "model": None},
        {"above": 0.55, "model": None},
        {"above": 0.30, "model": None},
        {"above": 0.00, "model": None},
    ],
}


def test_select_decoder_keeps_the_best_within_one_rule_on_dev():
    head = dict(SPLIT_MASS_HEAD)
    comparison = ordinal.select_decoder(head, np.zeros((4, 1)), [2, 2, 2, 2])
    # median reads 2 (exact); within_one reads 1 (still within one); argmax
    # reads 0 (two off) — median wins the tie by coming first
    assert head["decoder"] == "median"
    assert comparison["selected"] == "median"
    assert comparison["metric"] == "within_one"
    assert comparison["decoders"]["median"]["within_one"] == 1.0
    assert comparison["decoders"]["argmax"]["within_one"] == 0.0


def test_select_decoder_ties_resolve_to_the_stable_default():
    head = dict(SPLIT_MASS_HEAD)
    # references at 0: argmax is exact, within_one lands within one — both
    # score 1.0 on the selection metric, and argmax (listed first) wins
    comparison = ordinal.select_decoder(head, np.zeros((4, 1)), [0, 0, 0, 0])
    assert head["decoder"] == "argmax"
    assert comparison["decoders"]["within_one"]["within_one"] == 1.0


def test_select_decoder_pinned_setting_skips_the_dev_comparison():
    head = dict(SPLIT_MASS_HEAD)
    assert ordinal.select_decoder(head, np.zeros((1, 1)), [2], "median") is None
    assert head["decoder"] == "median"


def test_select_decoder_without_dev_rows_keeps_argmax():
    head = dict(SPLIT_MASS_HEAD)
    assert ordinal.select_decoder(head, None, []) is None
    assert head["decoder"] == "argmax"


def test_predict_honors_the_persisted_decoder():
    head = dict(SPLIT_MASS_HEAD)
    assert ordinal.predict(head, np.zeros((1, 1))) == [0]  # no key: argmax
    head["decoder"] = "median"
    assert ordinal.predict(head, np.zeros((1, 1))) == [2]
    head["decoder"] = "within_one"
    assert ordinal.predict(head, np.zeros((1, 1))) == [1]


def test_tfidf_selects_and_persists_a_decoder(tmp_path):
    import skops.io as sio

    from smallbatch.candidates import train_tfidf

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
    record = train_tfidf(spec, rows, tmp_path, dev_rows=rows)
    assert record["decode"]["score"] in ("argmax", "median", "within_one")
    assert record["dev_decode_comparison"]["score"]["selected"] == record["decode"]["score"]

    # the decoder key persists inside the head under skops' strict trust policy
    loaded = sio.load(tmp_path / "model.skops", trusted=[])
    assert loaded["score"]["head"]["decoder"] == record["decode"]["score"]


def test_tfidf_pinned_decoder_needs_no_dev_rows(tmp_path):
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
    record = train_tfidf(spec, rows, tmp_path, decode="median")
    assert record["decode"] == {"score": "median"}
    assert record["dev_decode_comparison"] == {}
    loaded = sio.load(tmp_path / "model.skops", trusted=[])
    assert loaded["score"]["head"]["decoder"] == "median"


def test_boundary_probabilities_expose_the_unrepaired_chain():
    """Diagnostics need the raw crossing, not the repaired one."""
    head = {
        "kind": ordinal.KIND,
        "values": [0, 1, 2],
        "steps": [{"above": 0.30, "model": None}, {"above": 0.80, "model": None}],
    }
    raw = ordinal.boundary_probabilities(head, np.zeros((1, 1)))
    assert np.allclose(raw, [[0.30, 0.80]])  # the violation survives here
    repaired = ordinal.class_distribution(head, np.zeros((1, 1)))
    assert np.allclose(repaired, [[0.70, 0.0, 0.30]])
