"""Ordinal heads for the sklearn-based candidates.

A multinomial head treats the levels of an integer scale as unrelated symbols:
predicting 4 when the decision was 0 costs exactly what predicting 1 costs. The
ordered decomposition here instead fits one binary model per boundary of the
scale — `P(y > level)` — and reconstructs the class distribution from those
cumulative probabilities, so the head learns the order and errors grow with
distance.

Everything persisted is a standard sklearn estimator inside plain containers:
the chain logic lives in code, so artifacts stay loadable under skops' strict
trust policy and generated packages need no Smallbatch class to unpickle.
"""

from __future__ import annotations

from typing import Any

KIND = "ordinal-cumulative"


def build(values: list[Any], features, labels: list[Any], fit_binary) -> dict:
    """Fit `P(y > v)` for every boundary except the last value.

    `fit_binary(features, targets)` returns a fitted binary estimator. A
    boundary with only one observed side has nothing to learn and is recorded
    as a constant instead, which keeps degenerate scales (a level with no rows)
    trainable rather than raising.
    """
    steps: list[dict[str, Any]] = []
    for value in values[:-1]:
        targets = [1 if label > value else 0 for label in labels]
        observed = set(targets)
        if len(observed) < 2:
            steps.append({"above": float(observed.pop()), "model": None})
            continue
        steps.append({"above": None, "model": fit_binary(features, targets)})
    return {"kind": KIND, "values": list(values), "steps": steps}


def boundary_probabilities(head: dict, features):
    """Raw `P(y > v)` per boundary, before any monotonic repair."""
    import numpy as np

    rows = features.shape[0]
    above = np.empty((rows, len(head["steps"])), dtype=float)
    for index, step in enumerate(head["steps"]):
        if step["model"] is None:
            above[:, index] = step["above"]
            continue
        probabilities = step["model"].predict_proba(features)
        classes = list(step["model"].classes_)
        above[:, index] = probabilities[:, classes.index(1)]
    return above


def class_distribution(head: dict, features):
    """Per-row probability of every level, after monotonic repair."""
    import numpy as np

    above = boundary_probabilities(head, features)

    # P(y > v) cannot rise as v rises; binary models are fit independently, so
    # enforce the monotonicity the scale guarantees before differencing.
    above = np.minimum.accumulate(above, axis=1)

    rows = features.shape[0]
    padded = np.concatenate(
        [np.ones((rows, 1)), above, np.zeros((rows, 1))], axis=1
    )
    return padded[:, :-1] - padded[:, 1:]


def predict(head: dict, features) -> list[Any]:
    """Class values via the cumulative chain, one row per feature row."""
    from . import decode

    return decode.decode_levels(
        class_distribution(head, features), head["values"], "argmax"
    )


def applies(spec, field_name: str) -> bool:
    """Ordered levels exist only for integer ranges, not for enum labels."""
    return spec.output.fields[field_name].type == "int"
