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


# the default first: ties resolve to it, and without dev rows it is the head
HEAD_GRID = (
    (1.0, None),
    (0.1, None),
    (10.0, None),
    (1.0, "balanced"),
    (0.1, "balanced"),
    (10.0, "balanced"),
)


def fit_head(
    values: list[Any],
    features,
    labels: list[Any],
    dev_features=None,
    dev_references: list | None = None,
) -> tuple[dict, dict | None]:
    """Fit the cumulative head, choosing boundary regularization on dev rows.

    A fixed C=1.0 leaves accuracy on the table when a scale is imbalanced —
    a rubric's extreme levels are usually rare — so with development rows the
    boundary models try a small C grid with and without balanced class
    weights, scored by dev within-one agreement under the argmax read (the
    decode rule is selected afterwards, on the winning head). Without dev
    rows the default fit stands. Returns (head, tuning record or None).
    """
    from sklearn.linear_model import LogisticRegression

    def build_with(c: float, weight: str | None) -> dict:
        return build(
            values,
            features,
            labels,
            lambda x, y: LogisticRegression(
                max_iter=1000, C=c, class_weight=weight
            ).fit(x, y),
        )

    if dev_features is None or not dev_references:
        return build_with(*HEAD_GRID[0]), None
    trials: list[dict[str, Any]] = []
    best: tuple[float, dict, dict] | None = None
    for c, weight in HEAD_GRID:
        head = build_with(c, weight)
        predictions = predict(head, dev_features)
        value = sum(
            abs(prediction - reference) <= 1
            for prediction, reference in zip(predictions, dev_references)
        ) / len(dev_references)
        trials.append({"C": c, "class_weight": weight, "dev_within_one": round(value, 4)})
        if best is None or value > best[0]:
            best = (value, head, {"C": c, "class_weight": weight})
    return best[1], {"selected": best[2], "metric": "within_one", "trials": trials}


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
    """Class values via the cumulative chain, one row per feature row.

    Heads persisted before decoder selection existed carry no `decoder` key
    and decode as argmax — exactly what they were evaluated with.
    """
    from . import decode

    return decode.decode_levels(
        class_distribution(head, features),
        head["values"],
        head.get("decoder", "argmax"),
    )


def select_decoder(head: dict, features, references: list, setting: str = "auto") -> dict | None:
    """Pick the decode rule for this head and persist it inside the head.

    A pinned setting is recorded as-is. `auto` measures every rule on the
    development rows — never the evaluation split — and keeps the one with the
    best within-one agreement, ties resolving to the stable default (argmax,
    listed first). With no development rows the head keeps argmax. Returns the
    development comparison when one was run, for the candidate record.
    """
    from . import decode

    if setting != "auto":
        head["decoder"] = setting
        return None
    if references is None or not len(references):
        head["decoder"] = "argmax"
        return None
    distribution = class_distribution(head, features)
    comparison: dict[str, dict[str, float]] = {}
    for name in decode.DECODERS:
        predictions = decode.decode_levels(distribution, head["values"], name)
        errors = [abs(p - r) for p, r in zip(predictions, references)]
        n = len(errors)
        comparison[name] = {
            "exact": round(sum(e == 0 for e in errors) / n, 4),
            "within_one": round(sum(e <= 1 for e in errors) / n, 4),
            "mae": round(sum(errors) / n, 4),
        }
    chosen = max(decode.DECODERS, key=lambda name: comparison[name]["within_one"])
    head["decoder"] = chosen
    return {"selected": chosen, "metric": "within_one", "decoders": comparison}


def head_diagnostics(
    head: dict, features, references: list, levels: list | None = None
) -> dict:
    """Aggregate development diagnostics for the cumulative head.

    Everything here is a summary — no inputs, no per-row values — so it can
    sit in public evidence. Violations are rows where an independently fitted
    later boundary scored above an earlier one before repair; the repair rate
    is how often forcing the chain monotonic changed the decoded level. Each
    boundary's mean predicted `P(y > v)` next to the observed rate shows
    calibration in the large.

    `references` live in the head's own value space; `levels` relabels the
    reported boundaries for heads that train on indices of the real scale.
    """
    import numpy as np

    from . import decode

    raw = boundary_probabilities(head, features)
    rows = raw.shape[0]
    values = head["values"]
    decoder = head.get("decoder", "argmax")

    rises = np.diff(raw, axis=1).clip(min=0.0)
    per_row = rises.max(axis=1, initial=0.0)
    violating = per_row > 1e-9

    padded = np.concatenate([np.ones((rows, 1)), raw, np.zeros((rows, 1))], axis=1)
    unrepaired = padded[:, :-1] - padded[:, 1:]
    before = decode.decode_levels(unrepaired, values, decoder)
    after = decode.decode_levels(class_distribution(head, features), values, decoder)
    changed = sum(one != two for one, two in zip(before, after))

    labels = values if levels is None else levels
    boundaries = [
        {
            "boundary": labels[index],
            "mean_predicted": round(float(raw[:, index].mean()), 4),
            "observed_rate": round(
                sum(reference > value for reference in references) / len(references), 4
            ),
        }
        for index, value in enumerate(values[:-1])
    ]
    return {
        "rows": rows,
        "decoder": decoder,
        "monotonicity_violations": {
            "row_rate": round(float(violating.mean()), 4),
            "mean_magnitude": round(float(per_row[violating].mean()), 4)
            if violating.any()
            else 0.0,
            "max_magnitude": round(float(per_row.max(initial=0.0)), 4),
        },
        "repair_changed_prediction_rate": round(changed / rows, 4),
        "boundaries": boundaries,
    }


def applies(spec, field_name: str) -> bool:
    """Ordered levels exist only for integer ranges, not for enum labels."""
    return spec.output.fields[field_name].type == "int"
