"""Backend-neutral evidence metrics for constrained decisions."""

from __future__ import annotations

import math
import random
from collections import Counter
from collections.abc import Callable
from typing import Any

from .spec import FieldSpec, FunctionSpec

# Bootstrap parameters for the metrics Wilson cannot cover (means, correlations,
# F1). Fixed seed so a build reproduces its own confidence intervals.
BOOTSTRAP_RESAMPLES = 2000
BOOTSTRAP_SEED = 17


def _percentile(ordered: list[float], q: float) -> float:
    """Linear-interpolated percentile of an already-sorted list, q in [0, 100]."""
    if len(ordered) == 1:
        return ordered[0]
    position = (q / 100) * (len(ordered) - 1)
    low = math.floor(position)
    high = math.ceil(position)
    if low == high:
        return ordered[low]
    fraction = position - low
    return ordered[low] * (1 - fraction) + ordered[high] * fraction


def bootstrap_ci(
    statistic: Callable[[list[int]], float | None],
    n: int,
    seed: int = BOOTSTRAP_SEED,
    resamples: int = BOOTSTRAP_RESAMPLES,
    q: tuple[float, float] = (2.5, 97.5),
) -> list[float] | None:
    """Percentile bootstrap CI for a statistic of `n` paired rows.

    `statistic(indices)` recomputes the scalar over the resampled row indices
    (paired: pass the same indices to both sides of a contrast) and returns
    None when it is undefined for that resample. Returns None when fewer than
    two rows or when too many resamples are undefined to trust the interval.
    Deterministic for a given seed.
    """
    if n is None or n < 2:
        return None
    rng = random.Random(seed)
    values: list[float] = []
    for _ in range(resamples):
        indices = [rng.randrange(n) for _ in range(n)]
        value = statistic(indices)
        if value is not None:
            values.append(value)
    if len(values) < resamples // 2:
        return None
    values.sort()
    return [round(_percentile(values, q[0]), 4), round(_percentile(values, q[1]), 4)]


def wilson_ci(k: int, n: int, z: float = 1.96) -> list[float] | None:
    if not n:
        return None
    p = k / n
    denominator = 1 + z**2 / n
    center = (p + z**2 / (2 * n)) / denominator
    half = z / denominator * math.sqrt(p * (1 - p) / n + z**2 / (4 * n**2))
    return [round(max(0.0, center - half), 4), round(min(1.0, center + half), 4)]


def pearson_r(xs: list[float], ys: list[float]) -> float | None:
    if len(xs) < 2:
        return None
    mean_x, mean_y = sum(xs) / len(xs), sum(ys) / len(ys)
    var_x = sum((value - mean_x) ** 2 for value in xs)
    var_y = sum((value - mean_y) ** 2 for value in ys)
    if not var_x or not var_y:
        return None
    covariance = sum((x - mean_x) * (y - mean_y) for x, y in zip(xs, ys))
    return covariance / math.sqrt(var_x * var_y)


def _ranks(values: list[float]) -> list[float]:
    order = sorted(range(len(values)), key=values.__getitem__)
    ranks = [0.0] * len(values)
    index = 0
    while index < len(order):
        end = index
        while end + 1 < len(order) and values[order[end + 1]] == values[order[index]]:
            end += 1
        rank = (index + end) / 2 + 1
        for position in range(index, end + 1):
            ranks[order[position]] = rank
        index = end + 1
    return ranks


def spearman_rho(xs: list[float], ys: list[float]) -> float | None:
    return pearson_r(_ranks(xs), _ranks(ys)) if len(xs) >= 2 else None


def nearest_rank_p90(values: list[float]) -> float | None:
    if not values:
        return None
    ordered = sorted(values)
    return ordered[math.ceil(0.9 * len(ordered)) - 1]


def field_decision_matches(field: FieldSpec, prediction: Any, reference: Any) -> bool:
    if prediction is None:
        return False
    return abs(prediction - reference) <= 1 if field.type == "int" else prediction == reference


def _int_bootstrap_cis(predictions: list, references: list) -> dict[str, list[float] | None]:
    """Bootstrap CIs for the integer metrics Wilson does not cover (mean-shaped
    and correlation statistics). Resamples rows, dropping invalid predictions
    within each resample exactly as the point estimates do."""
    n = len(references)

    def valid(indices: list[int]) -> list[tuple]:
        return [(predictions[i], references[i]) for i in indices if predictions[i] is not None]

    def mae(indices: list[int]) -> float | None:
        pairs = valid(indices)
        return sum(abs(p - r) for p, r in pairs) / len(pairs) if pairs else None

    def signed(indices: list[int]) -> float | None:
        pairs = valid(indices)
        return sum(p - r for p, r in pairs) / len(pairs) if pairs else None

    def pearson(indices: list[int]) -> float | None:
        pairs = valid(indices)
        return pearson_r([p for p, _ in pairs], [r for _, r in pairs])

    def spearman(indices: list[int]) -> float | None:
        pairs = valid(indices)
        return spearman_rho([p for p, _ in pairs], [r for _, r in pairs])

    return {
        "mae_ci": bootstrap_ci(mae, n),
        "mean_signed_error_ci": bootstrap_ci(signed, n),
        "pearson_r_ci": bootstrap_ci(pearson, n),
        "spearman_rho_ci": bootstrap_ci(spearman, n),
    }


def int_field_metrics(
    field: FieldSpec, predictions: list, references: list, bootstrap: bool = False
) -> dict[str, Any]:
    n = len(references)
    valid = [(p, r) for p, r in zip(predictions, references) if p is not None]
    errors = [abs(p - r) for p, r in valid]
    signed = [p - r for p, r in valid]
    exact_count = sum(p == r for p, r in zip(predictions, references))
    within_count = sum(
        field_decision_matches(field, p, r) for p, r in zip(predictions, references)
    )
    pearson = pearson_r([p for p, _ in valid], [r for _, r in valid])
    spearman = spearman_rho([p for p, _ in valid], [r for _, r in valid])
    # prediction bias and error by rubric level: support against predicted
    # counts shows where a candidate leans; signed error shows which way
    per_level: dict[str, dict[str, Any]] = {}
    for level in field.values():
        at_level = [(p, r) for p, r in valid if r == level]
        level_errors = [abs(p - r) for p, r in at_level]
        per_level[str(level)] = {
            "support": sum(r == level for r in references),
            "predicted": sum(p == level for p, _ in valid),
            "exact": round(sum(e == 0 for e in level_errors) / len(at_level), 4)
            if at_level
            else None,
            "within_one": round(sum(e <= 1 for e in level_errors) / len(at_level), 4)
            if at_level
            else None,
            "mean_signed_error": round(
                sum(p - r for p, r in at_level) / len(at_level), 4
            )
            if at_level
            else None,
        }
    result = {
        "n": n,
        "valid_n": len(valid),
        "invalid_rate": round((n - len(valid)) / n, 4) if n else 0.0,
        "exact": round(exact_count / n, 4) if n else 0.0,
        "exact_ci": wilson_ci(exact_count, n),
        "within_one": round(within_count / n, 4) if n else 0.0,
        "within_one_ci": wilson_ci(within_count, n),
        "mae": round(sum(errors) / len(errors), 4) if errors else None,
        "absolute_error_histogram": {
            str(key): value for key, value in sorted(Counter(errors).items())
        },
        "p90_absolute_error": nearest_rank_p90(errors),
        "max_absolute_error": max(errors) if errors else None,
        "mean_signed_error": round(sum(signed) / len(signed), 4) if signed else None,
        "pearson_r": round(pearson, 4) if pearson is not None else None,
        "spearman_rho": round(spearman, 4) if spearman is not None else None,
        "per_level": per_level,
    }
    if bootstrap:
        result.update(_int_bootstrap_cis(predictions, references))
    return result


def _enum_scalar_scores(labels: list, predictions: list, references: list):
    """(macro_f1, weighted_f1, balanced_accuracy) over every contract class.

    Absent classes count as zero, matching the point-estimate aggregation, so
    this is reusable both there and inside the bootstrap resampling loop."""
    n = len(references)
    if not n:
        return None, None, None
    f1s: list[float] = []
    recalls: list[float] = []
    weighted_numerator = 0.0
    for label in labels:
        true_positive = sum(p == label and r == label for p, r in zip(predictions, references))
        predicted = sum(p == label for p in predictions)
        support = sum(r == label for r in references)
        precision = true_positive / predicted if predicted else 0.0
        recall = true_positive / support if support else 0.0
        f1 = 2 * precision * recall / (precision + recall) if precision + recall else 0.0
        f1s.append(f1)
        recalls.append(recall)
        weighted_numerator += f1 * support
    return sum(f1s) / len(labels), weighted_numerator / n, sum(recalls) / len(labels)


def _enum_bootstrap_cis(
    labels: list, predictions: list, references: list
) -> dict[str, list[float] | None]:
    n = len(references)

    def scores(indices: list[int]):
        return _enum_scalar_scores(
            labels, [predictions[i] for i in indices], [references[i] for i in indices]
        )

    return {
        "macro_f1_ci": bootstrap_ci(lambda indices: scores(indices)[0], n),
        "weighted_f1_ci": bootstrap_ci(lambda indices: scores(indices)[1], n),
        "balanced_accuracy_ci": bootstrap_ci(lambda indices: scores(indices)[2], n),
    }


def enum_field_metrics(
    field: FieldSpec, predictions: list, references: list, bootstrap: bool = False
) -> dict[str, Any]:
    labels = field.values()
    n = len(references)
    valid_n = sum(prediction is not None for prediction in predictions)
    exact_count = sum(p == r for p, r in zip(predictions, references))
    per_class: dict[str, dict[str, Any]] = {}
    for label in labels:
        true_positive = sum(p == label and r == label for p, r in zip(predictions, references))
        predicted = sum(p == label for p in predictions)
        support = sum(r == label for r in references)
        precision = true_positive / predicted if predicted else 0.0
        recall = true_positive / support if support else 0.0
        f1 = 2 * precision * recall / (precision + recall) if precision + recall else 0.0
        per_class[str(label)] = {
            "precision": round(precision, 4),
            "recall": round(recall, 4),
            "f1": round(f1, 4),
            "support": support,
        }
    observed = [str(label) for label in labels if per_class[str(label)]["support"]]
    # macro averages span every contract class: an absent class counts as zero,
    # or the score inflates whenever a hard class misses this eval split
    class_names = [str(label) for label in labels]
    macro_f1 = (
        round(sum(per_class[label]["f1"] for label in class_names) / len(class_names), 4)
        if n
        else None
    )
    weighted_f1 = (
        round(
            sum(per_class[label]["f1"] * per_class[label]["support"] for label in observed)
            / n,
            4,
        )
        if n
        else None
    )
    balanced = (
        round(
            sum(per_class[label]["recall"] for label in class_names) / len(class_names), 4
        )
        if n
        else None
    )
    worst = min(observed, key=lambda label: per_class[label]["recall"]) if observed else None
    confusion = {
        "labels": [str(label) for label in labels],
        "matrix": [
            [sum(r == actual and p == predicted for p, r in zip(predictions, references)) for predicted in labels]
            for actual in labels
        ],
    }
    result = {
        "n": n,
        "valid_n": valid_n,
        "invalid_rate": round((n - valid_n) / n, 4) if n else 0.0,
        "decision_agreement": round(exact_count / n, 4) if n else 0.0,
        "decision_agreement_ci": wilson_ci(exact_count, n),
        "macro_f1": macro_f1,
        "weighted_f1": weighted_f1,
        "balanced_accuracy": balanced,
        "per_class": per_class,
        "worst_class_recall": (
            {"label": worst, "recall": per_class[worst]["recall"]} if worst else None
        ),
        "classes_absent": [str(label) for label in labels if str(label) not in observed],
        "confusion": confusion,
    }
    if bootstrap:
        result.update(_enum_bootstrap_cis(labels, predictions, references))
    return result


def field_metrics(
    field: FieldSpec, predictions: list, references: list, bootstrap: bool = False
) -> dict[str, Any]:
    if field.type == "int":
        return int_field_metrics(field, predictions, references, bootstrap)
    return enum_field_metrics(field, predictions, references, bootstrap)


def compare(
    spec: FunctionSpec, predictions: list, references: list, bootstrap: bool = False
) -> dict[str, Any]:
    """Backend-neutral evidence for one candidate on one split.

    `bootstrap` adds percentile-bootstrap CIs to the metrics Wilson cannot
    cover (MAE, mean signed error, correlations, F1s). It is off by default so
    the per-epoch dev scoring during training pays nothing; the report-time
    evaluation in `api.py` turns it on.
    """
    if len(predictions) != len(references):
        raise ValueError("predictions and references must have equal length")
    if spec.output.is_scalar:
        return field_metrics(spec.output.scalar, predictions, references, bootstrap)

    n = len(references)
    normalized = [prediction if isinstance(prediction, dict) else {} for prediction in predictions]
    fields = {
        name: field_metrics(
            field,
            [prediction.get(name) for prediction in normalized],
            [reference[name] for reference in references],
            bootstrap,
        )
        for name, field in spec.output.fields.items()
    }
    joint = sum(
        all(
            field_decision_matches(field, prediction.get(name), reference[name])
            for name, field in spec.output.fields.items()
        )
        for prediction, reference in zip(normalized, references)
    )
    exact = sum(prediction == reference for prediction, reference in zip(normalized, references))
    invalid = sum(
        any(prediction.get(name) not in field.values() for name, field in spec.output.fields.items())
        for prediction in normalized
    )
    return {
        "n": n,
        "valid_n": n - invalid,
        "invalid_rate": round(invalid / n, 4) if n else 0.0,
        "joint_decision_agreement": round(joint / n, 4) if n else 0.0,
        "joint_decision_agreement_ci": wilson_ci(joint, n),
        "joint_exact": round(exact / n, 4) if n else 0.0,
        "joint_exact_ci": wilson_ci(exact, n),
        "fields": fields,
    }
