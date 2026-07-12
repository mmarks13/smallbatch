"""One shared, backend-neutral metric layer.

Every comparison in the product — candidate vs teacher, candidate vs gold,
zero-shot, constant baseline, sweep cells, the all-failure decision prompt,
and benchmark outputs — computes its numbers here, so no consumer ever
recomputes (or disagrees about) a metric.

Conventions, applied everywhere:
- `n` is all compared rows; `valid_n` counts rows whose prediction parsed to a
  legal value. Invalid predictions count as failures for `agreement`/`exact`
  but are excluded from numeric-error and correlation calculations (which is
  why `valid_n` is always reported beside them).
- Correlations are JSON `null` (Python None) when undefined (< 2 valid pairs
  or a constant series) — never NaN.
- Metrics that don't apply to a field type are omitted, never rendered as 0.
- Enum per-class metrics cover the complete contract label set with a
  zero-division policy of 0.0; contract labels absent from the reference
  labels are listed in `classes_absent` (support 0) and excluded from
  macro/balanced averages so an unexercised label can't halve a score. A
  `None` prediction is "no prediction": it misses recall but enters no
  precision denominator.
"""

from __future__ import annotations

from typing import Any

from .spec import FieldSpec, FunctionSpec

# int outputs: |pred - reference| >= spec.gate.severe_delta is a severe miss
DEFAULT_SEVERE_DELTA = 3


def wilson_ci(k: int, n: int, z: float = 1.96) -> tuple[float, float] | None:
    """Wilson score interval for a proportion k/n — honest about small n,
    where a point verdict is otherwise statistical theater."""
    if n == 0:
        return None
    p = k / n
    denom = 1 + z**2 / n
    center = (p + z**2 / (2 * n)) / denom
    half = (z / denom) * ((p * (1 - p) / n + z**2 / (4 * n**2)) ** 0.5)
    return (round(max(0.0, center - half), 4), round(min(1.0, center + half), 4))


def pearson_r(xs: list[float], ys: list[float]) -> float | None:
    n = len(xs)
    if n < 2:
        return None
    mx, my = sum(xs) / n, sum(ys) / n
    vx = sum((x - mx) ** 2 for x in xs)
    vy = sum((y - my) ** 2 for y in ys)
    if vx == 0 or vy == 0:
        return None  # constant series: correlation undefined
    cov = sum((x - mx) * (y - my) for x, y in zip(xs, ys))
    return cov / (vx**0.5 * vy**0.5)


def _ranks(xs: list[float]) -> list[float]:
    """Average ranks (ties share their mean rank), 1-based."""
    order = sorted(range(len(xs)), key=lambda i: xs[i])
    ranks = [0.0] * len(xs)
    i = 0
    while i < len(order):
        j = i
        while j + 1 < len(order) and xs[order[j + 1]] == xs[order[i]]:
            j += 1
        mean_rank = (i + j) / 2 + 1
        for k in range(i, j + 1):
            ranks[order[k]] = mean_rank
        i = j + 1
    return ranks


def spearman_rho(xs: list[float], ys: list[float]) -> float | None:
    if len(xs) < 2:
        return None
    return pearson_r(_ranks(xs), _ranks(ys))


def nearest_rank_p90(values: list[float]) -> float | None:
    """Nearest-rank 90th percentile: the ceil(0.9*n)-th smallest value.
    Deterministic by definition — no interpolation-mode ambiguity."""
    if not values:
        return None
    ordered = sorted(values)
    import math

    return ordered[math.ceil(0.9 * len(ordered)) - 1]


def agrees(field: FieldSpec, pred: Any, ref: Any) -> bool:
    """Field-level agreement: ±1 for int fields, exact for enum fields.
    An invalid (None) prediction never agrees."""
    if pred is None:
        return False
    return abs(pred - ref) <= 1 if field.type == "int" else pred == ref


def int_field_metrics(
    field: FieldSpec, preds: list, refs: list, severe_delta: int = DEFAULT_SEVERE_DELTA
) -> dict[str, Any]:
    n = len(refs)
    valid = [(p, r) for p, r in zip(preds, refs) if p is not None]
    valid_n = len(valid)
    agree_k = sum(1 for p, r in zip(preds, refs) if agrees(field, p, r))
    exact_k = sum(1 for p, r in zip(preds, refs) if p == r)
    abs_errs = [abs(p - r) for p, r in valid]
    # a severe miss is a large error OR no legal prediction at all
    severe_k = sum(1 for p, r in zip(preds, refs) if p is None or abs(p - r) >= severe_delta)
    pr = pearson_r([p for p, _ in valid], [r for _, r in valid])
    rho = spearman_rho([p for p, _ in valid], [r for _, r in valid])
    ci = wilson_ci(agree_k, n)
    p90 = nearest_rank_p90(abs_errs)
    return {
        "n": n,
        "valid_n": valid_n,
        "invalid_rate": round((n - valid_n) / n, 4) if n else 0.0,
        "agreement": round(agree_k / n, 4) if n else 0.0,
        "agreement_ci": list(ci) if ci else None,
        "exact": round(exact_k / n, 4) if n else 0.0,
        "mae": round(sum(abs_errs) / valid_n, 4) if valid_n else None,
        "p90_absolute_error": p90,
        "max_absolute_error": max(abs_errs) if abs_errs else None,
        "mean_signed_error": (
            round(sum(p - r for p, r in valid) / valid_n, 4) if valid_n else None
        ),
        "pearson_r": round(pr, 4) if pr is not None else None,
        "spearman_rho": round(rho, 4) if rho is not None else None,
        "severe": {
            "threshold": severe_delta,
            "count": severe_k,
            "rate": round(severe_k / n, 4) if n else 0.0,
        },
    }


def enum_field_metrics(field: FieldSpec, preds: list, refs: list) -> dict[str, Any]:
    n = len(refs)
    labels = field.values()
    valid_n = sum(1 for p in preds if p is not None)
    agree_k = sum(1 for p, r in zip(preds, refs) if agrees(field, p, r))
    per_class: dict[str, dict[str, Any]] = {}
    for label in labels:
        tp = sum(1 for p, r in zip(preds, refs) if p == label and r == label)
        pred_k = sum(1 for p in preds if p == label)
        support = sum(1 for r in refs if r == label)
        precision = tp / pred_k if pred_k else 0.0
        recall = tp / support if support else 0.0
        f1 = 2 * precision * recall / (precision + recall) if precision + recall else 0.0
        per_class[str(label)] = {
            "precision": round(precision, 4),
            "recall": round(recall, 4),
            "f1": round(f1, 4),
            "support": support,
        }
    observed = {
        str(label)
        for label in labels
        if per_class[str(label)]["support"] or any(p == label for p in preds)
    }
    absent = [str(label) for label in labels if str(label) not in observed]
    scored = [c for c in observed if per_class[c]["support"]]
    macro_f1 = (
        round(sum(per_class[c]["f1"] for c in observed) / len(observed), 4)
        if observed
        else None
    )
    weighted_f1 = (
        round(sum(per_class[c]["f1"] * per_class[c]["support"] for c in scored) / n, 4)
        if n and scored
        else None
    )
    balanced_accuracy = (
        round(sum(per_class[c]["recall"] for c in scored) / len(scored), 4)
        if scored
        else None
    )
    worst = min(scored, key=lambda c: per_class[c]["recall"]) if scored else None
    ci = wilson_ci(agree_k, n)
    return {
        "n": n,
        "valid_n": valid_n,
        "invalid_rate": round((n - valid_n) / n, 4) if n else 0.0,
        "agreement": round(agree_k / n, 4) if n else 0.0,
        "agreement_ci": list(ci) if ci else None,
        "exact": round(agree_k / n, 4) if n else 0.0,  # enum: agreement is exact
        "macro_f1": macro_f1,  # point estimate; no interval in v0.2
        "weighted_f1": weighted_f1,
        "balanced_accuracy": balanced_accuracy,
        "per_class": per_class,
        "worst_class_recall": (
            {"label": worst, "recall": per_class[worst]["recall"]} if worst else None
        ),
        "classes_absent": absent,
    }


def field_metrics(
    field: FieldSpec, preds: list, refs: list, severe_delta: int = DEFAULT_SEVERE_DELTA
) -> dict[str, Any]:
    if field.type == "int":
        return int_field_metrics(field, preds, refs, severe_delta)
    return enum_field_metrics(field, preds, refs)


def compare(
    spec: FunctionSpec,
    preds: list,
    refs: list,
    severe_delta: int | None = None,
) -> dict[str, Any]:
    """The one comparison shape. Scalar contracts return the field's metric
    dict; structured contracts return joint agreement/exact/invalid plus the
    full per-field set under `fields` and the worst required field by gate
    margin under `worst_field`."""
    delta = severe_delta if severe_delta is not None else spec.gate.severe_delta
    if spec.output.is_scalar:
        return field_metrics(spec.output.scalar, preds, refs, delta)

    n = len(refs)
    fields = spec.output.fields
    dicts = [p if isinstance(p, dict) else {} for p in preds]
    per_field = {
        name: field_metrics(field, [d.get(name) for d in dicts], [g[name] for g in refs], delta)
        for name, field in fields.items()
    }
    joint_k = sum(
        1
        for d, g in zip(dicts, refs)
        if all(agrees(f, d.get(name), g[name]) for name, f in fields.items())
    )
    exact_k = sum(
        1 for d, g in zip(dicts, refs) if all(d.get(name) == g[name] for name in fields)
    )
    invalid = sum(1 for d in dicts if any(d.get(name) is None for name in fields))
    worst_name = min(
        per_field,
        key=lambda name: per_field[name]["agreement"] - spec.gate.field_threshold(name),
    )
    ci = wilson_ci(joint_k, n)
    return {
        "n": n,
        "valid_n": n - invalid,
        "invalid_rate": round(invalid / n, 4) if n else 0.0,
        "agreement": round(joint_k / n, 4) if n else 0.0,  # joint: all fields
        "agreement_ci": list(ci) if ci else None,
        "exact": round(exact_k / n, 4) if n else 0.0,
        "fields": per_field,
        "worst_field": {
            "name": worst_name,
            "agreement": per_field[worst_name]["agreement"],
            "threshold": spec.gate.field_threshold(worst_name),
        },
    }
