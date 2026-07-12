"""The all-candidates-failed decision: information-dense, never lossy.

When every completed candidate misses the gate, the work is not wasted — the
user gets a type-appropriate table of everything the manifest/report already
measured (values are read from them, NEVER recomputed) and may explicitly
accept the best candidate for deployment. Acceptance is candidate-scoped and
persistent (deployment.accepted_candidate); the quality verdict stays FAIL and
compile still exits 2.
"""

from __future__ import annotations

from typing import Any, Optional

from .spec import FunctionSpec

_NA = "-"


def _fmt(v: Any, pct: bool = False) -> str:
    if v is None:
        return _NA
    if pct:
        return f"{v:.0%}"
    return f"{v:g}" if isinstance(v, (int, float)) else str(v)


def _size(rec: dict) -> str:
    b = rec.get("artifact_size_bytes")
    if not b:
        return _NA
    return f"{b / 1e6:.1f} MB" if b >= 1e5 else f"{b / 1e3:.0f} KB"


def _int_columns() -> list[tuple[str, str, bool]]:
    # (header, metrics key, render-as-percent) — MAE/p90/max/correlation/
    # invalid must never be silently omitted for integer outputs
    return [
        ("agree ±1", "agreement", True),
        ("exact", "exact", True),
        ("MAE", "mae", False),
        ("p90 err", "p90_absolute_error", False),
        ("max err", "max_absolute_error", False),
        ("pearson", "pearson_r", False),
        ("invalid", "invalid_rate", True),
    ]


def _enum_columns() -> list[tuple[str, str, bool]]:
    return [
        ("accuracy", "agreement", True),
        ("macro F1", "macro_f1", False),
        ("bal acc", "balanced_accuracy", False),
        ("worst recall", "_worst_recall", False),
        ("invalid", "invalid_rate", True),
    ]


def _metric_row(name: str, metrics: dict, columns, extra: str = "") -> str:
    cells = [name]
    for _, key, pct in columns:
        if key == "_worst_recall":
            worst = metrics.get("worst_class_recall") or {}
            cells.append(
                f"{worst.get('recall'):.0%} ({worst.get('label')})"
                if worst.get("recall") is not None else _NA
            )
        else:
            cells.append(_fmt(metrics.get(key), pct))
    if extra:
        cells.append(extra)
    return "  ".join(f"{c:>12}" if i else f"{c:<12}" for i, c in enumerate(cells))


def _columns_for(spec: FunctionSpec):
    if spec.output.is_scalar and spec.output.scalar.type == "int":
        return _int_columns()
    if spec.output.is_scalar:
        return _enum_columns()
    # structured: joint headline + the worst required field; the report keeps
    # every field's full table
    return [
        ("joint agree", "agreement", True),
        ("joint exact", "exact", True),
        ("invalid", "invalid_rate", True),
    ]


def build_decision_text(
    spec: FunctionSpec,
    manifest: dict,
    report: Optional[dict],
    displaced: Optional[str] = None,
) -> str:
    """The complete decision summary shown before the acceptance prompt."""
    candidates = manifest.get("candidates") or {}
    selection = manifest.get("selection") or {}
    winner_name = selection.get("winner")
    winner = candidates.get(winner_name) or {}
    threshold = spec.gate.threshold
    columns = _columns_for(spec)
    n = (winner.get("metrics") or {}).get("n")
    if spec.output.is_scalar:
        kind = (
            "integer output; lower error is better"
            if spec.output.scalar.type == "int"
            else "enum output"
        )
    else:
        kind = "structured output (per-field table in the report)"

    header = ["candidate"] + [c[0] for c in columns] + ["size", "result"]
    lines = [
        f"No candidate met the {threshold:.0%} teacher-agreement gate.",
        "",
        f"Teacher-labeled gate (n={n}, {kind})",
        "",
        "  ".join(f"{h:>12}" if i else f"{h:<12}" for i, h in enumerate(header)),
    ]
    for name, rec in candidates.items():
        if rec.get("status") != "completed":
            lines.append(f"{name:<12}  ERROR — {rec.get('error')}")
            continue
        m = rec.get("metrics") or {}
        agree = m.get("agreement") or 0.0
        miss = f"FAIL {100 * (agree - threshold):+.0f}pt"
        mark = f"{name} *" if name == winner_name else name
        lines.append(_metric_row(mark, m, columns, f"{_size(rec):>8}  {miss}"))
    zeroshot = (manifest.get("metrics") or {}).get("zeroshot")
    if zeroshot:
        lines.append(_metric_row("zero-shot", zeroshot, columns, f"{_NA:>8}  baseline"))
    const = (winner.get("metrics") or {}).get("constant_baseline")
    if const is not None:
        lines.append(
            f"{'oracle const':<12}  agreement {const['agreement']:.0%} "
            f"(always \"{const['value']}\") — oracle on this split, baseline"
        )
        if (winner.get("metrics") or {}).get("agreement", 0) <= const["agreement"]:
            lines.append(
                f"NOTE: the best candidate does NOT beat the constant "
                f"\"{const['value']}\" baseline — it has learned the label "
                "prior, not the task."
            )

    gold = (report or {}).get("gold")
    if gold:
        lines += ["", f"Gold subset (n={gold['n']})", ""]
        header = ["comparison"] + [c[0] for c in columns]
        lines.append("  ".join(f"{h:>12}" if i else f"{h:<12}" for i, h in enumerate(header)))
        for key, label in (
            ("teacher_vs_gold", "teacher"),
            ("student_vs_gold", f"{winner_name} (win)"),
        ):
            lines.append(_metric_row(label, gold.get(key) or {}, columns))
        rep_candidates = (report or {}).get("candidates") or {}
        golds = {
            name: c.get("gold_agreement")
            for name, c in rep_candidates.items()
            if c.get("gold_agreement") is not None and name != winner_name
        }
        for name, ga in golds.items():
            lines.append(f"{name:<12}  gold agreement {ga:.0%}")
        best_gold = max(
            (
                (c.get("gold_agreement"), name)
                for name, c in rep_candidates.items()
                if c.get("gold_agreement") is not None
            ),
            default=(None, None),
        )
        if best_gold[1] and best_gold[1] != winner_name:
            lines.append(
                f"NOTE: gold prefers '{best_gold[1]}' while teacher agreement "
                f"prefers '{winner_name}' — the ranking is ambiguous; read the "
                "report before deciding."
            )

    win_m = winner.get("metrics") or {}
    lines += [
        "",
        f"Best available: {winner_name} ({selection.get('reason')})",
        "Failed requirement: "
        + "; ".join((winner.get("gate") or {}).get("reasons") or ["(none recorded)"]),
    ]
    runtime = "CPU (sklearn pipeline)" if winner.get("backend") == "tfidf" else \
        f"base model {winner.get('base_model')} required"
    lines.append(f"Runtime: {runtime}; candidate artifact {_size(winner)}.")
    if win_m.get("severe") is not None:
        s = win_m["severe"]
        lines.append(
            f"Tail error: {s['count']} severe miss(es) (|Δ| ≥ {s['threshold']}, "
            f"{s['rate']:.0%} of the gate)."
        )
    if displaced:
        lines.append(
            f"CURRENTLY DEPLOYED: {displaced} (a PASSING version). Accepting "
            "makes THIS failed build the default instead; declining keeps "
            f"{displaced} active."
        )
    return "\n".join(lines)


def all_completed_failed(manifest: dict) -> bool:
    """The acceptance offer's precondition: at least one candidate completed
    and every completed candidate failed its gate. Errors, stale-label
    refusals, and corrupt artifacts never reach this code path (they raise)."""
    candidates = manifest.get("candidates") or {}
    completed = [c for c in candidates.values() if c.get("status") == "completed"]
    return bool(completed) and not any(
        (c.get("gate") or {}).get("passed") for c in completed
    )
