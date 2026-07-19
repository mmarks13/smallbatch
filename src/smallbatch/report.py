"""Redacted comparison evidence and local disagreement details."""

from __future__ import annotations

import json
import math
from pathlib import Path

from .metrics import BOOTSTRAP_SEED, bootstrap_ci, field_decision_matches
from .spec import FunctionSpec


def _number(value) -> str:
    if isinstance(value, float):
        return f"{value:.4f}"
    return str(value)


def _mib(value) -> str:
    return f"{value / (1024 * 1024):.1f}MiB"


def evidence_summary(record: dict) -> str:
    """Compact, stable evidence line for long-running compile progress."""
    metrics = record.get("metrics") or {}
    parts: list[str] = []
    if "within_one" in metrics:
        fields = (
            ("exact", "exact"),
            ("within_one", "within_one"),
            ("mae", "mae"),
            ("p90_error", "p90_absolute_error"),
            ("max_error", "max_absolute_error"),
            ("signed_error", "mean_signed_error"),
            ("pearson", "pearson_r"),
            ("spearman", "spearman_rho"),
            ("invalid", "invalid_rate"),
        )
    elif "decision_agreement" in metrics:
        fields = (
            ("agreement", "decision_agreement"),
            ("macro_f1", "macro_f1"),
            ("balanced_accuracy", "balanced_accuracy"),
            ("invalid", "invalid_rate"),
        )
    else:
        fields = (
            ("joint_agreement", "joint_decision_agreement"),
            ("joint_exact", "joint_exact"),
            ("invalid", "invalid_rate"),
        )
    parts.extend(
        f"{label}={_number(metrics[key])}"
        for label, key in fields
        if metrics.get(key) is not None
    )

    profile = record.get("profile") or {}
    latency = profile.get("batch_one_latency_ms") or {}
    for label, value in (("p50_ms", latency.get("p50")), ("p95_ms", latency.get("p95"))):
        if value is not None:
            parts.append(f"{label}={_number(value)}")
    for label, key in (("peak_rss", "peak_rss_bytes"), ("owned", "candidate_owned_bytes")):
        if profile.get(key) is not None:
            parts.append(f"{label}={_mib(profile[key])}")
    return " ".join(parts) if parts else "no evaluation evidence"


def _row_errors(spec: FunctionSpec, predictions: list, references: list) -> list[list[float]]:
    errors: list[list[float]] = []
    for prediction, reference in zip(predictions, references):
        pred = prediction if isinstance(prediction, dict) else {next(iter(spec.output.fields)): prediction}
        ref = reference if isinstance(reference, dict) else {next(iter(spec.output.fields)): reference}
        row = []
        for name, field in spec.output.fields.items():
            value = pred.get(name)
            if value is None:
                row.append(float("inf"))
            elif field.type == "int":
                row.append(abs(value - ref[name]))
            else:
                row.append(0.0 if value == ref[name] else 1.0)
        errors.append(row)
    return errors


def observed_dominance(
    spec: FunctionSpec, completed: dict[str, dict], references: list
) -> list[dict[str, str]]:
    """Strict observed dominance over every row and operating dimension."""
    output: list[dict[str, str]] = []
    errors = {
        name: _row_errors(spec, record["predictions"], references)
        for name, record in completed.items()
    }
    for first, first_record in completed.items():
        for second, second_record in completed.items():
            if first == second:
                continue
            no_worse = all(
                all(a <= b for a, b in zip(first_row, second_row))
                for first_row, second_row in zip(errors[first], errors[second])
            )
            first_profile = first_record["profile"]
            second_profile = second_record["profile"]
            first_shared = first_profile["required_shared_bytes"]
            second_shared = second_profile["required_shared_bytes"]
            if first_shared is None or second_shared is None:
                continue
            first_ops = [
                first_profile["batch_one_latency_ms"]["p50"],
                first_profile["batch_one_latency_ms"]["p95"],
                first_profile["peak_rss_bytes"],
                first_profile["candidate_owned_bytes"] + first_shared,
            ]
            second_ops = [
                second_profile["batch_one_latency_ms"]["p50"],
                second_profile["batch_one_latency_ms"]["p95"],
                second_profile["peak_rss_bytes"],
                second_profile["candidate_owned_bytes"] + second_shared,
            ]
            no_worse = no_worse and all(a <= b for a, b in zip(first_ops, second_ops))
            strictly_better = any(
                any(a < b for a, b in zip(first_row, second_row))
                for first_row, second_row in zip(errors[first], errors[second])
            ) or any(a < b for a, b in zip(first_ops, second_ops))
            if no_worse and strictly_better:
                output.append({"dominates": second, "candidate": first})
    return output


def _decision_correct(spec: FunctionSpec, prediction, reference) -> bool:
    """Did this row's decision match within the contract's tolerance (±1 for
    integer fields, exact for enums, jointly for structured output)?"""
    if spec.output.is_scalar:
        return field_decision_matches(spec.output.scalar, prediction, reference)
    pred = prediction if isinstance(prediction, dict) else {}
    return all(
        field_decision_matches(field, pred.get(name), reference[name])
        for name, field in spec.output.fields.items()
    )


def _row_absolute_error(spec: FunctionSpec, prediction, reference) -> float | None:
    """Total absolute error over integer fields, or None when the spec has no
    integer field or this row's prediction is missing an integer field (so a
    paired MAE contrast can drop it complete-case rather than count it as zero)."""
    if spec.output.is_scalar:
        if spec.output.scalar.type != "int":
            return None
        return None if prediction is None else float(abs(prediction - reference))
    pred = prediction if isinstance(prediction, dict) else {}
    total = 0
    has_int = False
    for name, field in spec.output.fields.items():
        if field.type != "int":
            continue
        has_int = True
        value = pred.get(name)
        if value is None:
            return None
        total += abs(value - reference[name])
    return float(total) if has_int else None


def _mcnemar_p(discordant_a: int, discordant_b: int) -> float | None:
    """Two-sided exact McNemar p over the rows the two candidates disagree on
    (an exact binomial sign test at p=0.5, right for the small eval splits here)."""
    n = discordant_a + discordant_b
    if n == 0:
        return None
    tail = sum(math.comb(n, k) for k in range(min(discordant_a, discordant_b) + 1)) * 0.5**n
    return round(min(1.0, 2 * tail), 4)


def pairwise_comparisons(
    spec: FunctionSpec, completed: dict[str, dict], references: list, seed: int = BOOTSTRAP_SEED
) -> list[dict]:
    """Paired candidate-vs-candidate contrasts on the shared evaluation rows.

    Both candidates score the identical rows, so pairing cancels row-difficulty
    variance and is far more powerful than comparing two marginal intervals:
    overlapping per-candidate CIs do not imply the pair is indistinguishable.
    Reports the paired decision-agreement delta with a bootstrap CI, an exact
    McNemar test on discordant decisions, and — for integer-bearing specs — a
    paired MAE delta over complete-case rows. Deltas are first-minus-second in
    candidate-name order; a delta CI clear of zero is a robust win.
    """
    names = sorted(completed)
    n = len(references)
    correct = {
        name: [
            _decision_correct(spec, prediction, reference)
            for prediction, reference in zip(completed[name]["predictions"], references)
        ]
        for name in names
    }
    abs_error = {
        name: [
            _row_absolute_error(spec, prediction, reference)
            for prediction, reference in zip(completed[name]["predictions"], references)
        ]
        for name in names
    }
    comparisons: list[dict] = []
    for first_index in range(len(names)):
        for second_index in range(first_index + 1, len(names)):
            first, second = names[first_index], names[second_index]
            hits_a, hits_b = correct[first], correct[second]
            discordant = [
                sum(a and not b for a, b in zip(hits_a, hits_b)),
                sum(b and not a for a, b in zip(hits_a, hits_b)),
            ]
            entry: dict = {
                "a": first,
                "b": second,
                "agreement_delta": round((sum(hits_a) - sum(hits_b)) / n, 4) if n else None,
                "agreement_delta_ci": bootstrap_ci(
                    lambda idx, hits_a=hits_a, hits_b=hits_b: (
                        sum(hits_a[k] for k in idx) - sum(hits_b[k] for k in idx)
                    )
                    / len(idx),
                    n,
                    seed=seed,
                ),
                "mcnemar_discordant": discordant,
                "mcnemar_p": _mcnemar_p(*discordant),
            }
            paired = [
                (a, b)
                for a, b in zip(abs_error[first], abs_error[second])
                if a is not None and b is not None
            ]
            if paired:
                diffs = [a - b for a, b in paired]
                entry["mae_delta"] = round(sum(diffs) / len(diffs), 4)
                entry["mae_delta_ci"] = bootstrap_ci(
                    lambda idx, diffs=diffs: sum(diffs[k] for k in idx) / len(idx),
                    len(diffs),
                    seed=seed,
                )
                entry["mae_delta_paired_n"] = len(paired)
            comparisons.append(entry)
    return comparisons


def build_report(
    spec: FunctionSpec,
    eval_rows: list[dict],
    candidates: dict[str, dict],
    diagnostics: dict[str, dict],
) -> tuple[dict, dict]:
    references = [row["output"] for row in eval_rows]
    completed = {
        name: record for name, record in candidates.items() if record.get("status") == "completed"
    }
    public_candidates = {}
    for name, record in candidates.items():
        public_candidates[name] = {
            key: value
            for key, value in record.items()
            if key not in {"predictions"}
        }
    report = {
        "schema_version": 4,
        "function": spec.name,
        "evaluation_rows": len(eval_rows),
        "small_evaluation_warning": len(eval_rows) < 30,
        "candidates": public_candidates,
        "diagnostics": diagnostics,
        "observed_strict_dominance": observed_dominance(spec, completed, references),
        "pairwise": pairwise_comparisons(spec, completed, references),
        "selection_bias_note": (
            "These candidates share one evaluation split. Selecting after comparison makes "
            "the selected result optimistic; v0.2 does not provide an independent confirmation set."
        ),
        "claims": {
            "decision_correctness_validated": False,
            "energy_measured": False,
            "operating_proxies": ["CPU latency", "peak RSS", "artifact bytes"],
        },
    }
    details = {"function": spec.name, "largest_disagreements": {}}
    for name, record in completed.items():
        ranked = sorted(
            range(len(eval_rows)),
            key=lambda index: sum(
                value if value != float("inf") else 10**9
                for value in _row_errors(
                    spec, [record["predictions"][index]], [references[index]]
                )[0]
            ),
            reverse=True,
        )[:10]
        details["largest_disagreements"][name] = [
            {
                "eval_row": index,
                "input": eval_rows[index]["input"],
                "reference": references[index],
                "prediction": record["predictions"][index],
            }
            for index in ranked
        ]
    return report, details


def write_report(build: Path, report: dict, details: dict) -> Path:
    path = build / "report.json"
    path.write_text(json.dumps(report, indent=2, ensure_ascii=False))
    (build / "report_details.local.json").write_text(
        json.dumps(details, indent=2, ensure_ascii=False)
    )
    (build / "report.md").write_text(render_markdown(report))
    return path


def _decoder_evidence(record: dict) -> list[tuple[str, str, dict | None, dict | None]]:
    """(field, selected decoder, dev comparison, head diagnostics) per ordinal
    field, across the three backends' record shapes."""
    entries: list[tuple[str, str, dict | None, dict | None]] = []
    backend = record.get("backend")
    if backend == "lora" and record.get("decode"):
        training = record.get("training") or {}
        entries.append(("decode", record["decode"], training.get("dev_decode_comparison"), None))
    elif backend == "tfidf":
        for field, selected in (record.get("decode") or {}).items():
            entries.append(
                (
                    field,
                    selected,
                    (record.get("dev_decode_comparison") or {}).get(field),
                    (record.get("head_diagnostics") or {}).get(field),
                )
            )
    elif backend == "setfit":
        for field, training in (record.get("field_training") or {}).items():
            if training.get("decode"):
                entries.append(
                    (
                        field,
                        training["decode"],
                        training.get("dev_decode_comparison"),
                        training.get("head_diagnostics"),
                    )
                )
    return entries


def _decoder_lines(record: dict) -> list[str]:
    lines: list[str] = []
    for field, selected, comparison, diagnostics in _decoder_evidence(record):
        lines.append("")
        source = "dev-selected" if comparison else "pinned"
        lines.append(f"Decode `{field}`: **{selected}** ({source})")
        if comparison:
            decoders = comparison.get("decoders", comparison)
            lines.extend(["", "| decoder | exact | within_one | mae |", "|---|---:|---:|---:|"])
            for name, values in decoders.items():
                marker = " *" if name == selected else ""
                lines.append(
                    f"| {name}{marker} | {_number(values.get('exact'))} | "
                    f"{_number(values.get('within_one'))} | {_number(values.get('mae'))} |"
                )
        if diagnostics:
            capacity = "linear" if not diagnostics.get("hidden") else (
                f"{diagnostics['hidden']}-hidden"
            )
            lines.append(
                f"\nOrdinal head on dev: {capacity} capacity, mean confidence "
                f"{_number(diagnostics.get('mean_confidence'))}."
            )
            levels = diagnostics.get("levels") or []
            if levels:
                lines.extend(
                    [
                        "",
                        "| level | mean probability | observed rate |",
                        "|---|---:|---:|",
                    ]
                )
                lines.extend(
                    f"| {entry['level']} | {_number(entry['mean_probability'])} | "
                    f"{_number(entry['observed_rate'])} |"
                    for entry in levels
                )
    return lines


def _embedding_lines(record: dict) -> list[str]:
    """SetFit only: did fine-tuning the encoder beat leaving it frozen?"""
    lines: list[str] = []
    if record.get("backend") != "setfit":
        return lines
    for field, training in (record.get("field_training") or {}).items():
        comparison = training.get("frozen_vs_tuned")
        if not comparison:
            continue
        lines.append(
            f"\nEmbedding `{field}`: frozen {_number(comparison['frozen_dev_within_one'])} "
            f"-> best {_number(comparison['best_dev_within_one'])} dev within-one "
            f"(delta {_number(comparison['delta'])}, kept epoch {comparison['best_epoch']})"
        )
    return lines


def render_markdown(report: dict) -> str:
    lines = [
        f"# {report['function']} evidence",
        "",
        f"Evaluation decisions: **{report['evaluation_rows']}**",
        "",
        "> Decision agreement measures fidelity to the supplied decisions, not correctness.",
        "",
        "## Candidates",
        "",
        "| candidate | status | runtime | p50 ms | peak RSS | owned bytes |",
        "|---|---:|---:|---:|---:|---:|",
    ]
    for name, record in report["candidates"].items():
        profile = record.get("profile") or {}
        latency = (profile.get("batch_one_latency_ms") or {}).get("p50")
        lines.append(
            f"| {name} | {record.get('status')} | {record.get('backend', '-')} | "
            f"{latency if latency is not None else '-'} | "
            f"{profile.get('peak_rss_bytes', '-')} | {profile.get('candidate_owned_bytes', '-')} |"
        )
        if record.get("metrics"):
            lines.extend(["", f"### {name} metrics", "", "```json", json.dumps(record["metrics"], indent=2), "```"])
        lines.extend(_decoder_lines(record))
        lines.extend(_embedding_lines(record))
        if record.get("error"):
            lines.append(f"\nError: `{record['error']}`")
    lines.extend(_pairwise_lines(report.get("pairwise") or []))
    lines.extend(["", "## Interpretation", "", report["selection_bias_note"]])
    if report["observed_strict_dominance"]:
        lines.extend(["", "Observed strict dominance:"])
        lines.extend(
            f"- `{entry['candidate']}` dominates `{entry['dominates']}` on this evaluation."
            for entry in report["observed_strict_dominance"]
        )
    return "\n".join(lines) + "\n"


def _fmt_ci(interval) -> str:
    return f"[{_number(interval[0])}, {_number(interval[1])}]" if interval else "-"


def _ci_excludes_zero(interval) -> bool:
    return bool(interval) and (interval[0] > 0 or interval[1] < 0)


def _pairwise_lines(pairwise: list[dict]) -> list[str]:
    """Paired candidate contrasts, with the CI-clear-of-zero wins called out."""
    if not pairwise:
        return []
    lines = [
        "",
        "## Paired comparisons",
        "",
        "Same evaluation rows, so these contrasts are paired (deltas are A minus B). "
        "A delta CI clear of zero is a robust win; overlapping per-candidate CIs do not "
        "rule one out.",
        "",
        "| A | B | agreement Δ | 95% CI | McNemar p | MAE Δ | MAE Δ 95% CI |",
        "|---|---|---:|---:|---:|---:|---:|",
    ]
    for entry in pairwise:
        agreement = entry.get("agreement_delta")
        lines.append(
            f"| {entry['a']} | {entry['b']} | "
            f"{_number(agreement) if agreement is not None else '-'} | "
            f"{_fmt_ci(entry.get('agreement_delta_ci'))} | "
            f"{entry.get('mcnemar_p') if entry.get('mcnemar_p') is not None else '-'} | "
            f"{_number(entry['mae_delta']) if 'mae_delta' in entry else '-'} | "
            f"{_fmt_ci(entry.get('mae_delta_ci'))} |"
        )
    wins = []
    for entry in pairwise:
        interval = entry.get("agreement_delta_ci")
        if not _ci_excludes_zero(interval):
            continue
        better, worse = (entry["a"], entry["b"]) if interval[0] > 0 else (entry["b"], entry["a"])
        wins.append(
            f"- `{better}` beats `{worse}` on decision agreement "
            f"(Δ={_number(abs(entry['agreement_delta']))}, 95% CI {_fmt_ci(interval)})."
        )
    if wins:
        lines.extend(["", "Robust agreement wins (CI excludes zero):", *wins])
    return lines
