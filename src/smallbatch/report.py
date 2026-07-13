"""Redacted comparison evidence and local disagreement details."""

from __future__ import annotations

import json
from pathlib import Path

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
        "schema_version": 3,
        "function": spec.name,
        "evaluation_rows": len(eval_rows),
        "small_evaluation_warning": len(eval_rows) < 30,
        "candidates": public_candidates,
        "diagnostics": diagnostics,
        "observed_strict_dominance": observed_dominance(spec, completed, references),
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
        if record.get("error"):
            lines.append(f"\nError: `{record['error']}`")
    lines.extend(["", "## Interpretation", "", report["selection_bias_note"]])
    if report["observed_strict_dominance"]:
        lines.extend(["", "Observed strict dominance:"])
        lines.extend(
            f"- `{entry['candidate']}` dominates `{entry['dominates']}` on this evaluation."
            for entry in report["observed_strict_dominance"]
        )
    return "\n".join(lines) + "\n"
