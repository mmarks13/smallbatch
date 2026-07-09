"""Eval reports: the headline output of a compile.

The gate verdict is a recorded acceptance check; this report is how you
actually learn what the adapter does — agreement with a confidence interval,
per-value breakdown, confusion, training curve, and concrete failures.
Torch-free: works from metrics dicts + rows, unit-testable on CPU.
"""

from __future__ import annotations

import json
from pathlib import Path
from typing import Any, Optional

from .labeling import Row, row_output
from .spec import FieldSpec, FunctionSpec

SMALL_GATE_N = 50  # below this, the verdict is noise-dominated — say so
SEVERE_DELTA = 3  # int outputs: |pred - gold| >= this is a severe miss


def _input_excerpt(row: Row, limit: int = 110) -> str:
    text = " | ".join(f"{k}={v}" for k, v in row["input"].items() if v)
    return text[:limit] + ("…" if len(text) > limit else "")


def _field_breakdown(field: FieldSpec, preds: list, golds: list) -> dict:
    """per-value agreement + confusion (+severe for int) for one field."""
    values = field.values()
    per_value: dict[str, dict] = {}
    for v in values:
        idx = [i for i, g in enumerate(golds) if g == v]
        if not idx:
            continue
        if field.type == "int":
            ok = sum(1 for i in idx if preds[i] is not None and abs(preds[i] - v) <= 1)
        else:
            ok = sum(1 for i in idx if preds[i] == v)
        per_value[str(v)] = {"n": len(idx), "agreement": round(ok / len(idx), 4)}

    labels = [str(v) for v in values] + ["invalid"]
    matrix = [[0] * len(labels) for _ in values]
    for p, g in zip(preds, golds):
        col = len(labels) - 1 if p is None else values.index(p)
        matrix[values.index(g)][col] += 1

    severe = None
    if field.type == "int":
        n = len(golds)
        severe_k = sum(
            1
            for p, g in zip(preds, golds)
            if p is None or abs(p - g) >= SEVERE_DELTA
        )
        severe = {
            "threshold": SEVERE_DELTA,
            "rate": round(severe_k / n, 4) if n else 0.0,
            "count": severe_k,
        }
    return {
        "per_value": per_value,
        "confusion": {"labels": labels, "gold_order": [str(v) for v in values], "matrix": matrix},
        "severe": severe,
    }


def build_report(
    spec: FunctionSpec,
    gate_rows: list[Row],
    adapter: dict,
    zeroshot: Optional[dict],
    gate: dict,
    training: dict,
) -> dict:
    """Assemble the full eval report from compile outputs."""
    preds = adapter.get("preds", [])
    golds = [row_output(spec, r) for r in gate_rows]

    fields_section = None
    if spec.output.is_scalar:
        breakdown = _field_breakdown(spec.output.scalar, preds, golds)
        misses = []
        for i, (p, g) in enumerate(zip(preds, golds)):
            if spec.output.scalar.type == "int":
                delta = SEVERE_DELTA + 1 if p is None else abs(p - g)
                if delta > 1:
                    misses.append((delta, i))
            elif p != g:
                misses.append((1, i))
    else:
        dicts = [p if isinstance(p, dict) else {} for p in preds]
        fields_section = {
            name: {
                "metrics": (adapter.get("fields") or {}).get(name),
                **_field_breakdown(
                    field, [d.get(name) for d in dicts], [g[name] for g in golds]
                ),
            }
            for name, field in spec.output.fields.items()
        }
        breakdown = {"per_value": {}, "confusion": None, "severe": None}
        misses = []
        for i, (d, g) in enumerate(zip(dicts, golds)):
            wrong = [
                name
                for name, field in spec.output.fields.items()
                if not (
                    d.get(name) is not None
                    and (
                        abs(d[name] - g[name]) <= 1
                        if field.type == "int"
                        else d[name] == g[name]
                    )
                )
            ]
            if wrong:
                misses.append((len(wrong), i))
    misses.sort(reverse=True)
    failures = [
        {
            "input": _input_excerpt(gate_rows[i]),
            "teacher": golds[i],
            "adapter": preds[i],
            "teacher_reason": gate_rows[i].get("reason", ""),
        }
        for _, i in misses[:5]
    ]

    headline = {
        k: adapter.get(k)
        for k in ("n", "agreement", "agreement_ci", "exact", "invalid_rate", "pearson_r")
        if k in adapter
    }
    if zeroshot:
        headline["zeroshot_agreement"] = zeroshot.get("agreement")

    return {
        "function": spec.name,
        "base_model": spec.train.base,
        "precision": training.get("precision"),
        "headline": headline,
        "gate": gate,
        "small_gate_warning": len(gate_rows) < SMALL_GATE_N,
        "training": {
            k: training.get(k)
            for k in (
                "train_rows", "dev_rows", "epochs_run", "best_epoch",
                "best_dev_agreement", "stopped_reason", "train_loss", "curve",
            )
        },
        "per_value": breakdown["per_value"],
        "confusion": breakdown["confusion"],
        "severe": breakdown["severe"],
        "fields": fields_section,
        "failures": failures,
    }


def _pct(x) -> str:
    return "-" if x is None else f"{x:.0%}"


def _confusion_md(conf: dict) -> list[str]:
    lines = ["### Confusion (teacher rows × adapter cols)"]
    lines.append("| gold \\ pred | " + " | ".join(conf["labels"]) + " |")
    lines.append("|---|" + "---|" * len(conf["labels"]))
    for gold_label, row in zip(conf["gold_order"], conf["matrix"]):
        lines.append(f"| **{gold_label}** | " + " | ".join(str(c) for c in row) + " |")
    lines.append("")
    return lines


def render_markdown(report: dict) -> str:
    h = report["headline"]
    gate = report["gate"]
    tr = report["training"]
    lines = [f"# {report['function']} — eval report", ""]
    lines.append(f"base: `{report['base_model']}` ({report['precision']})")
    lines.append("")

    ci = h.get("agreement_ci")
    ci_txt = f" (95% CI {ci[0]:.0%}–{ci[1]:.0%})" if ci else ""
    lines.append(
        f"**agreement {h.get('agreement', 0):.1%}{ci_txt}** on {h.get('n', 0)} gate items"
        f" · exact {_pct(h.get('exact'))} · invalid {_pct(h.get('invalid_rate'))}"
        + (f" · pearson r {h['pearson_r']}" if h.get("pearson_r") is not None else "")
    )
    if "zeroshot_agreement" in h:
        lines.append(f"zero-shot base: {_pct(h['zeroshot_agreement'])} agreement")
    verdict = "PASS" if gate["passed"] else "FAIL"
    lines.append(f"gate: **{verdict}**" + (f" — {'; '.join(gate['reasons'])}" if gate["reasons"] else ""))
    if report["small_gate_warning"]:
        lines.append(
            f"\n> ⚠ gate split has only {h.get('n', 0)} items (< {SMALL_GATE_N}): "
            "the verdict is noise-dominated — treat the CI, not the point estimate, "
            "as the result, and grow the gate with more labeled real items."
        )
    lines.append("")

    lines.append("## Training")
    stop = tr.get("stopped_reason") or "-"
    best = tr.get("best_epoch")
    lines.append(
        f"{tr.get('train_rows')} train rows, {tr.get('dev_rows') or 0} dev rows · "
        f"ran {tr.get('epochs_run')} epochs ({stop})"
        + (f" · **best epoch {best}** (dev agreement {_pct(tr.get('best_dev_agreement'))})"
           if best is not None else "")
    )
    curve = tr.get("curve") or []
    if curve:
        lines += ["", "| epoch | train loss | dev agreement |", "|---|---|---|"]
        for pt in curve:
            loss = pt.get("train_loss")
            lines.append(
                f"| {pt['epoch']} | {loss if loss is not None else '-'} | "
                f"{pt['dev_agreement']:.4f} |"
            )
    lines.append("")

    if report["per_value"]:
        lines.append("## Agreement by teacher label")
        lines += ["| label | n | agreement |", "|---|---|---|"]
        for v, d in report["per_value"].items():
            lines.append(f"| {v} | {d['n']} | {d['agreement']:.0%} |")
        lines.append("")

    if report.get("severe"):
        s = report["severe"]
        lines.append(
            f"severe misses (|Δ| ≥ {s['threshold']}): {s['count']} ({s['rate']:.1%})"
        )
        lines.append("")

    if report.get("confusion"):
        lines += _confusion_md(report["confusion"])

    for name, section in (report.get("fields") or {}).items():
        lines.append(f"## Field `{name}`")
        m = section.get("metrics") or {}
        ci = m.get("agreement_ci")
        ci_txt = f" (95% CI {ci[0]:.0%}–{ci[1]:.0%})" if ci else ""
        lines.append(
            f"agreement {m.get('agreement', 0):.1%}{ci_txt}"
            f" · exact {_pct(m.get('exact'))} · invalid {_pct(m.get('invalid_rate'))}"
        )
        if section.get("severe"):
            s = section["severe"]
            lines.append(
                f"severe misses (|Δ| ≥ {s['threshold']}): {s['count']} ({s['rate']:.1%})"
            )
        lines.append("")
        if section.get("confusion"):
            lines += _confusion_md(section["confusion"])

    if report["failures"]:
        lines.append("## Largest disagreements")
        for f in report["failures"]:
            lines.append(
                f"- teacher **{f['teacher']}** / adapter **{f['adapter']}** — {f['input']}"
            )
            if f.get("teacher_reason"):
                lines.append(f"  - teacher: {f['teacher_reason']}")
        lines.append("")

    return "\n".join(lines)


def write_report(version_dir: Path, report: dict) -> Path:
    (version_dir / "report.json").write_text(json.dumps(report, indent=2))
    md = version_dir / "report.md"
    md.write_text(render_markdown(report))
    return md
