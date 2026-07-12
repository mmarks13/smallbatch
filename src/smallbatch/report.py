"""Eval reports: the headline output of a compile.

The gate verdict is a recorded acceptance check; this report is how you
actually learn what the adapter does — agreement with a confidence interval,
per-value breakdown, confusion, training curve, and concrete failures.
Torch-free: works from metrics dicts + rows, unit-testable on CPU.
"""

from __future__ import annotations

import json
import re
from pathlib import Path
from typing import Any, Optional

from .labeling import Row, row_output
from .metrics import DEFAULT_SEVERE_DELTA, spearman_rho
from .spec import FieldSpec, FunctionSpec

SMALL_GATE_N = 50  # below this, the verdict is noise-dominated — say so

# shortcut audit knobs
MIN_SLICE_N = 8  # a presence slice needs this many rows on BOTH sides
MAX_SLICE_VALUES = 8  # per-value slices only for fields this low-cardinality
LONG_TEXT_AVG = 40  # a field this long (avg chars) is "text" → length terciles
CORR_GAP_WARN = 0.25  # |student ρ| − |teacher ρ| above this flags a shortcut
_NUM_TOKEN = re.compile(r"([A-Za-z_][\w-]*):\s*(-?\d+(?:\.\d+)?)")


def _input_excerpt(row: Row, limit: int = 110) -> str:
    text = " | ".join(f"{k}={v}" for k, v in row["input"].items() if v)
    return text[:limit] + ("…" if len(text) > limit else "")


def _field_breakdown(
    field: FieldSpec, preds: list, golds: list, severe_delta: int = DEFAULT_SEVERE_DELTA
) -> dict:
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
            if p is None or abs(p - g) >= severe_delta
        )
        severe = {
            "threshold": severe_delta,
            "rate": round(severe_k / n, 4) if n else 0.0,
            "count": severe_k,
        }
    return {
        "per_value": per_value,
        "confusion": {"labels": labels, "gold_order": [str(v) for v in values], "matrix": matrix},
        "severe": severe,
    }


_spearman = spearman_rho  # shared metric layer (metrics.py is torch-free)


def _row_agreement(spec: FunctionSpec, preds: list, golds: list) -> list[bool]:
    """Per-row agreement under the headline rule (joint for multi-field)."""
    if spec.output.is_scalar:
        field = spec.output.scalar
        return [
            p is not None and (abs(p - g) <= 1 if field.type == "int" else p == g)
            for p, g in zip(preds, golds)
        ]
    fields = spec.output.fields
    dicts = [p if isinstance(p, dict) else {} for p in preds]
    out = []
    for d, g in zip(dicts, golds):
        out.append(
            all(
                d.get(name) is not None
                and (
                    abs(d[name] - g[name]) <= 1
                    if f.type == "int"
                    else d[name] == g[name]
                )
                for name, f in fields.items()
            )
        )
    return out


def _slice_stats(idx: list[int], flags: list[bool], deltas: list) -> dict:
    ds = [deltas[i] for i in idx if deltas[i] is not None]
    return {
        "n": len(idx),
        "agreement": round(sum(flags[i] for i in idx) / len(idx), 4),
        "mae": round(sum(ds) / len(ds), 4) if ds else None,
    }


def shortcut_audit(
    spec: FunctionSpec, gate_rows: list[Row], preds: list, golds: list
) -> dict:
    """Slice metrics + surface-feature correlations over the gate split.

    Slices show where agreement collapses; the correlation table is the
    smoking gun for shortcut reliance: a surface feature the STUDENT's
    predictions track notably harder than the TEACHER's labels do is a
    signal the model learned the feature, not the task. All auto-derived
    from the input schema — no configuration."""
    inputs = [r["input"] for r in gate_rows]
    flags = _row_agreement(spec, preds, golds)
    scalar_int = spec.output.is_scalar and spec.output.scalar.type == "int"
    deltas = [
        abs(p - g) if scalar_int and p is not None else None
        for p, g in zip(preds, golds)
    ]

    field_names = list(spec.input_schema)
    str_vals = {
        k: ["" if inputs[i].get(k) is None else str(inputs[i].get(k)).strip()
            for i in range(len(inputs))]
        for k in field_names
    }

    slices: list[dict] = []
    for k in field_names:
        vals = str_vals[k]
        present = [i for i, v in enumerate(vals) if v]
        empty = [i for i, v in enumerate(vals) if not v]
        if len(present) >= MIN_SLICE_N and len(empty) >= MIN_SLICE_N:
            slices.append({"slice": f"{k}: present", **_slice_stats(present, flags, deltas)})
            slices.append({"slice": f"{k}: empty", **_slice_stats(empty, flags, deltas)})
        distinct = {v for v in vals if v}
        if 2 <= len(distinct) <= MAX_SLICE_VALUES and all(len(v) <= 60 for v in distinct):
            for dv in sorted(distinct):
                idx = [i for i, v in enumerate(vals) if v == dv]
                slices.append({"slice": f"{k} = {dv}", **_slice_stats(idx, flags, deltas)})

    text_fields = [
        (k, sum(len(v) for v in str_vals[k]) / len(inputs))
        for k in field_names
        if any(str_vals[k])
    ]
    text_fields = [(k, avg) for k, avg in text_fields if avg >= LONG_TEXT_AVG]
    if text_fields and len(inputs) >= 3 * MIN_SLICE_N:
        k = max(text_fields, key=lambda t: t[1])[0]
        order = sorted(range(len(inputs)), key=lambda i: len(str_vals[k][i]))
        third = len(order) // 3
        for name, idx in (
            (f"{k}: shortest third", order[:third]),
            (f"{k}: middle third", order[third: 2 * third]),
            (f"{k}: longest third", order[2 * third:]),
        ):
            slices.append({"slice": name, **_slice_stats(idx, flags, deltas)})

    # surface correlations: only meaningful when there's a numeric output to
    # correlate against (scalar int contracts — the common scorer case)
    surface: list[dict] = []
    warnings: list[str] = []
    if scalar_int:
        features: dict[str, dict[int, float]] = {}
        features["input length (chars)"] = {
            i: float(sum(len(v) for v in (str_vals[k][i] for k in field_names)))
            for i in range(len(inputs))
        }
        for k in field_names:
            vals = str_vals[k]
            n_present = sum(1 for v in vals if v)
            if MIN_SLICE_N <= n_present <= len(vals) - MIN_SLICE_N:
                features[f"{k} present"] = {
                    i: float(bool(v)) for i, v in enumerate(vals)
                }
            token_rows: dict[str, dict[int, float]] = {}
            for i, v in enumerate(vals):
                for key, num in _NUM_TOKEN.findall(v):
                    token_rows.setdefault(key, {})[i] = float(num)
            for key, rows in token_rows.items():
                if len(rows) >= MIN_SLICE_N and len(set(rows.values())) >= 2:
                    features[f"{k}:{key}"] = rows

        for name, rows in features.items():
            idx = [i for i in sorted(rows) if preds[i] is not None]
            if len(idx) < MIN_SLICE_N:
                continue
            feat = [rows[i] for i in idx]
            t_rho = _spearman(feat, [float(golds[i]) for i in idx])
            s_rho = _spearman(feat, [float(preds[i]) for i in idx])
            if t_rho is None or s_rho is None:
                continue
            gap = round(abs(s_rho) - abs(t_rho), 4)
            flagged = gap > CORR_GAP_WARN
            surface.append({
                "feature": name,
                "n": len(idx),
                "teacher_rho": round(t_rho, 4),
                "student_rho": round(s_rho, 4),
                "gap": gap,
                "flag": flagged,
            })
            if flagged:
                warnings.append(
                    f"student predictions track `{name}` (ρ {s_rho:.2f}) far more "
                    f"than the teacher's labels do (ρ {t_rho:.2f}) — possible "
                    "shortcut; consider augment.field_dropout or more counter-"
                    "examples varying this feature"
                )
        surface.sort(key=lambda s: s["gap"], reverse=True)

    return {"slices": slices, "surface": surface, "warnings": warnings}


def build_report(
    spec: FunctionSpec,
    gate_rows: list[Row],
    adapter: dict,
    zeroshot: Optional[dict],
    gate: dict,
    training: dict,
    teacher_probe: Optional[dict] = None,
) -> dict:
    """Assemble the full eval report from compile outputs."""
    preds = adapter.get("preds", [])
    golds = [row_output(spec, r) for r in gate_rows]

    fields_section = None
    if spec.output.is_scalar:
        breakdown = _field_breakdown(
            spec.output.scalar, preds, golds, spec.gate.severe_delta
        )
        misses = []
        for i, (p, g) in enumerate(zip(preds, golds)):
            if spec.output.scalar.type == "int":
                delta = spec.gate.severe_delta + 1 if p is None else abs(p - g)
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
                    field,
                    [d.get(name) for d in dicts],
                    [g[name] for g in golds],
                    spec.gate.severe_delta,
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
        for k in (
            "n", "agreement", "agreement_ci", "exact", "invalid_rate",
            "pearson_r", "mae", "constant_baseline",
        )
        if k in adapter
    }
    if zeroshot:
        headline["zeroshot_agreement"] = zeroshot.get("agreement")
    if teacher_probe:
        headline["teacher_self_agreement"] = teacher_probe.get("self_agreement")
        headline["teacher_probe_n"] = teacher_probe.get("n")

    audit = shortcut_audit(spec, gate_rows, preds, golds)
    gold_section, gold_warnings = _gold_sections(spec, gate_rows, preds)

    return {
        "function": spec.name,
        "base_model": spec.train.base,
        "precision": training.get("precision"),
        "headline": headline,
        "gate": gate,
        "gold": gold_section,
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
        "shortcut_audit": {"slices": audit["slices"], "surface": audit["surface"]},
        "warnings": audit["warnings"]
        + gold_warnings
        + (
            [
                f"eval batch size reduced to {adapter['eval_batch_size_effective']} "
                "after OOM — set train.eval_batch_size to avoid the retry cost"
            ]
            if adapter.get("eval_batch_size_effective") is not None
            else []
        ),
    }


def _gold_sections(
    spec: FunctionSpec, gate_rows: list[Row], preds: list
) -> tuple[Optional[dict], list[str]]:
    """Three-way breakdown over the gate rows that carry a gold annotation:
    teacher-vs-gold (is the teacher even right?), student-vs-gold (is the
    compiled function right?), student-vs-teacher (imitation — the verdict's
    basis). Returns (section, warnings). The PASS/FAIL verdict itself remains
    teacher agreement; gold is independent evidence."""
    idx = [i for i, r in enumerate(gate_rows) if r.get("gold") is not None]
    if not idx or len(preds) < len(gate_rows):
        return None, []
    from . import metrics as m

    gold_refs = [gate_rows[i]["gold"] for i in idx]
    teacher_labels = [row_output(spec, gate_rows[i]) for i in idx]
    student_preds = [preds[i] for i in idx]
    section = {
        "n": len(idx),
        "teacher_vs_gold": m.compare(spec, teacher_labels, gold_refs),
        "student_vs_gold": m.compare(spec, student_preds, gold_refs),
        "student_vs_teacher": m.compare(spec, student_preds, teacher_labels),
    }
    warnings = []
    t_agree = section["teacher_vs_gold"]["agreement"]
    if t_agree < spec.gate.threshold:
        warnings.append(
            f"teacher agrees with gold only {t_agree:.0%} on {len(idx)} gold "
            "row(s) — a teacher-agreement PASS is NOT independent quality "
            "validation for this function; fix the rubric or the teacher"
        )
    return section, warnings


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
        + (f" · MAE {h['mae']}" if h.get("mae") is not None else "")
    )
    if "zeroshot_agreement" in h:
        lines.append(f"zero-shot base: {_pct(h['zeroshot_agreement'])} agreement")
    const = h.get("constant_baseline")
    if const:
        lines.append(
            f"best constant baseline: predicting \"{const['value']}\" scores "
            f"{_pct(const['agreement'])} agreement"
        )
    tsa = h.get("teacher_self_agreement")
    if tsa:
        ratio = h.get("agreement", 0) / tsa if tsa else 0
        lines.append(
            f"teacher self-agreement: {_pct(tsa)} (n={h.get('teacher_probe_n')}) — "
            f"the student is at {ratio:.0%} of the teacher's own ceiling"
        )
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

    if report.get("gold"):
        g = report["gold"]
        lines.append(f"## Gold labels (n={g['n']})")
        lines.append(
            "Independent evidence — the PASS/FAIL verdict above is teacher "
            "agreement (imitation), not ground truth."
        )
        lines += ["", "| comparison | agreement | 95% CI | invalid |", "|---|---|---|---|"]
        for key, label in (
            ("teacher_vs_gold", "teacher vs gold"),
            ("student_vs_gold", "student vs gold"),
            ("student_vs_teacher", "student vs teacher"),
        ):
            c = g[key]
            ci = c.get("agreement_ci")
            ci_txt = f"{ci[0]:.0%}–{ci[1]:.0%}" if ci else "-"
            lines.append(
                f"| {label} | {c['agreement']:.1%} | {ci_txt} | "
                f"{_pct(c.get('invalid_rate'))} |"
            )
        lines.append("")

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

    audit = report.get("shortcut_audit") or {}
    if audit.get("slices") or audit.get("surface"):
        lines.append("## Shortcut audit")
        if audit.get("slices"):
            lines += ["", "| slice | n | agreement | MAE |", "|---|---|---|---|"]
            for s in audit["slices"]:
                mae = s["mae"] if s["mae"] is not None else "-"
                lines.append(
                    f"| {s['slice']} | {s['n']} | {s['agreement']:.0%} | {mae} |"
                )
            lines.append("")
        if audit.get("surface"):
            lines.append(
                "Surface-feature correlations (a student ρ well above the "
                "teacher's is a shortcut signal):"
            )
            lines += ["", "| surface feature | n | teacher ρ | student ρ | gap |",
                      "|---|---|---|---|---|"]
            for s in audit["surface"]:
                flag = " ⚠" if s["flag"] else ""
                lines.append(
                    f"| {s['feature']} | {s['n']} | {s['teacher_rho']:.2f} | "
                    f"{s['student_rho']:.2f} | {s['gap']:.2f}{flag} |"
                )
            lines.append("")

    # warnings render regardless of which sections exist
    for w in report.get("warnings") or []:
        lines.append(f"> ⚠ {w}")
    if report.get("warnings"):
        lines.append("")

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
