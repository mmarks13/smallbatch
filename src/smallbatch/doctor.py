"""`smallbatch doctor <spec>`: preflight a compile before spending teacher
calls or GPU time.

Every check returns (level, message) findings — "ok", "warn", or "fail" —
so the pure checks are unit-testable; the CLI runner prints them and exits
1 only on hard failures. Heavy imports (torch) stay inside functions.
"""

from __future__ import annotations

import json
import os
import shutil as _shutil
from pathlib import Path
from typing import Optional

from . import prompts
from .labeling import read_jsonl, resolve_count
from .report import SMALL_GATE_N
from .spec import FunctionSpec

Finding = tuple[str, str]  # ("ok" | "warn" | "fail", message)


def contract_findings(spec: FunctionSpec) -> list[Finding]:
    out: list[Finding] = []
    fields = spec.output.fields
    kinds = ", ".join(f"{n}({f.type})" for n, f in fields.items())
    out.append(("ok", f"output contract: {len(fields)} field(s) — {kinds}"))
    completions = prompts.allowed_completions(spec)
    if spec.train.rationale_distillation:
        out.append(("warn", "rationale mode: decoding is unconstrained (parse-validated only)"))
    elif completions is None:
        out.append((
            "warn",
            "output cross product too large to constrain at decode time — "
            "outputs are parse-validated only (exported GBNF stays exact)",
        ))
    else:
        out.append(("ok", f"constrained decoding: {len(completions)} legal completions"))
    for name, field in fields.items():
        if field.type == "int":
            lo, hi = field.range
            if hi - lo + 1 > 1000:
                out.append(("fail", f"field {name}: int range {lo}..{hi} too large for a grammar"))
        elif len(field.labels) > 50:
            out.append(("warn", f"field {name}: {len(field.labels)} labels — teachers get sloppy past ~50"))
    return out


def items_findings(spec: FunctionSpec, items: list[dict]) -> list[Finding]:
    out: list[Finding] = []
    if not items:
        return [("fail", "items file is empty")]
    missing_counts = {k: 0 for k in spec.input_schema}
    for it in items:
        for k in spec.input_schema:
            if k not in it or it[k] in (None, ""):
                missing_counts[k] += 1
    out.append(("ok", f"{len(items)} items"))
    for k, miss in missing_counts.items():
        if miss == len(items):
            out.append(("fail", f"input field '{k}' missing from every item"))
        elif miss:
            out.append(("warn", f"input field '{k}' missing/empty in {miss}/{len(items)} items"))
    n_real = len(items)
    gate = resolve_count(spec.teacher.holdout, n_real)
    dev = resolve_count(spec.teacher.dev, n_real)
    train = n_real - gate - dev
    out.append(("ok" if train > 0 else "fail",
                f"planned split of reals: {train} train / {dev} dev / {gate} gate"))
    if gate < SMALL_GATE_N:
        out.append(("warn",
                    f"gate split would be {gate} < {SMALL_GATE_N} items — the verdict will be "
                    "noise-dominated (CI is shown, but more real items would help)"))
    batches = -(-len(items) // spec.teacher.batch_size)
    out.append(("ok", f"teacher budget: ~{batches} labeling call(s) for provided items"))
    if spec.augment:
        paraphrase = spec.augment.paraphrase.cap if spec.augment.paraphrase else 0
        dropout = (
            spec.augment.field_dropout.cap * len(spec.augment.field_dropout.fields)
            if spec.augment.field_dropout
            else 0
        )
        counterfactual = spec.augment.counterfactual.cap if spec.augment.counterfactual else 0
        retry = counterfactual
        out.append((
            "ok",
            "augmentation plan: up to "
            f"{paraphrase} paraphrases + {dropout} field dropouts + "
            f"{counterfactual} counterfactuals (+ up to {retry} retries); "
            "generation-call count depends on the observed label histogram",
        ))
    else:
        variants = max(0, spec.teacher.examples - n_real)
        variant_calls = -(-variants // 20) + -(-variants // spec.teacher.batch_size)
        if variants:
            out.append(("ok", f"legacy augmentation: ~{variant_calls} calls for ~{variants} variants"))
    return out


def data_findings(spec: FunctionSpec, data_dir: Path) -> list[Finding]:
    out: list[Finding] = []
    if not (data_dir / "labeled.jsonl").exists():
        return [("warn", f"no labeled dataset under {data_dir} — run `smallbatch label` first")]
    sizes = {}
    for split in ("train", "dev", "gate"):
        p = data_dir / f"{split}.jsonl"
        if not p.exists() and split == "gate":
            p = data_dir / "holdout.jsonl"  # pre-v0.2 layout
        sizes[split] = len(read_jsonl(p)) if p.exists() else 0
    out.append(("ok", f"dataset splits: {sizes['train']} train / {sizes['dev']} dev / {sizes['gate']} gate"))
    if sizes["gate"] == 0:
        out.append(("fail", "gate split is empty"))
    elif sizes["gate"] < SMALL_GATE_N:
        out.append(("warn", f"gate split has {sizes['gate']} < {SMALL_GATE_N} items — verdict is noisy"))
    if sizes["dev"] == 0:
        out.append(("warn", "no dev split — checkpoint selection will carve one from train at compile time"))
    meta_path = data_dir / "meta.json"
    if meta_path.exists():
        meta = json.loads(meta_path.read_text())
        hist = meta.get("label_histogram", {})
        empty = [k for k, v in hist.items() if v == 0]
        if empty:
            out.append(("warn", f"label bands with zero examples: {', '.join(empty)}"))
        if meta.get("spec_hash") and meta["spec_hash"] != spec.spec_hash():
            out.append(("warn", "dataset was labeled under a different spec/spec_files version"))
    return out


def teacher_findings(spec: FunctionSpec, probe: bool = True) -> list[Finding]:
    out: list[Finding] = []
    if spec.teacher.backend == "claude-cli":
        if not _shutil.which("claude"):
            return [("fail", "teacher backend claude-cli: `claude` not found on PATH")]
        out.append(("ok", "claude CLI found"))
    elif spec.teacher.backend == "codex-cli":
        if not _shutil.which("codex"):
            return [("fail", "teacher backend codex-cli: `codex` not found on PATH")]
        out.append(("ok", "Codex CLI found"))
    else:
        if not spec.teacher.base_url:
            return [("fail", "teacher backend openai-compatible needs base_url")]
        out.append(("ok", f"teacher endpoint: {spec.teacher.base_url}"))
    if not probe:
        out.append(("warn", "teacher probe skipped (--no-probe)"))
        return out
    from .teacher import make_teacher

    fake_item = {k: f"<{k} probe>" for k in spec.input_schema}
    try:
        reply = make_teacher(spec.teacher).complete(
            prompts.teacher_label_prompt(spec, [fake_item], spec.spec_files_text())
        )
        labels = prompts.extract_json(reply)
        entry = labels[0] if isinstance(labels, list) and labels else {}
        raw = entry.get("score") if spec.output.is_scalar else entry.get("output")
        from .labeling import _coerce_valid

        if _coerce_valid(spec, raw) is not None:
            out.append(("ok", "teacher probe: 1 item labeled and parsed against the contract"))
        else:
            out.append(("warn", f"teacher probe: reply parsed but label invalid: {str(raw)[:80]!r}"))
    except Exception as e:  # noqa: BLE001 - report, don't crash a diagnostic
        out.append(("fail", f"teacher probe failed: {type(e).__name__}: {str(e)[:200]}"))
    return out


def hardware_findings(spec: FunctionSpec) -> list[Finding]:
    out: list[Finding] = []
    try:
        import torch
    except ImportError:
        return [("fail", "torch not installed — compile/run need it (label does not)")]
    if not torch.cuda.is_available():
        out.append(("warn", "CUDA unavailable — training on CPU is impractical "
                            "(on Pascal cards check the torch/cuda wheel pin)"))
        return out
    name = torch.cuda.get_device_name(0)
    cap = torch.cuda.get_device_capability(0)
    vram = torch.cuda.get_device_properties(0).total_memory / 1e9
    out.append(("ok", f"GPU: {name} (sm_{cap[0]}{cap[1]}, {vram:.0f}GB)"))
    from .hardware import pick_precision

    resolved = pick_precision(spec.train.precision)
    out.append(("ok", f"precision {spec.train.precision} -> {resolved}"))
    if resolved == "qlora" or spec.train.precision == "qlora":
        try:
            import bitsandbytes as bnb

            out.append(("ok", f"bitsandbytes {bnb.__version__}"))
        except ImportError:
            out.append(("fail", "precision qlora needs bitsandbytes>=0.46.1"))
    return out


def disk_findings(min_free_gb: float = 30) -> list[Finding]:
    free = _shutil.disk_usage(".").free / 1e9
    hf_cache = os.environ.get("HF_HOME", str(Path.home() / ".cache" / "huggingface"))
    level = "ok" if free >= min_free_gb else "warn"
    msg = f"disk: {free:.0f}GB free (HF cache: {hf_cache})"
    if level == "warn":
        msg += f" — under {min_free_gb:.0f}GB; model downloads mid-sweep can fail on this"
    return [(level, msg)]


def export_findings() -> list[Finding]:
    out: list[Finding] = []
    llama = os.environ.get("LLAMA_CPP_DIR")
    if llama and Path(llama).is_dir():
        out.append(("ok", f"LLAMA_CPP_DIR: {llama}"))
    else:
        out.append(("warn", "LLAMA_CPP_DIR not set — `smallbatch export`/`serve` need a llama.cpp checkout"))
    try:
        import gguf  # noqa: F401

        out.append(("ok", "gguf package installed"))
    except ImportError:
        out.append(("warn", "gguf package missing — `smallbatch export` needs `pip install gguf`"))
    return out


_ICONS = {"ok": "  ok ", "warn": "WARN ", "fail": "FAIL "}


def run_doctor(
    spec: FunctionSpec,
    items: Optional[list[dict]] = None,
    data_dir: Optional[Path] = None,
    probe: bool = True,
) -> int:
    sections: list[tuple[str, list[Finding]]] = [
        ("output contract", contract_findings(spec)),
        ("teacher", teacher_findings(spec, probe=probe)),
        ("hardware", hardware_findings(spec)),
        ("disk", disk_findings()),
        ("export", export_findings()),
    ]
    if items is not None:
        sections.insert(1, ("items", items_findings(spec, items)))
    if data_dir is not None:
        sections.insert(1, ("dataset", data_findings(spec, data_dir)))

    worst = "ok"
    for title, findings in sections:
        print(f"-- {title}")
        for level, msg in findings:
            print(f"{_ICONS[level]} {msg}")
            if level == "fail" or (level == "warn" and worst == "ok"):
                worst = level
    print()
    if worst == "fail":
        print("doctor: FAIL — fix the failures above before label/compile")
        return 1
    print("doctor: ok" + (" (with warnings)" if worst == "warn" else ""))
    return 0
