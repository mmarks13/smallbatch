"""Preflight prompt-first specs, decision data, and candidate requirements."""

from __future__ import annotations

import importlib.util
import json
import shutil
from pathlib import Path

from .labeling import normalize_item_records
from .spec import FunctionSpec, LoraCandidateSpec, SetFitCandidateSpec

Finding = tuple[str, str]


def inspect_spec(spec: FunctionSpec) -> list[Finding]:
    findings: list[Finding] = [("ok", "prompt-first spec validated")]
    for name, candidate in spec.candidates.items():
        if isinstance(candidate, SetFitCandidateSpec):
            level = "ok" if importlib.util.find_spec("setfit") else "fail"
            findings.append((level, f"{name}: SetFit dependency {'available' if level == 'ok' else 'missing'}"))
        elif isinstance(candidate, LoraCandidateSpec):
            findings.append(("ok", f"{name}: LoRA base {candidate.model}, precision {candidate.precision}"))
        else:
            findings.append(("ok", f"{name}: CPU TF-IDF candidate"))
    return findings


def inspect_items(spec: FunctionSpec, records: list[dict]) -> list[Finding]:
    try:
        _inputs, outputs = normalize_item_records(spec, records)
    except ValueError as exc:
        return [("fail", str(exc))]
    source = "imported decisions" if outputs is not None else "unlabeled teacher inputs"
    findings = [("ok", f"{len(records)} valid {source}")]
    if len(records) < 20:
        findings.append(
            ("warn", "fewer than 20 rows can leave calibration, dev, or evaluation evidence too small")
        )
    if outputs is None and spec.teacher is None:
        findings.append(("fail", "unlabeled inputs require a teacher block"))
    if outputs is not None and spec.augmentation and spec.teacher is None:
        findings.append(("fail", "augmentation of imported decisions requires a teacher"))
    return findings


def inspect_data(spec: FunctionSpec, data_dir: Path) -> list[Finding]:
    meta = data_dir / "meta.json"
    if not meta.exists():
        return [("warn", f"no decision dataset at {data_dir}")]
    value = json.loads(meta.read_text())
    findings = []
    if value.get("schema_version") != 3:
        findings.append(("fail", "dataset is pre-v0.2 and must be regenerated"))
    elif value.get("decision_hash") != spec.decision_hash():
        findings.append(("fail", "dataset belongs to a different prompt, contract, or teacher"))
    else:
        findings.append(("ok", f"decision dataset matches: {value.get('counts')}"))
    for split in ("train", "dev", "eval"):
        if not (data_dir / f"{split}.jsonl").exists():
            findings.append(("fail", f"missing {split}.jsonl"))
    return findings


def inspect_environment(spec: FunctionSpec) -> list[Finding]:
    free = shutil.disk_usage(Path.cwd()).free / 1024**3
    findings = [("ok" if free >= 5 else "warn", f"disk free: {free:.1f} GiB")]
    if any(isinstance(candidate, LoraCandidateSpec) for candidate in spec.candidates.values()):
        try:
            import torch

            if torch.cuda.is_available():
                findings.append(("ok", f"CUDA training device: {torch.cuda.get_device_name()}"))
            else:
                findings.append(("warn", "LoRA training has no CUDA device and may be very slow"))
        except ImportError:
            findings.append(("fail", "Torch is missing for LoRA candidates"))
    return findings


def probe_teacher(spec: FunctionSpec, records: list[dict] | None) -> list[Finding]:
    if not records or spec.teacher is None:
        return []
    try:
        inputs, outputs = normalize_item_records(spec, records)
        if outputs is not None:
            return [("ok", "teacher probe unnecessary for imported decisions")]
        from .prompts import teacher_label_prompt
        from .teacher import make_teacher

        response = make_teacher(spec.teacher).complete(teacher_label_prompt(spec, inputs[:1]))
        return [("ok", f"teacher responded ({len(response)} characters)")]
    except Exception as exc:
        return [("fail", f"teacher probe failed: {type(exc).__name__}: {exc}")]


def run_doctor(
    spec: FunctionSpec,
    items: list[dict] | None = None,
    data_dir: Path | None = None,
    probe: bool = True,
) -> int:
    findings = inspect_spec(spec)
    if items is not None:
        findings.extend(inspect_items(spec, items))
    if data_dir is not None:
        findings.extend(inspect_data(spec, data_dir))
    findings.extend(inspect_environment(spec))
    if probe:
        findings.extend(probe_teacher(spec, items))
    for level, message in findings:
        print(f"{level.upper():4}  {message}")
    return 1 if any(level == "fail" for level, _ in findings) else 0
