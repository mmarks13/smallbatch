"""Preflight prompt-first specs, decision data, and candidate requirements."""

from __future__ import annotations

import importlib.util
import json
import math
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


def _encoder_tokenizer(model_id: str):
    """The encoder's tokenizer from local files only — doctor never downloads."""
    try:
        from transformers import AutoTokenizer

        return AutoTokenizer.from_pretrained(model_id, local_files_only=True)
    except Exception:
        return None


def _encoder_sequence_limit(model_id: str) -> int | None:
    """The SentenceTransformer input window (its own config truncates harder
    than the tokenizer's limit on some models), from local files only."""
    config_name = "sentence_bert_config.json"
    try:
        local = Path(model_id) / config_name
        if local.exists():
            value = json.loads(local.read_text()).get("max_seq_length")
            return int(value) if value else None
        from huggingface_hub import try_to_load_from_cache

        cached = try_to_load_from_cache(model_id, config_name)
        if isinstance(cached, str):
            value = json.loads(Path(cached).read_text()).get("max_seq_length")
            return int(value) if value else None
    except Exception:
        return None
    return None


def inspect_setfit_truncation(spec: FunctionSpec, records: list[dict]) -> list[Finding]:
    """SetFit encoders truncate silently: an item longer than the encoder's
    window loses its tail with no error anywhere, while TF-IDF reads all of
    it. Doctor is the only place that says so before training."""
    from . import prompts

    models = sorted(
        {
            candidate.model
            for candidate in spec.candidates.values()
            if isinstance(candidate, SetFitCandidateSpec)
        }
    )
    if not models or not records:
        return []
    try:
        inputs, _outputs = normalize_item_records(spec, records)
    except ValueError:
        return []  # inspect_items already failed these records
    texts = [prompts.render_input(item, spec.input_schema) for item in inputs]
    findings: list[Finding] = []
    for model_id in models:
        tokenizer = _encoder_tokenizer(model_id)
        if tokenizer is None:
            findings.append(
                ("warn", f"{model_id}: not cached locally; cannot check input truncation")
            )
            continue
        limit = _encoder_sequence_limit(model_id)
        if limit is None:
            claimed = int(getattr(tokenizer, "model_max_length", 0) or 0)
            limit = claimed if 0 < claimed < 100_000 else None
        if limit is None:
            findings.append(
                ("warn", f"{model_id}: encoder input window unknown; cannot check truncation")
            )
            continue
        lengths = [
            len(tokenizer(text, add_special_tokens=True)["input_ids"]) for text in texts
        ]
        truncated = sum(length > limit for length in lengths)
        if truncated:
            findings.append(
                (
                    "warn",
                    f"{model_id}: {truncated}/{len(lengths)} items exceed its "
                    f"{limit}-token window and would be silently truncated "
                    f"(longest {max(lengths)} tokens)",
                )
            )
        else:
            findings.append(
                (
                    "ok",
                    f"{model_id}: all items fit its {limit}-token window "
                    f"(longest {max(lengths)} tokens)",
                )
            )
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
    if outputs is None and spec.teacher is not None and spec.teacher.passes == 1:
        findings.append(
            (
                "warn",
                "teacher.passes: 1 draws each decision once, so the teacher's "
                "self-agreement ceiling stays unknown; passes: 2 measures it and "
                "tie-breaks unstable decisions",
            )
        )
    if outputs is not None and spec.augmentation and spec.teacher is None:
        findings.append(("fail", "augmentation of imported decisions requires a teacher"))
    findings.extend(_text_headroom(spec, outputs))
    return findings


# Warn when the p95 reference length crowds the declared limit. The contract
# is never changed and nothing fails while every reference is valid — but a
# limit this tight means generation will regularly run out of room.
TEXT_HEADROOM_FRACTION = 0.8


def _text_headroom(spec: FunctionSpec, outputs: list | None) -> list[Finding]:
    text_field = spec.output.text_field
    if text_field is None or not outputs:
        return []
    values = [
        output[text_field] if isinstance(output, dict) else output
        for output in outputs
    ]
    lengths = sorted(len(value) for value in values)
    p95 = lengths[min(len(lengths) - 1, math.ceil(0.95 * len(lengths)) - 1)]
    limit = spec.output.fields[text_field].max_chars
    if p95 > TEXT_HEADROOM_FRACTION * limit:
        return [
            (
                "warn",
                f"text field {text_field!r}: p95 reference length is {p95} of "
                f"max_chars {limit} (over {TEXT_HEADROOM_FRACTION:.0%}); "
                "generation will often run out of room — raise max_chars or "
                "ask the teacher for shorter text",
            )
        ]
    return [
        (
            "ok",
            f"text field {text_field!r}: p95 reference length {p95} fits "
            f"max_chars {limit}",
        )
    ]


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
    unresolved = data_dir / "unresolved.jsonl"
    if unresolved.exists():
        count = sum(1 for line in unresolved.read_text().splitlines() if line.strip())
        if count:
            findings.append(
                (
                    "warn",
                    f"{count} unresolved decisions in {unresolved} — resolution is "
                    "optional; fill in `output` per kept line and relabel with "
                    "--append, or tighten the prompt and relabel",
                )
            )
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
        findings.extend(inspect_setfit_truncation(spec, items))
    if data_dir is not None:
        findings.extend(inspect_data(spec, data_dir))
    findings.extend(inspect_environment(spec))
    if probe:
        findings.extend(probe_teacher(spec, items))
    for level, message in findings:
        print(f"{level.upper():4}  {message}")
    return 1 if any(level == "fail" for level, _ in findings) else 0
