"""Dataset generation: teacher labels real items, generates band-targeted
variants for coverage, and everything is relabeled through one path."""

from __future__ import annotations

import datetime
import json
import random
from pathlib import Path
from typing import Any

from . import prompts
from .spec import FunctionSpec
from .teacher import Teacher

Row = dict[str, Any]  # {"input": {...}, "score": ..., "reason": str, "origin": ...}


def _score_values(spec: FunctionSpec) -> list[Any]:
    if spec.output.type == "int":
        lo, hi = spec.output.range
        return list(range(lo, hi + 1))
    return list(spec.output.labels)


def _valid(spec: FunctionSpec, score: Any) -> bool:
    return score in _score_values(spec)


def label_items(
    teacher: Teacher, spec: FunctionSpec, items: list[dict], origin: str
) -> list[Row]:
    """Label items in batches; one retry pass for missing/invalid labels."""
    spec_text = spec.spec_files_text()
    rows: dict[int, Row] = {}
    pending = list(range(len(items)))
    for attempt in range(2):
        if not pending:
            break
        for start in range(0, len(pending), spec.teacher.batch_size):
            batch_ids = pending[start : start + spec.teacher.batch_size]
            batch = [items[i] for i in batch_ids]
            reply = teacher.complete(prompts.teacher_label_prompt(spec, batch, spec_text))
            try:
                labels = prompts.extract_json(reply)
            except ValueError:
                continue  # whole batch retried on next pass
            for entry in labels:
                try:
                    local_id = int(entry["id"])
                    score = entry["score"]
                    if spec.output.type == "int":
                        score = int(score)
                except (KeyError, TypeError, ValueError):
                    continue
                if 0 <= local_id < len(batch_ids) and _valid(spec, score):
                    gid = batch_ids[local_id]
                    rows[gid] = {
                        "input": {k: items[gid].get(k) for k in spec.input_schema},
                        "score": score,
                        "reason": str(entry.get("reason", "")).strip(),
                        "origin": origin,
                    }
        pending = [i for i in range(len(items)) if i not in rows]
    if pending:
        print(f"warning: {len(pending)} items failed labeling and were dropped")
    return [rows[i] for i in sorted(rows)]


def plan_variant_bands(spec: FunctionSpec, real: list[Row], target_total: int) -> dict[Any, int]:
    """How many variants to request per score value: fill toward uniform
    coverage so rare bands exist in training. (Holdout stays real-distribution;
    variants are relabeled afterward, so these are targets, not labels.)"""
    values = _score_values(spec)
    needed = max(0, target_total - len(real))
    if needed == 0:
        return {}
    counts = {v: sum(1 for r in real if r["score"] == v) for v in values}
    per_bin = target_total / len(values)
    deficits = {v: max(0.0, per_bin - counts[v]) for v in values}
    total_deficit = sum(deficits.values()) or 1.0
    plan = {v: round(needed * d / total_deficit) for v, d in deficits.items() if d > 0}
    return {v: n for v, n in plan.items() if n > 0}


def generate_variants(
    teacher: Teacher,
    spec: FunctionSpec,
    real: list[Row],
    target_total: int,
    per_call: int = 20,
    seed: int = 17,
) -> list[dict]:
    rng = random.Random(seed)
    spec_text = spec.spec_files_text()
    plan = plan_variant_bands(spec, real, target_total)
    variants: list[dict] = []
    for band, n in plan.items():
        remaining = n
        while remaining > 0:
            count = min(per_call, remaining)
            examples = [r["input"] for r in rng.sample(real, min(3, len(real)))]
            reply = teacher.complete(
                prompts.teacher_variant_prompt(spec, examples, str(band), count, spec_text)
            )
            try:
                new_items = prompts.extract_json(reply)
            except ValueError:
                remaining -= count  # give up on this chunk rather than loop forever
                continue
            usable = [it for it in new_items if isinstance(it, dict)][:count]
            variants.extend(usable)
            remaining -= count
    return variants


def split_holdout(rows: list[Row], frac: float, seed: int = 17) -> tuple[list[Row], list[Row]]:
    """Stratified holdout drawn from REAL rows only, so the gate is judged on
    the true input distribution, never on synthetic variants."""
    rng = random.Random(seed)
    real = [r for r in rows if r["origin"] == "real"]
    by_score: dict[Any, list[Row]] = {}
    for r in real:
        by_score.setdefault(r["score"], []).append(r)
    holdout: list[Row] = []
    for group in by_score.values():
        rng.shuffle(group)
        k = round(len(group) * frac)
        holdout.extend(group[:k])
    holdout_ids = {id(r) for r in holdout}
    train = [r for r in rows if id(r) not in holdout_ids]
    return train, holdout


def build_dataset(
    teacher: Teacher, spec: FunctionSpec, items: list[dict], out_dir: Path
) -> dict[str, Any]:
    """The full labeling pipeline. Writes labeled/train/holdout JSONL + meta."""
    out_dir.mkdir(parents=True, exist_ok=True)
    real = label_items(teacher, spec, items, origin="real")
    if not real:
        raise RuntimeError("labeling produced no usable rows")

    variants_raw = generate_variants(teacher, spec, real, spec.teacher.examples)
    variant_rows = (
        label_items(teacher, spec, variants_raw, origin="variant") if variants_raw else []
    )

    rows = real + variant_rows
    stamp = {
        "teacher_model": spec.teacher.model,
        "teacher_backend": spec.teacher.backend,
        "prompt_version": prompts.PROMPT_VERSION,
        "labeled_at": datetime.date.today().isoformat(),
    }
    for r in rows:
        r.update(stamp)

    train, holdout = split_holdout(rows, spec.teacher.holdout, seed=spec.train.seed)
    _write_jsonl(out_dir / "labeled.jsonl", rows)
    _write_jsonl(out_dir / "train.jsonl", train)
    _write_jsonl(out_dir / "holdout.jsonl", holdout)

    hist = {str(v): sum(1 for r in rows if r["score"] == v) for v in _score_values(spec)}
    meta = {
        "function": spec.name,
        "spec_hash": spec.spec_hash(),
        "real": len(real),
        "variants": len(variant_rows),
        "train": len(train),
        "holdout": len(holdout),
        "label_histogram": hist,
        **stamp,
    }
    (out_dir / "meta.json").write_text(json.dumps(meta, indent=2))
    return meta


def _write_jsonl(path: Path, rows: list[Row]) -> None:
    with path.open("w") as f:
        for r in rows:
            f.write(json.dumps(r, ensure_ascii=False) + "\n")


def read_jsonl(path: Path) -> list[Row]:
    return [json.loads(line) for line in path.read_text().splitlines() if line.strip()]
