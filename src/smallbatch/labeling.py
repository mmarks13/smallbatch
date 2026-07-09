"""Dataset generation: teacher labels real items, generates band-targeted
variants for coverage, and everything is relabeled through one path."""

from __future__ import annotations

import datetime
import hashlib
import json
import random
from pathlib import Path
from typing import Any, Optional

from . import prompts
from .spec import FunctionSpec
from .teacher import Teacher

# {"id", "input": {...}, "score", "reason", "origin", "split", ...provenance;
# variant rows also carry "source_ids": the train reals shown to the teacher
# when the variant was generated}
Row = dict[str, Any]


def _primary_field(spec: FunctionSpec):
    """The first declared output field: what histograms, stratified splits,
    and variant band-targeting key on for multi-field contracts."""
    return next(iter(spec.output.fields.values()))


def _score_values(spec: FunctionSpec) -> list[Any]:
    return _primary_field(spec).values()


def row_output(spec: FunctionSpec, row: Row) -> Any:
    """A row's labeled output: bare value (scalar) or {field: value} dict."""
    return row["score"] if spec.output.is_scalar else row["output"]


def primary_value(spec: FunctionSpec, row: Row) -> Any:
    out = row_output(spec, row)
    return out[next(iter(spec.output.fields))] if isinstance(out, dict) else out


def _coerce_valid(spec: FunctionSpec, raw: Any) -> Any:
    """Coerce + validate a teacher label against the contract.
    Returns the normalized output, or None if invalid/incomplete."""
    if spec.output.is_scalar:
        field = spec.output.scalar
        if field.type == "int":
            try:
                raw = int(raw)
            except (TypeError, ValueError):
                return None
        return raw if raw in field.values() else None
    if not isinstance(raw, dict):
        return None
    out = {}
    for name, field in spec.output.fields.items():
        v = raw.get(name)
        if field.type == "int":
            try:
                v = int(v)
            except (TypeError, ValueError):
                return None
        if v not in field.values():
            return None
        out[name] = v
    return out


def row_id(input_obj: dict) -> str:
    """Stable id for a row, keyed on its (projected) input."""
    canon = json.dumps(input_obj, sort_keys=True, ensure_ascii=False)
    return hashlib.sha256(canon.encode()).hexdigest()[:12]


def resolve_count(value: float | int, n_real: int) -> int:
    """A split target: float < 1 is a fraction of real rows, int is absolute."""
    return int(value) if isinstance(value, int) else round(n_real * value)


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
                except (KeyError, TypeError, ValueError):
                    continue
                raw = entry.get("score") if spec.output.is_scalar else entry.get("output")
                out = _coerce_valid(spec, raw)
                if 0 <= local_id < len(batch_ids) and out is not None:
                    gid = batch_ids[local_id]
                    inp = {k: items[gid].get(k) for k in spec.input_schema}
                    rows[gid] = {
                        "id": row_id(inp),
                        "input": inp,
                        "score" if spec.output.is_scalar else "output": out,
                        "reason": str(entry.get("reason", "")).strip(),
                        "origin": origin,
                    }
        pending = [i for i in range(len(items)) if i not in rows]
    if pending:
        print(f"warning: {len(pending)} items failed labeling and were dropped")
    return [rows[i] for i in sorted(rows)]


def plan_variant_bands(
    spec: FunctionSpec,
    real: list[Row],
    target_total: int,
    n_new: Optional[int] = None,
) -> dict[Any, int]:
    """How many variants to request per score value: fill toward uniform
    coverage so rare bands exist in training. `n_new` overrides the count
    derived from target_total (used for balance-driven top-ups on an already
    full dataset). Variants are relabeled afterward, so these are targets,
    not labels."""
    values = _score_values(spec)
    needed = n_new if n_new is not None else max(0, target_total - len(real))
    if needed <= 0:
        return {}
    counts = {v: sum(1 for r in real if primary_value(spec, r) == v) for v in values}
    per_bin = (len(real) + needed) / len(values)
    deficits = {v: max(0.0, per_bin - counts[v]) for v in values}
    total_deficit = sum(deficits.values()) or 1.0
    plan = {v: round(needed * d / total_deficit) for v, d in deficits.items() if d > 0}
    return {v: n for v, n in plan.items() if n > 0}


def generate_variants(
    teacher: Teacher,
    spec: FunctionSpec,
    real: list[Row],
    target_total: int,
    n_new: Optional[int] = None,
    per_call: int = 20,
    seed: int = 17,
    hist_rows: Optional[list[Row]] = None,
) -> tuple[list[dict], list[list[str]]]:
    """Generate band-targeted synthetic items from `real` example rows.

    `real` must be TRAIN-split real rows only: the style examples shown to the
    teacher leak into the variants, so dev/gate rows must never appear here.
    `hist_rows` (default: `real`) is what band coverage is measured against —
    pass all train rows so existing variants count toward their bands.
    Returns (items, source_ids) — source_ids[i] lists the ids of the example
    rows the teacher saw when writing items[i].
    """
    rng = random.Random(seed)
    spec_text = spec.spec_files_text()
    plan = plan_variant_bands(spec, hist_rows if hist_rows is not None else real,
                              target_total, n_new=n_new)
    variants: list[dict] = []
    sources: list[list[str]] = []
    for band, n in plan.items():
        remaining = n
        while remaining > 0:
            count = min(per_call, remaining)
            examples = rng.sample(real, min(3, len(real)))
            example_ids = [r.get("id", row_id(r["input"])) for r in examples]
            reply = teacher.complete(
                prompts.teacher_variant_prompt(
                    spec, [r["input"] for r in examples], str(band), count, spec_text
                )
            )
            try:
                new_items = prompts.extract_json(reply)
            except ValueError:
                remaining -= count  # give up on this chunk rather than loop forever
                continue
            usable = [it for it in new_items if isinstance(it, dict)][:count]
            variants.extend(usable)
            sources.extend([example_ids] * len(usable))
            remaining -= count
    return variants, sources


def _score_key(r: Row) -> Any:
    # stratification key: the bare score, or the first field of a multi-field
    # output (callers with a spec in hand pass key=primary_value instead)
    out = r.get("score", r.get("output"))
    return next(iter(out.values())) if isinstance(out, dict) else out


def split_holdout(rows: list[Row], frac: float, seed: int = 17) -> tuple[list[Row], list[Row]]:
    """Stratified holdout drawn from REAL rows only, so the gate is judged on
    the true input distribution, never on synthetic variants."""
    rng = random.Random(seed)
    real = [r for r in rows if r["origin"] == "real"]
    by_score: dict[Any, list[Row]] = {}
    for r in real:
        by_score.setdefault(_score_key(r), []).append(r)
    holdout: list[Row] = []
    for group in by_score.values():
        rng.shuffle(group)
        k = round(len(group) * frac)
        holdout.extend(group[:k])
    holdout_ids = {id(r) for r in holdout}
    train = [r for r in rows if id(r) not in holdout_ids]
    return train, holdout


def _stratified_take(pool: list[Row], k: int, rng: random.Random) -> list[Row]:
    """Take ~k rows from pool, stratified by score, exactly k when possible."""
    if k <= 0 or not pool:
        return []
    k = min(k, len(pool))
    by_score: dict[Any, list[Row]] = {}
    for r in pool:
        by_score.setdefault(_score_key(r), []).append(r)
    taken: list[Row] = []
    for group in by_score.values():
        rng.shuffle(group)
        taken.extend(group[: round(len(group) * k / len(pool))])
    rng.shuffle(taken)
    taken = taken[:k]
    if len(taken) < k:  # rounding shortfall: top up from anywhere
        taken_ids = {id(r) for r in taken}
        rest = [r for r in pool if id(r) not in taken_ids]
        rng.shuffle(rest)
        taken.extend(rest[: k - len(taken)])
    return taken


def assign_splits(rows: list[Row], gate_target: int, dev_target: int, seed: int = 17) -> None:
    """Assign a `split` (train/dev/gate) to every row that lacks one.

    Sticky: a row that already has a split keeps it — the gate must never be
    reshuffled once anything trained against the rest of the data. Only REAL
    rows are eligible for gate/dev (and, because assignment happens before
    variant generation, gate/dev rows never have variant siblings in train).
    Variants always land in train.
    """
    rng = random.Random(seed)
    for r in rows:
        if r["origin"] != "real":
            r["split"] = r.get("split") or "train"
    reals = [r for r in rows if r["origin"] == "real"]
    pool = [r for r in reals if not r.get("split")]
    for split_name, target in (("gate", gate_target), ("dev", dev_target)):
        have = sum(1 for r in reals if r.get("split") == split_name)
        for r in _stratified_take(pool, target - have, rng):
            r["split"] = split_name
        pool = [r for r in pool if not r.get("split")]
    for r in pool:
        r["split"] = "train"


def _migrate_legacy_splits(rows: list[Row], out_dir: Path) -> None:
    """Give pre-v0.2 datasets ids and splits: holdout.jsonl members become the
    (sticky) gate, everything else train. Dev gets carved from new rows later."""
    for r in rows:
        r.setdefault("id", row_id(r["input"]))
    if all(r.get("split") for r in rows):
        return
    legacy_holdout = out_dir / "holdout.jsonl"
    gate_ids = (
        {row_id(r["input"]) for r in read_jsonl(legacy_holdout)}
        if legacy_holdout.exists()
        else set()
    )
    for r in rows:
        if not r.get("split"):
            r["split"] = "gate" if r["id"] in gate_ids else "train"


def write_dataset(spec: FunctionSpec, rows: list[Row], out_dir: Path) -> dict[str, Any]:
    """Write labeled/train/dev/gate JSONL + meta for already-split rows."""
    out_dir.mkdir(parents=True, exist_ok=True)
    by_split = {s: [r for r in rows if r["split"] == s] for s in ("train", "dev", "gate")}
    _write_jsonl(out_dir / "labeled.jsonl", rows)
    for split_name, split_rows in by_split.items():
        _write_jsonl(out_dir / f"{split_name}.jsonl", split_rows)

    hist = {
        str(v): sum(1 for r in rows if primary_value(spec, r) == v)
        for v in _score_values(spec)
    }
    meta = {
        "function": spec.name,
        "spec_hash": spec.spec_hash(),
        "real": sum(1 for r in rows if r["origin"] == "real"),
        "variants": sum(1 for r in rows if r["origin"] == "variant"),
        "train": len(by_split["train"]),
        "dev": len(by_split["dev"]),
        "gate": len(by_split["gate"]),
        "label_histogram": hist,
        "teacher_model": spec.teacher.model,
        "teacher_backend": spec.teacher.backend,
        "prompt_version": prompts.PROMPT_VERSION,
        "labeled_at": datetime.date.today().isoformat(),
    }
    (out_dir / "meta.json").write_text(json.dumps(meta, indent=2))
    return meta


def build_dataset(
    teacher: Teacher,
    spec: FunctionSpec,
    items: list[dict],
    out_dir: Path,
    append: bool = False,
    max_variants: Optional[int] = None,
) -> dict[str, Any]:
    """The full labeling pipeline. Writes labeled/train/dev/gate JSONL + meta.

    With `append`, existing rows (and their split assignments — the gate is
    sticky) are kept; `items` already present are skipped. Splits are assigned
    BEFORE variants are generated, and variants are generated from train-split
    reals only, so gate/dev never contain or influence synthetic data.
    `max_variants` caps this call's newly generated variants (balance-driven).
    """
    out_dir.mkdir(parents=True, exist_ok=True)
    existing: list[Row] = []
    if append and (out_dir / "labeled.jsonl").exists():
        existing = read_jsonl(out_dir / "labeled.jsonl")
        _migrate_legacy_splits(existing, out_dir)

    known = {r["id"] for r in existing}
    new_items = []
    for it in items:
        proj = {k: it.get(k) for k in spec.input_schema}
        rid = row_id(proj)
        if rid not in known:
            known.add(rid)  # also dedupes within `items` itself
            new_items.append(it)

    stamp = {
        "teacher_model": spec.teacher.model,
        "teacher_backend": spec.teacher.backend,
        "prompt_version": prompts.PROMPT_VERSION,
        "labeled_at": datetime.date.today().isoformat(),
    }
    real_new = label_items(teacher, spec, new_items, origin="real") if new_items else []
    for r in real_new:
        r.update(stamp)
    rows = existing + real_new
    if not any(r["origin"] == "real" for r in rows):
        raise RuntimeError("labeling produced no usable rows")

    n_real = sum(1 for r in rows if r["origin"] == "real")
    assign_splits(
        rows,
        gate_target=resolve_count(spec.teacher.holdout, n_real),
        dev_target=resolve_count(spec.teacher.dev, n_real),
        seed=spec.train.seed,
    )

    train_reals = [r for r in rows if r["origin"] == "real" and r["split"] == "train"]
    train_rows = [r for r in rows if r["split"] == "train"]
    variants_raw, sources = generate_variants(
        teacher, spec, train_reals, spec.teacher.examples,
        n_new=max_variants, seed=spec.train.seed, hist_rows=train_rows,
    )
    variant_rows = (
        label_items(teacher, spec, variants_raw, origin="variant") if variants_raw else []
    )
    src_by_id = {
        row_id({k: it.get(k) for k in spec.input_schema}): src
        for it, src in zip(variants_raw, sources)
    }
    for r in variant_rows:
        r["split"] = "train"
        r["source_ids"] = src_by_id.get(r["id"], [])
        r.update(stamp)

    return write_dataset(spec, rows + variant_rows, out_dir)


def _write_jsonl(path: Path, rows: list[Row]) -> None:
    with path.open("w") as f:
        for r in rows:
            f.write(json.dumps(r, ensure_ascii=False) + "\n")


def read_jsonl(path: Path) -> list[Row]:
    return [json.loads(line) for line in path.read_text().splitlines() if line.strip()]
