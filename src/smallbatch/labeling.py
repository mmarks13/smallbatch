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
        batch_count = -(-len(pending) // spec.teacher.batch_size)
        for start in range(0, len(pending), spec.teacher.batch_size):
            batch_number = start // spec.teacher.batch_size + 1
            batch_ids = pending[start : start + spec.teacher.batch_size]
            batch = [items[i] for i in batch_ids]
            print(
                f"labeling {origin}: pass {attempt + 1}, "
                f"batch {batch_number}/{batch_count} ({len(batch)} items)",
                flush=True,
            )
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
    cap: Optional[int] = None,
) -> dict[Any, int]:
    """How many variants to request per score value: fill toward uniform
    coverage so rare bands exist in training. `cap` is a true ceiling on the
    count derived from target_total — it can only reduce the request, never
    force generation on an already-full dataset. Variants are relabeled
    afterward, so these are targets, not labels."""
    values = _score_values(spec)
    needed = max(0, target_total - len(real))
    if cap is not None:
        needed = min(needed, cap)
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
    cap: Optional[int] = None,
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
                              target_total, cap=cap)
    variants: list[dict] = []
    sources: list[list[str]] = []
    for band, n in plan.items():
        remaining = n
        while remaining > 0:
            count = min(per_call, remaining)
            examples = rng.sample(real, min(3, len(real)))
            example_ids = [r.get("id", row_id(r["input"])) for r in examples]
            print(
                f"generating variants: target {band}, {count} item(s)", flush=True
            )
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


def generate_field_dropout(
    spec: FunctionSpec,
    train_reals: list[Row],
    fields: list[str],
    cap: int,
    rng: random.Random,
) -> tuple[list[dict], list[list[str]]]:
    """Ablation probes: train reals with one input field blanked. The caller
    RELABELS them — copying the source label would teach exactly the wrong
    thing when the field is load-bearing."""
    items: list[dict] = []
    sources: list[list[str]] = []
    for field in fields:
        pool = [
            r for r in train_reals
            if str(r["input"].get(field) or "").strip()
        ]
        for r in rng.sample(pool, min(cap, len(pool))):
            item = dict(r["input"])
            item[field] = ""
            items.append(item)
            sources.append([r["id"]])
    return items, sources


def _cf_request(
    teacher: Teacher,
    spec: FunctionSpec,
    batch: list[Row],
    band: Any,
    spec_text: str,
    feedback: Optional[str] = None,
) -> list[tuple[dict, Row]]:
    """One counterfactual-edit call: (edited item, source row) pairs."""
    reply = teacher.complete(
        prompts.teacher_counterfactual_prompt(
            spec, [r["input"] for r in batch], str(band), spec_text,
            feedback=feedback,
        )
    )
    try:
        edits = prompts.extract_json(reply)
    except ValueError:
        return []
    out = []
    for e in edits:
        if not isinstance(e, dict):
            continue
        try:
            local_id = int(e.get("id"))
        except (TypeError, ValueError):
            continue
        if 0 <= local_id < len(batch):
            out.append(({k: e.get(k) for k in spec.input_schema}, batch[local_id]))
    return out


def counterfactual_rows(
    teacher: Teacher,
    spec: FunctionSpec,
    train_reals: list[Row],
    cap: int,
    seed: int,
    hist_rows: list[Row],
    per_call: int = 5,
) -> tuple[list[Row], Optional[float]]:
    """Minimal label-moving edits of train reals, targeted at thin bands
    (deficits measured over `hist_rows`) and independently relabeled.

    An edit that does not move CLOSER to its intended band is a miss: it gets
    one stronger retry — but the miss row is kept too (it's a paid-for
    invariance example). Returns (rows, hit_rate); rows carry
    `intended_band` + `source_ids`."""
    rng = random.Random(seed)
    spec_text = spec.spec_files_text()
    # counterfactuals always trace the boundary regardless of dataset size:
    # ask the band planner for exactly `cap` rows over the observed deficits
    plan = plan_variant_bands(spec, hist_rows, target_total=len(hist_rows) + cap)
    pairs: list[tuple[dict, Row, Any]] = []
    for band, n in plan.items():
        pool = [r for r in train_reals if primary_value(spec, r) != band]
        if not pool:
            continue
        chosen = [pool[rng.randrange(len(pool))] for _ in range(n)]
        for start in range(0, len(chosen), per_call):
            print(
                f"generating counterfactuals: target {band}, "
                f"{len(chosen[start : start + per_call])} item(s)",
                flush=True,
            )
            for item, src in _cf_request(
                teacher, spec, chosen[start : start + per_call], band, spec_text
            ):
                pairs.append((item, src, band))

    def label_pairs(ps: list[tuple[dict, Row, Any]]) -> tuple[list[Row], list[tuple[Row, Any]]]:
        rows = label_items(teacher, spec, [p[0] for p in ps], origin="counterfactual")
        by_id = {
            row_id({k: it.get(k) for k in spec.input_schema}): (src, band)
            for it, src, band in ps
        }
        misses: list[tuple[Row, Any]] = []
        for r in rows:
            src, band = by_id.get(r["id"], (None, None))
            r["intended_band"] = band
            r["source_ids"] = [src["id"]] if src else []
            if src is not None and not _closer_to_band(spec, r, src, band):
                misses.append((src, band))
        return rows, misses

    if not pairs:
        return [], None
    rows, misses = label_pairs(pairs)

    retry_pairs: list[tuple[dict, Row, Any]] = []
    by_band: dict[Any, list[Row]] = {}
    for src, band in misses:
        by_band.setdefault(band, []).append(src)
    for band, srcs in by_band.items():
        for start in range(0, len(srcs), per_call):
            print(
                f"retrying counterfactuals: target {band}, "
                f"{len(srcs[start : start + per_call])} item(s)",
                flush=True,
            )
            for item, src in _cf_request(
                teacher, spec, srcs[start : start + per_call], band, spec_text,
                feedback="the edit did not move the label closer to the requested "
                         "band — make a stronger (but still minimal) change to "
                         "what the rubric scores",
            ):
                retry_pairs.append((item, src, band))
    if retry_pairs:
        retry_rows, _ = label_pairs(retry_pairs)
        rows += retry_rows

    # hit rate: fraction independently judged closer to the intended band
    src_target = {}
    for it, src, band in pairs + retry_pairs:
        src_target[row_id({k: it.get(k) for k in spec.input_schema})] = (src, band)
    hits = sum(
        1 for r in rows
        if r["id"] in src_target
        and _closer_to_band(spec, r, *src_target[r["id"]])
    )
    hit_rate = round(hits / len(rows), 4) if rows else None
    return rows, hit_rate


def _closer_to_band(spec: FunctionSpec, row: Row, source: Row, band: Any) -> bool:
    """Whether the primary label moved strictly closer to a target band."""
    actual = primary_value(spec, row)
    original = primary_value(spec, source)
    field = _primary_field(spec)
    return abs(actual - band) < abs(original - band) if field.type == "int" else actual == band


def _out_agrees(spec: FunctionSpec, a: Any, b: Any) -> bool:
    """Two outputs agree under the gate rule (±1 int / exact enum; all
    fields for multi-field contracts). Local twin of evaluate._agrees —
    evaluate.py is a torch-heavy import this module must not pull in."""
    if spec.output.is_scalar:
        f = spec.output.scalar
        return abs(a - b) <= 1 if f.type == "int" else a == b
    return all(
        (abs(a[name] - b[name]) <= 1 if f.type == "int" else a[name] == b[name])
        for name, f in spec.output.fields.items()
    )


def consistency_probe(
    teacher: Teacher, spec: FunctionSpec, rows: list[Row], n: int, seed: int = 17
) -> Optional[dict[str, Any]]:
    """Double-label a stratified sample of real rows with the input fields in
    shuffled order, and measure how often the teacher agrees with itself.
    That self-agreement is the ceiling on any student's gate agreement.

    Each probed row gains `probe_output`; the original label stays
    authoritative (review --unstable steps through the disagreements)."""
    reals = [r for r in rows if r["origin"] == "real"]
    if n <= 0 or not reals:
        return None
    rng = random.Random(seed)
    sample = _stratified_take(reals, min(n, len(reals)), rng)
    order = list(spec.input_schema)
    if len(order) > 1:
        rng.shuffle(order)
        if order == list(spec.input_schema):
            order.reverse()

    spec_text = spec.spec_files_text()
    outputs: dict[int, Any] = {}
    for start in range(0, len(sample), spec.teacher.batch_size):
        batch = sample[start : start + spec.teacher.batch_size]
        reply = teacher.complete(
            prompts.teacher_label_prompt(
                spec, [r["input"] for r in batch], spec_text, field_order=order
            )
        )
        try:
            labels = prompts.extract_json(reply)
        except ValueError:
            continue
        for entry in labels:
            try:
                local_id = int(entry["id"])
            except (KeyError, TypeError, ValueError):
                continue
            raw = entry.get("score") if spec.output.is_scalar else entry.get("output")
            out = _coerce_valid(spec, raw)
            if 0 <= local_id < len(batch) and out is not None:
                outputs[id(batch[local_id])] = out

    probed = agree = unstable_in_gate = 0
    for r in sample:
        out = outputs.get(id(r))
        if out is None:
            continue
        probed += 1
        r["probe_output"] = out
        if _out_agrees(spec, out, row_output(spec, r)):
            agree += 1
        elif r.get("split") == "gate":
            unstable_in_gate += 1
    if not probed:
        return None
    return {
        "n": probed,
        "self_agreement": round(agree / probed, 4),
        "unstable": probed - agree,
        "unstable_in_gate": unstable_in_gate,
    }


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


def dataset_hash(rows: list[Row]) -> str:
    """Deterministic identity of the exact rows a compile consumes: id, input,
    split routing, origin, label, and gold annotation. Review edits, split
    changes, and gold changes all change it; row ordering does not."""
    h = hashlib.sha256()
    for r in sorted(rows, key=lambda r: str(r.get("id"))):
        key = {
            "id": r.get("id"),
            "input": r.get("input"),
            "split": r.get("split"),
            "origin": r.get("origin"),
            "label": r.get("score") if "score" in r else r.get("output"),
            "gold": r.get("gold"),
        }
        h.update(json.dumps(key, sort_keys=True, default=str).encode())
    return h.hexdigest()


def write_dataset(
    spec: FunctionSpec,
    rows: list[Row],
    out_dir: Path,
    probe: Optional[dict[str, Any]] = None,
) -> dict[str, Any]:
    """Write labeled/train/dev/gate JSONL + meta for already-split rows."""
    out_dir.mkdir(parents=True, exist_ok=True)
    rows, deduplicated = _dedupe_labeled_rows(rows)
    if deduplicated:
        print(f"deduplicated {deduplicated} exact-input row(s)", flush=True)
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
        "labeling_hash": spec.labeling_hash(),
        "dataset_hash": dataset_hash(rows),
        "real": sum(1 for r in rows if r["origin"] == "real"),
        "variants": sum(1 for r in rows if r["origin"] == "variant"),
        **{
            origin: n
            for origin in ("dropout", "counterfactual")
            if (n := sum(1 for r in rows if r["origin"] == origin))
        },
        "train": len(by_split["train"]),
        "dev": len(by_split["dev"]),
        "gate": len(by_split["gate"]),
        "label_histogram": hist,
        "teacher_model": spec.teacher.model,
        "teacher_backend": spec.teacher.backend,
        "prompt_version": prompts.PROMPT_VERSION,
        "labeled_at": datetime.date.today().isoformat(),
    }
    if probe:
        meta["teacher_self_agreement"] = probe["self_agreement"]
        meta["probe_n"] = probe["n"]
    if deduplicated:
        meta["deduplicated"] = deduplicated
    (out_dir / "meta.json").write_text(json.dumps(meta, indent=2))
    return meta


def _dedupe_labeled_rows(rows: list[Row]) -> tuple[list[Row], int]:
    """Keep one row per stable input ID while preserving duplicate judgments."""
    unique: list[Row] = []
    by_id: dict[str, Row] = {}
    duplicate_count = 0
    for row in rows:
        canonical = by_id.get(row["id"])
        if canonical is None:
            by_id[row["id"]] = row
            unique.append(row)
            continue
        duplicate_count += 1
        output_key = "score" if "score" in row else "output"
        observation = {
            "origin": row.get("origin"),
            output_key: row.get(output_key),
            "reason": row.get("reason", ""),
            "source_ids": row.get("source_ids", []),
        }
        if "intended_band" in row:
            observation["intended_band"] = row["intended_band"]
        canonical.setdefault("duplicate_observations", []).append(observation)
    return unique, duplicate_count


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
    `max_variants` is a global budget: the maximum total new synthetic rows
    this call may generate across all augmentation stages (0 = none).
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
    aug = spec.augment

    def _finish(new_rows: list[Row], sources: list[list[str]], items: list[dict]) -> list[Row]:
        src_by_id = {
            row_id({k: it.get(k) for k in spec.input_schema}): src
            for it, src in zip(items, sources)
        }
        for r in new_rows:
            r["split"] = "train"
            r.setdefault("source_ids", src_by_id.get(r["id"], []))
            r.update(stamp)
        return new_rows

    # `max_variants` is a GLOBAL budget: the maximum total new synthetic rows
    # this invocation may generate across paraphrase, field-dropout, and
    # counterfactual stages. Allocation is deterministic (pipeline order, each
    # stage consumes what it generates). 0 disables synthetic work entirely;
    # None means only the per-stage caps apply.
    budget = max_variants

    def _stage_cap(stage_cap: Optional[int]) -> Optional[int]:
        """Effective ceiling for a stage: min of its own cap and the global
        budget's remainder (None = unlimited)."""
        if budget is None:
            return stage_cap
        remaining = max(0, budget - len(extra_rows))
        return remaining if stage_cap is None else min(stage_cap, remaining)

    run_paraphrase, para_cap = True, None  # legacy: fill toward teacher.examples
    if aug is not None:
        run_paraphrase = aug.paraphrase is not None
        para_cap = aug.paraphrase.cap if aug.paraphrase else 0
    extra_rows: list[Row] = []
    effective = _stage_cap(para_cap)
    if run_paraphrase and effective != 0:
        variants_raw, sources = generate_variants(
            teacher, spec, train_reals, spec.teacher.examples,
            cap=effective, seed=spec.train.seed, hist_rows=train_rows,
        )
        variant_rows = (
            label_items(teacher, spec, variants_raw, origin="variant")
            if variants_raw else []
        )
        extra_rows += _finish(variant_rows, sources, variants_raw)

    if aug and aug.field_dropout and train_reals and _stage_cap(None) != 0:
        d_items, d_sources = generate_field_dropout(
            spec, train_reals, aug.field_dropout.fields,
            aug.field_dropout.cap, random.Random(spec.train.seed + 2),
        )
        remaining = _stage_cap(None)
        if remaining is not None:
            d_items, d_sources = d_items[:remaining], d_sources[:remaining]
        d_rows = label_items(teacher, spec, d_items, origin="dropout") if d_items else []
        extra_rows += _finish(d_rows, d_sources, d_items)

    cf_cap = _stage_cap(aug.counterfactual.cap) if aug and aug.counterfactual else 0
    if aug and aug.counterfactual and train_reals and cf_cap != 0:
        cf_rows, hit_rate = counterfactual_rows(
            teacher, spec, train_reals, cf_cap,
            seed=spec.train.seed + 3, hist_rows=train_rows + extra_rows,
        )
        for r in cf_rows:
            r["split"] = "train"
            r.update(stamp)
        extra_rows += cf_rows
        if hit_rate is not None:
            print(
                f"counterfactuals: {len(cf_rows)} rows, {hit_rate:.0%} moved "
                "closer to the intended band (misses kept as invariance examples)"
            )

    probe = consistency_probe(
        teacher, spec, rows, spec.teacher.consistency, seed=spec.train.seed
    )
    if probe:
        print(
            f"teacher self-agreement: {probe['self_agreement']:.0%} on "
            f"{probe['n']} re-labeled rows ({probe['unstable']} unstable)"
        )
        if probe["unstable_in_gate"]:
            print(
                f"warning: {probe['unstable_in_gate']} unstable row(s) sit in the "
                "gate split — the teacher itself wavers on them; "
                "`smallbatch review --unstable` to inspect"
            )

    meta = write_dataset(spec, rows + extra_rows, out_dir, probe=probe)
    usage = getattr(teacher, "usage", None)
    if isinstance(usage, dict) and usage:
        meta["teacher_usage"] = usage
        (out_dir / "meta.json").write_text(json.dumps(meta, indent=2))
    return meta


def _write_jsonl(path: Path, rows: list[Row]) -> None:
    with path.open("w") as f:
        for r in rows:
            f.write(json.dumps(r, ensure_ascii=False) + "\n")


def read_jsonl(path: Path) -> list[Row]:
    return [json.loads(line) for line in path.read_text().splitlines() if line.strip()]
