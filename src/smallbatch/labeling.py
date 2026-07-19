"""Decision ingestion, teacher labeling, splitting, and optional augmentation."""

from __future__ import annotations

import datetime
import hashlib
import json
import random
import sys
from collections import Counter, defaultdict
from pathlib import Path
from typing import Any

from . import prompts
from .atomic import atomic_json, atomic_jsonl
from .journal import LabelJournal, NullJournal
from .spec import FunctionSpec, validate_input, validate_output
from .teacher import Teacher

Row = dict[str, Any]
SPLIT_SEED = 17

# Origins whose rows are supplied decisions (as opposed to synthetic
# augmentation variants). "user-resolved" marks a decision the user made by
# resolving a three-way teacher split; the reference on that row is the user,
# not the teacher, and public evidence must disclose the count.
REAL_ORIGINS = {"real", "user-resolved"}

# Fraction of three-way splits above which hand-resolving is the wrong tool:
# the rubric under-determines a whole region of inputs, and tightening the
# prompt then relabeling beats annotating the symptom.
UNRESOLVED_RUBRIC_THRESHOLD = 0.03


def _progress(message: str) -> None:
    print(f"[smallbatch] {message}", file=sys.stderr, flush=True)


def row_id(input_obj: dict) -> str:
    canonical = json.dumps(input_obj, sort_keys=True, ensure_ascii=False, separators=(",", ":"))
    return hashlib.sha256(canonical.encode()).hexdigest()[:16]


def row_output(spec: FunctionSpec, row: Row) -> Any:
    return row["output"]


def primary_value(spec: FunctionSpec, row: Row) -> Any:
    output = row_output(spec, row)
    return output[next(iter(spec.output.fields))] if isinstance(output, dict) else output


def stratum_value(spec: FunctionSpec, row: Row) -> Any:
    """The value splits stratify on: the first bounded field.

    Unique free-text values are never treated as classes, so a text-only
    function returns None — one stratum, deterministic seeded splitting."""
    output = row_output(spec, row)
    for name, field in spec.output.fields.items():
        if field.type != "text":
            return output[name] if isinstance(output, dict) else output
    return None


def dataset_hash(rows: list[Row]) -> str:
    canonical = json.dumps(
        sorted(rows, key=lambda row: row["id"]),
        sort_keys=True,
        ensure_ascii=False,
        separators=(",", ":"),
    )
    return hashlib.sha256(canonical.encode()).hexdigest()


def normalize_item_records(
    spec: FunctionSpec, records: list[dict]
) -> tuple[list[dict], list[Any] | None]:
    if not isinstance(records, list) or not records:
        raise ValueError("items must be a non-empty JSON/JSONL list")
    inputs: list[dict] = []
    outputs: list[Any] = []
    decided: list[bool] = []
    for index, record in enumerate(records):
        if not isinstance(record, dict) or "input" not in record:
            raise ValueError(f"item {index} must use {{\"input\": {{...}}}}")
        extra = set(record) - {"input", "output", "provenance"}
        if extra:
            raise ValueError(f"item {index} has unexpected envelope keys {sorted(extra)}")
        inputs.append(validate_input(spec, record["input"]))
        has_output = "output" in record
        decided.append(has_output)
        if has_output:
            outputs.append(validate_output(spec, record["output"]))
    if any(decided) and not all(decided):
        raise ValueError("items cannot mix imported decisions and unlabeled inputs")
    return inputs, outputs if all(decided) else None


def label_items(
    teacher: Teacher,
    spec: FunctionSpec,
    items: list[dict],
    origin: str,
    journal=None,
    field_order: list[str] | None = None,
    allow_partial: bool = False,
) -> list[Row]:
    """Teacher-label validated input objects with one retry and journal replay.

    Real decisions are all-or-nothing; `allow_partial` lets synthetic
    augmentation variants degrade to fewer rows instead of aborting the run.
    """
    journal = journal or NullJournal()
    rows: dict[int, Row] = {}
    for index, item in enumerate(items):
        cached = journal.rows.get((row_id(item), origin))
        if cached is not None:
            rows[index] = dict(cached)
    pending = [index for index in range(len(items)) if index not in rows]
    if rows:
        _progress(f"teacher {origin} resumed rows={len(rows)} pending={len(pending)}")
    batch_size = spec.teacher.batch_size if spec.teacher else 40
    for attempt in range(2):
        if not pending:
            break
        next_pending: list[int] = []
        batch_total = (len(pending) + batch_size - 1) // batch_size
        for start in range(0, len(pending), batch_size):
            indices = pending[start : start + batch_size]
            batch = [items[index] for index in indices]
            batch_number = start // batch_size + 1
            _progress(
                f"teacher {origin} batch={batch_number}/{batch_total} rows={len(batch)} "
                f"attempt={attempt + 1}/2"
            )
            response = teacher.complete(
                prompts.teacher_label_prompt(spec, batch, field_order=field_order)
            )
            try:
                entries = prompts.extract_json(response)
            except (ValueError, json.JSONDecodeError):
                next_pending.extend(indices)
                continue
            accepted: set[int] = set()
            for entry in entries if isinstance(entries, list) else []:
                try:
                    local_index = int(entry["id"])
                    output = validate_output(spec, entry.get("output"))
                except (KeyError, TypeError, ValueError):
                    continue
                if not 0 <= local_index < len(indices) or local_index in accepted:
                    continue
                global_index = indices[local_index]
                row = {
                    "id": row_id(items[global_index]),
                    "input": items[global_index],
                    "output": output,
                    "origin": origin,
                }
                rows[global_index] = row
                accepted.add(local_index)
                journal.record_row(row)
            next_pending.extend(
                indices[local_index]
                for local_index in range(len(indices))
                if local_index not in accepted
            )
        pending = next_pending
    if pending:
        if not allow_partial:
            raise ValueError(
                f"teacher failed to return valid decisions for {len(pending)} of "
                f"{len(items)} items"
            )
        _progress(
            f"teacher {origin} dropped {len(pending)} of {len(items)} rows "
            "after 2 attempts"
        )
    elif items:
        _progress(f"teacher {origin} complete rows={len(items)}")
    return [rows[index] for index in range(len(items)) if index in rows]


def _output_key(output: Any) -> str:
    return json.dumps(output, sort_keys=True, ensure_ascii=False, separators=(",", ":"))


def _comparison_output(spec: FunctionSpec, output: Any) -> Any:
    """The projection compared across teacher passes: bounded fields only.

    Two draws whose decisions match but whose text is worded differently are
    not a disagreement — exact text match would overstate teacher
    inconsistency, and semantic comparison is out of scope. Text-only
    functions reject passes: 2 at the spec, so a projection always exists."""
    if not spec.output.has_text:
        return output
    return {name: output[name] for name in spec.output.bounded_fields}


def _draw_summary(row: Row) -> dict[str, Any]:
    return {"output": row["output"], "origin": row["origin"]}


def label_real_rows(
    teacher: Teacher,
    spec: FunctionSpec,
    items: list[dict],
    journal=None,
) -> tuple[list[Row], list[dict]]:
    """Label real decisions honoring `teacher.passes`.

    passes=1 is a single draw per item, unchanged. passes=2 draws every item
    twice — the second pass over a shuffled item order so batch neighbors
    differ — and tie-breaks flips with one targeted third draw. Rows carry
    `weight` (the fraction of draws matching the consensus) and `agreement`
    ("unanimous", "majority", or "median"); training may down-weight low
    agreement, and the unanimous fraction is the teacher's self-agreement
    ceiling reported as evidence. Items whose three draws are all distinct and
    not an ordered integer scale are returned separately as unresolved: they
    get no decision, never block the run, and may be resolved by the user via
    the imported-decisions path.

    Draws are compared on their bounded fields only: a text field that is
    worded differently between passes is variation, not disagreement, and the
    winning draw keeps its own text.
    """
    passes = spec.teacher.passes if spec.teacher else 1
    if passes == 1 or not items:
        return label_items(teacher, spec, items, "real", journal), []
    first = label_items(teacher, spec, items, "real#1", journal)
    order = list(range(len(items)))
    random.Random(SPLIT_SEED).shuffle(order)
    second = label_items(teacher, spec, [items[i] for i in order], "real#2", journal)
    first_by = {row["id"]: row for row in first}
    second_by = {row["id"]: row for row in second}
    flips = [
        item
        for item in items
        if _comparison_output(spec, first_by[row_id(item)]["output"])
        != _comparison_output(spec, second_by[row_id(item)]["output"])
    ]
    _progress(
        f"teacher pass agreement: {len(items) - len(flips)}/{len(items)} unanimous, "
        f"{len(flips)} flips to tie-break"
    )
    third_by = {
        row["id"]: row
        for row in (
            label_items(teacher, spec, flips, "real#tiebreak", journal) if flips else []
        )
    }
    ordered_scale = spec.output.is_scalar and spec.output.scalar.type == "int"
    rows: list[Row] = []
    unresolved: list[dict] = []
    for item in items:
        rid = row_id(item)
        draws = [first_by[rid], second_by[rid]]
        if _comparison_output(spec, draws[0]["output"]) == _comparison_output(
            spec, draws[1]["output"]
        ):
            rows.append({**draws[0], "origin": "real", "weight": 1.0, "agreement": "unanimous"})
            continue
        draws.append(third_by[rid])
        counts = Counter(
            _output_key(_comparison_output(spec, draw["output"])) for draw in draws
        )
        key, votes = counts.most_common(1)[0]
        if votes >= 2:
            winner = next(
                draw
                for draw in draws
                if _output_key(_comparison_output(spec, draw["output"])) == key
            )
            rows.append(
                {**winner, "origin": "real", "weight": round(2 / 3, 4), "agreement": "majority"}
            )
        elif ordered_scale:
            # the median of three distinct levels is always one of the draws
            winner = sorted(draws, key=lambda draw: draw["output"])[1]
            rows.append(
                {**winner, "origin": "real", "weight": round(1 / 3, 4), "agreement": "median"}
            )
        else:
            unresolved.append(
                {
                    "input": item,
                    "provenance": {
                        "unresolved": True,
                        "draws": [_draw_summary(draw) for draw in draws],
                    },
                }
            )
    return rows, unresolved


def _strata(spec: FunctionSpec, rows: list[Row], seed: int) -> dict[str, list[Row]]:
    rng = random.Random(seed)
    groups: dict[str, list[Row]] = defaultdict(list)
    for row in rows:
        groups[json.dumps(stratum_value(spec, row), sort_keys=True)].append(row)
    for group in groups.values():
        rng.shuffle(group)
    return dict(sorted(groups.items()))


def _proportional_allocation(
    groups: dict[str, list[Row]], quota: int, taken: dict[str, int]
) -> dict[str, int]:
    """Largest-remainder share of `quota` per class, after `taken` rows."""
    remaining = {name: len(rows) - taken[name] for name, rows in groups.items()}
    total = sum(remaining.values())
    if not total or quota <= 0:
        return {name: 0 for name in groups}
    exact = {name: quota * count / total for name, count in remaining.items()}
    allocation = {name: min(int(value), remaining[name]) for name, value in exact.items()}
    leftover = quota - sum(allocation.values())
    by_remainder = sorted(
        groups,
        key=lambda name: (allocation[name] - exact[name], name),
    )
    for name in by_remainder:
        if leftover <= 0:
            break
        if allocation[name] < remaining[name]:
            allocation[name] += 1
            leftover -= 1
    return allocation


def assign_splits(
    spec: FunctionSpec,
    rows: list[Row],
    *,
    existing: dict[str, str] | None = None,
    force_train_ids: set[str] | None = None,
) -> None:
    """Deterministic proportional 70/10/20 split with sticky assignments.

    Every split receives a proportional share of each decision class: the
    class-interleaved order is walked once and each row goes to the split
    with the largest remaining quota deficit. Slicing the interleaved head
    into eval would instead concentrate rare classes there, starving dev
    (breaking early stopping) and train (hiding classes from candidates).
    """
    existing = existing or {}
    force_train_ids = force_train_ids or set()
    for row in rows:
        if row["id"] in existing:
            row["split"] = existing[row["id"]]
        elif row["id"] in force_train_ids:
            row["split"] = "train"

    target_eval = round(len(rows) * 0.2)
    target_dev = round(len(rows) * 0.1)
    current = Counter(row.get("split") for row in rows)
    unassigned = [row for row in rows if not row.get("split")]
    groups = _strata(spec, unassigned, SPLIT_SEED)
    taken = {name: 0 for name in groups}
    for split, quota in (
        ("eval", max(0, target_eval - current["eval"])),
        ("dev", max(0, target_dev - current["dev"])),
    ):
        allocation = _proportional_allocation(groups, quota, taken)
        for name, count in allocation.items():
            for row in groups[name][taken[name] : taken[name] + count]:
                row["split"] = split
            taken[name] += count
    for name, group in groups.items():
        for row in group[taken[name] :]:
            row["split"] = "train"


def _imported_rows(
    spec: FunctionSpec, inputs: list[dict], outputs: list[Any], records: list[dict]
) -> list[Row]:
    """Imported decisions; a record whose provenance marks it unresolved is a
    user resolution of a three-way teacher split and keeps that identity."""
    rows = []
    for item, output, record in zip(inputs, outputs, records):
        provenance = record.get("provenance")
        resolved = isinstance(provenance, dict) and provenance.get("unresolved")
        rows.append(
            {
                "id": row_id(item),
                "input": item,
                "output": output,
                "origin": "user-resolved" if resolved else "real",
                **({"weight": 1.0} if resolved else {}),
            }
        )
    return rows


def _dedupe(rows: list[Row]) -> list[Row]:
    by_id: dict[str, Row] = {}
    for row in rows:
        if row["id"] in by_id and row["output"] != by_id[row["id"]]["output"]:
            raise ValueError(f"conflicting decisions for duplicate input {row['id']}")
        by_id[row["id"]] = row
    return list(by_id.values())


def _augment(
    teacher: Teacher,
    spec: FunctionSpec,
    train_reals: list[Row],
    journal: LabelJournal,
    max_variants: int | None,
) -> list[Row]:
    plan = spec.augmentation
    if not plan or not train_reals or max_variants == 0:
        return []
    budget = max_variants if max_variants is not None else 10**9
    generated: list[tuple[dict, str, list[str]]] = []
    rng = random.Random(SPLIT_SEED)

    if plan.paraphrase and budget > 0:
        cap = min(plan.paraphrase.cap, budget)
        stage = f"paraphrase:{cap}"
        cached = journal.cached_stage(stage)
        stage_rows: list[tuple[dict, str, list[str]]] = []
        if cached is not None:
            stage_rows = [
                (event["item"], "paraphrase", event["source_ids"])
                for event in cached
            ]
            _progress(f"augmentation target-band resumed generated={len(stage_rows)}")
        else:
            _progress(f"augmentation target-band generation start requested={cap}")
            values = list(next(iter(spec.output.fields.values())).values())
            for index in range(cap):
                source = train_reals[index % len(train_reals)]
                target = values[index % len(values)]
                response = teacher.complete(
                    prompts.teacher_variant_prompt(spec, [source["input"]], str(target), 1)
                )
                try:
                    values_out = prompts.extract_json(response)
                    item = validate_input(spec, values_out[0])
                except (IndexError, TypeError, ValueError, json.JSONDecodeError):
                    continue
                sources = [source["id"]]
                stage_rows.append((item, "paraphrase", sources))
                journal.record_stage_item(stage, item, source_ids=sources)
            journal.record_stage_done(stage)
            _progress(f"augmentation target-band generation complete generated={len(stage_rows)}")
        generated.extend(stage_rows)
        budget -= len(stage_rows)

    if plan.field_dropout and budget > 0:
        stage = f"field-dropout:{budget}"
        cached = journal.cached_stage(stage)
        stage_rows = []
        if cached is not None:
            stage_rows = [
                (event["item"], "field-dropout", event["source_ids"])
                for event in cached
            ]
            _progress(f"augmentation field-dropout resumed generated={len(stage_rows)}")
        else:
            _progress("augmentation field-dropout generation start")
            remaining = budget
            for field in plan.field_dropout.fields:
                if spec.input_schema[field] != "string":
                    raise ValueError("field_dropout supports string input fields only")
                pool = [row for row in train_reals if row["input"][field]]
                rng.shuffle(pool)
                for source in pool[: min(plan.field_dropout.cap, remaining)]:
                    item = dict(source["input"])
                    item[field] = ""
                    sources = [source["id"]]
                    stage_rows.append((item, "field-dropout", sources))
                    journal.record_stage_item(stage, item, source_ids=sources)
                    remaining -= 1
                    if remaining <= 0:
                        break
                if remaining <= 0:
                    break
            journal.record_stage_done(stage)
            _progress(f"augmentation field-dropout generation complete generated={len(stage_rows)}")
        generated.extend(stage_rows)
        budget -= len(stage_rows)

    if plan.counterfactual and budget > 0:
        cap = min(plan.counterfactual.cap, budget)
        stage = f"counterfactual:{cap}"
        cached = journal.cached_stage(stage)
        stage_rows = []
        if cached is not None:
            stage_rows = [
                (event["item"], "counterfactual", event["source_ids"])
                for event in cached
            ]
            _progress(f"augmentation counterfactual resumed generated={len(stage_rows)}")
        else:
            _progress(f"augmentation counterfactual generation start requested={cap}")
            values = list(next(iter(spec.output.fields.values())).values())
            for source in train_reals[:cap]:
                target = values[(values.index(primary_value(spec, source)) + 1) % len(values)]
                response = teacher.complete(
                    prompts.teacher_counterfactual_prompt(
                        spec, [source["input"]], str(target)
                    )
                )
                try:
                    entries = prompts.extract_json(response)
                    raw = {name: entries[0][name] for name in spec.input_schema}
                    item = validate_input(spec, raw)
                except (IndexError, KeyError, TypeError, ValueError, json.JSONDecodeError):
                    continue
                sources = [source["id"]]
                stage_rows.append((item, "counterfactual", sources))
                journal.record_stage_item(stage, item, source_ids=sources)
            journal.record_stage_done(stage)
            _progress(f"augmentation counterfactual generation complete generated={len(stage_rows)}")
        generated.extend(stage_rows)
        budget -= len(stage_rows)

    unique: dict[str, tuple[dict, str, list[str]]] = {}
    real_ids = {row["id"] for row in train_reals}
    for item, origin, sources in generated:
        if row_id(item) not in real_ids:
            unique[row_id(item)] = (item, origin, sources)
    rows: list[Row] = []
    for origin in ("paraphrase", "field-dropout", "counterfactual"):
        chosen = [entry for entry in unique.values() if entry[1] == origin]
        if not chosen:
            continue
        labeled = label_items(
            teacher, spec, [entry[0] for entry in chosen], origin, journal,
            allow_partial=True,
        )
        provenance = {row_id(entry[0]): entry[2] for entry in chosen}
        for row in labeled:
            row["source_ids"] = provenance[row["id"]]
            row["split"] = "train"
        rows.extend(labeled)
    return rows


def _maintain_unresolved(
    spec: FunctionSpec,
    out_dir: Path,
    new_unresolved: list[dict],
    real_rows: list[Row],
    *,
    append: bool,
) -> list[dict]:
    """Fold this run's three-way splits into `unresolved.jsonl`.

    Records whose input has since received a decision (a user resolution or a
    later stable teacher draw) drop out. Resolution is always optional: the
    file never blocks a run, it only records what has no stable answer yet.
    """
    path = out_dir / "unresolved.jsonl"
    prior = read_jsonl(path) if append else []
    decided = {row["id"] for row in real_rows}
    kept = [record for record in prior if row_id(record["input"]) not in decided]
    known = {row_id(record["input"]) for record in kept}
    kept.extend(
        record for record in new_unresolved if row_id(record["input"]) not in known
    )
    if not kept:
        path.unlink(missing_ok=True)
        return []
    atomic_jsonl(path, kept)
    spec_path = spec._source_path or f"{spec.name}.yaml"
    _progress(
        f"{len(kept)} items had three distinct teacher answers and were set aside in "
        f"{path} — to resolve any of them, fill in a top-level \"output\" on each "
        f"line you keep (delete the rest), then run: "
        f"smallbatch label {spec_path} --items {path} --append"
    )
    total = len(kept) + len(real_rows)
    if total and len(kept) / total > UNRESOLVED_RUBRIC_THRESHOLD:
        _progress(
            "high three-way rate: the prompt likely under-determines these inputs; "
            "tightening the decision instructions and relabeling will help more "
            "than resolving items by hand"
        )
    return kept


def _noise_meta(
    spec: FunctionSpec,
    decision_source: str,
    real_rows: list[Row],
    unresolved_rows: list[dict],
) -> dict[str, Any] | None:
    """Teacher-noise evidence: self-agreement per split plus disclosure counts.

    Self-agreement is the fraction of measured rows whose two independent
    draws matched — the ceiling candidate agreement is judged against. Rows
    labeled before measurement was on (or imported) carry no agreement data
    and are excluded from the ceiling, never guessed at.
    """
    measured = [row for row in real_rows if "agreement" in row]
    user_resolved = sum(row.get("origin") == "user-resolved" for row in real_rows)
    if decision_source != "teacher" and not measured and not user_resolved and not unresolved_rows:
        return None

    def unanimous_fraction(rows: list[Row]) -> float | None:
        if not rows:
            return None
        return round(sum(row["agreement"] == "unanimous" for row in rows) / len(rows), 4)

    return {
        "passes": spec.teacher.passes if spec.teacher else 1,
        "measured_rows": len(measured),
        "self_agreement": {
            "overall": unanimous_fraction(measured),
            **{
                split: unanimous_fraction(
                    [row for row in measured if row.get("split") == split]
                )
                for split in ("train", "dev", "eval")
            },
        },
        "agreement_counts": dict(Counter(row["agreement"] for row in measured)),
        "unresolved": len(unresolved_rows),
        "user_resolved": user_resolved,
    }


def build_dataset(
    spec: FunctionSpec,
    records: list[dict],
    out_dir: Path,
    *,
    teacher: Teacher | None = None,
    append: bool = False,
    max_variants: int | None = None,
    force_train_ids: set[str] | None = None,
) -> dict[str, Any]:
    inputs, imported = normalize_item_records(spec, records)
    out_dir.mkdir(parents=True, exist_ok=True)
    if append and (out_dir / "meta.json").exists():
        prior = json.loads((out_dir / "meta.json").read_text())
        if prior.get("decision_hash") != spec.decision_hash():
            raise ValueError(
                "existing dataset was labeled under a different prompt, "
                "contract, or teacher (or a pre-v0.3 schema); appending would "
                "mix decision formats. Move the data directory aside or "
                "relabel from scratch"
            )
    existing_rows = read_jsonl(out_dir / "labeled.jsonl") if append else []
    existing_variants = [row for row in existing_rows if row.get("origin") not in REAL_ORIGINS]
    existing_splits = {row["id"]: row["split"] for row in existing_rows}
    existing_by_id = {row["id"]: row for row in existing_rows}

    journal = LabelJournal.open(out_dir, spec.decision_hash())
    try:
        new_unresolved: list[dict] = []
        if imported is not None:
            new_rows = _imported_rows(spec, inputs, imported, records)
            decision_source = "imported"
        else:
            if teacher is None:
                raise ValueError("unlabeled inputs require a configured teacher")
            unseen = [item for item in inputs if row_id(item) not in existing_by_id]
            new_rows, new_unresolved = label_real_rows(teacher, spec, unseen, journal)
            decision_source = "teacher"
        real_rows = _dedupe([*existing_rows, *new_rows])
        real_rows = [row for row in real_rows if row.get("origin") in REAL_ORIGINS]
        assign_splits(
            spec,
            real_rows,
            existing=existing_splits,
            force_train_ids=force_train_ids,
        )
        split_counts = Counter(row["split"] for row in real_rows)
        _progress(
            f"decisions ready source={decision_source} real={len(real_rows)} "
            f"train={split_counts['train']} dev={split_counts['dev']} eval={split_counts['eval']}"
        )
        unresolved_rows = _maintain_unresolved(
            spec, out_dir, new_unresolved, real_rows, append=append
        )
        variants: list[Row] = list(existing_variants)
        if spec.augmentation and (not append or max_variants is not None):
            if teacher is None:
                raise ValueError("augmentation requires a callable teacher")
            train_reals = [row for row in real_rows if row["split"] == "train"]
            variants.extend(_augment(teacher, spec, train_reals, journal, max_variants))
            _progress(f"augmentation complete variants={len(variants)}")
        rows = _dedupe([*real_rows, *variants])
        rows.sort(key=lambda row: (row["split"], row["id"]))
        for split in ("train", "dev", "eval"):
            atomic_jsonl(out_dir / f"{split}.jsonl", [row for row in rows if row["split"] == split])
        atomic_jsonl(out_dir / "labeled.jsonl", rows)
        has_bounded = bool(spec.output.bounded_fields)
        histogram = (
            Counter(str(stratum_value(spec, row)) for row in real_rows)
            if has_bounded
            else Counter()
        )
        meta = {
            "schema_version": 3,
            "function": spec.name,
            "decision_hash": spec.decision_hash(),
            "dataset_hash": dataset_hash(rows),
            "decision_source": decision_source,
            "teacher": spec.teacher.model_dump(mode="json") if decision_source == "teacher" else None,
            "counts": {split: sum(row["split"] == split for row in rows) for split in ("train", "dev", "eval")},
            "real": len(real_rows),
            "variants": sum(row.get("origin") != "real" for row in rows),
            "label_histogram": dict(sorted(histogram.items())),
            "teacher_noise": _noise_meta(spec, decision_source, real_rows, unresolved_rows),
            "split_label_histograms": {
                split: dict(
                    sorted(
                        Counter(
                            str(stratum_value(spec, row))
                            for row in real_rows
                            if row["split"] == split
                        ).items()
                    )
                )
                if has_bounded
                else {}
                for split in ("train", "dev", "eval")
            },
            "created_at": datetime.datetime.now(datetime.timezone.utc).isoformat(),
        }
        atomic_json(out_dir / "meta.json", meta)
        journal.archive()
        return meta
    except BaseException:
        journal.close()
        raise


def read_jsonl(path: Path) -> list[Row]:
    if not path.exists():
        return []
    return [json.loads(line) for line in path.read_text().splitlines() if line.strip()]


