"""`smallbatch review <spec>`: step through teacher labels before training.

Accept / reject / edit / note / skip, with filters. Decisions are written
back into labeled.jsonl under a `review` key; the split files are rewritten
with rejected rows excluded and edits applied, so the next compile trains on
the curated set. The interactive loop is a thin shell over pure helpers
(injectable input/print) so everything is unit-testable.
"""

from __future__ import annotations

import datetime
import json
from pathlib import Path
from typing import Callable, Optional

from .labeling import Row, _coerce_valid, _write_jsonl, primary_value, read_jsonl, row_output
from .spec import FunctionSpec


def matches(spec: FunctionSpec, row: Row, args) -> bool:
    if args.split and row.get("split") != args.split:
        return False
    if args.origin and row.get("origin") != args.origin:
        return False
    if args.label is not None and str(primary_value(spec, row)) != args.label:
        return False
    if args.field:
        out = row_output(spec, row)
        if not isinstance(out, dict):
            return False
        name, _, want = args.field.partition("=")
        if name not in out:
            return False
        if want and str(out[name]) != want:
            return False
    status = (row.get("review") or {}).get("status")
    if args.status == "unreviewed":
        return status is None
    if args.status and args.status != "all":
        return status == args.status
    return True


def format_row(spec: FunctionSpec, row: Row, pos: int, total: int) -> str:
    head = (
        f"[{pos}/{total}] {row.get('origin')} | split={row.get('split')} | "
        f"{row.get('teacher_model')} @ {row.get('labeled_at')}"
    )
    status = (row.get("review") or {}).get("status")
    if status:
        head += f" | review: {status}"
    lines = [head]
    for k, v in row["input"].items():
        v = ", ".join(map(str, v)) if isinstance(v, (list, tuple)) else v
        lines.append(f"  {k}: {v}")
    out = row_output(spec, row)
    if isinstance(out, dict):
        for name, v in out.items():
            lines.append(f"  -> {name}: {v}")
    else:
        lines.append(f"  -> score: {out}")
    if row.get("reason"):
        lines.append(f"  teacher: {row['reason']}")
    return "\n".join(lines)


def apply_edit(
    spec: FunctionSpec, row: Row, ask: Callable[[str], str]
) -> Optional[str]:
    """Prompt for new value(s); returns an error message or None on success."""
    original = row_output(spec, row)
    if spec.output.is_scalar:
        raw = ask(f"new value [{original}]: ").strip()
        if not raw:
            return None
        new = _coerce_valid(spec, raw)
        if new is None:
            return f"invalid value {raw!r} for the output contract"
        row["score"] = new
    else:
        new_out = dict(original)
        for name in spec.output.fields:
            raw = ask(f"{name} [{original[name]}]: ").strip()
            if raw:
                new_out[name] = raw
        new = _coerce_valid(spec, new_out)
        if new is None:
            return "edited fields do not satisfy the output contract"
        row["output"] = new
    review = row.setdefault("review", {})
    review["status"] = "edited"
    review.setdefault("original", original)
    return None


def save(spec: FunctionSpec, rows: list[Row], out_dir: Path) -> dict[str, int]:
    """Rewrite labeled.jsonl (all rows, audit trail intact) and the split
    files (rejected rows excluded). Returns counts by review status."""
    _write_jsonl(out_dir / "labeled.jsonl", rows)
    kept = [r for r in rows if (r.get("review") or {}).get("status") != "rejected"]
    for split in ("train", "dev", "gate"):
        _write_jsonl(out_dir / f"{split}.jsonl", [r for r in kept if r.get("split") == split])
    counts: dict[str, int] = {}
    for r in rows:
        status = (r.get("review") or {}).get("status") or "unreviewed"
        counts[status] = counts.get(status, 0) + 1
    meta_path = out_dir / "meta.json"
    if meta_path.exists():
        meta = json.loads(meta_path.read_text())
        meta["review"] = {**counts, "reviewed_at": datetime.date.today().isoformat()}
        meta_path.write_text(json.dumps(meta, indent=2))
    return counts


_HELP = "[a]ccept  [r]eject  [e]dit  [n]ote  [s]kip  [q]uit+save"


def run_review(
    spec: FunctionSpec,
    out_dir: Path,
    args,
    input_fn: Callable[[str], str] = input,
    print_fn: Callable[[str], None] = print,
) -> int:
    rows = read_jsonl(out_dir / "labeled.jsonl")
    queue = [r for r in rows if matches(spec, r, args)]
    if not queue:
        print_fn("nothing matches the filters")
        return 0
    print_fn(f"{len(queue)} row(s) to review — {_HELP}")
    stamp = datetime.date.today().isoformat()
    dirty = False
    for pos, row in enumerate(queue, 1):
        print_fn("")
        print_fn(format_row(spec, row, pos, len(queue)))
        while True:
            try:
                cmd = input_fn("> ").strip().lower()
            except EOFError:
                cmd = "q"
            if cmd in ("a", "accept"):
                row["review"] = {**(row.get("review") or {}), "status": "accepted", "at": stamp}
                dirty = True
            elif cmd in ("r", "reject"):
                row["review"] = {**(row.get("review") or {}), "status": "rejected", "at": stamp}
                dirty = True
            elif cmd in ("e", "edit"):
                err = apply_edit(spec, row, input_fn)
                if err:
                    print_fn(err)
                    continue
                row["review"]["at"] = stamp
                dirty = True
            elif cmd in ("n", "note"):
                note = input_fn("note: ").strip()
                review = row.setdefault("review", {})
                review["note"] = note
                dirty = True
                continue  # a note isn't a verdict; stay on this row
            elif cmd in ("s", "skip", ""):
                pass
            elif cmd in ("q", "quit"):
                if dirty:
                    counts = save(spec, rows, out_dir)
                    print_fn(f"saved: {counts}")
                return 0
            else:
                print_fn(_HELP)
                continue
            break
    if dirty:
        counts = save(spec, rows, out_dir)
        print_fn(f"saved: {counts}")
    return 0
