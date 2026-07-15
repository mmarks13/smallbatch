"""Teacher calibration before committing to a full labeling run."""

from __future__ import annotations

import hashlib
import json
import random
import sys
from collections.abc import Callable
from dataclasses import dataclass
from pathlib import Path

from .atomic import atomic_json
from .spec import FunctionSpec


class CalibrationDeclined(RuntimeError):
    """The user inspected the teacher and chose not to continue."""


@dataclass
class CalibrationResult:
    status: str
    row_ids: list[str]
    batches_reviewed: int
    path: Path


def _input_fingerprint(items: list[dict]) -> str:
    payload = json.dumps(items, sort_keys=True, ensure_ascii=False, separators=(",", ":"))
    return hashlib.sha256(payload.encode()).hexdigest()


def _diverse_order(items: list[dict]) -> list[dict]:
    """Deterministically alternate short and long inputs."""
    ordered = sorted(
        items,
        key=lambda item: (
            sum(len(str(value)) for value in item.values()),
            hashlib.sha256(
                json.dumps(item, sort_keys=True, ensure_ascii=False).encode()
            ).hexdigest(),
        ),
    )
    spread: list[dict] = []
    left, right = 0, len(ordered) - 1
    while left <= right:
        spread.append(ordered[left])
        left += 1
        if left <= right:
            spread.append(ordered[right])
            right -= 1
    return spread


def calibrate_teacher(
    teacher,
    spec: FunctionSpec,
    items: list[dict],
    out_dir: Path,
    *,
    skip: bool = False,
    interactive: bool | None = None,
    input_fn: Callable[[str], str] = input,
    print_fn: Callable[[str], None] = print,
) -> CalibrationResult:
    """Show repeated decisions in batches of ten and persist the decision."""
    from .labeling import label_items, row_id, row_output
    from .metrics import compare

    out_dir.mkdir(parents=True, exist_ok=True)
    path = out_dir / "calibration.json"
    identity = {
        "decision_hash": spec.decision_hash(),
        "input_fingerprint": _input_fingerprint(items),
    }
    if path.exists():
        recorded = json.loads(path.read_text())
        if all(recorded.get(key) == value for key, value in identity.items()):
            if recorded.get("status") in {"approved", "bypassed"}:
                return CalibrationResult(
                    recorded["status"],
                    recorded.get("row_ids", []),
                    recorded.get("batches_reviewed", 0),
                    path,
                )

    if skip:
        payload = {**identity, "status": "bypassed", "row_ids": [], "batches_reviewed": 0}
        atomic_json(path, payload)
        return CalibrationResult("bypassed", [], 0, path)

    if interactive is None:
        interactive = sys.stdin.isatty() and sys.stdout.isatty()
    if not interactive:
        raise ValueError(
            "teacher labeling requires interactive calibration; import decisions or pass "
            "--skip-calibration explicitly"
        )

    ordered = _diverse_order(items)
    offset = 0
    reviewed: list[str] = []
    batches = 0
    while offset < len(ordered):
        batch = ordered[offset : offset + 10]
        if not batch:
            break
        first = label_items(teacher, spec, batch, "calibration-first")
        second_batch = list(batch)
        shuffle_seed = int(
            hashlib.sha256(
                f"{spec.decision_hash()}\n{offset}\ncalibration-second".encode()
            ).hexdigest()[:16],
            16,
        )
        random.Random(shuffle_seed).shuffle(second_batch)
        if len(second_batch) > 1 and second_batch == batch:
            second_batch = second_batch[1:] + second_batch[:1]
        reversed_order = list(reversed(spec.input_schema))
        second = label_items(
            teacher,
            spec,
            second_batch,
            "calibration-second",
            field_order=reversed_order,
        )
        first_by_id = {row["id"]: row for row in first}
        second_by_id = {row["id"]: row for row in second}
        common = [row_id(item) for item in batch if row_id(item) in first_by_id and row_id(item) in second_by_id]
        refs = [row_output(spec, first_by_id[rid]) for rid in common]
        preds = [row_output(spec, second_by_id[rid]) for rid in common]
        metrics = compare(spec, preds, refs)
        print_fn("\nTeacher calibration")
        for index, rid in enumerate(common, 1):
            print_fn(
                f"{index:>2}. first={row_output(spec, first_by_id[rid])!r}  "
                f"repeat={row_output(spec, second_by_id[rid])!r}"
            )
        print_fn(json.dumps(metrics, indent=2))
        reviewed.extend(common)
        batches += 1
        atomic_json(
            path,
            {
                **identity,
                "status": "pending",
                "row_ids": reviewed,
                "batches_reviewed": batches,
                "last_metrics": metrics,
            },
        )
        answer = input_fn("Approve teacher behavior, review more, or decline? [a/m/d] ").strip().lower()
        if answer in {"a", "approve", "y", "yes"}:
            atomic_json(
                path,
                {
                    **identity,
                    "status": "approved",
                    "row_ids": reviewed,
                    "batches_reviewed": batches,
                    "last_metrics": metrics,
                },
            )
            return CalibrationResult("approved", reviewed, batches, path)
        if answer not in {"m", "more"}:
            atomic_json(
                path,
                {
                    **identity,
                    "status": "declined",
                    "row_ids": reviewed,
                    "batches_reviewed": batches,
                    "last_metrics": metrics,
                },
            )
            raise CalibrationDeclined(
                "teacher behavior was declined; revise the prompt or teacher before labeling"
            )
        offset += len(batch)

    raise CalibrationDeclined("no additional calibration inputs remain")
