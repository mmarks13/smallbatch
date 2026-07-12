"""Durable stage-event journal for teacher labeling.

Every paid unit of work — a labeled row, a teacher-generated synthetic input,
a consistency-probe result — is appended to `data/<fn>/journal/events.jsonl`
the moment it completes, so a crash (transport failure, OOM, ^C) loses at most
the in-flight batch. Rerunning the same `smallbatch label` command replays the
journal instead of re-spending teacher calls, then folds everything into the
dataset files and archives itself.

Events are keyed by the spec's labeling fingerprint (labeling_hash): a journal
written under a different rubric/contract/teacher/prompt version is ignored,
never replayed.
"""

from __future__ import annotations

import datetime
import json
import os
from pathlib import Path
from typing import Any

EVENTS_FILE = "events.jsonl"
LOCK_FILE = "lock"


class JournalLocked(RuntimeError):
    pass


class LabelJournal:
    """Append-only event log with replay state.

    Replay state exposed to the pipeline:
    - `rows`: labeled rows by row id (real and synthetic) — skip these calls
    - `cached_stage(stage)`: the generated inputs of a COMPLETED generation
      stage (partial generations are regenerated; their finished labels still
      replay when the regenerated input matches by content hash)
    - `probe`: consistency-probe outputs by row id
    """

    def __init__(self, journal_dir: Path, fingerprint: str):
        self.dir = journal_dir
        self.fingerprint = fingerprint
        self.rows: dict[str, dict] = {}
        self._stage_items: dict[str, list[dict]] = {}
        self._stage_done: set[str] = set()
        self.probe: dict[str, Any] = {}
        self._fh = None
        self._locked = False

    # -- lifecycle -----------------------------------------------------------

    @classmethod
    def open(cls, out_dir: Path, fingerprint: str) -> LabelJournal:
        journal_dir = out_dir / "journal"
        journal_dir.mkdir(parents=True, exist_ok=True)
        j = cls(journal_dir, fingerprint)
        j._acquire_lock()
        j._load()
        j._fh = (journal_dir / EVENTS_FILE).open("a", encoding="utf-8")
        return j

    def _acquire_lock(self) -> None:
        lock = self.dir / LOCK_FILE
        try:
            fd = os.open(lock, os.O_CREAT | os.O_EXCL | os.O_WRONLY)
        except FileExistsError:
            raise JournalLocked(
                f"another labeling process appears to hold {lock} "
                f"(pid {lock.read_text().strip() or '?'}); remove the file if "
                "no other `smallbatch label` is running"
            ) from None
        os.write(fd, str(os.getpid()).encode())
        os.close(fd)
        self._locked = True

    def _release_lock(self) -> None:
        if self._locked:
            (self.dir / LOCK_FILE).unlink(missing_ok=True)
            self._locked = False

    def _load(self) -> None:
        path = self.dir / EVENTS_FILE
        if not path.exists():
            return
        for line in path.read_text(encoding="utf-8").splitlines():
            if not line.strip():
                continue
            try:
                ev = json.loads(line)
            except json.JSONDecodeError:
                continue  # torn final line from a crash mid-append
            if ev.get("fingerprint") != self.fingerprint:
                continue  # stale: written under a different labeling identity
            kind = ev.get("event")
            if kind == "row":
                self.rows[ev["row"]["id"]] = ev["row"]
            elif kind == "stage_item":
                self._stage_items.setdefault(ev["stage"], []).append(ev)
            elif kind == "stage_done":
                self._stage_done.add(ev["stage"])
            elif kind == "probe":
                self.probe[ev["row_id"]] = ev["output"]

    def close(self) -> None:
        if self._fh is not None:
            self._fh.close()
            self._fh = None
        self._release_lock()

    def archive(self) -> None:
        """Called only after the dataset files were atomically replaced: the
        journal's work is folded in, so retire it (kept for audit, ignored by
        future runs because a fresh events file starts empty)."""
        if self._fh is not None:
            self._fh.close()
            self._fh = None
        events = self.dir / EVENTS_FILE
        if events.exists():
            stamp = datetime.datetime.now().strftime("%Y%m%d-%H%M%S")
            os.replace(events, self.dir / f"archived-{stamp}.jsonl")
        self._release_lock()

    # -- recording (each record is durable before the call is "complete") ----

    def _record(self, event: str, **payload) -> None:
        line = json.dumps(
            {"event": event, "fingerprint": self.fingerprint, **payload},
            ensure_ascii=False,
        )
        self._fh.write(line + "\n")
        self._fh.flush()
        os.fsync(self._fh.fileno())

    def record_row(self, row: dict) -> None:
        self._record("row", row=row)
        self.rows[row["id"]] = row

    def record_stage_item(self, stage: str, item: dict, **provenance) -> None:
        ev = {"stage": stage, "item": item, **provenance}
        self._record("stage_item", **ev)
        self._stage_items.setdefault(stage, []).append(
            {"fingerprint": self.fingerprint, "event": "stage_item", **ev}
        )

    def record_stage_done(self, stage: str) -> None:
        self._record("stage_done", stage=stage)
        self._stage_done.add(stage)

    def record_probe(self, row_id: str, output: Any) -> None:
        self._record("probe", row_id=row_id, output=output)
        self.probe[row_id] = output

    # -- replay ---------------------------------------------------------------

    def cached_stage(self, stage: str) -> list[dict] | None:
        """The stage's generated items (with provenance) iff generation
        COMPLETED before the crash; None means regenerate."""
        if stage not in self._stage_done:
            return None
        return self._stage_items.get(stage, [])


class NullJournal:
    """No-persistence stand-in so the pipeline reads one code path."""

    rows: dict[str, dict] = {}
    probe: dict[str, Any] = {}

    def record_row(self, row: dict) -> None:
        pass

    def record_stage_item(self, stage: str, item: dict, **provenance) -> None:
        pass

    def record_stage_done(self, stage: str) -> None:
        pass

    def record_probe(self, row_id: str, output: Any) -> None:
        pass

    def cached_stage(self, stage: str) -> list[dict] | None:
        return None

    def close(self) -> None:
        pass

    def archive(self) -> None:
        pass
