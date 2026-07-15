"""Crash-safe writes shared by every module that persists JSON state.

Content is staged beside the target as `<full name>.tmp` and moved into place
with an atomic rename, so readers never observe a partial file and
interrupted-write cleanup has exactly one naming pattern to look for.
"""

from __future__ import annotations

import json
from pathlib import Path
from typing import Any


def atomic_text(path: Path, text: str, *, mkdir: bool = False) -> None:
    if mkdir:
        path.parent.mkdir(parents=True, exist_ok=True)
    tmp = path.with_name(path.name + ".tmp")
    tmp.write_text(text)
    tmp.replace(path)


def atomic_json(
    path: Path, value: Any, *, indent: int | None = 2, mkdir: bool = False
) -> None:
    atomic_text(path, json.dumps(value, indent=indent, ensure_ascii=False), mkdir=mkdir)


def atomic_jsonl(path: Path, rows: list[dict]) -> None:
    atomic_text(path, "".join(json.dumps(row, ensure_ascii=False) + "\n" for row in rows))
