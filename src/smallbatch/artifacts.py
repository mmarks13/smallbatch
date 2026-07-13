"""Immutable builds, active selections, integrity, and resume state."""

from __future__ import annotations

import datetime
import hashlib
import json
import os
import re
import shutil
from pathlib import Path

from .spec import load_spec

DEFAULT_ROOT = Path("artifacts")
MANIFEST_SCHEMA_VERSION = 3
SOURCE_UNAVAILABLE = "source comparison unavailable"


def _atomic_json(path: Path, value: dict) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    tmp = path.with_suffix(path.suffix + ".tmp")
    tmp.write_text(json.dumps(value, indent=2, ensure_ascii=False))
    tmp.replace(path)


def dir_size(path: Path) -> int:
    return sum(file.stat().st_size for file in path.rglob("*") if file.is_file())


def _version_key(path: Path) -> tuple[str, int]:
    match = re.fullmatch(r"(.*?)(?:-r(\d+))?", path.name)
    return match.group(1), int(match.group(2) or 1)


def versions(root: Path, name: str) -> list[Path]:
    base = root / name / "builds"
    if not base.is_dir():
        return []
    return sorted(
        (path for path in base.iterdir() if (path / "manifest.json").exists()),
        key=_version_key,
    )


def latest(root: Path, name: str) -> Path | None:
    found = [
        path
        for path in versions(root, name)
        if read_build_state(path).get("status") == "complete"
    ]
    return found[-1] if found else None


def _allocate_build(
    base: Path, build_hash: str, dataset_hash: str, *, stem: str | None = None
) -> Path:
    stem = stem or f"{datetime.date.today().isoformat()}-{build_hash[:8]}"
    path = base / stem
    revision = 1
    while path.exists():
        revision += 1
        path = base / f"{stem}-r{revision}"
    path.mkdir(parents=True)
    write_build_state(
        path,
        {
            "schema_version": MANIFEST_SCHEMA_VERSION,
            "status": "running",
            "build_hash": build_hash,
            "dataset_hash": dataset_hash,
            "candidates": {},
            "created_at": datetime.datetime.now(datetime.timezone.utc).isoformat(),
        },
    )
    return path


def _link_or_copy(source: str, destination: str) -> str:
    try:
        os.link(source, destination)
    except OSError:
        shutil.copy2(source, destination)
    return destination


def _seed_retry(source: Path, destination: Path) -> None:
    """Seed a revision without modifying or duplicating completed candidates."""
    manifest = read_manifest(source)
    state = read_build_state(destination)
    state["retry_of"] = source.name
    for candidate, record in (manifest.get("candidates") or {}).items():
        source_dir = source / "candidates" / candidate
        if not source_dir.exists():
            continue
        completed = record.get("status") == "completed"
        shutil.copytree(
            source_dir,
            destination / "candidates" / candidate,
            copy_function=_link_or_copy if completed else shutil.copy2,
        )
        state["candidates"][candidate] = {
            "stage": "completed" if completed else "retry-pending",
            "reused_from": source.name if completed else None,
            "previous_error": None if completed else record.get("error"),
        }
    for diagnostic in (manifest.get("diagnostics") or {}):
        source_record = source / f"{diagnostic}.local.json"
        if source_record.exists():
            _link_or_copy(str(source_record), str(destination / source_record.name))
    write_build_state(destination, state)


def build_dir(root: Path, name: str, build_hash: str, dataset_hash: str) -> Path:
    """Return a resumable build or allocate an immutable retry revision."""
    base = root / name / "builds"
    base.mkdir(parents=True, exist_ok=True)
    matching: list[Path] = []
    for path in sorted(base.iterdir(), key=_version_key):
        state_path = path / "build_state.json"
        if not state_path.exists():
            continue
        state = json.loads(state_path.read_text())
        if state.get("build_hash") == build_hash and state.get("dataset_hash") == dataset_hash:
            matching.append(path)

    running = [path for path in matching if read_build_state(path).get("status") != "complete"]
    if running:
        return running[-1]

    intact = [path for path in matching if artifact_integrity(path) is None]
    if intact:
        source = intact[-1]
        records = list((read_manifest(source).get("candidates") or {}).values())
        if records and all(record.get("status") == "completed" for record in records):
            return source
        retry = _allocate_build(
            base, build_hash, dataset_hash, stem=_version_key(source)[0]
        )
        _seed_retry(source, retry)
        return retry

    stem = _version_key(matching[-1])[0] if matching else None
    return _allocate_build(base, build_hash, dataset_hash, stem=stem)


def read_build_state(path: Path) -> dict:
    state = path / "build_state.json"
    return json.loads(state.read_text()) if state.exists() else {}


def write_build_state(path: Path, state: dict) -> None:
    _atomic_json(path / "build_state.json", state)


def update_candidate_state(path: Path, candidate: str, **values) -> dict:
    state = read_build_state(path)
    state.setdefault("candidates", {}).setdefault(candidate, {}).update(values)
    state["updated_at"] = datetime.datetime.now(datetime.timezone.utc).isoformat()
    write_build_state(path, state)
    return state


def write_manifest(path: Path, manifest: dict) -> None:
    _atomic_json(path / "manifest.json", manifest)


def read_manifest(path: Path) -> dict:
    manifest = json.loads((path / "manifest.json").read_text())
    if manifest.get("manifest_schema_version") != MANIFEST_SCHEMA_VERSION:
        raise ValueError(
            f"{path} uses an unsupported pre-v0.2 artifact; recompile from a prompt-first spec"
        )
    return manifest


def candidate_record(manifest: dict, candidate: str) -> dict:
    record = (manifest.get("candidates") or {}).get(candidate)
    if record is None:
        raise ValueError(f"build has no candidate {candidate!r}")
    return record


def completed_candidates(manifest: dict) -> dict[str, dict]:
    return {
        name: record
        for name, record in (manifest.get("candidates") or {}).items()
        if record.get("status") == "completed"
    }


def file_hashes(path: Path) -> dict[str, str]:
    excluded = {"manifest.json", "build_state.json", "provenance.local.json"}
    return {
        str(file.relative_to(path)): hashlib.sha256(file.read_bytes()).hexdigest()
        for file in sorted(path.rglob("*"))
        if file.is_file() and file.name not in excluded
    }


def artifact_integrity(path: Path) -> str | None:
    try:
        manifest = read_manifest(path)
    except (FileNotFoundError, ValueError, json.JSONDecodeError) as exc:
        return str(exc)
    expected = manifest.get("artifact_files") or {}
    actual = file_hashes(path)
    if expected != actual:
        missing = sorted(set(expected) - set(actual))
        changed = sorted(key for key in expected.keys() & actual.keys() if expected[key] != actual[key])
        extra = sorted(set(actual) - set(expected))
        return f"artifact contents changed (missing={missing}, changed={changed}, extra={extra})"
    return None


def source_drift(path: Path) -> str | None:
    provenance = path / "provenance.local.json"
    if not provenance.exists():
        return SOURCE_UNAVAILABLE
    source = json.loads(provenance.read_text()).get("source_spec")
    if not source or not Path(source).exists():
        return SOURCE_UNAVAILABLE
    try:
        live = load_spec(source)
        manifest = read_manifest(path)
    except (FileNotFoundError, ValueError):
        return SOURCE_UNAVAILABLE
    if live.decision_hash() != manifest.get("decision_hash"):
        return "prompt, contract, teacher, or decision recipe changed"
    if live.build_hash() != manifest.get("build_hash"):
        return "candidate build settings changed"
    return None


def active_path(root: Path, name: str) -> Path:
    return root / name / "active.json"


def read_active(root: Path, name: str) -> dict | None:
    path = active_path(root, name)
    return json.loads(path.read_text()) if path.exists() else None


def activate(root: Path, name: str, selection: dict) -> None:
    _atomic_json(active_path(root, name), selection)
    history = root / name / "selection-history.jsonl"
    with history.open("a", encoding="utf-8") as handle:
        handle.write(json.dumps(selection, ensure_ascii=False) + "\n")


def clear_active(root: Path, name: str, method: str = "cli") -> dict:
    event = {
        "function": name,
        "candidate": None,
        "build": None,
        "package": None,
        "selected_at": datetime.datetime.now(datetime.timezone.utc).isoformat(),
        "method": method,
    }
    active_path(root, name).unlink(missing_ok=True)
    history = root / name / "selection-history.jsonl"
    history.parent.mkdir(parents=True, exist_ok=True)
    with history.open("a", encoding="utf-8") as handle:
        handle.write(json.dumps(event) + "\n")
    return event


def resolve_build(root: Path, name: str, version: str | None = None) -> Path:
    if version:
        path = root / name / "builds" / version
    else:
        path = latest(root, name)
        if path is None:
            raise FileNotFoundError(f"no completed build for {name!r}")
    if not (path / "manifest.json").exists():
        raise FileNotFoundError(f"no completed build at {path}")
    if read_build_state(path).get("status") != "complete":
        raise ValueError(f"build {path.name!r} is not complete")
    return path


def resolve_runtime(
    root: Path,
    name: str,
    version: str | None = None,
    candidate: str | None = None,
) -> tuple[Path, str]:
    if version or candidate:
        if not version or not candidate:
            raise ValueError("explicit runtime requires both version and candidate")
        path = resolve_build(root, name, version)
        record = candidate_record(read_manifest(path), candidate)
        if record.get("status") != "completed":
            raise ValueError(f"candidate {candidate!r} did not complete")
        return path, candidate
    active = read_active(root, name)
    if active is None:
        raise FileNotFoundError(
            f"{name!r} has no active candidate; run `smallbatch select {name} <candidate>`"
        )
    path = root / name / "builds" / active["build"]
    if artifact_integrity(path):
        raise ValueError(f"active build failed integrity: {artifact_integrity(path)}")
    return path, active["candidate"]


def package_dir(root: Path, name: str, build: str, candidate: str) -> Path:
    base = root / name / "packages" / f"{build}--{candidate}"
    path = base
    revision = 1
    while path.exists():
        revision += 1
        path = base.with_name(f"{base.name}-r{revision}")
    return path
