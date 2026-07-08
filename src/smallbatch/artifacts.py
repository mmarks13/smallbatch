"""Compiled-function artifacts: versioned adapter dirs with manifests."""

from __future__ import annotations

import datetime
import json
import shutil
from pathlib import Path
from typing import Optional

from .spec import FunctionSpec, load_spec

DEFAULT_ROOT = Path("artifacts")


def new_version_dir(root: Path, name: str) -> Path:
    today = datetime.date.today().isoformat()
    d = root / name / today
    n = 1
    while d.exists():
        n += 1
        d = root / name / f"{today}-r{n}"
    d.mkdir(parents=True)
    return d


def sweep_run_dir(root: Path, name: str, sweep: str, tag: str) -> Path:
    """Artifact dir for one sweep run: root/<function>/<sweep>/<tag>.

    Stateless — an existing dir is wiped so a rerun fully replaces it. These
    nested dirs never become the deployed `smallbatch run` version because
    versions() only looks one level under the function dir.
    """
    d = root / name / sweep / tag
    if d.exists():
        shutil.rmtree(d)
    d.mkdir(parents=True)
    return d


def sweep_runs(root: Path, name: str) -> list[Path]:
    """All sweep run dirs (depth-2 manifests) under a function, for `status`."""
    base = root / name
    if not base.is_dir():
        return []
    out = []
    for sweep_dir in sorted(p for p in base.iterdir() if p.is_dir()):
        out.extend(
            sorted(p for p in sweep_dir.iterdir() if (p / "manifest.json").exists())
        )
    return out


def write_manifest(version_dir: Path, manifest: dict) -> None:
    (version_dir / "manifest.json").write_text(json.dumps(manifest, indent=2))


def read_manifest(version_dir: Path) -> dict:
    return json.loads((version_dir / "manifest.json").read_text())


def versions(root: Path, name: str) -> list[Path]:
    base = root / name
    if not base.is_dir():
        return []
    return sorted(p for p in base.iterdir() if (p / "manifest.json").exists())


def latest(root: Path, name: str, passing_only: bool = True) -> Optional[Path]:
    for v in reversed(versions(root, name)):
        if not passing_only or read_manifest(v)["gate"]["passed"]:
            return v
    return None


def resolve_version(
    root: Path, name: str, version: Optional[str], allow_failed: bool
) -> Path:
    """Pick an artifact version dir (explicit name or latest), refusing
    gate-failed artifacts unless allow_failed."""
    if version:
        d = root / name / version
        if not (d / "manifest.json").exists():
            raise FileNotFoundError(f"no manifest under {d}")
    else:
        d = latest(root, name, passing_only=not allow_failed)
        if d is None:
            raise FileNotFoundError(
                f"no {'passing ' if not allow_failed else ''}artifact for "
                f"'{name}' under {root}"
            )
    if not read_manifest(d)["gate"]["passed"] and not allow_failed:
        raise ValueError(f"{d} failed its gate; use --allow-failed to override")
    return d


def staleness(version_dir: Path) -> Optional[str]:
    """None if fresh; otherwise a human-readable reason the artifact is stale.

    Compares the manifest's recorded spec_hash against a re-hash of the spec
    copy's referenced spec_files today, plus the live spec if it still exists.
    """
    manifest = read_manifest(version_dir)
    spec_path = version_dir / "spec.yaml"
    if not spec_path.exists():
        return "no spec.yaml archived with artifact"
    try:
        current = load_spec(spec_path).spec_hash()
    except FileNotFoundError as e:
        return f"spec file missing: {e}"
    if current != manifest["spec_hash"]:
        return "spec or a spec_file changed since compile"
    return None
