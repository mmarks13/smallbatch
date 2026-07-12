"""Compiled-function artifacts: versioned adapter dirs with manifests."""

from __future__ import annotations

import datetime
import json
import re
import shutil
from pathlib import Path
from typing import Optional

from .spec import FunctionSpec, load_spec

DEFAULT_ROOT = Path("artifacts")

# manifest schema history:
#   1 (implicit): LoRA-only flat manifest, gate/metrics at top level
#   2: adds manifest_schema_version, per-candidate records under `candidates`,
#      a `selection` block naming the winner, and an optional `deployment`
#      block recording an explicit user acceptance of a gate-failed candidate
MANIFEST_SCHEMA_VERSION = 2


def winner(manifest: dict) -> str:
    """The selected candidate's name ('lora' for pre-v2 manifests)."""
    return (manifest.get("selection") or {}).get("winner", "lora")


def candidate_record(manifest: dict, name: Optional[str] = None) -> Optional[dict]:
    """A candidate's result record; synthesizes one for pre-v2 manifests so
    every consumer can speak the v2 shape."""
    name = name or winner(manifest)
    candidates = manifest.get("candidates")
    if candidates is not None:
        return candidates.get(name)
    if name != "lora":  # pre-v2 manifests only ever contain the adapter
        return None
    return {
        "backend": "lora",
        "status": "completed",
        "artifact_path": "adapter",
        "base_model": manifest.get("base_model"),
        "inference_precision": manifest.get("inference_precision"),
        "metrics": (manifest.get("metrics") or {}).get("adapter"),
        "gate": manifest.get("gate", {}),
        "error": None,
    }


def candidate_is_usable(
    manifest: dict, candidate: Optional[str] = None, allow_failed: bool = False
) -> bool:
    """One predicate for every consumer (run/load_fn/status/serve/export):
    a candidate is usable when its own gate passed, or the user explicitly
    accepted THIS candidate despite a failed gate, or the caller opted into
    failed artifacts. Acceptance never leaks to other retained candidates."""
    rec = candidate_record(manifest, candidate)
    if rec is None or rec.get("status") != "completed":
        return False
    if allow_failed:
        return True
    if (rec.get("gate") or {}).get("passed"):
        return True
    accepted = (manifest.get("deployment") or {}).get("accepted_candidate")
    return accepted is not None and accepted == (candidate or winner(manifest))


def artifact_is_usable(manifest: dict) -> bool:
    """Usability of the artifact's *selected* candidate — never a grant to
    every retained candidate."""
    return candidate_is_usable(manifest, winner(manifest))


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


def dir_size(path: Path) -> int:
    """Total bytes of files under `path` (a candidate's incremental artifact size)."""
    return sum(f.stat().st_size for f in path.rglob("*") if f.is_file())


def write_manifest(version_dir: Path, manifest: dict) -> None:
    (version_dir / "manifest.json").write_text(json.dumps(manifest, indent=2))


def read_manifest(version_dir: Path) -> dict:
    return json.loads((version_dir / "manifest.json").read_text())


def _version_key(p: Path) -> tuple[str, int]:
    """Sort '<date>' < '<date>-r2' < ... < '<date>-r10' correctly: plain
    lexicographic ordering puts -r10 before -r9."""
    m = re.fullmatch(r"(.*?)(?:-r(\d+))?", p.name)
    return (m.group(1), int(m.group(2) or 1))


def versions(root: Path, name: str) -> list[Path]:
    base = root / name
    if not base.is_dir():
        return []
    return sorted(
        (p for p in base.iterdir() if (p / "manifest.json").exists()),
        key=_version_key,
    )


def latest(root: Path, name: str, passing_only: bool = True) -> Optional[Path]:
    for v in reversed(versions(root, name)):
        if not passing_only or artifact_is_usable(read_manifest(v)):
            return v
    return None


def resolve_version(
    root: Path,
    name: str,
    version: Optional[str],
    allow_failed: bool,
    candidate: Optional[str] = None,
) -> Path:
    """Pick an artifact version dir (explicit name or latest), refusing
    unusable candidates unless allow_failed. `candidate` scopes the usability
    check to an explicitly requested candidate (an accepted winner never makes
    an unaccepted secondary usable)."""
    if version:
        d = root / name / version
        if not (d / "manifest.json").exists():
            raise FileNotFoundError(f"no manifest under {d}")
    else:
        d = latest(root, name, passing_only=not allow_failed)
        if d is None:
            raise FileNotFoundError(
                f"no {'usable ' if not allow_failed else ''}artifact for "
                f"'{name}' under {root}"
            )
    manifest = read_manifest(d)
    if candidate and candidate_record(manifest, candidate) is None:
        raise ValueError(f"{d} has no '{candidate}' candidate")
    if not candidate_is_usable(manifest, candidate, allow_failed=allow_failed):
        raise ValueError(
            f"{d}: candidate '{candidate or winner(manifest)}' failed its gate "
            "and was not accepted; use --allow-failed to override"
        )
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
