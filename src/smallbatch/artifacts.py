"""Compiled-function artifacts: versioned adapter dirs with manifests."""

from __future__ import annotations

import datetime
import json
import re
import shutil
from pathlib import Path

from .spec import load_spec

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


def candidate_record(manifest: dict, name: str | None = None) -> dict | None:
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
    manifest: dict, candidate: str | None = None, allow_failed: bool = False
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


def select_winner(candidates: dict[str, dict], tie_margin: float) -> dict:
    """Explicit selection order over candidate result records:

    1. only completed candidates compete;
    2. if any candidate passed every gate check, choose among passing ones
       (a candidate passing all required fields always beats a field-failing
       one, whatever their joint headlines);
    3. highest gate agreement wins;
    4. within `tie_margin` (a fixed pragmatic margin, not CI equivalence),
       the smaller artifact wins.
    """
    completed = {n: c for n, c in candidates.items() if c.get("status") == "completed"}
    if not completed:
        raise ValueError("no completed candidates to select from")
    passing = {n: c for n, c in completed.items() if (c.get("gate") or {}).get("passed")}
    pool = passing or completed

    def agreement(rec: dict) -> float:
        return (rec.get("metrics") or {}).get("agreement") or 0.0

    def size(name: str) -> float:
        return pool[name].get("artifact_size_bytes") or float("inf")

    best = max(agreement(c) for c in pool.values())
    tied = [n for n, c in pool.items() if best - agreement(c) <= tie_margin]
    scope = "passing candidates" if passing else "completed candidates (none passed)"
    if len(tied) == 1:
        return {"winner": tied[0], "reason": f"highest gate agreement among {scope}"}
    winner = min(tied, key=size)
    return {
        "winner": winner,
        "reason": f"tie within {tie_margin:.0%} among {scope} — smallest artifact",
    }


def accept_candidate(version_dir: Path, candidate: str, via: str) -> dict:
    """Record the user's explicit decision to deploy `candidate` despite a
    failed gate. Never rewrites the gate result; usability comes from the
    deployment block (candidate-scoped)."""
    manifest = read_manifest(version_dir)
    manifest["deployment"] = {
        "accepted_despite_gate": True,
        "accepted_candidate": candidate,
        "accepted_at": datetime.datetime.now(datetime.timezone.utc).isoformat(),
        "accepted_via": via,
    }
    write_manifest(version_dir, manifest)
    return manifest


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


def latest(root: Path, name: str, passing_only: bool = True) -> Path | None:
    for v in reversed(versions(root, name)):
        if not passing_only or artifact_is_usable(read_manifest(v)):
            return v
    return None


def resolve_version(
    root: Path,
    name: str,
    version: str | None,
    allow_failed: bool,
    candidate: str | None = None,
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


SOURCE_UNAVAILABLE = "source comparison unavailable"


def artifact_integrity(version_dir: Path) -> str | None:
    """None when the ARCHIVED spec + spec_files still reproduce the manifest's
    recorded spec_hash; otherwise the reason the artifact itself is damaged.
    Self-contained: never consults the original project files."""
    manifest = read_manifest(version_dir)
    spec_path = version_dir / "spec.yaml"
    if not spec_path.exists():
        return "no spec.yaml archived with artifact"
    try:
        current = load_spec(spec_path).spec_hash()
    except FileNotFoundError as e:
        return f"archived spec file missing: {e}"
    if manifest.get("spec_hash") and current != manifest["spec_hash"]:
        return "archived spec/spec_files no longer match the manifest"
    return None


def source_drift(version_dir: Path) -> str | None:
    """None when the live source spec still matches the artifact snapshot;
    SOURCE_UNAVAILABLE when the original project can't be found (a moved or
    deleted source never makes a self-contained artifact unusable); otherwise
    what drifted. Reads provenance.local.json, which stays on this machine."""
    prov_path = version_dir / "provenance.local.json"
    if not prov_path.exists():
        return SOURCE_UNAVAILABLE
    source = json.loads(prov_path.read_text()).get("source_spec")
    if not source or not Path(source).exists():
        return SOURCE_UNAVAILABLE
    try:
        live = load_spec(source)
        archived = load_spec(version_dir / "spec.yaml")
        if live.labeling_hash() != archived.labeling_hash():
            return "rubric/contract/teacher changed since compile — labels and artifact no longer describe the live spec"
        if live.spec_hash() != archived.spec_hash():
            return "build settings changed since compile"
    except (FileNotFoundError, ValueError):
        return SOURCE_UNAVAILABLE
    return None


def staleness(version_dir: Path) -> str | None:
    """Legacy single-string view: an integrity failure, else source drift.
    A missing source project is NOT staleness — the artifact is an immutable
    snapshot and stays valid."""
    broken = artifact_integrity(version_dir)
    if broken:
        return broken
    drift = source_drift(version_dir)
    return None if drift == SOURCE_UNAVAILABLE else drift
