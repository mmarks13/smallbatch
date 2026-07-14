"""Verify the frozen CFPB case inputs and compile all configured candidates."""

from __future__ import annotations

import argparse
import datetime
import hashlib
import json
import platform
import shutil
import sys
from pathlib import Path

HERE = Path(__file__).resolve().parent
WORK = HERE / "work"
RESULTS = HERE / "results"
EXPECTED_CANDIDATES = {"tfidf", "bge-small", "qwen35-08b", "qwen35-2b", "qwen35-4b"}
EXPECTED_COUNT = 600
EXPECTED_SOURCE = "https://www.consumerfinance.gov/data-research/consumer-complaints/search/api/v1/"
EXPECTED_API_LICENSE = "CC0"
EXPECTED_SELECTION = (
    "normalize, narrative length 200-4000, content dedupe, product round-robin"
)


def atomic_copy(source: Path, destination: Path) -> None:
    tmp = destination.with_suffix(destination.suffix + ".tmp")
    shutil.copyfile(source, tmp)
    tmp.replace(destination)


def atomic_json(path: Path, value: dict) -> None:
    tmp = path.with_suffix(path.suffix + ".tmp")
    tmp.write_text(json.dumps(value, indent=2) + "\n")
    tmp.replace(path)


def verify_frozen() -> dict:
    frozen = json.loads((HERE / "frozen_ids.json").read_text())
    checks = {
        "source": EXPECTED_SOURCE,
        "api_license": EXPECTED_API_LICENSE,
        "selection": EXPECTED_SELECTION,
        "spec_sha256": hashlib.sha256((HERE / "spec.yaml").read_bytes()).hexdigest(),
        "items_sha256": hashlib.sha256((HERE / "items.jsonl").read_bytes()).hexdigest(),
    }
    for key, value in checks.items():
        if frozen.get(key) != value:
            raise RuntimeError(f"frozen {key} mismatch")
    extracted_at = frozen.get("extracted_at")
    try:
        extracted = datetime.datetime.fromisoformat(extracted_at)
    except (TypeError, ValueError) as exc:
        raise RuntimeError("frozen extraction timestamp is invalid") from exc
    if extracted.tzinfo is None:
        raise RuntimeError("frozen extraction timestamp must include a timezone")
    items = [json.loads(line) for line in (HERE / "items.jsonl").read_text().splitlines()]
    complaints = frozen.get("complaints") or []
    if frozen.get("count") != EXPECTED_COUNT or len(items) != EXPECTED_COUNT:
        raise RuntimeError(f"frozen case must contain exactly {EXPECTED_COUNT} inputs")
    if len(complaints) != EXPECTED_COUNT:
        raise RuntimeError(f"frozen case must contain exactly {EXPECTED_COUNT} complaint records")
    for index, (item, complaint) in enumerate(zip(items, complaints)):
        if not isinstance(item, dict) or set(item) != {"input", "provenance"}:
            raise RuntimeError(f"frozen item schema mismatch at row {index}")
        input_value = item.get("input")
        if not isinstance(input_value, dict) or set(input_value) != {
            "product",
            "issue",
            "narrative",
        }:
            raise RuntimeError(f"frozen input schema mismatch at row {index}")
        if any(
            not isinstance(input_value[field], str) or not input_value[field].strip()
            for field in ("product", "issue", "narrative")
        ):
            raise RuntimeError(f"frozen input value missing at row {index}")
        narrative = input_value["narrative"]
        if not 200 <= len(narrative) <= 4000:
            raise RuntimeError(f"frozen narrative length invalid at row {index}")
        provenance = item.get("provenance")
        if not isinstance(provenance, dict) or set(provenance) != {
            "complaint_id",
            "content_sha256",
        }:
            raise RuntimeError(f"frozen complaint provenance schema mismatch at row {index}")
        if not isinstance(complaint, dict) or set(complaint) != set(provenance):
            raise RuntimeError(f"frozen complaint record schema mismatch at row {index}")
        if provenance != complaint:
            raise RuntimeError(f"frozen complaint provenance mismatch at row {index}")
        complaint_id = complaint.get("complaint_id")
        content_hash = complaint.get("content_sha256")
        if not isinstance(complaint_id, str) or not complaint_id.isdecimal():
            raise RuntimeError(f"frozen complaint ID invalid at row {index}")
        if (
            not isinstance(content_hash, str)
            or len(content_hash) != 64
            or any(character not in "0123456789abcdef" for character in content_hash)
        ):
            raise RuntimeError(f"frozen complaint content hash invalid at row {index}")
        if hashlib.sha256(narrative.encode()).hexdigest() != complaint.get("content_sha256"):
            raise RuntimeError(f"frozen complaint content hash mismatch at row {index}")
    if len({row["complaint_id"] for row in complaints}) != EXPECTED_COUNT:
        raise RuntimeError("frozen complaint IDs are not unique")
    if len({row["content_sha256"] for row in complaints}) != EXPECTED_COUNT:
        raise RuntimeError("frozen complaint narratives are not unique")
    return frozen


def require_all_candidates(report: dict) -> None:
    records = report.get("candidates") or {}
    missing = EXPECTED_CANDIDATES - set(records)
    incomplete = {
        name: records.get(name, {}).get("status", "missing")
        for name in EXPECTED_CANDIDATES
        if records.get(name, {}).get("status") != "completed"
    }
    if missing or incomplete:
        details = ", ".join(f"{name}={status}" for name, status in sorted(incomplete.items()))
        raise RuntimeError(
            "CFPB release evidence requires all configured candidates to complete"
            + (f": {details}" if details else "")
        )


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--select", help="explicit candidate ID to package after compile")
    parser.add_argument("--cpu-threads", type=int, default=4)
    args = parser.parse_args()
    frozen = verify_frozen()
    data = WORK / "data"
    if not (data / "meta.json").exists():
        raise SystemExit("label the frozen items first; see README.md")
    from smallbatch import __version__
    from smallbatch.api import compile as compile_fn
    from smallbatch.api import select

    artifact_root = WORK / "artifacts"
    result = compile_fn(
        HERE / "spec.yaml",
        data_dir=data,
        artifacts_root=artifact_root,
        cpu_threads=args.cpu_threads,
    )
    require_all_candidates(result.report)
    if args.select:
        select(
            "complaint-review-priority",
            args.select,
            version=result.build_id,
            artifacts_root=artifact_root,
        )
    RESULTS.mkdir(exist_ok=True)
    atomic_copy(result.report_path, RESULTS / "results.json")
    atomic_copy(result.version_dir / "report.md", RESULTS / "report.md")
    atomic_json(
        RESULTS / "protocol.json",
        {
            "executed_at": datetime.datetime.now(datetime.timezone.utc).isoformat(),
            "frozen": frozen,
            "build": result.build_id,
            "smallbatch_version": __version__,
            "python_version": platform.python_version(),
            "platform": platform.platform(),
            "cpu_threads": args.cpu_threads,
            "candidate_statuses": {
                name: record.get("status")
                for name, record in result.report["candidates"].items()
            },
            "explicit_selection": args.select,
            "correctness_claim": False,
            "energy_measured": False,
            "command": [sys.executable, str(Path(__file__).resolve()), "--cpu-threads", str(args.cpu_threads)],
        },
    )
    print(f"aggregate results -> {RESULTS}")


if __name__ == "__main__":
    main()
