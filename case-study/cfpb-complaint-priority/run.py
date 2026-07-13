"""Verify the frozen CFPB case inputs and compile all configured candidates."""

from __future__ import annotations

import argparse
import hashlib
import json
import shutil
import subprocess
from pathlib import Path

HERE = Path(__file__).resolve().parent
WORK = HERE / "work"
RESULTS = HERE / "results"


def verify_frozen() -> dict:
    frozen = json.loads((HERE / "frozen_ids.json").read_text())
    checks = {
        "spec_sha256": hashlib.sha256((HERE / "spec.yaml").read_bytes()).hexdigest(),
        "items_sha256": hashlib.sha256((HERE / "items.jsonl").read_bytes()).hexdigest(),
    }
    for key, value in checks.items():
        if frozen.get(key) != value:
            raise RuntimeError(f"frozen {key} mismatch")
    return frozen


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--select", help="explicit candidate ID to package after compile")
    args = parser.parse_args()
    frozen = verify_frozen()
    data = WORK / "data"
    if not (data / "meta.json").exists():
        raise SystemExit("label the frozen items first; see README.md")
    artifacts = WORK / "artifacts"
    command = [
        "smallbatch",
        "compile",
        str(HERE / "spec.yaml"),
        "--data",
        str(data),
        "--artifacts",
        str(artifacts),
    ]
    subprocess.run(command, check=True)
    builds = sorted((artifacts / "complaint-review-priority" / "builds").iterdir())
    build = builds[-1]
    if args.select:
        subprocess.run(
            [
                "smallbatch",
                "select",
                "complaint-review-priority",
                args.select,
                "--version",
                build.name,
                "--artifacts",
                str(artifacts),
            ],
            check=True,
        )
    RESULTS.mkdir(exist_ok=True)
    shutil.copy(build / "report.json", RESULTS / "results.json")
    shutil.copy(build / "report.md", RESULTS / "report.md")
    (RESULTS / "protocol.json").write_text(
        json.dumps(
            {
                "frozen": frozen,
                "build": build.name,
                "explicit_selection": args.select,
                "correctness_claim": False,
            },
            indent=2,
        )
    )
    print(f"aggregate results -> {RESULTS}")


if __name__ == "__main__":
    main()
