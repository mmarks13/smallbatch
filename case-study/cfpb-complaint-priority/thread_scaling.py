#!/usr/bin/env python3
"""Measure how each candidate's CPU latency scales with the threads it is given.

The report profiles every candidate at a fixed thread count so the candidates
are comparable to each other. That says nothing about what a candidate costs on
a bigger machine, and the answer is not obvious: Smallbatch times one decision
at a time and torch parallelizes *within* a single forward pass, so more threads
can shorten one decision -- but only for the candidates whose time is spent in
large matrix multiplies.

Writes results/thread_scaling.json (aggregate latency only, no inputs or
decisions). Refuses to profile more threads than the machine actually owns,
because oversubscribed threads measure contention, not scaling.
"""

from __future__ import annotations

import argparse
import json
import os
import platform
from pathlib import Path

HERE = Path(__file__).resolve().parent
WORK = HERE / "work"
RESULTS = HERE / "results"
THREAD_COUNTS = (4, 8, 16, 32)


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--rows", type=int, default=40, help="evaluation rows to time")
    parser.add_argument(
        "--threads",
        type=int,
        nargs="*",
        default=list(THREAD_COUNTS),
        help="thread counts to sweep",
    )
    args = parser.parse_args()

    from smallbatch import artifacts
    from smallbatch.labeling import read_jsonl
    from smallbatch.profiling import profile_candidate

    cores = os.cpu_count() or 1
    counts = [count for count in args.threads if count <= cores]
    skipped = [count for count in args.threads if count > cores]

    root = WORK / "artifacts"
    build = artifacts.latest(root, "complaint-review-priority")
    manifest = artifacts.read_manifest(build)
    candidates = [
        name
        for name, record in manifest["candidates"].items()
        if record.get("status") == "completed"
    ]
    rows = read_jsonl(build / "evaluation.local.jsonl")[: args.rows]
    items = [row["input"] for row in rows]

    print(f"[scaling] {len(candidates)} candidates x {counts} threads on {cores} cores")
    measured: dict[str, dict[str, dict]] = {}
    for candidate in candidates:
        measured[candidate] = {}
        for threads in counts:
            profile = profile_candidate(build, candidate, items, threads=threads)
            latency = profile["profile"]["batch_one_latency_ms"]
            measured[candidate][str(threads)] = {
                "p50_ms": latency["p50"],
                "p95_ms": latency["p95"],
                "cold_load_seconds": profile["profile"]["cold_load_seconds"],
            }
            print(
                f"[scaling] {candidate:12} threads={threads:>2} "
                f"p50={latency['p50']:.1f}ms p95={latency['p95']:.1f}ms",
                flush=True,
            )

    baseline = str(counts[0])
    speedup = {
        candidate: {
            threads: round(
                values[baseline]["p50_ms"] / values[threads]["p50_ms"], 2
            )
            for threads in values
        }
        for candidate, values in measured.items()
    }

    RESULTS.mkdir(exist_ok=True)
    payload = {
        "build": build.name,
        "rows_timed": len(items),
        "machine_cores": cores,
        "platform": platform.platform(),
        "threads_skipped_over_core_count": skipped,
        "latency": measured,
        "speedup_vs_lowest": speedup,
        "note": (
            "One decision is timed at a time; torch parallelizes within a single "
            "forward pass (interop threads are pinned to 1). Speedup is therefore "
            "intra-decision, and only candidates dominated by large matrix "
            "multiplies can use the extra threads."
        ),
    }
    tmp = RESULTS / "thread_scaling.json.tmp"
    tmp.write_text(json.dumps(payload, indent=2) + "\n")
    tmp.replace(RESULTS / "thread_scaling.json")
    print(f"[scaling] wrote {RESULTS / 'thread_scaling.json'}")


if __name__ == "__main__":
    main()
