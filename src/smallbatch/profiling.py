"""Isolated full-evaluation CPU profiling for completed candidates."""

from __future__ import annotations

import importlib.metadata
import json
import os
import platform
import resource
import subprocess
import sys
import time
from pathlib import Path
from typing import Any


def profile_candidate(
    build: Path,
    candidate: str,
    items: list[dict],
    *,
    threads: int | None = None,
) -> dict[str, Any]:
    threads = max(1, threads or min(4, os.cpu_count() or 1))
    request = build / f".profile-{candidate}.json"
    result = build / f".profile-{candidate}-result.json"
    request.write_text(json.dumps({"items": items, "threads": threads}))
    env = {
        **os.environ,
        "CUDA_VISIBLE_DEVICES": "",
        "OMP_NUM_THREADS": str(threads),
        "MKL_NUM_THREADS": str(threads),
        "OPENBLAS_NUM_THREADS": str(threads),
        "TOKENIZERS_PARALLELISM": "false",
    }
    command = [
        sys.executable,
        "-m",
        "smallbatch.profiling",
        "--worker",
        str(build),
        candidate,
        str(request),
        str(result),
    ]
    try:
        completed = subprocess.run(command, env=env, text=True, capture_output=True, check=False)
        if completed.returncode:
            detail = completed.stderr.strip() or completed.stdout.strip()
            raise RuntimeError(f"CPU profile failed for {candidate}: {detail}")
        return json.loads(result.read_text())
    finally:
        request.unlink(missing_ok=True)
        result.unlink(missing_ok=True)


def profile_zeroshot(
    build: Path,
    diagnostic_id: str,
    base_model: str,
    items: list[dict],
    *,
    threads: int | None = None,
) -> dict[str, Any]:
    threads = max(1, threads or min(4, os.cpu_count() or 1))
    request = build / f".profile-{diagnostic_id}.json"
    result = build / f".profile-{diagnostic_id}-result.json"
    request.write_text(
        json.dumps(
            {
                "mode": "zeroshot",
                "items": items,
                "threads": threads,
                "base_model": base_model,
            }
        )
    )
    env = {
        **os.environ,
        "CUDA_VISIBLE_DEVICES": "",
        "OMP_NUM_THREADS": str(threads),
        "MKL_NUM_THREADS": str(threads),
        "OPENBLAS_NUM_THREADS": str(threads),
        "TOKENIZERS_PARALLELISM": "false",
    }
    command = [
        sys.executable,
        "-m",
        "smallbatch.profiling",
        "--worker",
        str(build),
        diagnostic_id,
        str(request),
        str(result),
    ]
    try:
        completed = subprocess.run(command, env=env, text=True, capture_output=True, check=False)
        if completed.returncode:
            detail = completed.stderr.strip() or completed.stdout.strip()
            raise RuntimeError(f"CPU zero-shot profile failed: {detail}")
        return json.loads(result.read_text())
    finally:
        request.unlink(missing_ok=True)
        result.unlink(missing_ok=True)


def _worker(build: Path, candidate: str, request_path: Path, result_path: Path) -> None:
    from . import artifacts
    from .runtime import load_candidate

    request = json.loads(request_path.read_text())
    items = request["items"]
    threads = int(request["threads"])
    try:
        import torch

        torch.set_num_threads(threads)
        torch.set_num_interop_threads(1)
    except (ImportError, RuntimeError):
        pass

    started = time.perf_counter()
    if request.get("mode") == "zeroshot":
        function = _load_zeroshot(build, request["base_model"])
        record = {"backend": "zeroshot", "base_model": request["base_model"]}
        model_dir = None
    else:
        function = load_candidate(build, candidate)
        manifest = artifacts.read_manifest(build)
        record = artifacts.candidate_record(manifest, candidate)
        model_dir = build / record["artifact_path"]
    cold_load = time.perf_counter() - started
    for item in items[: min(3, len(items))]:
        function(item)
    latencies: list[float] = []
    outputs: list[Any] = []
    for item in items:
        started = time.perf_counter()
        outputs.append(function(item))
        latencies.append((time.perf_counter() - started) * 1000)

    dependencies = _runtime_dependencies(record["backend"])
    peak = resource.getrusage(resource.RUSAGE_SELF).ru_maxrss
    if sys.platform != "darwin":
        peak *= 1024
    result = {
        "predictions": outputs,
        "profile": {
            "runtime": record["backend"],
            "cold_load_seconds": round(cold_load, 4),
            "batch_one_latency_ms": {
                "p50": round(_percentile(latencies, 0.5), 4) if latencies else None,
                "p95": round(_percentile(latencies, 0.95), 4) if latencies else None,
                "n": len(latencies),
            },
            "peak_rss_bytes": int(peak),
            "candidate_owned_bytes": artifacts.dir_size(model_dir) if model_dir else 0,
            "required_shared_bytes": _shared_bytes(record),
            "required_base_model": record.get("base_model"),
            "dependencies": dependencies,
            "cpu": _cpu_name(),
            "threads": threads,
            "os": platform.platform(),
            "python": platform.python_version(),
            "offline_after_install": record["backend"] in {"tfidf", "setfit"},
        },
    }
    result_path.write_text(json.dumps(result, ensure_ascii=False))


def _load_zeroshot(build: Path, base_model: str):
    from . import prompts
    from .evaluate import generate_batch
    from .spec import load_spec, validate_input, validate_output
    from .training import load_base_model

    spec = load_spec(build / "spec.yaml")
    tokenizer, model = load_base_model(base_model, "fp32")
    model.eval()

    def call(item):
        normalized = validate_input(spec, item)
        raw, _ = generate_batch(
            model,
            tokenizer,
            [prompts.zeroshot_prompt(spec, normalized)],
            max(16, prompts.completion_budget(spec)),
            batch_size=1,
            allowed_completions=prompts.allowed_completions(spec),
        )
        return validate_output(spec, prompts.parse_output(spec, raw[0]))

    return call


def _percentile(values: list[float], quantile: float) -> float:
    ordered = sorted(values)
    index = max(0, min(len(ordered) - 1, int((len(ordered) - 1) * quantile + 0.999999)))
    return ordered[index]


def _runtime_dependencies(backend: str) -> dict[str, str]:
    names = {
        "tfidf": ["scikit-learn", "skops", "numpy", "scipy"],
        "setfit": ["setfit", "sentence-transformers", "torch", "transformers"],
        "lora": ["torch", "transformers", "peft"],
        "zeroshot": ["torch", "transformers"],
    }[backend]
    versions = {}
    for name in names:
        try:
            versions[name] = importlib.metadata.version(name)
        except importlib.metadata.PackageNotFoundError:
            versions[name] = "unknown"
    return versions


def _shared_bytes(record: dict) -> int | None:
    model_id = record.get("base_model")
    if not model_id:
        return 0
    try:
        from huggingface_hub import scan_cache_dir

        cache = scan_cache_dir()
        return sum(repo.size_on_disk for repo in cache.repos if repo.repo_id == model_id)
    except Exception:  # cache inspection is best-effort metadata
        return None


def _cpu_name() -> str:
    if Path("/proc/cpuinfo").exists():
        for line in Path("/proc/cpuinfo").read_text(errors="ignore").splitlines():
            if line.lower().startswith("model name"):
                return line.split(":", 1)[1].strip()
    return platform.processor() or "unknown"


def main(argv: list[str] | None = None) -> int:
    argv = list(argv or sys.argv[1:])
    if len(argv) == 5 and argv[0] == "--worker":
        _worker(Path(argv[1]), argv[2], Path(argv[3]), Path(argv[4]))
        return 0
    raise SystemExit("profiling is an internal worker module")


if __name__ == "__main__":
    raise SystemExit(main())
