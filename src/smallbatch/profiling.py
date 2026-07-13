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

PROFILE_HEARTBEAT_SECONDS = 30


def _profile_environment(threads: int) -> dict[str, str]:
    return {
        **os.environ,
        "CUDA_VISIBLE_DEVICES": "",
        "OMP_NUM_THREADS": str(threads),
        "MKL_NUM_THREADS": str(threads),
        "OPENBLAS_NUM_THREADS": str(threads),
        "TOKENIZERS_PARALLELISM": "false",
    }


def _read_progress(path: Path) -> dict[str, Any]:
    try:
        return json.loads(path.read_text())
    except (FileNotFoundError, json.JSONDecodeError):
        return {}


def _wait_for_profile(process: subprocess.Popen, progress: Path, label: str) -> tuple[str, str]:
    started = time.perf_counter()
    while True:
        try:
            stdout, stderr = process.communicate(timeout=PROFILE_HEARTBEAT_SECONDS)
            return stdout, stderr
        except subprocess.TimeoutExpired:
            state = _read_progress(progress)
            completed = state.get("completed", 0)
            total = state.get("total", "?")
            elapsed = time.perf_counter() - started
            print(
                f"[smallbatch] CPU evaluation progress {label}: "
                f"rows={completed}/{total} elapsed={elapsed:.1f}s",
                file=sys.stderr,
                flush=True,
            )


def _run_profile(
    build: Path,
    profile_id: str,
    request_value: dict[str, Any],
    *,
    threads: int,
    error_prefix: str,
) -> dict[str, Any]:
    request = build / f".profile-{profile_id}.json"
    result = build / f".profile-{profile_id}-result.json"
    progress = build / f".profile-{profile_id}-progress.json"
    request.write_text(
        json.dumps(
            {
                **request_value,
                "threads": threads,
                "progress_path": str(progress),
            }
        )
    )
    command = [
        sys.executable,
        "-m",
        "smallbatch.profiling",
        "--worker",
        str(build),
        profile_id,
        str(request),
        str(result),
    ]
    try:
        process = subprocess.Popen(
            command,
            env=_profile_environment(threads),
            text=True,
            stdout=subprocess.PIPE,
            stderr=subprocess.PIPE,
        )
        stdout, stderr = _wait_for_profile(process, progress, profile_id)
        if process.returncode:
            detail = stderr.strip() or stdout.strip()
            raise RuntimeError(f"{error_prefix}: {detail}")
        return json.loads(result.read_text())
    finally:
        request.unlink(missing_ok=True)
        result.unlink(missing_ok=True)
        progress.unlink(missing_ok=True)
        progress.with_suffix(progress.suffix + ".tmp").unlink(missing_ok=True)


def profile_candidate(
    build: Path,
    candidate: str,
    items: list[dict],
    *,
    threads: int | None = None,
) -> dict[str, Any]:
    threads = max(1, threads or min(4, os.cpu_count() or 1))
    return _run_profile(
        build,
        candidate,
        {"items": items},
        threads=threads,
        error_prefix=f"CPU profile failed for {candidate}",
    )


def profile_zeroshot(
    build: Path,
    diagnostic_id: str,
    base_model: str,
    items: list[dict],
    *,
    threads: int | None = None,
) -> dict[str, Any]:
    threads = max(1, threads or min(4, os.cpu_count() or 1))
    return _run_profile(
        build,
        diagnostic_id,
        {"mode": "zeroshot", "items": items, "base_model": base_model},
        threads=threads,
        error_prefix="CPU zero-shot profile failed",
    )


def _write_progress(path: Path, completed: int, total: int) -> None:
    tmp = path.with_suffix(path.suffix + ".tmp")
    tmp.write_text(json.dumps({"completed": completed, "total": total}))
    tmp.replace(path)


def _worker(build: Path, candidate: str, request_path: Path, result_path: Path) -> None:
    from . import artifacts
    from .runtime import load_candidate

    request = json.loads(request_path.read_text())
    items = request["items"]
    threads = int(request["threads"])
    progress_path = Path(request["progress_path"]) if request.get("progress_path") else None
    if progress_path:
        _write_progress(progress_path, 0, len(items))
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
    progress_written = time.perf_counter()
    for index, item in enumerate(items, 1):
        started = time.perf_counter()
        outputs.append(function(item))
        latencies.append((time.perf_counter() - started) * 1000)
        now = time.perf_counter()
        if progress_path and (index == len(items) or now - progress_written >= 1):
            _write_progress(progress_path, index, len(items))
            progress_written = now

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
            "candidate_owned_bytes": (
                artifacts.deployable_size(model_dir, record["backend"]) if model_dir else 0
            ),
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
    from .spec import load_spec, validate_input
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
        return prompts.parse_output(spec, raw[0])

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
