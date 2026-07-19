"""gpu-release: the release gate. Precision paths, resume, and OOM backoff.

Everything genuinely untestable without a GPU and too slow for the PR gate.
Run the whole tier (smoke included) with

    SMALLBATCH_GPU_TESTS=1 pytest tests_gpu -q
"""

from __future__ import annotations

import json
import os
import signal
import subprocess
import sys
import time
from pathlib import Path

import pytest
from gpu_cases import build_spec, lora_candidate, triage_records

pytestmark = pytest.mark.gpu_release


def _bf16_supported() -> bool:
    import torch

    return torch.cuda.get_device_capability()[0] >= 8


def _bitsandbytes_available() -> bool:
    import importlib.util

    return importlib.util.find_spec("bitsandbytes") is not None


def _train_bounded(tmp_path, **candidate_overrides) -> dict:
    from smallbatch.api import compile as compile_fn
    from smallbatch.api import label

    spec = build_spec(
        "gpu-precision",
        {"type": "int", "range": [0, 4]},
        candidate=lora_candidate(max_epochs=3, patience=None, **candidate_overrides),
    )
    label(spec, triage_records(text=False)[:60], out_dir=tmp_path / "data")
    compiled = compile_fn(
        spec, data_dir=tmp_path / "data", artifacts_root=tmp_path / "artifacts",
        cpu_threads=4,
    )
    record = compiled.candidates["student"]
    assert record["status"] == "completed", record.get("error")
    return record


def test_fp32_precision_trains(tmp_path):
    record = _train_bounded(tmp_path, precision="fp32")
    assert record["train_precision"] == "fp32"


@pytest.mark.skipif(not _bf16_supported(), reason="GPU predates bf16 (capability < 8)")
def test_bf16_precision_trains(tmp_path):
    record = _train_bounded(tmp_path, precision="bf16")
    assert record["train_precision"] == "bf16"


@pytest.mark.skipif(
    not _bitsandbytes_available(), reason="bitsandbytes is not installed"
)
def test_qlora_precision_trains(tmp_path):
    record = _train_bounded(tmp_path, precision="qlora")
    assert record["train_precision"] == "qlora"


def test_gradient_checkpointing_trains(tmp_path):
    record = _train_bounded(tmp_path, gradient_checkpointing=True)
    assert record["metrics"]["invalid_rate"] == 0.0


def test_interrupted_compile_resumes_from_checkpoint(tmp_path):
    """Kill a compile mid-training; the rerun must resume the trainer
    checkpoint and complete instead of starting over or failing."""
    from smallbatch.api import compile as compile_fn
    from smallbatch.api import label

    spec = build_spec(
        "gpu-resume",
        {"type": "int", "range": [0, 4]},
        candidate=lora_candidate(max_epochs=6, patience=None),
    )
    data = tmp_path / "data"
    root = tmp_path / "artifacts"
    label(spec, triage_records(text=False), out_dir=data)
    spec_path = tmp_path / "spec.yaml"
    import yaml

    spec_path.write_text(yaml.safe_dump(spec.model_dump(mode="json"), sort_keys=False))

    script = (
        "from smallbatch.api import compile as c\n"
        f"c({str(spec_path)!r}, data_dir={str(data)!r}, artifacts_root={str(root)!r}, cpu_threads=1)\n"
    )
    process = subprocess.Popen(
        [sys.executable, "-c", script],
        cwd=Path.cwd(),
        env={**os.environ},
    )
    # wait for the first trainer checkpoint, then kill without warning
    deadline = time.time() + 600
    checkpoint = None
    try:
        while time.time() < deadline:
            candidates = list(root.glob("*/builds/*/candidates/student/trainer/checkpoint-*"))
            if candidates:
                checkpoint = candidates[0]
                break
            if process.poll() is not None:
                pytest.fail("compile finished before a checkpoint appeared; slow the config down")
            time.sleep(2)
    finally:
        if process.poll() is None:
            process.send_signal(signal.SIGKILL)
            process.wait()
    assert checkpoint is not None, "no trainer checkpoint was ever written"

    compiled = compile_fn(spec, data_dir=data, artifacts_root=root, cpu_threads=4)
    record = compiled.candidates["student"]
    assert record["status"] == "completed", record.get("error")
    assert record["training"]["epochs_run"] >= 1
    state = json.loads(
        (compiled.version_dir / "build_state.json").read_text()
    )
    assert state["status"] == "complete"


def test_eval_oom_backoff_halves_the_batch_and_finishes(tmp_path):
    """Starve the allocator and evaluate with an oversized batch: the OOM
    backoff must halve down and still return one output per prompt."""
    import torch

    from smallbatch import prompts
    from smallbatch.evaluate import generate_batch
    from smallbatch.training import load_base_model

    spec = build_spec("gpu-oom", {"type": "int", "range": [0, 4]})
    tokenizer, model = load_base_model(lora_candidate()["model"], "fp32")
    model.eval()
    texts = [
        prompts.student_prompt(spec, record["input"])
        for record in triage_records(text=False)[:32]
    ]
    torch.cuda.empty_cache()
    fraction = 0.35  # small enough to break batch-32 fp32 generation
    torch.cuda.set_per_process_memory_fraction(fraction)
    try:
        outs, effective = generate_batch(
            model, tokenizer, texts, prompts.completion_budget(spec),
            batch_size=32,
            allowed_completions=prompts.allowed_completions(spec),
        )
    finally:
        torch.cuda.set_per_process_memory_fraction(1.0)
    assert len(outs) == len(texts)
    if effective == 32:
        pytest.skip(
            f"memory fraction {fraction} did not force an OOM on this card; "
            "lower it for this GPU"
        )
    assert effective < 32
