"""Programmatic labeling, compilation, selection, and loading APIs."""

from __future__ import annotations

import datetime
import hashlib
import json
import os
import sys
import time
from collections.abc import Callable
from dataclasses import dataclass
from pathlib import Path

import yaml

from . import artifacts
from .atomic import atomic_json
from .labeling import dataset_hash, normalize_item_records, read_jsonl
from .spec import (
    FunctionSpec,
    LoraCandidateSpec,
    SetFitCandidateSpec,
    TfidfCandidateSpec,
    load_spec,
)


@dataclass
class LabelResult:
    out_dir: Path
    meta: dict

    @property
    def compressed(self) -> bool:
        histogram = self.meta.get("label_histogram") or {}
        total = sum(histogram.values())
        return bool(total) and max(histogram.values()) / total > 0.5


@dataclass
class CompileResult:
    version_dir: Path
    build_id: str
    candidates: dict
    manifest: dict
    report: dict
    report_path: Path


@dataclass
class SelectionResult:
    function: str
    build_id: str
    candidate: str
    package_dir: Path
    wheel: Path
    evidence: dict
    active: dict


def _as_spec(spec: FunctionSpec | str | Path) -> FunctionSpec:
    return spec if isinstance(spec, FunctionSpec) else load_spec(spec)


def _progress(message: str) -> None:
    print(f"[smallbatch] {message}", file=sys.stderr, flush=True)


def label(
    spec: FunctionSpec | str | Path,
    items: list[dict],
    out_dir: str | Path | None = None,
    append: bool = False,
    max_variants: int | None = None,
    *,
    skip_calibration: bool = False,
    interactive: bool | None = None,
    input_fn: Callable[[str], str] = input,
) -> LabelResult:
    """Import complete decisions or generate them with an approved teacher."""
    from .calibration import calibrate_teacher
    from .labeling import build_dataset
    from .teacher import make_teacher

    spec = _as_spec(spec)
    out = Path(out_dir or f"data/{spec.name}")
    inputs, imported = normalize_item_records(spec, items)
    source = "imported" if imported is not None else "teacher"
    _progress(
        f"label start function={spec.name} source={source} items={len(items)} "
        f"augmentation={'yes' if spec.augmentation else 'no'}"
    )
    teacher = None
    force_train_ids: set[str] = set()
    if imported is None:
        if spec.teacher is None:
            raise ValueError("unlabeled inputs require a `teacher` block")
        teacher = make_teacher(spec.teacher)
        calibration = calibrate_teacher(
            teacher,
            spec,
            inputs,
            out,
            skip=skip_calibration,
            interactive=interactive,
            input_fn=input_fn,
        )
        force_train_ids = set(calibration.row_ids)
    elif spec.augmentation:
        if spec.teacher is None:
            raise ValueError("augmentation of imported decisions requires a `teacher` block")
        teacher = make_teacher(spec.teacher)
    meta = build_dataset(
        spec,
        items,
        out,
        teacher=teacher,
        append=append,
        max_variants=max_variants,
        force_train_ids=force_train_ids,
    )
    counts = meta["counts"]
    _progress(
        f"label complete real={meta['real']} variants={meta['variants']} "
        f"train={counts['train']} dev={counts['dev']} eval={counts['eval']} out={out}"
    )
    return LabelResult(out, meta)


def _archive_spec(spec: FunctionSpec, build: Path) -> None:
    (build / "spec.yaml").write_text(
        yaml.safe_dump(spec.model_dump(mode="json"), sort_keys=False)
    )
    (build / "provenance.local.json").write_text(
        json.dumps(
            {"source_spec": str(spec._source_path) if spec._source_path else None},
            indent=2,
        )
    )


def _candidate_public(record: dict) -> dict:
    return {key: value for key, value in record.items() if key != "predictions"}


def _write_provisional_manifest(
    spec: FunctionSpec,
    build: Path,
    data_meta: dict,
    exact_dataset_hash: str,
    candidates: dict,
    diagnostics: dict,
) -> dict:
    state = artifacts.read_build_state(build)
    manifest = {
        "manifest_schema_version": artifacts.MANIFEST_SCHEMA_VERSION,
        "function": spec.name,
        "version": build.name,
        "decision_hash": spec.decision_hash(),
        "dataset_hash": exact_dataset_hash,
        "build_hash": spec.build_hash(),
        "data": data_meta,
        "candidates": {
            name: _candidate_public(record) for name, record in candidates.items()
        },
        "diagnostics": diagnostics,
        "smallbatch_version": _version(),
    }
    if state.get("retry_of"):
        manifest["retry_of"] = state["retry_of"]
    artifacts.write_manifest(build, manifest)
    return manifest


def _version() -> str:
    from . import __version__

    return __version__


def _train_candidate(
    spec: FunctionSpec,
    candidate_id: str,
    config,
    train_rows: list[dict],
    dev_rows: list[dict],
    candidate_dir: Path,
) -> dict:
    import time

    started = time.perf_counter()
    model_dir = candidate_dir / "model"
    if isinstance(config, TfidfCandidateSpec):
        from .candidates import train_tfidf

        metadata = train_tfidf(spec, train_rows, model_dir)
    elif isinstance(config, SetFitCandidateSpec):
        from .setfit_candidate import train_setfit

        metadata = train_setfit(spec, config, train_rows, dev_rows, model_dir)
    elif isinstance(config, LoraCandidateSpec):
        from .training import train

        info = train(spec, config, train_rows, candidate_dir, dev_rows)
        metadata = {
            "format": "peft",
            "base_model": config.model,
            "train_precision": info["precision"],
            "inference_precision": "fp32",
            "rationale_distillation": config.rationale_distillation,
            "eval_batch_size": config.eval_batch_size,
            # the dev-selected decoder must reach the record: runtime.py and
            # the standalone package fall back to argmax when it is absent
            "decode": info["decode"],
            "training": {
                key: info.get(key)
                for key in (
                    "train_loss",
                    "curve",
                    "best_epoch",
                    "best_dev_agreement",
                    "epochs_run",
                    "stopped_reason",
                    "train_rows",
                    "dev_rows",
                    "dev_decode_comparison",
                )
            },
        }
    else:  # pragma: no cover - Pydantic discriminator makes this unreachable
        raise TypeError(f"unsupported candidate config {type(config).__name__}")
    return {
        "candidate": candidate_id,
        "backend": config.type,
        "status": "completed",
        "artifact_path": str(model_dir.relative_to(candidate_dir.parent.parent)),
        "train_seconds": round(time.perf_counter() - started, 3),
        **metadata,
        "error": None,
    }


def _release_accelerator_memory() -> None:
    """HF Trainer holds the trained model in reference cycles; collect them
    between candidates or consecutive trainings won't coexist on a 12GB card."""
    import gc

    gc.collect()
    torch = sys.modules.get("torch")
    if torch is not None and torch.cuda.is_available():
        torch.cuda.empty_cache()


def _load_local_record(path: Path) -> dict | None:
    return json.loads(path.read_text()) if path.exists() else None


def _write_local_record(path: Path, record: dict) -> None:
    atomic_json(path, record)


def compile(  # noqa: A001
    spec: FunctionSpec | str | Path,
    data_dir: str | Path | None = None,
    artifacts_root: str | Path = artifacts.DEFAULT_ROOT,
    *,
    cpu_threads: int | None = None,
) -> CompileResult:
    """Train, CPU-evaluate, and compare every configured candidate."""
    from .evaluate import compute_metrics, train_fitted_constant
    from .profiling import profile_candidate, profile_zeroshot
    from .report import build_report, evidence_summary, write_report

    compile_started = time.perf_counter()
    spec = _as_spec(spec)
    data = Path(data_dir or f"data/{spec.name}")
    meta_path = data / "meta.json"
    if not meta_path.exists():
        raise ValueError(f"no v0.2 decision dataset under {data}; run `smallbatch label` first")
    data_meta = json.loads(meta_path.read_text())
    if data_meta.get("schema_version") != 3:
        raise ValueError("pre-v0.2 datasets are unsupported; re-run `smallbatch label`")
    if data_meta.get("decision_hash") != spec.decision_hash():
        raise ValueError(
            "dataset decisions belong to a different prompt, contract, or teacher; "
            "re-run `smallbatch label`"
        )
    train_rows = read_jsonl(data / "train.jsonl")
    dev_rows = read_jsonl(data / "dev.jsonl")
    eval_rows = read_jsonl(data / "eval.jsonl")
    if not train_rows or not dev_rows or not eval_rows:
        raise ValueError(
            f"dataset must have non-empty train/dev/eval splits; got "
            f"{len(train_rows)}/{len(dev_rows)}/{len(eval_rows)}"
        )
    exact_dataset_hash = dataset_hash([*train_rows, *dev_rows, *eval_rows])
    if data_meta.get("dataset_hash") != exact_dataset_hash:
        raise ValueError("dataset files no longer match meta.json")

    root = Path(artifacts_root)
    build = artifacts.build_dir(root, spec.name, spec.build_hash(), exact_dataset_hash)
    threads = max(1, cpu_threads or min(4, os.cpu_count() or 1))
    _progress(
        f"compile start function={spec.name} build={build.name} candidates={len(spec.candidates)} "
        f"train={len(train_rows)} dev={len(dev_rows)} eval={len(eval_rows)} cpu_threads={threads}"
    )
    if (build / "manifest.json").exists():
        manifest = artifacts.read_manifest(build)
        state = artifacts.read_build_state(build)
        if state.get("status") == "complete" and artifacts.artifact_integrity(build) is None:
            report = json.loads((build / "report.json").read_text())
            _progress(f"compile resume build={build.name} already complete")
            for candidate_id, record in manifest["candidates"].items():
                if record.get("status") == "completed":
                    _progress(
                        f"candidate {candidate_id} resumed complete: {evidence_summary(record)}"
                    )
            return CompileResult(
                build,
                build.name,
                manifest["candidates"],
                manifest,
                report,
                build / "report.json",
            )

    _archive_spec(spec, build)
    (build / "evaluation.local.jsonl").write_text(
        "".join(json.dumps(row, ensure_ascii=False) + "\n" for row in eval_rows)
    )
    candidates: dict[str, dict] = {}
    diagnostics: dict[str, dict] = {}
    eval_inputs = [row["input"] for row in eval_rows]
    references = [row["output"] for row in eval_rows]

    candidate_total = len(spec.candidates)
    for candidate_index, (candidate_id, config) in enumerate(spec.candidates.items(), 1):
        prefix = f"candidate {candidate_index}/{candidate_total} {candidate_id} ({config.type})"
        candidate_dir = build / "candidates" / candidate_id
        candidate_dir.mkdir(parents=True, exist_ok=True)
        final_record_path = candidate_dir / "result.local.json"
        record = _load_local_record(final_record_path)
        if record and record.get("status") == "completed" and record.get("profile"):
            candidates[candidate_id] = record
            _progress(f"{prefix} resumed complete: {evidence_summary(record)}")
            continue
        trained_path = candidate_dir / "trained.json"
        try:
            record = _load_local_record(trained_path)
            if record is None:
                _progress(f"{prefix} training start rows={len(train_rows)} dev={len(dev_rows)}")
                artifacts.update_candidate_state(build, candidate_id, stage="training")
                record = _train_candidate(
                    spec, candidate_id, config, train_rows, dev_rows, candidate_dir
                )
                _write_local_record(trained_path, record)
                _progress(f"{prefix} training complete seconds={record['train_seconds']}")
            else:
                _progress(f"{prefix} resumed trained state")
            candidates[candidate_id] = record
            _write_provisional_manifest(
                spec, build, data_meta, exact_dataset_hash, candidates, diagnostics
            )
            _progress(f"{prefix} CPU evaluation start rows={len(eval_rows)} threads={threads}")
            evaluation_started = time.perf_counter()
            artifacts.update_candidate_state(build, candidate_id, stage="cpu-evaluation")
            profiled = profile_candidate(
                build,
                candidate_id,
                eval_inputs,
                threads=threads,
            )
            record["predictions"] = profiled["predictions"]
            record["metrics"] = compute_metrics(spec, record["predictions"], references)
            record["profile"] = profiled["profile"]
            _write_local_record(final_record_path, record)
            candidates[candidate_id] = record
            artifacts.update_candidate_state(build, candidate_id, stage="completed")
            _progress(
                f"{prefix} complete seconds={time.perf_counter() - evaluation_started:.1f}: "
                f"{evidence_summary(record)}"
            )
        except Exception as exc:  # candidate isolation is intentional
            record = {
                "candidate": candidate_id,
                "backend": config.type,
                "status": "error",
                "error": f"{type(exc).__name__}: {exc}",
            }
            candidates[candidate_id] = record
            _write_local_record(final_record_path, record)
            artifacts.update_candidate_state(
                build, candidate_id, stage="error", error=record["error"]
            )
            _progress(f"{prefix} ERROR {record['error']}")
        finally:
            _release_accelerator_memory()

    seen_bases: set[str] = set()
    diagnostic_configs = []
    for candidate_id, config in spec.candidates.items():
        if not isinstance(config, LoraCandidateSpec):
            continue
        if candidates.get(candidate_id, {}).get("status") != "completed":
            continue
        if config.model in seen_bases:
            continue
        seen_bases.add(config.model)
        diagnostic_configs.append(config)

    for diagnostic_index, config in enumerate(diagnostic_configs, 1):
        diagnostic_id = "zero-shot-" + hashlib.sha256(
            f"{config.model}\n{spec.prompt}".encode()
        ).hexdigest()[:8]
        prefix = (
            f"diagnostic {diagnostic_index}/{len(diagnostic_configs)} "
            f"{diagnostic_id} ({config.model})"
        )
        local_path = build / f"{diagnostic_id}.local.json"
        local = _load_local_record(local_path)
        if local and local.get("metrics") and local.get("profile") and "predictions" in local:
            diagnostics[diagnostic_id] = _candidate_public(local)
            _progress(f"{prefix} resumed complete: {evidence_summary(local)}")
            continue
        try:
            _progress(f"{prefix} CPU evaluation start rows={len(eval_rows)} threads={threads}")
            evaluation_started = time.perf_counter()
            profiled = profile_zeroshot(
                build,
                diagnostic_id,
                config.model,
                eval_inputs,
                threads=threads,
            )
            diagnostics[diagnostic_id] = {
                "selectable": False,
                "backend": "zeroshot",
                "base_model": config.model,
                "metrics": compute_metrics(spec, profiled["predictions"], references),
                "profile": profiled["profile"],
            }
            _write_local_record(
                local_path,
                {**diagnostics[diagnostic_id], "predictions": profiled["predictions"]},
            )
            _progress(
                f"{prefix} complete seconds={time.perf_counter() - evaluation_started:.1f}: "
                f"{evidence_summary(diagnostics[diagnostic_id])}"
            )
        except Exception as exc:
            diagnostics[diagnostic_id] = {
                "selectable": False,
                "backend": "zeroshot",
                "base_model": config.model,
                "status": "error",
                "error": f"{type(exc).__name__}: {exc}",
            }
            _write_local_record(local_path, diagnostics[diagnostic_id])
            _progress(f"{prefix} ERROR {diagnostics[diagnostic_id]['error']}")

    constant = train_fitted_constant(spec, train_rows)
    constant_predictions = [constant for _ in eval_rows]
    diagnostics["train-fitted-constant"] = {
        "selectable": False,
        "value": constant,
        "metrics": compute_metrics(spec, constant_predictions, references),
    }
    _progress(
        "diagnostic train-fitted-constant complete: "
        + evidence_summary(diagnostics["train-fitted-constant"])
    )
    if not any(record.get("status") == "completed" for record in candidates.values()):
        _write_provisional_manifest(
            spec, build, data_meta, exact_dataset_hash, candidates, diagnostics
        )
        state = artifacts.read_build_state(build)
        state["status"] = "error"
        artifacts.write_build_state(build, state)
        raise ValueError("all configured candidates failed; inspect build_state.json")

    report, details = build_report(spec, eval_rows, candidates, diagnostics)
    report_path = write_report(build, report, details)
    manifest = _write_provisional_manifest(
        spec, build, data_meta, exact_dataset_hash, candidates, diagnostics
    )
    manifest["artifact_files"] = artifacts.file_hashes(build)
    artifacts.write_manifest(build, manifest)
    state = artifacts.read_build_state(build)
    state["status"] = "complete"
    state["completed_at"] = datetime.datetime.now(datetime.timezone.utc).isoformat()
    artifacts.write_build_state(build, state)
    _progress(
        f"compile complete build={build.name} seconds={time.perf_counter() - compile_started:.1f} "
        f"report={report_path}"
    )
    return CompileResult(
        build,
        build.name,
        manifest["candidates"],
        manifest,
        report,
        report_path,
    )


def select(
    name: str,
    candidate: str,
    *,
    version: str | None = None,
    artifacts_root: str | Path = artifacts.DEFAULT_ROOT,
    accept_package_drift: bool = False,
    interactive: bool | None = None,
    input_fn: Callable[[str], str] = input,
) -> SelectionResult:
    """Package, validate, and atomically activate one completed candidate."""
    from .standalone import package_selection

    return package_selection(
        Path(artifacts_root),
        name,
        candidate,
        version=version,
        accept_drift=accept_package_drift,
        interactive=interactive,
        input_fn=input_fn,
    )
