"""Programmatic pipeline API: `label` and `compile`.

The CLI (`cli.py`) is a thin wrapper over these; import them directly for
notebook or pipeline use:

    import smallbatch
    result = smallbatch.label("spec.yaml", items)
    compiled = smallbatch.compile("spec.yaml")
    fn = smallbatch.load_fn(compiled.manifest["function"])

Heavy imports (torch/peft/trl) happen inside `compile`, so importing this
module — and running `label` — stays light.
"""

from __future__ import annotations

import json
import shutil
from dataclasses import dataclass
from pathlib import Path

import yaml

from . import artifacts
from .labeling import read_jsonl
from .spec import FunctionSpec, load_spec


@dataclass
class LabelResult:
    out_dir: Path
    meta: dict  # counts, label_histogram, teacher/prompt provenance

    @property
    def compressed(self) -> bool:
        """True when >50% of labels landed in one bin — a sign the rubric
        anchors need sharpening before the dataset is worth training on."""
        hist = self.meta["label_histogram"]
        total = sum(hist.values())
        return bool(total) and max(hist.values()) / total > 0.5


@dataclass
class CompileResult:
    passed: bool  # gate verdict; a False is an honest FAIL, not an error
    gate: dict
    metrics: dict  # {"adapter": {...}, "zeroshot": {...} | None}
    version_dir: Path
    manifest: dict
    report: dict | None = None  # full eval report (also report.json on disk)
    report_path: Path | None = None


def _as_spec(spec: FunctionSpec | str | Path) -> FunctionSpec:
    return spec if isinstance(spec, FunctionSpec) else load_spec(spec)


def label(
    spec: FunctionSpec | str | Path,
    items: list[dict],
    out_dir: str | Path | None = None,
    append: bool = False,
    max_variants: int | None = None,
) -> LabelResult:
    """Teacher-label `items` into a train/dev/gate dataset for `spec`.

    `append` keeps an existing dataset's rows and split assignments (the gate
    is sticky) and only labels unseen items; `max_variants` caps how many new
    balance-driven synthetic variants this call generates.
    """
    from .labeling import build_dataset
    from .teacher import make_teacher

    spec = _as_spec(spec)
    out = Path(out_dir or f"data/{spec.name}")
    teacher = make_teacher(spec.teacher)
    meta = build_dataset(teacher, spec, items, out, append=append, max_variants=max_variants)
    return LabelResult(out_dir=out, meta=meta)


def compile(  # noqa: A001 - deliberate: `smallbatch.compile` is the product verb
    spec: FunctionSpec | str | Path,
    data_dir: str | Path | None = None,
    artifacts_root: str | Path = artifacts.DEFAULT_ROOT,
    base: str | None = None,
    precision: str | None = None,
    sweep_name: str | None = None,
    tag: str | None = None,
    arm: str | None = None,
) -> CompileResult:
    """Train + evaluate + gate one adapter for `spec`.

    Returns a CompileResult whose `passed` reflects the gate; raises on real
    errors (missing data, training failure). Requires a labeled dataset from
    `label` under `data_dir` (default data/<name>).
    """
    from .evaluate import run_gate, score_holdout
    from .training import load_base_model, train

    spec = _as_spec(spec)
    if base:
        spec.train.base = base
    if precision:
        spec.train.precision = precision

    data = Path(data_dir or f"data/{spec.name}")
    train_rows = read_jsonl(data / "train.jsonl")
    gate_path = data / "gate.jsonl"
    if not gate_path.exists():  # pre-v0.2 dataset layout
        gate_path = data / "holdout.jsonl"
    gate_rows = read_jsonl(gate_path) if gate_path.exists() else []
    if not gate_rows:
        raise ValueError(f"empty gate split in {data} — run `smallbatch label` first")

    dev_path = data / "dev.jsonl"
    if dev_path.exists():
        dev_rows = read_jsonl(dev_path)
    else:
        # legacy dataset without a dev split: carve one from the train reals
        # deterministically so checkpoint selection still works. Relabeling
        # with `smallbatch label --append` gives a proper, persistent split.
        from .labeling import resolve_count, split_holdout

        n_real = sum(1 for r in train_rows if r.get("origin") == "real")
        frac = resolve_count(spec.teacher.dev, n_real) / n_real if n_real else 0
        train_rows, dev_rows = split_holdout(train_rows, frac, seed=spec.train.seed)
        print(
            f"warning: no dev.jsonl in {data} — carved {len(dev_rows)} dev rows "
            "out of train for checkpoint selection (re-run `smallbatch label "
            "--append` to persist a proper split)"
        )

    root = Path(artifacts_root)
    if sweep_name and tag:
        version_dir = artifacts.sweep_run_dir(root, spec.name, sweep_name, tag)
    else:
        version_dir = artifacts.new_version_dir(root, spec.name)
    print(f"compiling {spec.name} -> {version_dir}")

    info = train(spec, train_rows, version_dir, dev_rows=dev_rows)
    shutil.rmtree(version_dir / "trainer", ignore_errors=True)

    # HF Trainer holds the training model in reference cycles; collect them
    # before eval loads a second copy of the base model or the two won't
    # coexist on a 12GB card
    import gc

    import torch

    gc.collect()
    if torch.cuda.is_available():
        torch.cuda.empty_cache()

    # evaluate adapter (student prompts) and zero-shot base (full-spec prompts)
    from peft import PeftModel

    from . import prompts

    # eval in the training precision: a qlora adapter was trained against the
    # 4-bit base, and reloading in fp32 needs 4x the VRAM (36GB for a 9B)
    inference_precision = info["precision"]
    tokenizer, base_model = load_base_model(spec.train.base, inference_precision)
    max_new = 80 if spec.train.rationale_distillation else 8

    spec_text = spec.spec_files_text()
    zeroshot = None
    if spec.gate.must_beat_zeroshot:
        zeroshot = score_holdout(
            spec, base_model, tokenizer, gate_rows,
            lambda it: prompts.zeroshot_prompt(spec, it, spec_text), max_new_tokens=16,
        )
    student = PeftModel.from_pretrained(base_model, info["adapter_dir"])
    student.eval()
    adapter_metrics = score_holdout(
        spec, student, tokenizer, gate_rows,
        lambda it: prompts.student_prompt(spec, it), max_new_tokens=max_new,
    )

    gate = run_gate(spec, adapter_metrics, zeroshot)
    if spec._source_path is not None:
        shutil.copy(spec._source_path, version_dir / "spec.yaml")
    else:  # spec built programmatically: serialize it so the artifact is complete
        (version_dir / "spec.yaml").write_text(
            yaml.safe_dump(spec.model_dump(mode="json"), sort_keys=False)
        )
    data_meta = json.loads((data / "meta.json").read_text()) if (data / "meta.json").exists() else {}
    from . import __version__

    manifest = {
        "function": spec.name,
        "version": f"{sweep_name}/{tag}" if (sweep_name and tag) else version_dir.name,
        "sweep_name": sweep_name,
        "tag": tag,
        "arm": arm,
        "spec_hash": spec.spec_hash(),
        "base_model": spec.train.base,
        "train_precision": info["precision"],
        "inference_precision": inference_precision,
        "use_dora": spec.train.use_dora,
        "rationale_distillation": spec.train.rationale_distillation,
        "data": data_meta,
        "metrics": {"adapter": adapter_metrics, "zeroshot": zeroshot},
        "gate": gate,
        "train_loss": info["train_loss"],
        "epochs_run": info.get("epochs_run"),
        "best_epoch": info.get("best_epoch"),
        "stopped_reason": info.get("stopped_reason"),
        "smallbatch_version": __version__,
    }
    artifacts.write_manifest(version_dir, manifest)

    from .report import build_report, write_report

    report = build_report(spec, gate_rows, adapter_metrics, zeroshot, gate, info)
    report_path = write_report(version_dir, report)

    return CompileResult(
        passed=gate["passed"],
        gate=gate,
        metrics=manifest["metrics"],
        version_dir=version_dir,
        manifest=manifest,
        report=report,
        report_path=report_path,
    )
