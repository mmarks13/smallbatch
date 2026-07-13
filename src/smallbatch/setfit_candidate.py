"""SetFit candidate training and CPU inference."""

from __future__ import annotations

import json
from pathlib import Path
from typing import Any

from . import prompts
from .candidates import _check_class_coverage
from .labeling import Row, row_output
from .spec import FunctionSpec, SetFitCandidateSpec


def _labels(spec: FunctionSpec, field_name: str) -> list[Any]:
    return spec.output.fields[field_name].values()


def _resolved_args(config: SetFitCandidateSpec, output_dir: Path):
    from setfit import TrainingArguments

    values = {
        "output_dir": str(output_dir),
        "report_to": "none",
        "show_progress_bar": False,
        **config.training_args,
    }
    try:
        return TrainingArguments(**values)
    except TypeError as exc:
        raise ValueError(f"invalid SetFit training_args: {exc}") from exc


def train_setfit(
    spec: FunctionSpec,
    config: SetFitCandidateSpec,
    train_rows: list[Row],
    dev_rows: list[Row],
    out_dir: Path,
) -> dict[str, Any]:
    """Train and save one SetFit model per constrained output field."""
    import setfit
    from datasets import Dataset
    from setfit import SetFitModel, Trainer

    _check_class_coverage(spec, train_rows)
    train_texts = [prompts.render_input(row["input"], spec.input_schema) for row in train_rows]
    dev_texts = [prompts.render_input(row["input"], spec.input_schema) for row in dev_rows]
    out_dir.mkdir(parents=True, exist_ok=True)
    resolved: dict[str, Any] | None = None
    for field_name in spec.output.fields:
        field_dir = out_dir / field_name
        values = _labels(spec, field_name)
        value_to_index = {json.dumps(value): index for index, value in enumerate(values)}

        def encode(row: Row) -> int:
            output = row_output(spec, row)
            value = output[field_name] if isinstance(output, dict) else output
            return value_to_index[json.dumps(value)]

        train_dataset = Dataset.from_dict(
            {"text": train_texts, "label": [encode(row) for row in train_rows]}
        )
        eval_dataset = Dataset.from_dict(
            {"text": dev_texts, "label": [encode(row) for row in dev_rows]}
        )
        model = SetFitModel.from_pretrained(
            config.model,
            labels=[str(index) for index in range(len(values))],
        )
        args = _resolved_args(config, field_dir / "checkpoints")
        trainer = Trainer(
            model=model,
            args=args,
            train_dataset=train_dataset,
            eval_dataset=eval_dataset if dev_rows else None,
        )
        trainer.train()
        model.save_pretrained(field_dir / "model")
        (field_dir / "labels.json").write_text(
            json.dumps(values, indent=2, ensure_ascii=False)
        )
        if resolved is None:
            resolved = json.loads(json.dumps(args.to_dict(), default=str))
    return {
        "format": "setfit",
        "setfit_version": setfit.__version__,
        "model": config.model,
        "fields": list(spec.output.fields),
        "training_args": resolved or {},
    }


def predict_setfit(
    model_dir: Path, spec: FunctionSpec, items: list[dict], device: str = "cpu"
) -> list[Any]:
    return predict_setfit_models(load_setfit_models(model_dir, spec, device), spec, items)


def load_setfit_models(model_dir: Path, spec: FunctionSpec, device: str = "cpu") -> dict:
    from setfit import SetFitModel

    loaded = {}
    for field_name in spec.output.fields:
        field_dir = model_dir / field_name
        values = json.loads((field_dir / "labels.json").read_text())
        model = SetFitModel.from_pretrained(field_dir / "model", device=device)
        loaded[field_name] = (model, values)
    return loaded


def predict_setfit_models(models: dict, spec: FunctionSpec, items: list[dict]) -> list[Any]:
    if not items:
        return []
    texts = [prompts.render_input(item, spec.input_schema) for item in items]
    per_field: dict[str, list[Any]] = {}
    for field_name, (model, values) in models.items():
        raw = model.predict(texts, use_labels=False)
        indices = raw.tolist() if hasattr(raw, "tolist") else list(raw)
        per_field[field_name] = [values[int(index)] for index in indices]
    if spec.output.is_scalar:
        return per_field[next(iter(spec.output.fields))]
    return [
        {field_name: per_field[field_name][index] for field_name in spec.output.fields}
        for index in range(len(items))
    ]
