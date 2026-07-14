"""SetFit candidate training and CPU inference."""

from __future__ import annotations

import json
import math
import random
import shutil
import sys
from collections import Counter, defaultdict
from pathlib import Path
from typing import Any

from . import ordinal, prompts
from .candidates import _check_class_coverage
from .labeling import Row, row_output
from .spec import FunctionSpec, SetFitCandidateSpec

ORDINAL_HEAD_FILE = "ordinal_head.skops"
TARGET_CONTRASTIVE_PAIRS = 1024
MAX_PAIR_ITERATIONS = 20


def _labels(spec: FunctionSpec, field_name: str) -> list[Any]:
    return spec.output.fields[field_name].values()


def _embedding_indices(labels: list[int], per_class: int, seed: int) -> list[int]:
    grouped: dict[int, list[int]] = defaultdict(list)
    for index, label in enumerate(labels):
        grouped[label].append(index)
    rng = random.Random(seed)
    selected: list[int] = []
    for label in sorted(grouped):
        indices = grouped[label].copy()
        rng.shuffle(indices)
        selected.extend(indices[:per_class])
    return sorted(selected)


def _has_positive_pair(labels: list[int], indices: list[int]) -> bool:
    return any(count >= 2 for count in Counter(labels[index] for index in indices).values())


def _resolved_args(
    config: SetFitCandidateSpec, output_dir: Path, embedding_rows: int
):
    from setfit import TrainingArguments

    values = {
        "output_dir": str(output_dir),
        "report_to": "none",
        "save_strategy": "no",
        "show_progress_bar": False,
    }
    if not {"num_iterations", "sampling_strategy"} & set(config.training_args):
        values["num_iterations"] = min(
            MAX_PAIR_ITERATIONS,
            max(1, math.ceil(TARGET_CONTRASTIVE_PAIRS / (2 * embedding_rows))),
        )
    values.update(config.training_args)
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
    field_training: dict[str, dict[str, Any]] = {}
    for field_name in spec.output.fields:
        field_dir = out_dir / field_name
        values = _labels(spec, field_name)
        value_to_index = {json.dumps(value): index for index, value in enumerate(values)}

        def encode(row: Row) -> int:
            output = row_output(spec, row)
            value = output[field_name] if isinstance(output, dict) else output
            return value_to_index[json.dumps(value)]

        train_labels = [encode(row) for row in train_rows]
        dev_labels = [encode(row) for row in dev_rows]
        train_dataset = Dataset.from_dict({"text": train_texts, "label": train_labels})
        eval_dataset = Dataset.from_dict({"text": dev_texts, "label": dev_labels})
        default_seed = int(config.training_args.get("seed", 42))
        embedding_train = _embedding_indices(
            train_labels, config.embedding_samples_per_class, default_seed
        )
        embedding_eval = _embedding_indices(
            dev_labels, config.embedding_samples_per_class, default_seed
        )
        if not _has_positive_pair(train_labels, embedding_train):
            embedding_train = []
        if not _has_positive_pair(dev_labels, embedding_eval):
            embedding_eval = []
        model = SetFitModel.from_pretrained(
            config.model,
            labels=[str(index) for index in range(len(values))],
        )
        checkpoint_dir = field_dir / "checkpoints"
        args = _resolved_args(config, checkpoint_dir, max(1, len(embedding_train)))
        resolved = json.loads(json.dumps(args.to_dict(), default=str))
        embedding_status = "trained" if embedding_train else "skipped-no-positive-pair"
        print(
            f"[smallbatch] SetFit field={field_name} "
            f"embedding_train={len(embedding_train)} "
            f"embedding_eval={len(embedding_eval)} classifier_train={len(train_rows)} "
            f"pair_iterations={resolved.get('num_iterations')} status={embedding_status}",
            file=sys.stderr,
            flush=True,
        )
        trainer = Trainer(
            model=model,
            args=args,
            train_dataset=train_dataset,
            eval_dataset=eval_dataset if dev_rows else None,
        )
        if embedding_train:
            trainer.train_embeddings(
                [train_texts[index] for index in embedding_train],
                [train_labels[index] for index in embedding_train],
                [dev_texts[index] for index in embedding_eval] if embedding_eval else None,
                [dev_labels[index] for index in embedding_eval] if embedding_eval else None,
                args=args,
            )
        objective = "multinomial"
        if ordinal.applies(spec, field_name):
            # SetFit's own head is multinomial over unrelated symbols. Fit the
            # ordered head on the same tuned embeddings instead, and persist it
            # as stock sklearn parts so packages need no Smallbatch class.
            import skops.io as sio
            from sklearn.linear_model import LogisticRegression

            embeddings = model.encode(train_texts, show_progress_bar=False)
            head = ordinal.build(
                list(range(len(values))),
                embeddings,
                train_labels,
                lambda x, y: LogisticRegression(max_iter=1000).fit(x, y),
            )
            field_dir.mkdir(parents=True, exist_ok=True)
            sio.dump(head, field_dir / ORDINAL_HEAD_FILE)
            objective = "ordinal"
        else:
            trainer.train_classifier(train_texts, train_labels, args=args)
        shutil.rmtree(checkpoint_dir, ignore_errors=True)
        model.save_pretrained(field_dir / "model")
        (field_dir / "labels.json").write_text(
            json.dumps(values, indent=2, ensure_ascii=False)
        )
        field_training[field_name] = {
            "embedding_train_rows": len(embedding_train),
            "embedding_eval_rows": len(embedding_eval),
            "embedding_status": embedding_status,
            "classifier_train_rows": len(train_rows),
            "objective": objective,
            "resolved_args": resolved,
        }
    return {
        "format": "setfit",
        "setfit_version": setfit.__version__,
        "model": config.model,
        "fields": list(spec.output.fields),
        "embedding_samples_per_class": config.embedding_samples_per_class,
        "field_training": field_training,
    }


def predict_setfit(
    model_dir: Path, spec: FunctionSpec, items: list[dict], device: str = "cpu"
) -> list[Any]:
    return predict_setfit_models(load_setfit_models(model_dir, spec, device), spec, items)


def load_setfit_models(model_dir: Path, spec: FunctionSpec, device: str = "cpu") -> dict:
    import skops.io as sio
    from setfit import SetFitModel

    loaded = {}
    for field_name in spec.output.fields:
        field_dir = model_dir / field_name
        values = json.loads((field_dir / "labels.json").read_text())
        model = SetFitModel.from_pretrained(field_dir / "model", device=device)
        head_path = field_dir / ORDINAL_HEAD_FILE
        head = None
        if head_path.exists():
            untrusted = sio.get_untrusted_types(file=head_path)
            if untrusted:
                raise ValueError(
                    f"refusing to load {head_path}: unexpected types {sorted(untrusted)}"
                )
            head = sio.load(head_path, trusted=[])
        loaded[field_name] = (model, values, head)
    return loaded


def predict_setfit_models(models: dict, spec: FunctionSpec, items: list[dict]) -> list[Any]:
    if not items:
        return []
    texts = [prompts.render_input(item, spec.input_schema) for item in items]
    per_field: dict[str, list[Any]] = {}
    for field_name, (model, values, head) in models.items():
        if head is not None:
            embeddings = model.encode(texts, show_progress_bar=False)
            indices = ordinal.predict(head, embeddings)
        else:
            raw = model.predict(texts, use_labels=False)
            indices = raw.tolist() if hasattr(raw, "tolist") else list(raw)
        per_field[field_name] = [values[int(index)] for index in indices]
    if spec.output.is_scalar:
        return per_field[next(iter(spec.output.fields))]
    return [
        {field_name: per_field[field_name][index] for field_name in spec.output.fields}
        for index in range(len(items))
    ]
