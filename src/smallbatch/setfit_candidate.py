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

from . import heads, prompts
from .candidates import _check_class_coverage
from .labeling import Row, row_output
from .spec import FunctionSpec, SetFitCandidateSpec

ORDINAL_HEAD_FILE = "ordinal_head.skops"
# the pair budget backs off num_iterations as embedding rows grow, so lifting
# the per-class cap scales training volume with the data instead of the row
# count times a fixed few-shot iteration count
TARGET_CONTRASTIVE_PAIRS = 4096
MAX_PAIR_ITERATIONS = 20


def _labels(spec: FunctionSpec, field_name: str) -> list[Any]:
    return spec.output.fields[field_name].values()


def _embedding_indices(labels: list[int], per_class: int | None, seed: int) -> list[int]:
    """Rows the contrastive phase trains on: everything by default, an even
    per-class sample only when the spec restricts it."""
    if per_class is None:
        return list(range(len(labels)))
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


def _graded_pairs(
    texts: list[str], labels: list[int], span: int, budget: int, seed: int
) -> list[tuple[str, str, float]]:
    """Level-stratified pairs with cosine targets of `1 - |Δlevel| / span`.

    Both endpoints draw a level uniformly from the observed ones before
    drawing a row, so every level distance — including zero — appears at a
    rate independent of class imbalance."""
    rng = random.Random(seed)
    by_level: dict[int, list[int]] = defaultdict(list)
    for index, label in enumerate(labels):
        by_level[label].append(index)
    levels = sorted(by_level)
    pairs: list[tuple[str, str, float]] = []
    for _ in range(budget):
        first_level, second_level = rng.choice(levels), rng.choice(levels)
        first = rng.choice(by_level[first_level])
        second = rng.choice(by_level[second_level])
        if first == second:
            continue  # a row paired with itself teaches nothing
        pairs.append(
            (texts[first], texts[second], 1.0 - abs(first_level - second_level) / span)
        )
    return pairs


def _train_graded_embeddings(
    model,
    texts: list[str],
    labels: list[int],
    span: int,
    batch_size: int,
    pair_budget: int,
    checkpoint_dir: Path,
    seed: int,
    epochs: int = 1,
    dev_texts: list[str] | None = None,
    dev_labels: list[int] | None = None,
) -> dict[str, Any]:
    """Fine-tune the SentenceTransformer body with graded cosine targets.

    Binary same/different pairs push adjacent levels apart exactly as hard as
    opposite ends of the scale, which destroys the very ordering the ordinal
    head must recover from the embedding. Cosine targets that decay with level
    distance train the space to keep the scale's geometry instead.

    With development rows, every epoch — including epoch 0, the frozen body —
    is scored by the shared head's fixed probe on dev within-one agreement,
    and the best-scoring weights are what survives, the same snapshot-best
    protection the LoRA path has. A frozen body that beats its own fine-tuning
    is kept, and the curve says so."""
    from datasets import Dataset
    from sentence_transformers import (
        SentenceTransformerTrainer,
        SentenceTransformerTrainingArguments,
    )
    from sentence_transformers.losses import CosineSimilarityLoss
    from transformers import TrainerCallback

    pairs = _graded_pairs(texts, labels, span, pair_budget, seed)
    dataset = Dataset.from_dict(
        {
            "sentence_1": [pair[0] for pair in pairs],
            "sentence_2": [pair[1] for pair in pairs],
            "label": [pair[2] for pair in pairs],
        }
    )
    body = model.model_body
    select = bool(dev_texts) and bool(dev_labels)
    curve: list[dict[str, Any]] = []
    best: dict[str, Any] = {"epoch": None, "score": None, "state": None}

    def dev_within_one() -> float:
        return heads.probe_within_one(
            list(range(span + 1)),
            body.encode(texts, show_progress_bar=False),
            labels,
            body.encode(dev_texts, show_progress_bar=False),
            dev_labels,
        )

    def snapshot(epoch: int) -> None:
        value = dev_within_one()
        curve.append({"epoch": epoch, "dev_within_one": round(value, 4)})
        print(
            f"[smallbatch] embedding epoch {epoch}: dev_within_one={value:.4f}",
            file=sys.stderr,
            flush=True,
        )
        if best["score"] is None or value > best["score"]:
            best["epoch"] = epoch
            best["score"] = value
            best["state"] = {
                key: tensor.detach().cpu().clone()
                for key, tensor in body.state_dict().items()
            }

    class KeepBest(TrainerCallback):
        def on_epoch_end(self, args, state, control, **kwargs):
            snapshot(int(round(state.epoch)))

    if select:
        snapshot(0)
    st_args = SentenceTransformerTrainingArguments(
        output_dir=str(checkpoint_dir),
        num_train_epochs=epochs,
        per_device_train_batch_size=batch_size,
        save_strategy="no",
        logging_strategy="no",
        report_to="none",
        seed=seed,
        disable_tqdm=True,
    )
    trainer = SentenceTransformerTrainer(
        model=body,
        args=st_args,
        train_dataset=dataset,
        loss=CosineSimilarityLoss(body),
        callbacks=[KeepBest()] if select else None,
    )
    trainer.train()
    if select and best["epoch"] != curve[-1]["epoch"]:
        body.load_state_dict(best["state"])
    return {
        "pairs": len(pairs),
        "curve": curve or None,
        "best_epoch": best["epoch"],
    }


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
        is_ordinal = heads.applies(spec, field_name)
        if is_ordinal:
            # graded pairs need two distinct levels, not two same-level rows
            if len({train_labels[index] for index in embedding_train}) < 2:
                embedding_train = []
            skip_reason = "skipped-single-level"
        else:
            if not _has_positive_pair(train_labels, embedding_train):
                embedding_train = []
            skip_reason = "skipped-no-positive-pair"
        if not _has_positive_pair(dev_labels, embedding_eval):
            embedding_eval = []
        model = SetFitModel.from_pretrained(
            config.model,
            labels=[str(index) for index in range(len(values))],
        )
        checkpoint_dir = field_dir / "checkpoints"
        args = _resolved_args(config, checkpoint_dir, max(1, len(embedding_train)))
        resolved = json.loads(json.dumps(args.to_dict(), default=str))
        embedding_status = "trained" if embedding_train else skip_reason
        embedding_loss = None
        if embedding_train:
            embedding_loss = "graded-cosine" if is_ordinal else "contrastive-pairs"
        print(
            f"[smallbatch] SetFit field={field_name} "
            f"embedding_train={len(embedding_train)} "
            f"embedding_eval={len(embedding_eval)} classifier_train={len(train_rows)} "
            f"pair_iterations={resolved.get('num_iterations')} status={embedding_status}"
            + (f" loss={embedding_loss}" if embedding_loss else ""),
            file=sys.stderr,
            flush=True,
        )
        trainer = Trainer(
            model=model,
            args=args,
            train_dataset=train_dataset,
            eval_dataset=eval_dataset if dev_rows else None,
        )
        embedding_training = None
        if embedding_train and is_ordinal:
            iterations = resolved.get("num_iterations")
            pair_budget = (
                2 * int(iterations) * len(embedding_train)
                if iterations
                else min(
                    TARGET_CONTRASTIVE_PAIRS,
                    2 * MAX_PAIR_ITERATIONS * len(embedding_train),
                )
            )
            embedding_training = _train_graded_embeddings(
                model,
                [train_texts[index] for index in embedding_train],
                [train_labels[index] for index in embedding_train],
                span=len(values) - 1,
                batch_size=int(getattr(args, "embedding_batch_size", None) or 16),
                pair_budget=pair_budget,
                checkpoint_dir=checkpoint_dir,
                seed=default_seed,
                epochs=int(getattr(args, "embedding_num_epochs", None) or 1),
                dev_texts=dev_texts if dev_rows else None,
                dev_labels=dev_labels if dev_rows else None,
            )
        elif embedding_train:
            trainer.train_embeddings(
                [train_texts[index] for index in embedding_train],
                [train_labels[index] for index in embedding_train],
                [dev_texts[index] for index in embedding_eval] if embedding_eval else None,
                [dev_labels[index] for index in embedding_eval] if embedding_eval else None,
                args=args,
            )
        objective = "multinomial"
        decoder = None
        dev_decode_comparison = None
        head_diagnostics = None
        head_tuning = None
        if is_ordinal:
            # SetFit's own head is multinomial over unrelated symbols. Fit the
            # shared softmax ordinal head on the same tuned embeddings instead,
            # and persist it as plain numpy arrays so packages need no
            # Smallbatch class — and no torch beyond the encoder's own.
            import skops.io as sio

            from .candidates import dev_distribution_rows, write_dev_distributions

            embeddings = model.encode(train_texts, show_progress_bar=False)
            dev_embeddings = (
                model.encode(dev_texts, show_progress_bar=False) if dev_rows else None
            )
            head, head_tuning = heads.fit_head(
                list(range(len(values))),
                embeddings,
                train_labels,
                dev_embeddings,
                dev_labels,
                setting=config.head,
            )
            dev_decode_comparison = heads.select_decoder(
                head, dev_embeddings, dev_labels, config.decode
            )
            decoder = head["decoder"]
            field_dir.mkdir(parents=True, exist_ok=True)
            if dev_embeddings is not None and dev_labels:
                head_diagnostics = heads.head_diagnostics(
                    head, dev_embeddings, dev_labels, levels=values
                )
                write_dev_distributions(
                    field_dir,
                    {
                        field_name: dev_distribution_rows(
                            head,
                            dev_embeddings,
                            dev_rows,
                            [values[index] for index in dev_labels],
                            levels=values,
                        )
                    },
                )
            sio.dump(head, field_dir / ORDINAL_HEAD_FILE)
            objective = "ordinal"
        else:
            trainer.train_classifier(train_texts, train_labels, args=args)
        shutil.rmtree(checkpoint_dir, ignore_errors=True)
        model.save_pretrained(field_dir / "model")
        (field_dir / "labels.json").write_text(
            json.dumps(values, indent=2, ensure_ascii=False)
        )
        # the epoch-0 curve entry scored the frozen body with the same
        # throwaway head as every later epoch: the delta to the surviving
        # epoch is the direct answer to "did fine-tuning the encoder help?"
        frozen_vs_tuned = None
        curve = (embedding_training or {}).get("curve")
        if curve:
            best_epoch = embedding_training["best_epoch"]
            frozen = curve[0]["dev_within_one"]
            tuned = next(
                point["dev_within_one"] for point in curve if point["epoch"] == best_epoch
            )
            frozen_vs_tuned = {
                "frozen_dev_within_one": frozen,
                "best_dev_within_one": tuned,
                "delta": round(tuned - frozen, 4),
                "best_epoch": best_epoch,
            }
        field_training[field_name] = {
            "embedding_train_rows": len(embedding_train),
            "embedding_eval_rows": len(embedding_eval),
            "embedding_status": embedding_status,
            "embedding_loss": embedding_loss,
            "embedding_pairs": (embedding_training or {}).get("pairs"),
            "embedding_curve": (embedding_training or {}).get("curve"),
            "embedding_best_epoch": (embedding_training or {}).get("best_epoch"),
            "frozen_vs_tuned": frozen_vs_tuned,
            "head_tuning": head_tuning,
            "classifier_train_rows": len(train_rows),
            "objective": objective,
            "decode": decoder,
            "dev_decode_comparison": dev_decode_comparison,
            "head_diagnostics": head_diagnostics,
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
            if not isinstance(head, dict) or head.get("kind") != heads.KIND:
                raise ValueError(
                    f"unsupported ordinal head kind in {head_path}: this build "
                    "predates the v0.3 softmax head; re-run compile"
                )
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
            indices = heads.predict(head, embeddings)
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
