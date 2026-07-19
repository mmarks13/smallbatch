"""LoRA fine-tuning of the student on teacher-labeled data."""

from __future__ import annotations

import json
from pathlib import Path

from . import prompts
from .hardware import pick_precision
from .labeling import Row, _progress
from .spec import FunctionSpec, LoraCandidateSpec


def _load_tokenizer(base: str):
    """Bridge tokenizer.json-only Transformers 5 configs on our pinned v4 stack."""
    from transformers import AutoTokenizer, PreTrainedTokenizerFast

    try:
        return AutoTokenizer.from_pretrained(base)
    except ValueError as exc:
        if "Tokenizer class TokenizersBackend does not exist" not in str(exc):
            raise
        # SetFit 1.1.x currently requires Transformers 4.x. Newer Hub repos may
        # identify the same tokenizer.json backend by its Transformers 5 name.
        from transformers.utils import cached_file

        config_path = cached_file(base, "tokenizer_config.json")
        tokenizer_config = json.loads(Path(config_path).read_text())
        extra_tokens = tokenizer_config.get("extra_special_tokens")
        if isinstance(extra_tokens, list) and extra_tokens:
            raise ValueError(
                "tokenizer requires non-empty Transformers 5 extra_special_tokens; "
                "Smallbatch cannot translate them safely to its Transformers 4 runtime"
            ) from exc
        compatibility = {"extra_special_tokens": {}} if isinstance(extra_tokens, list) else {}
        return PreTrainedTokenizerFast.from_pretrained(base, **compatibility)


def load_base_model(base: str, precision: str):
    """Load tokenizer + base model at the given precision. Shared with eval."""
    import torch
    from transformers import AutoModelForCausalLM

    tokenizer = _load_tokenizer(base)
    if tokenizer.pad_token is None:
        tokenizer.pad_token = tokenizer.eos_token

    kwargs: dict = {}
    if precision == "bf16":
        kwargs["dtype"] = torch.bfloat16
    elif precision == "fp32":
        kwargs["dtype"] = torch.float32
    elif precision == "qlora":
        from transformers import BitsAndBytesConfig

        # fp16 compute is ~64x slower than fp32 on Pascal (sm_61, no bf16), so
        # match the base-model dtype logic: bf16-capable cards compute in fp16,
        # Pascal in fp32.
        compute_dtype = torch.float16
        if torch.cuda.is_available() and torch.cuda.get_device_capability()[0] < 8:
            compute_dtype = torch.float32
        kwargs["quantization_config"] = BitsAndBytesConfig(
            load_in_4bit=True,
            bnb_4bit_quant_type="nf4",
            bnb_4bit_compute_dtype=compute_dtype,
        )
    try:
        model = AutoModelForCausalLM.from_pretrained(base, **kwargs)
    except ValueError:
        # multimodal wrappers (Qwen3.5, Gemma 4, Ministral 3 are
        # *ForConditionalGeneration) aren't in the causal-LM mapping; text-only
        # training works fine through the image-text class with the tower idle
        from transformers import AutoModelForImageTextToText

        model = AutoModelForImageTextToText.from_pretrained(base, **kwargs)
    if precision != "qlora" and torch.cuda.is_available():
        model = model.cuda()
    return tokenizer, model


def ordinal_decision_tokens(spec: FunctionSpec, tokenizer) -> tuple[int, list[int]] | None:
    """The position where the legal completions diverge, and the token each
    contributes there.

    Tokenized completions share a prefix (the leading space) and a suffix (the
    end token) and differ in one token: " 3" is [space, "3", eos] for Granite
    and Qwen. So the whole class distribution is readable from the logits at
    that one position — keep those token ids and renormalize. Everything the
    completions share cancels in the softmax, and the distribution is exactly
    the one constrained decoding sees at inference.

    Returns None when several tokens are needed to tell the values apart (a
    0-10 scale splits "10" into two digits), because then no single position
    carries the decision.
    """
    tokenized = [
        tokenizer(
            prompts.student_completion(spec, value), add_special_tokens=False
        )["input_ids"]
        for value in spec.output.scalar.values()
    ]
    shortest = min(len(tokens) for tokens in tokenized)
    prefix = 0
    while prefix < shortest and len({tokens[prefix] for tokens in tokenized}) == 1:
        prefix += 1

    deciding = [tokens[prefix:] for tokens in tokenized]
    if any(len(tokens) != 1 for tokens in deciding):
        return None
    ids = [tokens[0] for tokens in deciding]
    if len(set(ids)) != len(ids):
        return None
    return prefix, ids


def _last_logged_loss(log_history: list[dict]) -> float | None:
    for entry in reversed(log_history):
        if "loss" in entry:
            return round(entry["loss"], 4)
    return None


def _latest_checkpoint(trainer_dir: Path) -> Path | None:
    """HF names checkpoints checkpoint-<step>: compare steps numerically, or a
    crash between save and rotation resumes 'checkpoint-999' over '-1000'."""
    checkpoints = [
        path
        for path in trainer_dir.glob("checkpoint-*")
        if path.name.rsplit("-", 1)[-1].isdigit()
    ]
    if not checkpoints:
        return None
    return max(checkpoints, key=lambda path: int(path.name.rsplit("-", 1)[-1]))



def train(
    spec: FunctionSpec,
    config: LoraCandidateSpec,
    train_rows: list[Row],
    out_dir: Path,
    dev_rows: list[Row] | None = None,
) -> dict:
    """Fine-tune a LoRA adapter under the universal per-field objective.

    Every output field trains with its type-appropriate loss (int: class NLL
    + RPS, enum: legal-value NLL, text: mean next-token NLL), each normalized
    by this candidate's untuned-base development loss for that field, and
    combined as a weighted mean plus the fixed-weight format loss (see
    `objective`). The same weighted normalized score — teacher-forced on the
    development split, never behavioral agreement — selects the best
    checkpoint and drives early stopping for every function shape.

    The adapter written to out_dir/model is the best-scoring epoch's.
    """
    from peft import LoraConfig, get_peft_model, prepare_model_for_kbit_training
    from transformers import Trainer, TrainerCallback, TrainingArguments

    from . import objective
    from .labeling import row_output
    from .spec import effective_loss_weights

    if not dev_rows:
        raise ValueError(
            "LoRA training requires a development split: the checkpoint rule "
            "scores every epoch on teacher-forced development losses. Label "
            "more data"
        )
    precision = pick_precision(config.precision)
    tokenizer, model = load_base_model(config.model, precision)
    codecs = objective.field_codecs(spec, tokenizer)
    weights = effective_loss_weights(spec)

    def encode(rows: list[Row]) -> list[dict]:
        return [
            objective.encode_row(
                spec,
                tokenizer,
                codecs,
                prompts.student_prompt(spec, row["input"]),
                row_output(spec, row),
                config.max_seq_len,
            )
            for row in rows
        ]

    train_encoded = encode(train_rows)
    dev_encoded = encode(dev_rows)

    # candidate- and field-specific normalization baselines: the untuned base
    # model's teacher-forced development losses, fixed before training and
    # recorded as evidence. An unsafe baseline fails here, before any training
    model.eval()
    baselines = objective.evaluate_field_losses(
        model, spec, codecs, dev_encoded, tokenizer.pad_token_id, config.eval_batch_size
    )
    objective.check_baselines(baselines)
    _progress(
        "untuned base dev losses: "
        + " ".join(f"{name}={value:.4f}" for name, value in sorted(baselines.items()))
    )

    if precision == "qlora":
        # prepare_model_for_kbit_training upcasts every non-quantized module
        # to fp32; for huge-vocab models the tied embedding alone can be a
        # ~4GB fp32 tensor and OOMs 12GB cards. In that case do the two
        # things we actually need by hand and keep the embedding load dtype.
        embed = model.get_input_embeddings()
        if embed.weight.numel() * 4 > 3_000_000_000:
            model.gradient_checkpointing_enable()
            model.enable_input_require_grads()
        else:
            model = prepare_model_for_kbit_training(model)

    lora = LoraConfig(
        r=config.lora_r,
        lora_alpha=config.alpha,
        lora_dropout=config.lora_dropout,
        use_dora=config.use_dora,
        target_modules="all-linear",
        task_type="CAUSAL_LM",
    )
    model = get_peft_model(model, lora)
    if config.gradient_checkpointing:
        model.enable_input_require_grads()

    args = TrainingArguments(
        output_dir=str(out_dir / "trainer"),
        num_train_epochs=config.max_epochs,
        gradient_checkpointing=config.gradient_checkpointing,
        learning_rate=config.learning_rate,
        per_device_train_batch_size=config.batch_size,
        bf16=(precision == "bf16"),
        fp16=False,
        seed=config.seed,
        logging_steps=20,
        save_strategy="epoch",
        save_total_limit=1,
        report_to=[],
        remove_unused_columns=False,
    )
    adapter_dir = out_dir / "model"

    class UniversalTrainer(Trainer):
        """One loss for every LoRA shape; see the module objective."""

        def compute_loss(
            self, model, inputs, return_outputs=False, num_items_in_batch=None
        ):
            logits, start = objective.completion_logits(model, inputs)
            losses = objective.row_losses(
                logits, inputs["input_ids"], inputs["owners"], spec, codecs, start
            )
            return objective.combine_losses(losses, weights, baselines).mean()

    class DevScore(TrainerCallback):
        """Score dev each epoch, snapshot the best adapter, stop on patience.

        The saved adapter is always the best-so-far, so early stopping never
        ships a worse-than-seen checkpoint. Behavioral metrics play no part."""

        def __init__(self):
            self.curve: list[dict] = []
            self.best: float | None = None
            self.best_epoch: int | None = None
            self.stale = 0
            self.stopped_reason = "max_epochs"

        def on_epoch_end(self, args, state, control, model=None, **kwargs):
            import torch

            epoch = int(round(state.epoch))
            was_training = model.training
            model.eval()
            with torch.no_grad():
                dev = objective.evaluate_field_losses(
                    model,
                    spec,
                    codecs,
                    dev_encoded,
                    tokenizer.pad_token_id,
                    config.eval_batch_size,
                )
            if was_training:
                model.train()
            score = objective.checkpoint_score(dev, weights, baselines)
            entry = {
                "epoch": epoch,
                "train_loss": _last_logged_loss(state.log_history),
                "checkpoint_score": round(score, 4),
                "normalized_dev_losses": {
                    name: round(dev[name] / baselines[name], 4) for name in weights
                },
                "normalized_format_loss": round(
                    dev[objective.FORMAT_KEY] / baselines[objective.FORMAT_KEY], 4
                ),
            }
            self.curve.append(entry)
            print(f"epoch {epoch}: checkpoint_score={score:.4f}", flush=True)
            if self.best is None or score < self.best - config.min_delta:
                self.best = score
                self.best_epoch = epoch
                self.stale = 0
                model.save_pretrained(str(adapter_dir))
            else:
                self.stale += 1
                if config.patience is not None and self.stale >= config.patience:
                    self.stopped_reason = f"early_stop(patience={config.patience})"
                    control.should_training_stop = True
            return control

    dev_cb = DevScore()
    trainer = UniversalTrainer(
        model=model,
        args=args,
        train_dataset=train_encoded,
        data_collator=lambda rows: objective.collate(rows, tokenizer.pad_token_id),
        processing_class=tokenizer,
        callbacks=[dev_cb],
    )
    checkpoint = _latest_checkpoint(out_dir / "trainer")
    result = trainer.train(resume_from_checkpoint=str(checkpoint) if checkpoint else None)

    if dev_cb.best_epoch is None:
        # dev never scored (e.g. zero epochs): fall back to the final adapter
        trainer.model.save_pretrained(str(adapter_dir))
    tokenizer.save_pretrained(str(adapter_dir))
    epochs_run = int(round(trainer.state.epoch or config.max_epochs))
    return {
        "precision": precision,
        "objective": "per-field",
        "loss_weights": weights,
        "untuned_baselines": {
            name: round(value, 4) for name, value in sorted(baselines.items())
        },
        "format_loss_weight": objective.FORMAT_LOSS_WEIGHT,
        "train_rows": len(train_rows),
        "train_loss": round(result.training_loss, 4),
        "adapter_dir": str(adapter_dir),
        "curve": dev_cb.curve,
        "best_epoch": dev_cb.best_epoch,
        "best_checkpoint_score": dev_cb.best,
        "epochs_run": epochs_run,
        "stopped_reason": dev_cb.stopped_reason,
        "dev_rows": len(dev_rows),
    }
