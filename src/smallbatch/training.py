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


def ordinal_objective_applies(spec: FunctionSpec, config: LoraCandidateSpec) -> bool:
    """Ordinal training needs one scalar int decision and no free-text rationale."""
    if config.objective == "token":
        return False
    scalar = spec.output.is_scalar and spec.output.scalar.type == "int"
    if config.objective == "ordinal":
        if not scalar:
            raise ValueError(
                "objective: ordinal requires a scalar integer output; use objective: token"
            )
        if config.rationale_distillation:
            raise ValueError(
                "objective: ordinal cannot score free-text rationales; "
                "disable rationale_distillation or use objective: token"
            )
        return True
    return scalar and not config.rationale_distillation


def _ordinal_trainer_class(spec: FunctionSpec, tokenizer):
    """SFTTrainer whose loss knows the levels are ordered.

    Token cross-entropy treats "3" and "4" as unrelated symbols, so a student
    is punished the same for a near miss and a far one. Here the logits at the
    deciding position are renormalized over the legal levels into a proper
    distribution, and the loss combines class NLL with the ranked probability
    score (RPS), which accumulates error across the ordered levels and so
    penalizes distant predictions more than adjacent ones. It costs one
    ordinary forward pass: everything the completions share cancels in the
    softmax, so nothing is gained by scoring them one at a time.
    """
    import torch
    import torch.nn.functional as F
    from trl import SFTTrainer

    decision = ordinal_decision_tokens(spec, tokenizer)
    if decision is None:
        raise ValueError(
            "the tokenizer needs more than one token to tell this scale's levels "
            "apart, so no single decision can be trained or scored; keep integer "
            "ranges within 0-9 or set objective: token"
        )
    prefix, legal_ids = decision
    classes = len(legal_ids)

    class OrdinalSFTTrainer(SFTTrainer):
        def compute_loss(
            self, model, inputs, return_outputs=False, num_items_in_batch=None
        ):
            input_ids = inputs["input_ids"]
            labels = inputs["labels"]
            device = input_ids.device

            supervised = labels != -100
            if not bool(supervised.any()):
                return input_ids.sum() * 0.0

            logits = model(
                input_ids=input_ids, attention_mask=inputs.get("attention_mask")
            ).logits.float()
            rows = torch.arange(input_ids.size(0), device=device)
            ids = torch.tensor(legal_ids, device=device)
            # the completions diverge `prefix` tokens into the answer; the logits
            # one step earlier are the ones that predict the deciding token
            decides_at = supervised.float().argmax(dim=1) + prefix
            class_logits = logits[rows, decides_at - 1][:, ids]
            answers = input_ids[rows, decides_at]
            target = (answers.unsqueeze(1) == ids.unsqueeze(0)).float().argmax(dim=1)

            nll = F.cross_entropy(class_logits, target)
            probabilities = F.softmax(class_logits, dim=-1)
            cumulative = probabilities.cumsum(dim=-1)
            steps = (
                torch.arange(classes, device=device).unsqueeze(0) >= target.unsqueeze(1)
            ).to(cumulative.dtype)
            rps = ((cumulative - steps) ** 2).sum(dim=-1).mean() / max(classes - 1, 1)
            return nll + rps

    return OrdinalSFTTrainer


def _last_logged_loss(log_history: list[dict]) -> float | None:
    for entry in reversed(log_history):
        if "loss" in entry:
            return round(entry["loss"], 4)
    return None


def _quality_value(spec: FunctionSpec, metrics: dict) -> float:
    if not spec.output.is_scalar:
        return metrics["joint_decision_agreement"]
    if spec.output.scalar.type == "int":
        return metrics["within_one"]
    return metrics["decision_agreement"]


def _make_dev_callback(
    spec: FunctionSpec,
    config: LoraCandidateSpec,
    tokenizer,
    dev_rows: list[Row],
    adapter_dir: Path,
):
    """TrainerCallback: score the dev split each epoch (constrained decode,
    task agreement), snapshot the adapter whenever it improves, and stop after
    `patience` epochs without improvement. The saved adapter is always the
    best-so-far, so early stopping never ships a worse-than-seen checkpoint."""
    import torch
    from transformers import TrainerCallback

    from . import prompts
    from .evaluate import compute_metrics, generate_batch
    from .labeling import row_output

    dev_texts = [prompts.student_prompt(spec, r["input"]) for r in dev_rows]
    golds = [row_output(spec, r) for r in dev_rows]
    allowed = prompts.allowed_completions(spec, config.rationale_distillation)
    max_new = prompts.completion_budget(spec, config.rationale_distillation)

    class DevEval(TrainerCallback):
        def __init__(self):
            self.curve: list[dict] = []
            self.best: float | None = None
            self.best_epoch: int | None = None
            self.stale = 0
            self.stopped_reason = "max_epochs"

        def on_epoch_end(self, args, state, control, model=None, **kwargs):
            epoch = int(round(state.epoch))
            was_training = model.training
            # generate_batch flips padding_side to left; the training collator
            # needs it back or every later epoch trains on left-padded batches
            pad_side = tokenizer.padding_side
            model.eval()
            with torch.no_grad():
                # dev-eval reductions print generate_batch's one-liner; the
                # final eval re-derives its own effective size for the report
                raw, _ = generate_batch(
                    model, tokenizer, dev_texts, max_new,
                    batch_size=config.eval_batch_size,
                    allowed_completions=allowed,
                )
            tokenizer.padding_side = pad_side
            if was_training:
                model.train()
            preds = [prompts.parse_output(spec, t) for t in raw]
            agreement = _quality_value(spec, compute_metrics(spec, preds, golds))
            self.curve.append({
                "epoch": epoch,
                "train_loss": _last_logged_loss(state.log_history),
                "dev_agreement": agreement,
            })
            print(f"epoch {epoch}: dev_agreement={agreement:.4f}", flush=True)

            if self.best is None or agreement > self.best + config.min_delta:
                self.best = agreement
                self.best_epoch = epoch
                self.stale = 0
                model.save_pretrained(str(adapter_dir))
            else:
                self.stale += 1
                if config.patience is not None and self.stale >= config.patience:
                    self.stopped_reason = f"early_stop(patience={config.patience})"
                    control.should_training_stop = True
            return control

    return DevEval()


def train(
    spec: FunctionSpec,
    config: LoraCandidateSpec,
    train_rows: list[Row],
    out_dir: Path,
    dev_rows: list[Row] | None = None,
) -> dict:
    """Fine-tune a LoRA adapter; saves it to out_dir/adapter. Returns run info.

    With `dev_rows`, the adapter written is the best-dev-agreement epoch (with
    early stopping per spec.train.patience), not necessarily the final one.
    """
    from datasets import Dataset
    from peft import LoraConfig, prepare_model_for_kbit_training
    from trl import SFTConfig, SFTTrainer

    precision = pick_precision(config.precision)
    tokenizer, model = load_base_model(config.model, precision)
    if precision == "qlora":
        # prepare_model_for_kbit_training upcasts every non-quantized module
        # to fp32; for huge-vocab models the tied embedding alone can be a
        # ~4GB fp32 tensor (Qwen3.5-9B: 151936x6656) and OOMs 12GB cards.
        # In that case do the two things we actually need by hand and keep
        # the embedding in its load dtype.
        embed = model.get_input_embeddings()
        if embed.weight.numel() * 4 > 3_000_000_000:
            model.gradient_checkpointing_enable()
            model.enable_input_require_grads()
        else:
            model = prepare_model_for_kbit_training(model)

    from .labeling import row_output

    ds = Dataset.from_list(
        [
            {
                "prompt": prompts.student_prompt(spec, r["input"]),
                "completion": prompts.student_completion(
                    spec,
                    row_output(spec, r),
                    r.get("reason", ""),
                    config.rationale_distillation,
                )
                + tokenizer.eos_token,
            }
            for r in train_rows
        ]
    )

    lora = LoraConfig(
        r=config.lora_r,
        lora_alpha=config.alpha,
        lora_dropout=config.lora_dropout,
        use_dora=config.use_dora,
        target_modules="all-linear",
        task_type="CAUSAL_LM",
    )
    ordinal = ordinal_objective_applies(spec, config)
    cfg = SFTConfig(
        output_dir=str(out_dir / "trainer"),
        num_train_epochs=config.max_epochs,
        learning_rate=config.learning_rate,
        per_device_train_batch_size=config.batch_size,
        max_length=config.max_seq_len,
        bf16=(precision == "bf16"),
        fp16=False,
        seed=config.seed,
        logging_steps=20,
        save_strategy="epoch",
        save_total_limit=1,
        report_to=[],
        # MoE load-balancing aux loss is meaningless for a frozen-base LoRA
        # student, and TRL's nonzero default crashes dense models whose config
        # merely carries the router attribute (e.g. Granite 4.0 hybrids)
        router_aux_loss_coef=0.0,
        **({"loss_type": config.loss_type} if config.loss_type else {}),
    )
    adapter_dir = out_dir / "model"
    dev_cb = (
        _make_dev_callback(spec, config, tokenizer, dev_rows, adapter_dir)
        if dev_rows
        else None
    )
    trainer_class = _ordinal_trainer_class(spec, tokenizer) if ordinal else SFTTrainer
    trainer = trainer_class(
        model=model, args=cfg, train_dataset=ds, processing_class=tokenizer,
        peft_config=lora, callbacks=[dev_cb] if dev_cb else None,
    )
    checkpoints = sorted((out_dir / "trainer").glob("checkpoint-*"))
    result = trainer.train(resume_from_checkpoint=str(checkpoints[-1]) if checkpoints else None)

    if dev_cb is None or dev_cb.best_epoch is None:
        # no dev split (or it never scored): fall back to the final adapter
        trainer.model.save_pretrained(str(adapter_dir))
    tokenizer.save_pretrained(str(adapter_dir))
    epochs_run = int(round(trainer.state.epoch or config.max_epochs))
    decoder, comparison = _select_decoder(
        spec, config, trainer.model, tokenizer, dev_rows
    )
    return {
        "precision": precision,
        "objective": "ordinal" if ordinal else "token",
        "decode": decoder,
        "dev_decode_comparison": comparison,
        "train_rows": len(train_rows),
        "train_loss": round(result.training_loss, 4),
        "adapter_dir": str(adapter_dir),
        "curve": dev_cb.curve if dev_cb else [],
        "best_epoch": dev_cb.best_epoch if dev_cb else None,
        "best_dev_agreement": dev_cb.best if dev_cb else None,
        "epochs_run": epochs_run,
        "stopped_reason": dev_cb.stopped_reason if dev_cb else "max_epochs",
        "dev_rows": len(dev_rows or []),
    }


def _select_decoder(
    spec: FunctionSpec, config: LoraCandidateSpec, model, tokenizer, dev_rows
) -> tuple[str, dict | None]:
    """Which point of the level distribution to report, decided on dev.

    The mode maximizes exact agreement and the median minimizes absolute error;
    which one reproduces the supplied decisions better is an empirical question,
    so `auto` measures both on the development split and keeps the winner. It
    is never decided on the evaluation split.
    """
    from . import decode
    from .labeling import row_output
    from .metrics import compare

    levels = decode.scale_levels(spec)
    if levels is None or config.rationale_distillation:
        return "argmax", None
    if config.decode != "auto":
        return config.decode, None
    if not dev_rows:
        return "argmax", None

    texts = [prompts.student_prompt(spec, row["input"]) for row in dev_rows]
    references = [row_output(spec, row) for row in dev_rows]
    distributions = decode.score_levels(
        model, tokenizer, spec, texts, config.eval_batch_size
    )
    comparison = {}
    for candidate in decode.DECODERS:
        predictions = decode.decode_levels(distributions, levels, candidate)
        comparison[candidate] = compare(spec, predictions, references)
    chosen = max(
        decode.DECODERS, key=lambda name: _quality_value(spec, comparison[name])
    )
    _progress(
        "decoder selected on dev: "
        + " ".join(
            f"{name}={_quality_value(spec, comparison[name]):.4f}"
            for name in decode.DECODERS
        )
        + f" -> {chosen}"
    )
    return chosen, comparison
