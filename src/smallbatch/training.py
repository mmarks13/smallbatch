"""LoRA fine-tuning of the student on teacher-labeled data."""

from __future__ import annotations

from pathlib import Path

from . import prompts
from .hardware import pick_precision
from .labeling import Row
from .spec import FunctionSpec


def load_base_model(base: str, precision: str):
    """Load tokenizer + base model at the given precision. Shared with eval."""
    import torch
    from transformers import AutoModelForCausalLM, AutoTokenizer

    tokenizer = AutoTokenizer.from_pretrained(base)
    if tokenizer.pad_token is None:
        tokenizer.pad_token = tokenizer.eos_token

    kwargs: dict = {}
    if precision == "bf16":
        kwargs["torch_dtype"] = torch.bfloat16
    elif precision == "fp32":
        kwargs["torch_dtype"] = torch.float32
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


def train(spec: FunctionSpec, train_rows: list[Row], out_dir: Path) -> dict:
    """Fine-tune a LoRA adapter; saves it to out_dir/adapter. Returns run info."""
    from datasets import Dataset
    from peft import LoraConfig, prepare_model_for_kbit_training
    from trl import SFTConfig, SFTTrainer

    precision = pick_precision(spec.train.precision)
    tokenizer, model = load_base_model(spec.train.base, precision)
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

    ds = Dataset.from_list(
        [
            {
                "prompt": prompts.student_prompt(spec, r["input"]),
                "completion": prompts.student_completion(spec, r["score"], r.get("reason", ""))
                + tokenizer.eos_token,
            }
            for r in train_rows
        ]
    )

    lora = LoraConfig(
        r=spec.train.lora_r,
        lora_alpha=spec.train.alpha,
        lora_dropout=spec.train.lora_dropout,
        use_dora=spec.train.use_dora,
        target_modules="all-linear",
        task_type="CAUSAL_LM",
    )
    cfg = SFTConfig(
        output_dir=str(out_dir / "trainer"),
        num_train_epochs=spec.train.epochs,
        learning_rate=spec.train.learning_rate,
        per_device_train_batch_size=spec.train.batch_size,
        max_length=spec.train.max_seq_len,
        bf16=(precision == "bf16"),
        fp16=False,
        seed=spec.train.seed,
        logging_steps=20,
        save_strategy="no",
        report_to=[],
        # MoE load-balancing aux loss is meaningless for a frozen-base LoRA
        # student, and TRL's nonzero default crashes dense models whose config
        # merely carries the router attribute (e.g. Granite 4.0 hybrids)
        router_aux_loss_coef=0.0,
        **({"loss_type": spec.train.loss_type} if spec.train.loss_type else {}),
    )
    trainer = SFTTrainer(
        model=model, args=cfg, train_dataset=ds, processing_class=tokenizer, peft_config=lora
    )
    result = trainer.train()

    adapter_dir = out_dir / "adapter"
    trainer.model.save_pretrained(str(adapter_dir))
    tokenizer.save_pretrained(str(adapter_dir))
    return {
        "precision": precision,
        "train_rows": len(train_rows),
        "train_loss": round(result.training_loss, 4),
        "adapter_dir": str(adapter_dir),
    }
