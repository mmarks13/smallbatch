"""Constrained generation and backend-neutral evaluation."""

from __future__ import annotations

from collections.abc import Callable
from typing import Any

from . import prompts
from .labeling import Row
from .metrics import (  # noqa: F401 (public compatibility)
    field_decision_matches,
    pearson_r,
    wilson_ci,
)
from .spec import FunctionSpec


def completion_trie(tokenizer, completions: list[str]) -> tuple[dict, set]:
    """Token-prefix trie over the legal completion strings.

    Returns (trie, terminals): trie maps a generated-so-far id tuple to the
    set of legal next ids; terminals is the set of complete sequences (where
    EOS becomes legal — a state can be both, e.g. " 1" while " 10" exists).
    Uses each string's canonical standalone tokenization, which is what the
    training completions were built from.
    """
    trie: dict[tuple, set] = {}
    terminals: set[tuple] = set()
    for c in completions:
        ids = tuple(tokenizer(c, add_special_tokens=False)["input_ids"])
        for i in range(len(ids)):
            trie.setdefault(ids[:i], set()).add(ids[i])
        trie.setdefault(ids, set())
        terminals.add(ids)
    return trie, terminals


def _prefix_allowed_fn(trie: dict, terminals: set, prompt_len: int, eos_id: int):
    def fn(batch_id, input_ids):
        gen = tuple(input_ids[prompt_len:].tolist())
        allowed = set(trie.get(gen, set()))
        if gen in terminals or not allowed:
            allowed.add(eos_id)
        return list(allowed)

    return fn


def _oom_backoff(process, items: list, batch_size: int, is_oom, on_oom=None):
    """Run `process(slice)` over successive slices of `items`, halving the
    batch size (floor 1) and retrying the SAME slice whenever `is_oom(exc)`.

    The reduction is sticky for the remainder of the call: prompts in one
    eval are similar length, so growing back up just re-pays the OOM. OOM at
    size 1 re-raises — that is a genuine capacity failure and must stay a
    loud error, not an infinite loop. Non-OOM exceptions propagate untouched.

    Returns (results, final_batch_size). Pure: torch-free and unit-testable
    with a stub exception class.
    """
    results: list = []
    size = max(1, batch_size)
    start = 0
    while start < len(items):
        chunk = items[start : start + size]
        try:
            results.extend(process(chunk))
        except Exception as e:  # noqa: BLE001 - is_oom decides; others re-raise
            if not is_oom(e) or size == 1:
                raise
            if on_oom is not None:
                on_oom()
            size = max(1, size // 2)
            continue  # retry the same slice at the new size
        start += len(chunk)
    return results, size


def generate_batch(
    model,
    tokenizer,
    texts: list[str],
    max_new_tokens: int,
    batch_size: int = 16,
    allowed_completions: list[str] | None = None,
) -> tuple[list[str], int]:
    """Greedy generation; returns (newly generated text per prompt, the
    effective batch size after any OOM backoff).

    With `allowed_completions`, decoding is constrained token-by-token to
    those exact strings (plus EOS), so an invalid output is impossible.
    Eval-time OOM is data-dependent (prompt length), so the batch loop
    self-heals by halving instead of asking the user to predict a safe size;
    results are unaffected because decoding is greedy with left padding.
    """
    import torch

    tokenizer.padding_side = "left"
    constrain = None
    if allowed_completions:
        trie, terminals = completion_trie(tokenizer, allowed_completions)
        eos_id = tokenizer.eos_token_id
        if eos_id is None:
            eos_id = tokenizer.pad_token_id
        constrain = (trie, terminals, eos_id)

    # some architectures (e.g. granite-4.0's GraniteMoeHybrid class with
    # all-attention layers, transformers 5.13) crash generate() with any KV
    # cache; completions here are tiny, so falling back to cache-free
    # generation is cheap. Sticky once tripped.
    gen_extra: dict = {}

    def process(batch: list[str]) -> list[str]:
        enc = tokenizer(batch, return_tensors="pt", padding=True, truncation=True, max_length=2048)
        enc = {k: v.to(model.device) for k, v in enc.items()}
        extra = dict(gen_extra)
        if constrain:
            # left padding makes the prompt length uniform within the batch
            extra["prefix_allowed_tokens_fn"] = _prefix_allowed_fn(
                *constrain[:2], enc["input_ids"].shape[1], constrain[2]
            )

        def _generate():
            with torch.no_grad():
                gen = model.generate(
                    **enc,
                    max_new_tokens=max_new_tokens,
                    do_sample=False,
                    pad_token_id=tokenizer.pad_token_id,
                    **extra,
                )
            return gen

        try:
            gen = _generate()
        except ValueError as e:
            if "has_previous_state" not in str(e):
                raise
            print(
                "note: this architecture's KV cache is incompatible with "
                "generate() on this transformers version — continuing without "
                "a cache (fine for short constrained completions)",
                flush=True,
            )
            gen_extra["use_cache"] = False
            extra["use_cache"] = False
            gen = _generate()
        new_tokens = gen[:, enc["input_ids"].shape[1] :]
        return tokenizer.batch_decode(new_tokens, skip_special_tokens=True)

    outs, effective = _oom_backoff(
        process,
        texts,
        batch_size,
        is_oom=lambda e: isinstance(e, torch.OutOfMemoryError),
        on_oom=lambda: torch.cuda.empty_cache() if torch.cuda.is_available() else None,
    )
    if effective != batch_size:
        print(
            f"eval batch {batch_size} OOM'd — continuing at {effective}; set "
            f"train.eval_batch_size: {effective} to avoid the retry cost",
            flush=True,
        )
    return outs, effective


def score_holdout(
    spec: FunctionSpec,
    model,
    tokenizer,
    holdout: list[Row],
    prompt_fn: Callable[[dict], str],
    max_new_tokens: int,
    batch_size: int = 16,
    rationale: bool = False,
) -> dict[str, Any]:
    texts = [prompt_fn(r["input"]) for r in holdout]
    # Candidate and zero-shot paths use the same output constraint so their
    # decision-fidelity metrics remain comparable.
    raw, effective_batch = generate_batch(
        model, tokenizer, texts, max_new_tokens,
        batch_size=batch_size,
        allowed_completions=prompts.allowed_completions(spec, rationale),
    )
    from .labeling import row_output

    preds = [prompts.parse_output(spec, t) for t in raw]
    metrics = compute_metrics(spec, preds, [row_output(spec, r) for r in holdout])
    metrics["preds"] = preds  # per-item, aligned with the holdout file order
    if effective_batch != batch_size:
        metrics["eval_batch_size_effective"] = effective_batch
    return metrics




def train_fitted_constant(spec: FunctionSpec, train_rows: list[Row]) -> Any:
    """Fit a diagnostic constant using only training decisions."""
    from .labeling import row_output

    outputs = [row_output(spec, row) for row in train_rows]
    values: dict[str, Any] = {}
    for name, field in spec.output.fields.items():
        references = [output[name] if isinstance(output, dict) else output for output in outputs]
        values[name] = max(
            field.values(),
            key=lambda candidate: sum(
                field_decision_matches(field, candidate, reference)
                for reference in references
            ),
        )
    return values[next(iter(values))] if spec.output.is_scalar else values


def compute_metrics(spec: FunctionSpec, preds: list, golds: list) -> dict[str, Any]:
    """Compute the complete shared metric set."""
    from . import metrics as m

    return m.compare(spec, preds, golds)
