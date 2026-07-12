"""Holdout evaluation and the compile gate."""

from __future__ import annotations

from typing import Any, Callable

from . import prompts
from .labeling import Row
from .metrics import agrees as _agrees
from .metrics import pearson_r, wilson_ci  # noqa: F401 (legacy import sites)
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


def generate_batch(
    model,
    tokenizer,
    texts: list[str],
    max_new_tokens: int,
    batch_size: int = 16,
    allowed_completions: list[str] | None = None,
) -> list[str]:
    """Greedy generation; returns only the newly generated text per prompt.

    With `allowed_completions`, decoding is constrained token-by-token to
    those exact strings (plus EOS), so an invalid output is impossible.
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
    outs: list[str] = []
    for start in range(0, len(texts), batch_size):
        batch = texts[start : start + batch_size]
        enc = tokenizer(batch, return_tensors="pt", padding=True, truncation=True, max_length=2048)
        enc = {k: v.to(model.device) for k, v in enc.items()}
        extra = {}
        if constrain:
            # left padding makes the prompt length uniform within the batch
            extra["prefix_allowed_tokens_fn"] = _prefix_allowed_fn(
                *constrain[:2], enc["input_ids"].shape[1], constrain[2]
            )
        with torch.no_grad():
            gen = model.generate(
                **enc,
                max_new_tokens=max_new_tokens,
                do_sample=False,
                pad_token_id=tokenizer.pad_token_id,
                **extra,
            )
        new_tokens = gen[:, enc["input_ids"].shape[1] :]
        outs.extend(tokenizer.batch_decode(new_tokens, skip_special_tokens=True))
    return outs


def score_holdout(
    spec: FunctionSpec,
    model,
    tokenizer,
    holdout: list[Row],
    prompt_fn: Callable[[dict], str],
    max_new_tokens: int,
) -> dict[str, Any]:
    texts = [prompt_fn(r["input"]) for r in holdout]
    # both the adapter and the zero-shot baseline decode under the same
    # output-contract constraint (None in rationale mode), so the gate
    # comparison stays apples-to-apples and invalid outputs are impossible
    raw = generate_batch(
        model, tokenizer, texts, max_new_tokens,
        batch_size=spec.train.eval_batch_size,
        allowed_completions=prompts.allowed_completions(spec),
    )
    from .labeling import row_output

    preds = [prompts.parse_output(spec, t) for t in raw]
    metrics = compute_metrics(spec, preds, [row_output(spec, r) for r in holdout])
    metrics["preds"] = preds  # per-item, aligned with the holdout file order
    return metrics




def constant_baseline(field, golds: list) -> dict[str, Any] | None:
    """The ORACLE constant on this split: the single constant prediction that
    scores best on these exact labels under the field's agreement rule. It is
    a conservative gate hurdle, not a deployable train-fitted model — a model
    that can't beat it has learned the label prior, not the task."""
    if not golds:
        return None
    candidates = field.values()
    best_v, best_k = None, -1
    for v in candidates:
        k = sum(1 for g in golds if _agrees(field, v, g))
        if k > best_k:
            best_v, best_k = v, k
    return {"value": best_v, "agreement": round(best_k / len(golds), 4)}


def compute_metrics(spec: FunctionSpec, preds: list, golds: list) -> dict[str, Any]:
    """The full shared metric set (see metrics.compare) plus the gate's
    constant-baseline comparison attached per field. Multi-field contracts
    report per-field metrics under `fields` with the headline `agreement`
    being the JOINT rate (every field agreeing)."""
    from . import metrics as m

    out = m.compare(spec, preds, golds)
    if spec.output.is_scalar:
        out["constant_baseline"] = constant_baseline(spec.output.scalar, golds)
        return out
    for name, field in spec.output.fields.items():
        out["fields"][name]["constant_baseline"] = constant_baseline(
            field, [g[name] for g in golds]
        )
    return out


def run_gate(spec: FunctionSpec, adapter: dict, zeroshot: dict | None) -> dict[str, Any]:
    """Scalar: agreement vs threshold (+ must beat zero-shot). Multi-field:
    every field must clear its own threshold (gate.fields overrides
    gate.agreement) and beat zero-shot per field; joint is reported, not gated."""
    reasons = []
    if spec.output.is_scalar:
        if adapter["agreement"] < spec.gate.threshold:
            reasons.append(
                f"agreement {adapter['agreement']:.2%} < required {spec.gate.threshold:.0%}"
            )
        if spec.gate.must_beat_zeroshot and zeroshot is not None:
            if adapter["agreement"] <= zeroshot["agreement"]:
                reasons.append(
                    f"adapter agreement {adapter['agreement']:.2%} does not beat "
                    f"zero-shot base {zeroshot['agreement']:.2%}"
                )
        const = adapter.get("constant_baseline")
        if spec.gate.must_beat_constant and const is not None:
            if adapter["agreement"] <= const["agreement"]:
                reasons.append(
                    f"adapter agreement {adapter['agreement']:.2%} does not beat the "
                    f"oracle constant \"{const['value']}\" on this split "
                    f"({const['agreement']:.2%})"
                )
        return {"passed": not reasons, "reasons": reasons}

    for name in spec.output.fields:
        a = adapter["fields"][name]["agreement"]
        threshold = spec.gate.field_threshold(name)
        if a < threshold:
            reasons.append(f"{name}: agreement {a:.2%} < required {threshold:.0%}")
        if spec.gate.must_beat_zeroshot and zeroshot is not None:
            z = zeroshot["fields"][name]["agreement"]
            if a <= z:
                reasons.append(
                    f"{name}: adapter agreement {a:.2%} does not beat zero-shot {z:.2%}"
                )
        const = adapter["fields"][name].get("constant_baseline")
        if spec.gate.must_beat_constant and const is not None:
            if a <= const["agreement"]:
                reasons.append(
                    f"{name}: adapter agreement {a:.2%} does not beat the oracle "
                    f"constant \"{const['value']}\" on this split ({const['agreement']:.2%})"
                )
    return {"passed": not reasons, "reasons": reasons}
