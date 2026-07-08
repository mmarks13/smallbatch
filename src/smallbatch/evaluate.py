"""Holdout evaluation and the compile gate."""

from __future__ import annotations

from typing import Any, Callable

from . import prompts
from .labeling import Row
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
    preds = [prompts.parse_output(spec, t) for t in raw]
    metrics = compute_metrics(spec, preds, [r["score"] for r in holdout])
    metrics["preds"] = preds  # per-item, aligned with the holdout file order
    return metrics


def pearson_r(xs: list[float], ys: list[float]) -> float | None:
    n = len(xs)
    if n < 2:
        return None
    mx, my = sum(xs) / n, sum(ys) / n
    vx = sum((x - mx) ** 2 for x in xs)
    vy = sum((y - my) ** 2 for y in ys)
    if vx == 0 or vy == 0:
        return None  # constant series: correlation undefined
    cov = sum((x - mx) * (y - my) for x, y in zip(xs, ys))
    return cov / (vx**0.5 * vy**0.5)


def compute_metrics(spec: FunctionSpec, preds: list, golds: list) -> dict[str, Any]:
    n = len(golds)
    invalid = sum(1 for p in preds if p is None)
    exact = sum(1 for p, g in zip(preds, golds) if p == g)
    metrics: dict[str, Any] = {"n": n}
    if spec.output.type == "int":
        within1 = sum(1 for p, g in zip(preds, golds) if p is not None and abs(p - g) <= 1)
        metrics["agreement"] = round(within1 / n, 4) if n else 0.0  # gate metric: ±1
        valid = [(p, g) for p, g in zip(preds, golds) if p is not None]
        r = pearson_r([p for p, _ in valid], [g for _, g in valid])
        metrics["pearson_r"] = round(r, 4) if r is not None else None
    else:
        metrics["agreement"] = round(exact / n, 4) if n else 0.0  # gate metric: exact
    metrics["exact"] = round(exact / n, 4) if n else 0.0
    metrics["invalid_rate"] = round(invalid / n, 4) if n else 0.0
    return metrics


def run_gate(spec: FunctionSpec, adapter: dict, zeroshot: dict | None) -> dict[str, Any]:
    reasons = []
    if adapter["agreement"] < spec.gate.agreement_pm1:
        reasons.append(
            f"agreement {adapter['agreement']:.2%} < required {spec.gate.agreement_pm1:.0%}"
        )
    if spec.gate.must_beat_zeroshot and zeroshot is not None:
        if adapter["agreement"] <= zeroshot["agreement"]:
            reasons.append(
                f"adapter agreement {adapter['agreement']:.2%} does not beat "
                f"zero-shot base {zeroshot['agreement']:.2%}"
            )
    return {"passed": not reasons, "reasons": reasons}
