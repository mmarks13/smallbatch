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
    from .labeling import row_output

    preds = [prompts.parse_output(spec, t) for t in raw]
    metrics = compute_metrics(spec, preds, [row_output(spec, r) for r in holdout])
    metrics["preds"] = preds  # per-item, aligned with the holdout file order
    return metrics


def wilson_ci(k: int, n: int, z: float = 1.96) -> tuple[float, float] | None:
    """Wilson score interval for a proportion k/n — honest about small n,
    where the gate verdict is otherwise statistical theater."""
    if n == 0:
        return None
    p = k / n
    denom = 1 + z**2 / n
    center = (p + z**2 / (2 * n)) / denom
    half = (z / denom) * ((p * (1 - p) / n + z**2 / (4 * n**2)) ** 0.5)
    return (round(max(0.0, center - half), 4), round(min(1.0, center + half), 4))


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


def _agrees(field, p, g) -> bool:
    """Field-level agreement: ±1 for int fields, exact for enum fields."""
    if p is None:
        return False
    return abs(p - g) <= 1 if field.type == "int" else p == g


def _scalar_metrics(field, preds: list, golds: list) -> dict[str, Any]:
    n = len(golds)
    invalid = sum(1 for p in preds if p is None)
    exact = sum(1 for p, g in zip(preds, golds) if p == g)
    agree_k = sum(1 for p, g in zip(preds, golds) if _agrees(field, p, g))
    metrics: dict[str, Any] = {"n": n}
    metrics["agreement"] = round(agree_k / n, 4) if n else 0.0
    if field.type == "int":
        valid = [(p, g) for p, g in zip(preds, golds) if p is not None]
        r = pearson_r([p for p, _ in valid], [g for _, g in valid])
        metrics["pearson_r"] = round(r, 4) if r is not None else None
    ci = wilson_ci(agree_k, n)
    metrics["agreement_ci"] = list(ci) if ci else None
    metrics["exact"] = round(exact / n, 4) if n else 0.0
    metrics["invalid_rate"] = round(invalid / n, 4) if n else 0.0
    return metrics


def compute_metrics(spec: FunctionSpec, preds: list, golds: list) -> dict[str, Any]:
    """Scalar contracts: agreement/CI/exact/invalid (+pearson for int).
    Multi-field contracts: the same per field under `fields`, with the
    headline `agreement` being the JOINT rate (every field agreeing)."""
    if spec.output.is_scalar:
        return _scalar_metrics(spec.output.scalar, preds, golds)

    n = len(golds)
    fields = spec.output.fields
    dicts = [p if isinstance(p, dict) else {} for p in preds]
    per_field = {
        name: _scalar_metrics(
            field, [d.get(name) for d in dicts], [g[name] for g in golds]
        )
        for name, field in fields.items()
    }
    joint_k = sum(
        1
        for d, g in zip(dicts, golds)
        if all(_agrees(f, d.get(name), g[name]) for name, f in fields.items())
    )
    exact_k = sum(
        1
        for d, g in zip(dicts, golds)
        if all(d.get(name) == g[name] for name in fields)
    )
    invalid = sum(1 for d in dicts if any(d.get(name) is None for name in fields))
    ci = wilson_ci(joint_k, n)
    return {
        "n": n,
        "agreement": round(joint_k / n, 4) if n else 0.0,  # joint: all fields
        "agreement_ci": list(ci) if ci else None,
        "exact": round(exact_k / n, 4) if n else 0.0,
        "invalid_rate": round(invalid / n, 4) if n else 0.0,
        "fields": per_field,
    }


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
    return {"passed": not reasons, "reasons": reasons}
