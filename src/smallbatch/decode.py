"""Reading an ordered decision out of a student's logits.

A level is one token, so the model's whole distribution over the scale sits in
the logits at one position: renormalize the legal levels and the result is a
proper distribution, which constrained generation would have collapsed to its
mode anyway. Scoring it directly costs one forward pass instead of a decoding
loop, and it exposes the choice generation hides — which point of the
distribution to report.

`argmax` returns the most likely level, which is what constrained greedy
decoding returns and what minimizes exact disagreement. `median` returns the
first level whose cumulative probability reaches one half, which minimizes
absolute error, so it is the better reading when a near miss matters more than
an exact hit. The candidate's `decode` setting picks one, and `auto` lets the
development split decide.
"""

from __future__ import annotations

from typing import Any

DECODERS = ("argmax", "median")


def scale_levels(spec) -> list[Any] | None:
    """The ordered levels of a scalar integer decision, else None."""
    if not spec.output.is_scalar or spec.output.scalar.type != "int":
        return None
    return spec.output.scalar.values()


def score_levels(model, tokenizer, spec, texts: list[str], batch_size: int = 16):
    """Probability of every level for every text, in one forward pass each."""
    import torch

    from .training import ordinal_decision_tokens

    decision = ordinal_decision_tokens(spec, tokenizer)
    if decision is None:
        raise ValueError("this scale needs more than one token to name a level")
    prefix, ids = decision

    from . import prompts

    tokenizer.padding_side = "left"
    completion = prompts.student_completion(spec, spec.output.scalar.values()[0])
    lead = tokenizer(completion, add_special_tokens=False)["input_ids"][:prefix]

    distributions = []
    for start in range(0, len(texts), batch_size):
        batch = texts[start : start + batch_size]
        encoded = tokenizer(
            [text + tokenizer.decode(lead) if lead else text for text in batch],
            return_tensors="pt",
            padding=True,
            add_special_tokens=False,
        )
        encoded = {key: value.to(model.device) for key, value in encoded.items()}
        with torch.no_grad():
            logits = model(**encoded).logits[:, -1, :].float()
        legal = logits[:, torch.tensor(ids, device=logits.device)]
        distributions.append(torch.softmax(legal, dim=-1).cpu())
    return torch.cat(distributions) if distributions else torch.empty(0, len(ids))


def decode_levels(distributions, levels: list[Any], decoder: str) -> list[Any]:
    """Read one level per row out of the distributions."""
    if decoder not in DECODERS:
        raise ValueError(f"unknown decoder {decoder!r}; use one of {DECODERS}")
    if decoder == "argmax":
        return [levels[int(index)] for index in distributions.argmax(dim=-1)]
    cumulative = distributions.cumsum(dim=-1)
    reached = (cumulative >= 0.5).float().argmax(dim=-1)
    return [levels[int(index)] for index in reached]
