"""Reading an ordered decision out of a student's logits.

A level is one token, so the model's whole distribution over the scale sits in
the logits at one position: renormalize the legal levels and the result is a
proper distribution. Scoring it directly costs one forward pass instead of a
decoding loop.

Every ordinal field decodes by argmax — the most likely level, which is
exactly what constrained greedy decoding would emit, so the scored
distribution and generation always agree. v0.3 removed the configurable
median/within-one decoders and their development-time selection: one decode
rule for every candidate family, chosen by the field's type, never by a
search.

The decode step is backend-neutral: any candidate that can produce a
distribution over the ordered levels — a student's renormalized logits, the
shared softmax head — reads a decision out of it the same way.
"""

from __future__ import annotations

from typing import Any


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
        )
        encoded = {key: value.to(model.device) for key, value in encoded.items()}
        with torch.no_grad():
            logits = model(**encoded).logits[:, -1, :].float()
        legal = logits[:, torch.tensor(ids, device=logits.device)]
        distributions.append(torch.softmax(legal, dim=-1).cpu())
    return torch.cat(distributions) if distributions else torch.empty(0, len(ids))


def decode_levels(distributions, levels: list[Any]) -> list[Any]:
    """Read the argmax level per row out of an array of level distributions.

    Accepts anything numpy can view as a rows-by-levels array (a CPU torch
    tensor included). Ties resolve to the lower level.
    """
    import numpy as np

    probabilities = np.asarray(distributions, dtype=float)
    chosen = probabilities.argmax(axis=1)
    return [levels[int(index)] for index in chosen]
