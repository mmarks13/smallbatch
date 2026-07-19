"""Per-field training objective: span attribution, losses, and normalization.

Every LoRA function — single enum, single integer, single text, structured
bounded, or decision-plus-text — trains and checkpoints under one rule:

    semantic_loss = sum(weight_f x field_loss_f / untuned_base_dev_loss_f)
                    / sum(weight_f)
    total_loss    = semantic_loss + 0.1 x normalized_format_loss

Each output field gets a type-appropriate semantic loss:

- int:   class NLL + ranked probability score over the legal level tokens
         (`heads.ordinal_loss` semantics), so the scale trains as ordered
         levels even beside a text field.
- enum:  legal-value NLL — token NLL renormalized at every step over the
         label trie's legal continuations, which is exactly -log P(label |
         legal labels) factored autoregressively. Tokens every label shares
         renormalize to probability one and cancel.
- text:  mean next-token NLL over the field's tokens. The per-row mean keeps
         a longer text from earning more influence just by having more
         tokens.

Serialization structure (JSON braces, key names, quotes, the `name:` lines of
the compact bounded format, and the final EOS) is a separate internal format
objective at fixed weight 0.1. It never enters user field weights and never
enters checkpoint selection.

Normalization divides every loss by the untuned base model's loss for the
same field on the development split, measured once before training. Values
below 1.0 mean the adapter improved on the base for that field, and the
weighted mean stays comparable as fields are added. A baseline at or below
`BASELINE_EPS` fails the candidate before training — never silently clamped.

Token attribution: the canonical completion is built as (owner, text)
segments whose boundaries sit on BPE pretokenizer boundaries (a value keeps
its leading space; JSON quotes and punctuation stay in format), then each
token is owned by the segment holding the majority of its characters, ties to
the later segment. Every supervised token is owned by exactly one objective —
none are dropped or double-counted. An integer field whose levels tokenize
with a shared prefix contributes that constant prefix to format and its one
deciding token to the ordinal loss.
"""

from __future__ import annotations

import json
from typing import Any

from . import prompts
from .spec import SCALAR_FIELD, FunctionSpec

FORMAT_LOSS_WEIGHT = 0.1
FORMAT_KEY = "__format__"
# An untuned dev loss this small leaves nothing meaningful to normalize by;
# dividing by it would let noise dominate the objective.
BASELINE_EPS = 1e-3

_PROMPT_OWNER = -1
_FORMAT_OWNER = 0


def completion_segments(spec: FunctionSpec, output: Any) -> list[tuple[str, str]]:
    """(owner, text) pieces whose concatenation is the canonical completion.

    Owner "" is format; otherwise an output field name. Matches
    `prompts.student_completion` exactly (tested), so training supervises the
    same bytes inference must produce.
    """
    fields = spec.output.fields
    values = {SCALAR_FIELD: output} if spec.output.is_scalar else output
    if not spec.output.has_text:
        if spec.output.is_scalar:
            return [(SCALAR_FIELD, f" {output}")]
        segments: list[tuple[str, str]] = []
        for index, name in enumerate(fields):
            lead = " " if index == 0 else "\n"
            segments.append(("", f"{lead}{name}:"))
            segments.append((name, f" {values[name]}"))
        return segments
    if spec.output.is_scalar:
        encoded = json.dumps(output, ensure_ascii=False)
        return [("", ' "'), (SCALAR_FIELD, encoded[1:-1]), ("", '"')]
    segments = [("", " {")]
    for index, (name, field) in enumerate(fields.items()):
        prefix = "" if index == 0 else ", "
        key = json.dumps(name, ensure_ascii=False)
        if field.type == "int":
            segments.append(("", f"{prefix}{key}:"))
            segments.append((name, f" {values[name]}"))
        else:  # enum and text values are JSON strings; the quotes are format
            encoded = json.dumps(values[name], ensure_ascii=False)
            segments.append(("", f'{prefix}{key}: "'))
            segments.append((name, encoded[1:-1]))
            segments.append(("", '"'))
    segments.append(("", "}"))
    return segments


def _tokenize_with_offsets(tokenizer, text: str):
    encoded = tokenizer(text, add_special_tokens=False, return_offsets_mapping=True)
    offsets = encoded.get("offset_mapping")
    if offsets is None:
        raise ValueError(
            "per-field training needs a fast tokenizer with offset mapping; "
            f"{type(tokenizer).__name__} provides none. Use a model whose "
            "tokenizer has a tokenizer.json backend"
        )
    return encoded["input_ids"], offsets


def attribute_tokens(
    tokenizer, segments: list[tuple[str, str]]
) -> tuple[list[int], list[int]]:
    """Tokenize the concatenated segments and own each token.

    Returns (token_ids, owner_index_per_token) where owner index is the
    position of the owning segment in `segments`. A token spanning a boundary
    goes to the segment holding the majority of its characters, ties to the
    later segment (so a value's merged leading character stays with the
    value).
    """
    completion = "".join(text for _, text in segments)
    boundaries: list[int] = []
    position = 0
    for _, text in segments:
        boundaries.append(position)
        position += len(text)
    token_ids, offsets = _tokenize_with_offsets(tokenizer, completion)

    def segment_of(char_index: int) -> int:
        chosen = 0
        for index, start in enumerate(boundaries):
            if start <= char_index:
                chosen = index
        return chosen

    owners: list[int] = []
    for start, end in offsets:
        counts: dict[int, int] = {}
        for char_index in range(start, max(end, start + 1)):
            segment = segment_of(min(char_index, position - 1))
            counts[segment] = counts.get(segment, 0) + 1
        best = max(counts.values())
        owners.append(max(segment for segment, count in counts.items() if count == best))
    return token_ids, owners


def field_codecs(spec: FunctionSpec, tokenizer) -> dict[str, dict]:
    """Per-field token vocabulary for the bounded semantic losses.

    int fields: the single deciding token id per level (a shared multi-token
    prefix is tolerated and supervised as format). enum fields: a trie over
    each label's token sequence, in the exact serialized form the completion
    uses. text fields need no codec (full-vocabulary NLL).

    Fails closed with an actionable error when a tokenizer cannot represent a
    level as one deciding token.
    """
    codecs: dict[str, dict] = {}
    json_form = spec.output.has_text
    for name, field in spec.output.fields.items():
        if field.type == "text":
            continue
        if field.type == "int":
            variants = [
                tokenizer(f" {value}", add_special_tokens=False)["input_ids"]
                for value in field.values()
            ]
            shortest = min(len(tokens) for tokens in variants)
            prefix = 0
            while prefix < shortest and len({t[prefix] for t in variants}) == 1:
                prefix += 1
            deciding = [tokens[prefix:] for tokens in variants]
            ids = [tokens[0] for tokens in deciding if len(tokens) == 1]
            if len(ids) != len(variants) or len(set(ids)) != len(ids):
                raise ValueError(
                    f"the tokenizer needs more than one token to tell field "
                    f"{name!r}'s levels apart, so no single position carries the "
                    "decision; keep integer ranges within 0-9"
                )
            codecs[name] = {
                "kind": "int",
                "prefix_len": prefix,
                "legal_ids": ids,
                "target_of_id": {token: index for index, token in enumerate(ids)},
                "values": field.values(),
            }
        else:
            sequences = {}
            for label in field.values():
                serialized = (
                    json.dumps(label, ensure_ascii=False)[1:-1] if json_form else f" {label}"
                )
                sequences[label] = tuple(
                    tokenizer(serialized, add_special_tokens=False)["input_ids"]
                )
            trie: dict[tuple, set] = {}
            for tokens in sequences.values():
                for index in range(len(tokens)):
                    trie.setdefault(tokens[:index], set()).add(tokens[index])
            codecs[name] = {"kind": "enum", "sequences": sequences, "trie": trie}
    return codecs


def encode_row(
    spec: FunctionSpec,
    tokenizer,
    codecs: dict[str, dict],
    prompt: str,
    output: Any,
    max_seq_len: int | None = None,
) -> dict[str, list[int]]:
    """One training/scoring example: ids, labels, and per-token owners.

    owners: -1 for prompt and padding (never supervised), 0 for format
    (including EOS), i+1 for the i-th declared output field. Verifies that
    every bounded field's owned span tokenizes exactly as its codec expects,
    so attribution errors fail loudly at encoding time, not silently in the
    loss.
    """
    field_names = list(spec.output.fields)
    segments = completion_segments(spec, output)
    completion_ids, segment_owner = attribute_tokens(tokenizer, segments)
    owner_of_segment = [
        0 if owner == "" else field_names.index(owner) + 1 for owner, _ in segments
    ]
    owners = [owner_of_segment[index] for index in segment_owner]

    values = {SCALAR_FIELD: output} if spec.output.is_scalar else output
    for name, codec in codecs.items():
        field_index = field_names.index(name) + 1
        span = [i for i, owner in enumerate(owners) if owner == field_index]
        span_ids = [completion_ids[i] for i in span]
        if codec["kind"] == "int":
            # a shared multi-token prefix carries no decision: it is format
            expected_prefix = codec["prefix_len"]
            if len(span_ids) != expected_prefix + 1 or span_ids[-1] not in codec["target_of_id"]:
                raise ValueError(
                    f"tokenizer split field {name!r} unexpectedly in context "
                    f"({span_ids}); cannot attribute its deciding token"
                )
            for i in span[:-1]:
                owners[i] = _FORMAT_OWNER
        else:
            expected = codec["sequences"][values[name]]
            if tuple(span_ids) != expected:
                raise ValueError(
                    f"tokenizer split enum field {name!r} value {values[name]!r} "
                    f"differently in context ({span_ids} vs {list(expected)}); "
                    "cannot attribute its tokens"
                )

    prompt_ids = tokenizer(prompt)["input_ids"]
    eos = tokenizer.eos_token_id
    input_ids = [*prompt_ids, *completion_ids, eos]
    if max_seq_len is not None and len(input_ids) > max_seq_len:
        raise ValueError(
            f"row needs {len(input_ids)} tokens but max_seq_len is {max_seq_len}; "
            "completions are never truncated (a cut completion is an invalid "
            "output) — raise max_seq_len or shorten the inputs"
        )
    labels = [-100] * len(prompt_ids) + completion_ids + [eos]
    owners_full = [_PROMPT_OWNER] * len(prompt_ids) + owners + [_FORMAT_OWNER]
    return {"input_ids": input_ids, "labels": labels, "owners": owners_full}


def _renormalized_nll(logits_row, span_ids: list[int], trie: dict) -> Any:
    """-log P(label | legal labels): token NLL renormalized over the trie's
    legal continuations at every step of the reference sequence."""
    import torch
    import torch.nn.functional as F

    total = None
    for step, token in enumerate(span_ids):
        allowed = sorted(trie.get(tuple(span_ids[:step]), {token}))
        if len(allowed) == 1:
            continue  # every legal label shares this token: it cancels
        legal = logits_row[step][torch.tensor(allowed, device=logits_row.device)]
        value = -F.log_softmax(legal, dim=-1)[allowed.index(token)]
        total = value if total is None else total + value
    if total is None:
        total = logits_row.sum() * 0.0
    return total


def row_losses(
    logits,
    input_ids,
    owners,
    spec: FunctionSpec,
    codecs: dict[str, dict],
    logits_start: int,
):
    """Per-row, per-field semantic losses plus the format loss.

    `logits[:, j]` scores position `logits_start + j + 1` of `input_ids` (the
    usual one-step shift). Returns {name: tensor[batch]} including
    FORMAT_KEY; int fields return (nll + rps) per row, enums the
    renormalized label NLL, text and format the mean token NLL over their
    spans.
    """
    import torch
    import torch.nn.functional as F

    field_names = list(spec.output.fields)
    batch, _, _ = logits.shape
    device = logits.device
    losses: dict[str, list] = {name: [] for name in field_names}
    losses[FORMAT_KEY] = []

    for row in range(batch):
        row_ids = input_ids[row]
        row_owner = owners[row]
        positions = {
            name: [] for name in (*field_names, FORMAT_KEY)
        }
        for position in range(logits_start + 1, row_ids.shape[0]):
            owner = int(row_owner[position])
            if owner == _PROMPT_OWNER:
                continue
            key = FORMAT_KEY if owner == _FORMAT_OWNER else field_names[owner - 1]
            positions[key].append(position)

        def logit_at(position: int):
            return logits[row, position - 1 - logits_start]

        for name in field_names:
            field = spec.output.fields[name]
            span = positions[name]
            if field.type == "int":
                codec = codecs[name]
                ids = torch.tensor(codec["legal_ids"], device=device)
                class_logits = logit_at(span[-1])[ids].float().unsqueeze(0)
                target = torch.tensor(
                    [codec["target_of_id"][int(row_ids[span[-1]])]], device=device
                )
                from .heads import ordinal_loss

                losses[name].append(ordinal_loss(class_logits, target))
            elif field.type == "enum":
                span_logits = torch.stack([logit_at(p).float() for p in span])
                span_ids = [int(row_ids[p]) for p in span]
                losses[name].append(
                    _renormalized_nll(span_logits, span_ids, codecs[name]["trie"])
                )
            else:  # text: mean full-vocabulary next-token NLL
                span_logits = torch.stack([logit_at(p).float() for p in span])
                targets = torch.tensor([int(row_ids[p]) for p in span], device=device)
                losses[name].append(
                    F.cross_entropy(span_logits, targets, reduction="mean")
                )
        span = positions[FORMAT_KEY]
        span_logits = torch.stack([logit_at(p).float() for p in span])
        targets = torch.tensor([int(row_ids[p]) for p in span], device=device)
        losses[FORMAT_KEY].append(F.cross_entropy(span_logits, targets, reduction="mean"))

    return {name: torch.stack(values) for name, values in losses.items()}


def combine_losses(field_losses: dict, weights: dict[str, float], baselines: dict[str, float]):
    """The universal objective: weighted mean of normalized semantic losses
    plus the fixed-weight normalized format loss. `field_losses` values may be
    tensors (training) or floats (reporting)."""
    total_weight = sum(weights.values())
    semantic = sum(
        weights[name] * field_losses[name] / baselines[name] for name in weights
    ) / total_weight
    return semantic + FORMAT_LOSS_WEIGHT * (
        field_losses[FORMAT_KEY] / baselines[FORMAT_KEY]
    )


def checkpoint_score(
    field_losses: dict[str, float], weights: dict[str, float], baselines: dict[str, float]
) -> float:
    """Weighted mean of normalized per-field development losses. Lower is
    better. Format loss is excluded: structure is a training aid, not the
    task. This one score drives best-checkpoint selection, early stopping,
    patience, and minimum-improvement behavior for every LoRA shape."""
    total_weight = sum(weights.values())
    return (
        sum(weights[name] * field_losses[name] / baselines[name] for name in weights)
        / total_weight
    )


def check_baselines(baselines: dict[str, float], candidate: str = "") -> None:
    """Fail a candidate before training when normalization would be unsafe."""
    import math

    for name, value in baselines.items():
        if value is None or not math.isfinite(value) or value <= BASELINE_EPS:
            label = "format" if name == FORMAT_KEY else f"field {name!r}"
            raise ValueError(
                f"untuned base development loss for {label} is {value!r}, at or "
                f"below the safe normalization threshold {BASELINE_EPS}; "
                "normalized training would divide by it. The development split "
                "is too small or degenerate for this candidate — label more "
                "varied data"
            )


def collate(rows: list[dict], pad_token_id: int):
    """Right-pad a batch of encode_row outputs into tensors."""
    import torch

    width = max(len(row["input_ids"]) for row in rows)

    def pad(key: str, value: int):
        return torch.tensor(
            [row[key] + [value] * (width - len(row[key])) for row in rows]
        )

    return {
        "input_ids": pad("input_ids", pad_token_id),
        "attention_mask": torch.tensor(
            [
                [1] * len(row["input_ids"]) + [0] * (width - len(row["input_ids"]))
                for row in rows
            ]
        ),
        "labels": pad("labels", -100),
        "owners": pad("owners", _PROMPT_OWNER),
    }


def completion_logits(model, batch: dict):
    """Model logits for the completion region only, plus the region's start.

    Only positions from the earliest supervised token onward are ever read,
    so the model is asked to project just those (`logits_to_keep`) — the full
    sequence-by-vocabulary logits are what evict 1B+ fp32 students from 11GB
    cards. An integer keep-count keeps the LAST k positions, so when a model
    ignores the kwarg or predates it the shapes still disambiguate: width k
    means kept, width seq_len means full (and k == seq_len needs no
    disambiguation at all).
    """
    input_ids = batch["input_ids"]
    supervised = batch["labels"] != -100
    first = int(supervised.float().argmax(dim=1).min())
    keep = input_ids.shape[1] - first + 1
    keep = min(keep, input_ids.shape[1])
    try:
        logits = model(
            input_ids=input_ids,
            attention_mask=batch["attention_mask"],
            logits_to_keep=keep,
        ).logits
    except TypeError:  # model predates the logits_to_keep contract
        logits = model(
            input_ids=input_ids, attention_mask=batch["attention_mask"]
        ).logits
    start = input_ids.shape[1] - logits.shape[1]
    return logits, start


def evaluate_field_losses(
    model,
    spec: FunctionSpec,
    codecs: dict[str, dict],
    encoded_rows: list[dict],
    pad_token_id: int,
    batch_size: int = 8,
) -> dict[str, float]:
    """Teacher-forced mean per-field losses over already-encoded rows.

    Every field is scored conditioned on the correct preceding reference
    fields (teacher forcing) — including a bounded field that follows a text
    field. Used for the untuned baselines and for per-epoch development
    scoring; free-running behavior is measured separately at evaluation.
    """
    import torch

    totals: dict[str, float] = {}
    counts = 0
    for start in range(0, len(encoded_rows), batch_size):
        batch = collate(encoded_rows[start : start + batch_size], pad_token_id)
        batch = {key: value.to(model.device) for key, value in batch.items()}
        with torch.no_grad():
            logits, region_start = completion_logits(model, batch)
            losses = row_losses(
                logits, batch["input_ids"], batch["owners"], spec, codecs, region_start
            )
        rows = batch["input_ids"].shape[0]
        for name, values in losses.items():
            totals[name] = totals.get(name, 0.0) + float(values.sum())
        counts += rows
    return {name: value / counts for name, value in totals.items()}


def text_fidelity(
    model,
    tokenizer,
    spec: FunctionSpec,
    codecs: dict[str, dict],
    rows: list[dict],
    batch_size: int = 8,
) -> dict[str, Any] | None:
    """Held-out teacher-text prediction metrics for the text field.

    These measure how well the candidate predicts the reference text token by
    token under teacher forcing — reference fidelity and nothing more. They
    are report-only and never claim correctness, factuality, usefulness, or
    semantic equivalence. Bits per byte is the primary cross-model metric;
    raw NLL and perplexity depend on the tokenizer and are not comparable
    across candidates with different tokenizers.
    """
    import math

    import torch
    import torch.nn.functional as F

    text_field = spec.output.text_field
    if text_field is None:
        return None
    field_index = list(spec.output.fields).index(text_field) + 1
    encoded = [
        encode_row(
            spec, tokenizer, codecs, prompts.student_prompt(spec, row["input"]), row["output"]
        )
        for row in rows
    ]
    per_example: list[dict] = []
    for start in range(0, len(encoded), batch_size):
        batch = collate(encoded[start : start + batch_size], tokenizer.pad_token_id)
        batch = {key: value.to(model.device) for key, value in batch.items()}
        with torch.no_grad():
            logits, region_start = completion_logits(model, batch)
        for row in range(batch["input_ids"].shape[0]):
            row_ids = batch["input_ids"][row]
            row_owner = batch["owners"][row]
            span = [
                position
                for position in range(region_start + 1, row_ids.shape[0])
                if int(row_owner[position]) == field_index
            ]
            span_logits = torch.stack(
                [logits[row, p - 1 - region_start].float() for p in span]
            )
            targets = torch.tensor([int(row_ids[p]) for p in span], device=span_logits.device)
            nll = F.cross_entropy(span_logits, targets, reduction="none")
            segment = next(
                text
                for owner, text in completion_segments(
                    spec, rows[start + row]["output"]
                )
                if owner == text_field
            )
            per_example.append(
                {
                    "nll_sum": float(nll.sum()),
                    "tokens": len(span),
                    "bytes": len(segment.encode("utf-8")),
                    "top1": int((span_logits.argmax(dim=-1) == targets).sum()),
                }
            )

    total_nll = sum(example["nll_sum"] for example in per_example)
    total_tokens = sum(example["tokens"] for example in per_example)
    total_bytes = sum(example["bytes"] for example in per_example)
    example_bpb = sorted(
        example["nll_sum"] / math.log(2) / example["bytes"] for example in per_example
    )

    def percentile(q: float) -> float:
        index = max(0, min(len(example_bpb) - 1, math.ceil(q * len(example_bpb)) - 1))
        return example_bpb[index]

    token_nll = total_nll / total_tokens
    return {
        "rows": len(per_example),
        "bits_per_byte": round(total_nll / math.log(2) / total_bytes, 4),
        "token_nll": round(token_nll, 4),
        "perplexity": round(math.exp(token_nll), 4),
        "top1_accuracy": round(
            sum(example["top1"] for example in per_example) / total_tokens, 4
        ),
        "median_example_bpb": round(percentile(0.5), 4),
        "p90_example_bpb": round(percentile(0.9), 4),
        "measures": (
            "held-out teacher-text prediction under teacher forcing; not "
            "correctness, factuality, usefulness, or downstream success"
        ),
    }
