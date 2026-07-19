"""Canonical input serialization, teacher prompts, and constrained parsing."""

from __future__ import annotations

import itertools
import json
import math
import re
from typing import Any

from .spec import SCALAR_FIELD, FunctionSpec, validate_output

# 3: v0.3 removed the teacher "reason" channel and added strict-JSON
# completions for text-bearing outputs. Bumping the version changes every
# decision hash, so pre-v0.3 datasets fail closed instead of mixing formats.
PROMPT_VERSION = 3
_MAX_COMPLETIONS = 5000


class InvalidOutputError(ValueError):
    """A generated output violated the declared contract.

    The full output is atomic: no partial result exists when this is raised.
    `category` names the structural failure for evidence counting:
    `malformed_json`, `contract`, `empty_text`, `char_limit`, or
    `budget_exhausted`.
    """

    def __init__(self, message: str, category: str):
        super().__init__(message)
        self.category = category


def render_input(item: dict[str, Any], input_schema: dict[str, str]) -> str:
    """Stable text seen by every candidate, in declared field order."""
    return "\n".join(
        f"{name}: {json.dumps(item[name], ensure_ascii=False)}" for name in input_schema
    )


def student_prompt(spec: FunctionSpec, item: dict[str, Any]) -> str:
    return f"[{spec.name}]\n{render_input(item, spec.input_schema)}\noutput:"


def canonical_json(spec: FunctionSpec, output: Any) -> str:
    """The exact JSON a text-bearing function's completion must contain.

    Scalar text is one JSON string; structured output is one JSON object in
    declared field order (order is generation order, so it is semantically
    meaningful). Uses json.dumps' default separators — a space after `:` and
    `,` keeps values on BPE pretokenizer boundaries, so field tokens never
    merge into the surrounding JSON syntax.
    """
    if spec.output.is_scalar:
        return json.dumps(output, ensure_ascii=False)
    return json.dumps(
        {name: output[name] for name in spec.output.fields}, ensure_ascii=False
    )


def student_completion(spec: FunctionSpec, output: Any) -> str:
    if spec.output.has_text:
        return " " + canonical_json(spec, output)
    if spec.output.is_scalar:
        return f" {output}"
    return " " + "\n".join(f"{name}: {output[name]}" for name in spec.output.fields)


def allowed_completions(spec: FunctionSpec) -> list[str] | None:
    """Legal completion strings for constrained decoding, or None when the
    output contains generated text (no finite completion set exists)."""
    if spec.output.has_text:
        return None
    if spec.output.is_scalar:
        return [f" {value}" for value in spec.output.scalar.values()]
    total = 1
    for field in spec.output.fields.values():
        total *= len(field.values())
        if total > _MAX_COMPLETIONS:
            return None
    names = list(spec.output.fields)
    combos = itertools.product(*(field.values() for field in spec.output.fields.values()))
    return [
        " " + "\n".join(f"{name}: {value}" for name, value in zip(names, combo))
        for combo in combos
    ]


def _field_instruction(field) -> str:
    if field.type == "int":
        return f"an integer from {field.range[0]} to {field.range[1]}"
    if field.type == "text":
        # the character limit rides in every labeling and zero-shot prompt so
        # over-limit answers stay rare; validation still enforces it exactly
        return (
            f"a non-empty string of at most {field.max_chars} characters"
        )
    return "exactly one of: " + ", ".join(field.labels)


def output_instruction(spec: FunctionSpec) -> str:
    if spec.output.is_scalar:
        if spec.output.has_text:
            return "a JSON string containing " + _field_instruction(spec.output.scalar)
        return _field_instruction(spec.output.scalar)
    if spec.output.has_text:
        inner = ", ".join(
            f'"{name}": <{_field_instruction(field)}>'
            for name, field in spec.output.fields.items()
        )
        return f"a JSON object with exactly these keys in this order: {{{inner}}}"
    return "one `name: value` line per field, in this order: " + "; ".join(
        f"{name}: <{_field_instruction(field)}>"
        for name, field in spec.output.fields.items()
    )


def zeroshot_prompt(spec: FunctionSpec, item: dict[str, Any]) -> str:
    return (
        f"Decision instructions:\n{spec.prompt.strip()}\n\n"
        f"Input:\n{render_input(item, spec.input_schema)}\n"
        f"Respond with only {output_instruction(spec)}.\noutput:"
    )


def teacher_label_prompt(
    spec: FunctionSpec,
    items: list[dict[str, Any]],
    field_order: list[str] | None = None,
) -> str:
    order = field_order or list(spec.input_schema)
    numbered = json.dumps(
        [{"id": index, **{name: item[name] for name in order}} for index, item in enumerate(items)],
        indent=1,
        ensure_ascii=False,
    )
    if spec.output.is_scalar:
        form = '{"id": 0, "output": <' + output_instruction(spec) + ">}"
    else:
        inner = ", ".join(
            f'"{name}": <{_field_instruction(field)}>'
            for name, field in spec.output.fields.items()
        )
        form = f'{{"id": 0, "output": {{{inner}}}}}'
    return (
        "Apply the decision instructions to every item. Use only the supplied facts.\n"
        f"Decision instructions:\n{spec.prompt.strip()}\n\n"
        f"Items:\n{numbered}\n\n"
        f"Reply with only a JSON array in this form: [{form}, ...].\n"
        "Every id must appear exactly once."
    )


def teacher_variant_prompt(
    spec: FunctionSpec, examples: list[dict[str, Any]], band: str, count: int
) -> str:
    rendered = json.dumps(examples, indent=1, ensure_ascii=False)
    fields = ", ".join(f'"{name}"' for name in spec.input_schema)
    return (
        "Generate realistic new inputs in the same distribution as the examples.\n"
        f"Decision instructions:\n{spec.prompt.strip()}\n\n"
        f"Examples:\n{rendered}\n\n"
        f"Write {count} new items likely to receive decision {band}. Reply only with "
        f"a JSON array of objects containing {fields}."
    )


def teacher_counterfactual_prompt(
    spec: FunctionSpec,
    sources: list[dict[str, Any]],
    band: str,
    feedback: str | None = None,
) -> str:
    numbered = json.dumps(
        [{"id": index, **item} for index, item in enumerate(sources)],
        indent=1,
        ensure_ascii=False,
    )
    fields = ", ".join(f'"{name}"' for name in spec.input_schema)
    retry = f"\nPrevious attempt feedback: {feedback}\n" if feedback else ""
    return (
        "Make the smallest realistic edit to each item that would change its decision "
        f"toward {band}. Preserve irrelevant content.\n"
        f"Decision instructions:\n{spec.prompt.strip()}\n{retry}\n"
        f"Items:\n{numbered}\n\nReply only with a JSON array containing the original "
        f'"id" and fields {fields}.'
    )


_SCORE_RE = re.compile(r"score:\s*(-?\d+)", re.IGNORECASE)
_SCORE_VALUE_RE = re.compile(r"^\s*score\s*:\s*(.+)$", re.IGNORECASE | re.MULTILINE)
_INT_RE = re.compile(r"-?\d+")


def _parse_field(field, text: str) -> Any | None:
    if field.type == "int":
        match = _INT_RE.search(text)
        if not match:
            return None
        value = int(match.group())
        return value if value in field.values() else None
    lowered = text.casefold()
    hits = [
        (lowered.find(label.casefold()), -len(label), label)
        for label in sorted(field.labels, key=len, reverse=True)
        if label.casefold() in lowered
    ]
    return min(hits)[2] if hits else None


def _strict_json_value(text: str) -> Any:
    """Parse `text` as exactly one JSON value, tolerating only outer whitespace.

    Rejects code fences, preambles, trailing commentary, partial JSON, and
    duplicate object keys. Insertion order of object keys is preserved so the
    caller can check it against the declared field order.
    """

    def no_duplicates(pairs):
        keys = [key for key, _ in pairs]
        if len(keys) != len(set(keys)):
            raise ValueError(f"duplicate keys {sorted(set(k for k in keys if keys.count(k) > 1))}")
        return dict(pairs)

    stripped = text.strip()
    if not stripped:
        raise ValueError("empty completion")
    value, end = json.JSONDecoder(object_pairs_hook=no_duplicates).raw_decode(stripped)
    if stripped[end:].strip():
        raise ValueError(f"trailing content after the JSON value: {stripped[end:][:80]!r}")
    return value


def parse_generated(
    spec: FunctionSpec, text: str, exhausted: bool = False
) -> tuple[Any | None, str | None]:
    """Strictly parse and validate one generated completion of a text-bearing
    function. Returns (validated output, None) or (None, failure category).

    The output is atomic: any invalid field invalidates the whole result.
    `exhausted` marks a generation that spent its whole token budget without
    ending; an invalid completion is then attributed to the budget, not to
    the malformed tail it produced.
    """
    try:
        value = _strict_json_value(text)
    except ValueError:
        return None, "budget_exhausted" if exhausted else "malformed_json"

    fields = spec.output.fields
    if spec.output.is_scalar:
        candidate = {SCALAR_FIELD: value}
    else:
        if not isinstance(value, dict):
            return None, "contract"
        if list(value) != list(fields):
            # wrong, missing, extra, or misordered keys: order is generation
            # order, so a reordered object is a different function
            return None, "contract"
        candidate = value

    for name, field in fields.items():
        if field.type != "text":
            continue
        raw = candidate.get(name)
        if not isinstance(raw, str):
            return None, "contract"
        if not raw.strip():
            return None, "empty_text"
        if len(raw.strip()) > field.max_chars:
            return None, "char_limit"
    try:
        validated = validate_output(spec, value)
    except ValueError:
        return None, "contract"
    return validated, None


def parse_output(spec: FunctionSpec, text: str) -> Any | None:
    if spec.output.has_text:
        output, _ = parse_generated(spec, text)
        return output
    if spec.output.is_scalar:
        if spec.output.scalar.type == "int":
            match = _SCORE_RE.search(text)
            if match and int(match.group(1)) in spec.output.scalar.values():
                return int(match.group(1))
        else:
            match = _SCORE_VALUE_RE.search(text)
            if match:
                return _parse_field(spec.output.scalar, match.group(1))
        return _parse_field(spec.output.scalar, text)
    output: dict[str, Any] = {}
    for name, field in spec.output.fields.items():
        match = re.search(
            rf"^\s*{re.escape(name)}\s*:\s*(.+)$",
            text,
            re.MULTILINE | re.IGNORECASE,
        )
        output[name] = _parse_field(field, match.group(1)) if match else None
    return None if all(value is None for value in output.values()) else output


def incomplete_fields(spec: FunctionSpec, output: Any | None) -> list[str]:
    if output is None:
        return list(spec.output.fields)
    if spec.output.is_scalar:
        return [] if output in spec.output.scalar.values() else [SCALAR_FIELD]
    if not isinstance(output, dict):
        return list(spec.output.fields)
    return [
        name
        for name, field in spec.output.fields.items()
        if output.get(name) not in field.values()
    ]


def completion_budget(spec: FunctionSpec) -> int:
    """Safe max_new_tokens for one completion.

    For text-bearing outputs the budget is a safety mechanism, not the
    contract: `max_chars` (Unicode code points, enforced after parsing) is
    the contract. The tokenizer-aware estimate is 1.25 tokens per allowed
    character — generous for BPE English (~0.3) and adequate for CJK (~1) —
    plus fixed JSON overhead. Deterministic EOS ends generation long before
    the budget in the normal case; spending the whole budget without a valid
    output is recorded as a `budget_exhausted` structural failure.
    """
    if spec.output.has_text:
        text_chars = sum(
            field.max_chars
            for field in spec.output.fields.values()
            if field.type == "text"
        )
        return 24 + 8 * len(spec.output.bounded_fields) + math.ceil(1.25 * text_chars)
    return 8 if spec.output.is_scalar else 8 + 8 * len(spec.output.fields)


def extract_json(text: str) -> Any:
    text = text.strip()
    if text.startswith("```"):
        text = re.sub(r"^```[a-zA-Z]*\n|\n```$", "", text.strip("`").strip())
    for opener, closer in (("[", "]"), ("{", "}")):
        start = text.find(opener)
        end = text.rfind(closer)
        if start >= 0 and end > start:
            return json.loads(text[start : end + 1])
    raise ValueError(f"no JSON found in teacher output: {text[:200]!r}")
