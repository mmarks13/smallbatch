"""Canonical input serialization, teacher prompts, and constrained parsing."""

from __future__ import annotations

import itertools
import json
import re
from typing import Any

from .spec import SCALAR_FIELD, FunctionSpec

PROMPT_VERSION = 2
_MAX_COMPLETIONS = 5000


def render_input(item: dict[str, Any], input_schema: dict[str, str]) -> str:
    """Stable text seen by every candidate, in declared field order."""
    return "\n".join(
        f"{name}: {json.dumps(item[name], ensure_ascii=False)}" for name in input_schema
    )


def student_prompt(spec: FunctionSpec, item: dict[str, Any]) -> str:
    return f"[{spec.name}]\n{render_input(item, spec.input_schema)}\noutput:"


def student_completion(
    spec: FunctionSpec, output: Any, reason: str = "", rationale: bool = False
) -> str:
    if spec.output.is_scalar:
        if rationale and reason:
            return f" reason: {reason}\nscore: {output}"
        return f" {output}"
    lines = "\n".join(f"{name}: {output[name]}" for name in spec.output.fields)
    if rationale and reason:
        return f" rationale: {reason}\n{lines}"
    return f" {lines}"


def allowed_completions(spec: FunctionSpec, rationale: bool = False) -> list[str] | None:
    if rationale:
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
    return "exactly one of: " + ", ".join(field.labels)


def output_instruction(spec: FunctionSpec) -> str:
    if spec.output.is_scalar:
        return _field_instruction(spec.output.scalar)
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
        form = (
            '{"id": 0, "output": <'
            + output_instruction(spec)
            + '>, "reason": "<one short sentence>"}'
        )
    else:
        inner = ", ".join(
            f'"{name}": <{_field_instruction(field)}>'
            for name, field in spec.output.fields.items()
        )
        form = f'{{"id": 0, "output": {{{inner}}}, "reason": "<one short sentence>"}}'
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


def parse_output(spec: FunctionSpec, text: str) -> Any | None:
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


def completion_budget(spec: FunctionSpec, rationale: bool = False) -> int:
    base = 8 if spec.output.is_scalar else 8 + 8 * len(spec.output.fields)
    return base + (72 if rationale else 0)


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
