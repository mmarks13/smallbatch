"""Prompt rendering and output parsing.

Two prompt families:
- student prompts are minimal (input only): the spec is compiled INTO the
  adapter weights, PAW-style, so inference never re-sends the rubric.
- teacher/zero-shot prompts carry the full spec text, since those models have
  no adapter to lean on.
"""

from __future__ import annotations

import json
import re
from typing import Any

from .spec import SCALAR_FIELD, FunctionSpec

PROMPT_VERSION = 1


def render_input(item: dict[str, Any], input_schema: dict[str, str]) -> str:
    lines = []
    for field in input_schema:
        v = item.get(field, "")
        if isinstance(v, (list, tuple)):
            v = ", ".join(str(x) for x in v)
        lines.append(f"{field}: {v}")
    return "\n".join(lines)


def student_prompt(spec: FunctionSpec, item: dict[str, Any]) -> str:
    return f"[{spec.name}]\n{render_input(item, spec.input_schema)}\noutput:"


def student_completion(spec: FunctionSpec, output: Any, reason: str = "") -> str:
    """The training target. Scalar contracts emit the bare value (legacy
    format — existing datasets/adapters keep working); multi-field contracts
    emit one `name: value` line per field in spec order."""
    if spec.output.is_scalar:
        if spec.train.rationale_distillation and reason:
            return f" reason: {reason}\nscore: {output}"
        return f" {output}"
    lines = "\n".join(f"{name}: {output[name]}" for name in spec.output.fields)
    if spec.train.rationale_distillation and reason:
        return f" rationale: {reason}\n{lines}"
    return f" {lines}"


# beyond this many enumerated completions, skip the decoding-time constraint
# (parse_output still validates, and the exported GBNF stays exact)
_MAX_COMPLETIONS = 5000


def allowed_completions(spec: FunctionSpec) -> list[str] | None:
    """Every completion the student may legally emit (see student_completion),
    for constrained decoding. None in rationale mode (free text can't be
    enumerated) or when the multi-field cross product is too large — those
    paths decode unconstrained and rely on parse_output."""
    if spec.train.rationale_distillation:
        return None
    if spec.output.is_scalar:
        return [f" {v}" for v in spec.output.scalar.values()]
    fields = spec.output.fields
    total = 1
    for f in fields.values():
        total *= len(f.values())
        if total > _MAX_COMPLETIONS:
            return None
    import itertools

    combos = itertools.product(*(f.values() for f in fields.values()))
    names = list(fields)
    return [
        " " + "\n".join(f"{n}: {v}" for n, v in zip(names, combo)) for combo in combos
    ]


def _field_instruction(field) -> str:
    if field.type == "int":
        lo, hi = field.range
        return f"an integer from {lo} to {hi}"
    return "exactly one of: " + ", ".join(field.labels)


def output_instruction(spec: FunctionSpec) -> str:
    if spec.output.is_scalar:
        return _field_instruction(spec.output.scalar)
    lines = "; ".join(
        f"{name}: <{_field_instruction(f)}>" for name, f in spec.output.fields.items()
    )
    return f"one `name: value` line per field, in this order — {lines}"


def zeroshot_prompt(spec: FunctionSpec, item: dict[str, Any], spec_files_text: str) -> str:
    """The un-tuned base model's best shot: full spec in the prompt."""
    ref = f"\nReference files:\n{spec_files_text}\n" if spec_files_text else ""
    return (
        f"Task: {spec.description.strip()}\n"
        f"Scoring rubric:\n{spec.rubric.strip()}\n{ref}"
        f"Input:\n{render_input(item, spec.input_schema)}\n"
        f"Respond with only {output_instruction(spec)}.\n"
        "output:"
    )


def teacher_label_prompt(
    spec: FunctionSpec,
    items: list[dict[str, Any]],
    spec_files_text: str,
    field_order: list[str] | None = None,
) -> str:
    ref = f"\nReference files:\n{spec_files_text}\n" if spec_files_text else ""
    order = field_order if field_order is not None else list(spec.input_schema)
    numbered = json.dumps(
        [{"id": i, **{k: it.get(k) for k in order}} for i, it in enumerate(items)],
        indent=1,
        ensure_ascii=False,
    )
    if spec.output.is_scalar:
        form = f'{{"id": 0, "score": <{output_instruction(spec)}>, "reason": "<one short sentence>"}}'
    else:
        inner = ", ".join(
            f'"{name}": <{_field_instruction(f)}>'
            for name, f in spec.output.fields.items()
        )
        form = f'{{"id": 0, "output": {{{inner}}}, "reason": "<one short sentence>"}}'
    return (
        "You are a careful data labeler. Label every item below.\n"
        f"Task: {spec.description.strip()}\n"
        f"Scoring rubric:\n{spec.rubric.strip()}\n{ref}"
        f"\nItems:\n{numbered}\n\n"
        f"Reply with ONLY a JSON array, one entry per item, in the form:\n"
        f"[{form}, ...]\n"
        "Every id above must appear exactly once. No other text."
    )


def teacher_variant_prompt(
    spec: FunctionSpec,
    examples: list[dict[str, Any]],
    band: str,
    count: int,
    spec_files_text: str,
) -> str:
    ref = f"\nReference files:\n{spec_files_text}\n" if spec_files_text else ""
    ex = json.dumps(
        [{k: it.get(k) for k in spec.input_schema} for it in examples],
        indent=1,
        ensure_ascii=False,
    )
    fields = ", ".join(f'"{k}"' for k in spec.input_schema)
    return (
        "You generate realistic synthetic inputs for a labeling task. "
        "They must be plausible variations in the same style and domain as the "
        "real examples — not copies, not fantasy.\n"
        f"Task the labels are for: {spec.description.strip()}\n"
        f"Scoring rubric:\n{spec.rubric.strip()}\n{ref}"
        f"\nReal examples of the input distribution:\n{ex}\n\n"
        f"Write {count} NEW items that would plausibly score around {band} "
        "under the rubric.\n"
        f"Reply with ONLY a JSON array of {count} objects, each with keys "
        f"{fields}. No other text."
    )


def teacher_counterfactual_prompt(
    spec: FunctionSpec,
    sources: list[dict[str, Any]],
    band: str,
    spec_files_text: str,
    feedback: str | None = None,
) -> str:
    """Ask for a MINIMAL edit of each source item aimed at a target label.
    Minimal edits trace the rubric's decision boundary — they teach the
    student which change moves the label, instead of which surface features
    co-occur with it."""
    ref = f"\nReference files:\n{spec_files_text}\n" if spec_files_text else ""
    numbered = json.dumps(
        [{"id": i, **{k: it.get(k) for k in spec.input_schema}}
         for i, it in enumerate(sources)],
        indent=1,
        ensure_ascii=False,
    )
    fields = ", ".join(f'"{k}"' for k in spec.input_schema)
    fb = f"\nPrevious attempt was judged unchanged: {feedback}\n" if feedback else ""
    return (
        "You write COUNTERFACTUAL versions of labeled items: for each item "
        "below, make the SMALLEST realistic change to its content so that it "
        f"would now score around {band} under the rubric. Keep everything "
        "irrelevant to the label identical — same style, same length, same "
        "formatting.\n"
        f"Task the labels are for: {spec.description.strip()}\n"
        f"Scoring rubric:\n{spec.rubric.strip()}\n{ref}{fb}"
        f"\nItems to edit:\n{numbered}\n\n"
        "Reply with ONLY a JSON array, one object per item, each with the "
        f"original \"id\" plus keys {fields}. No other text."
    )


_SCORE_RE = re.compile(r"score:\s*(-?\d+)", re.IGNORECASE)
_INT_RE = re.compile(r"-?\d+")


def _parse_field(field, text: str) -> Any | None:
    """Parse and validate one field's value out of `text`."""
    if field.type == "int":
        m = _INT_RE.search(text)
        if not m:
            return None
        val = int(m.group(0))
        lo, hi = field.range
        return val if lo <= val <= hi else None
    # enum: first label that appears, longest-first to avoid prefix collisions
    low = text.lower()
    hits = [
        (low.find(lb.lower()), lb)
        for lb in sorted(field.labels, key=len, reverse=True)
        if lb.lower() in low
    ]
    return min(hits)[1] if hits else None


def parse_output(spec: FunctionSpec, text: str) -> Any | None:
    """Parse a student/zero-shot generation into a validated output value:
    a scalar for legacy contracts, a {field: value} dict (missing/invalid
    fields are None) for multi-field contracts — or None if nothing parsed."""
    if spec.output.is_scalar:
        field = spec.output.scalar
        if field.type == "int":
            m = _SCORE_RE.search(text)
            if m:
                lo, hi = field.range
                val = int(m.group(1))
                return val if lo <= val <= hi else None
        return _parse_field(field, text)
    out: dict[str, Any] = {}
    for name, field in spec.output.fields.items():
        m = re.search(
            rf"^\s*{re.escape(name)}\s*:\s*(.+)$", text, re.MULTILINE | re.IGNORECASE
        )
        out[name] = _parse_field(field, m.group(1)) if m else None
    return None if all(v is None for v in out.values()) else out


def incomplete_fields(spec: FunctionSpec, output: Any | None) -> list[str]:
    """The contract fields `output` fails to satisfy — [] means fully valid.

    The ONE completeness check for every runtime boundary (Python runtime,
    HTTP serve): a multi-field output with any None/missing field is a
    contract violation, not a partial success.
    """
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
    """max_new_tokens for a student generation under this contract."""
    base = 8 if spec.output.is_scalar else 8 + 8 * len(spec.output.fields)
    return base + (72 if spec.train.rationale_distillation else 0)


def extract_json(text: str) -> Any:
    """Pull the first JSON array/object out of teacher output (fences etc.)."""
    text = text.strip()
    if text.startswith("```"):
        text = re.sub(r"^```[a-zA-Z]*\n|\n```$", "", text.strip("`").strip())
    for opener, closer in (("[", "]"), ("{", "}")):
        start = text.find(opener)
        if start == -1:
            continue
        end = text.rfind(closer)
        if end > start:
            return json.loads(text[start : end + 1])
    raise ValueError(f"no JSON found in teacher output: {text[:200]!r}")
