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
from typing import Any, Optional

from .spec import FunctionSpec

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


def student_completion(spec: FunctionSpec, score: Any, reason: str = "") -> str:
    if spec.train.rationale_distillation and reason:
        return f" reason: {reason}\nscore: {score}"
    return f" {score}"


def allowed_completions(spec: FunctionSpec) -> Optional[list[str]]:
    """Every completion the student may legally emit (see student_completion),
    for constrained decoding. None in rationale mode: the free-text reason
    can't be enumerated, so that path decodes unconstrained and relies on
    parse_output."""
    if spec.train.rationale_distillation:
        return None
    if spec.output.type == "int":
        lo, hi = spec.output.range
        return [f" {v}" for v in range(lo, hi + 1)]
    return [f" {lb}" for lb in spec.output.labels]


def output_instruction(spec: FunctionSpec) -> str:
    if spec.output.type == "int":
        lo, hi = spec.output.range
        return f"an integer from {lo} to {hi}"
    return "exactly one of: " + ", ".join(spec.output.labels)


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
    spec: FunctionSpec, items: list[dict[str, Any]], spec_files_text: str
) -> str:
    ref = f"\nReference files:\n{spec_files_text}\n" if spec_files_text else ""
    numbered = json.dumps(
        [{"id": i, **{k: it.get(k) for k in spec.input_schema}} for i, it in enumerate(items)],
        indent=1,
        ensure_ascii=False,
    )
    return (
        "You are a careful data labeler. Label every item below.\n"
        f"Task: {spec.description.strip()}\n"
        f"Scoring rubric:\n{spec.rubric.strip()}\n{ref}"
        f"\nItems:\n{numbered}\n\n"
        f'Reply with ONLY a JSON array, one entry per item, in the form:\n'
        f'[{{"id": 0, "score": <{output_instruction(spec)}>, "reason": "<one short sentence>"}}, ...]\n'
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


_SCORE_RE = re.compile(r"score:\s*(-?\d+)", re.IGNORECASE)
_INT_RE = re.compile(r"-?\d+")


def parse_output(spec: FunctionSpec, text: str) -> Optional[Any]:
    """Parse a student/zero-shot generation into a validated output value."""
    if spec.output.type == "int":
        m = _SCORE_RE.search(text)
        if m:
            val = int(m.group(1))
        else:
            m = _INT_RE.search(text)
            if not m:
                return None
            val = int(m.group(0))
        lo, hi = spec.output.range
        return val if lo <= val <= hi else None
    # enum: first label that appears, longest-first to avoid prefix collisions
    low = text.lower()
    hits = [
        (low.find(lb.lower()), lb)
        for lb in sorted(spec.output.labels, key=len, reverse=True)
        if lb.lower() in low
    ]
    return min(hits)[1] if hits else None


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
