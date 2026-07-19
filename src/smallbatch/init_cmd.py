"""Prompt-first project templates."""

from __future__ import annotations

import json
from pathlib import Path

_TEACHER = """\
teacher:
  backend: openai-compatible
  model: qwen3:8b
  base_url: http://localhost:11434/v1
"""

_CANDIDATES = """\
candidates:
  tfidf:
    type: tfidf
  bge-small:
    type: setfit
    model: BAAI/bge-small-en-v1.5
  granite-350m:
    type: lora
    model: ibm-granite/granite-4.0-350m
    precision: auto
"""

# a text field must be generated, not selected, so text templates configure
# only LoRA candidates — the spec would reject TF-IDF or SetFit
_LORA_ONLY_CANDIDATES = """\
candidates:
  granite-350m:
    type: lora
    model: ibm-granite/granite-4.0-350m
    precision: auto
"""


TEMPLATES = {
    "classifier": {
        "description": "TODO: what repeated decision this function makes.",
        "input_schema": {"title": "string", "body": "string"},
        "output": "output:\n  type: enum\n  labels: [urgent, normal, low]\n",
        "prompt": (
            "Choose urgent, normal, or low.\n"
            "urgent: TODO concrete criteria and tie-breakers\n"
            "normal: TODO\nlow: TODO\n"
        ),
    },
    "scorer": {
        "description": "TODO: what repeated decision this function makes.",
        "input_schema": {"title": "string", "summary": "string"},
        # integer scales run 0-9: one token per level, so the decision is one
        # ordered choice a student can be trained and scored on
        "output": "output:\n  type: int\n  range: [0, 9]\n",
        "prompt": (
            "Assign an integer from 0 to 9. Return the highest level whose\n"
            "complete definition is supported by the input.\n"
            "0: TODO\n1: TODO\n2: TODO\n3: TODO\n4: TODO\n"
            "5: TODO\n6: TODO\n7: TODO\n8: TODO\n9: TODO\n"
        ),
    },
    "structured": {
        "description": "TODO: what repeated decision this function makes.",
        "input_schema": {"title": "string", "body": "string"},
        "output": (
            "output:\n"
            "  priority:\n    labels: [urgent, normal, low]\n"
            "  reason:\n    labels: [outage, billing, question]\n"
            "  confidence:\n    range: [1, 5]\n"
        ),
        "prompt": "Decide every output field using these criteria:\nTODO\n",
    },
    "rewriter": {
        "description": "TODO: what stable text transformation this function makes.",
        "input_schema": {"query": "string"},
        # one length-bounded text output; max_chars counts Unicode code
        # points in the decoded text and over-limit outputs are invalid,
        # never truncated
        "output": "output:\n  type: text\n  max_chars: 300\n",
        "prompt": (
            "Rewrite the query as TODO: define the one narrow transformation\n"
            "(e.g. a search rewrite, a normalized message, a short title).\n"
            "Keep it under 300 characters. State what must be preserved and\n"
            "what must be dropped.\n"
        ),
        "candidates": _LORA_ONLY_CANDIDATES,
    },
}


def _spec(name: str, template: dict) -> str:
    fields = "\n".join(
        f"  {field}: {kind}" for field, kind in template["input_schema"].items()
    )
    prompt = "\n".join(f"  {line}" for line in template["prompt"].splitlines())
    # the description contains a colon, which YAML forbids in a plain scalar
    return (
        f"name: {name}\n"
        f"description: {json.dumps(template['description'])}\n"
        f"input_schema:\n{fields}\n"
        f"{template['output']}"
        f"prompt: |\n{prompt}\n"
        f"{_TEACHER}"
        f"{template.get('candidates', _CANDIDATES)}"
    )


def _items(schema: dict[str, str], count: int = 20) -> str:
    def value(kind: str, index: int):
        return {
            "string": f"TODO representative value {index + 1}",
            "integer": index,
            "number": float(index),
            "boolean": bool(index % 2),
        }[kind]

    return json.dumps(
        [
            {"input": {field: value(kind, index) for field, kind in schema.items()}}
            for index in range(count)
        ],
        indent=2,
    )


def init(template: str, name: str, directory: str | None = None) -> Path:
    from .spec import validate_id

    if template not in TEMPLATES:
        raise ValueError(f"template must be one of {sorted(TEMPLATES)}")
    validate_id(name, "function name")
    out = Path(directory or name)
    out.mkdir(parents=True, exist_ok=False)
    (out / "spec.yaml").write_text(_spec(name, TEMPLATES[template]))
    (out / "items.json").write_text(_items(TEMPLATES[template]["input_schema"]))
    return out
