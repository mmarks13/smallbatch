"""`smallbatch init <template> [name]`: start a new function from a working
skeleton instead of a blank page. Writes <name>/spec.yaml + items.json with
TODOs where judgment is needed."""

from __future__ import annotations

import json
from pathlib import Path

_TEACHER_BLOCK = """\
# Pick the teacher you are authorized to use (docs/responsible-use.md):
teacher:
  backend: openai-compatible        # any /chat/completions endpoint
  model: qwen3:8b                   # TODO: your labeling model
  base_url: http://localhost:11434/v1
  # backend: claude-cli             # or: logged-in Claude Code CLI
  # model: sonnet
  # backend: codex-cli              # or: logged-in OpenAI Codex CLI
  # model: gpt-5.6-terra
  examples: 150                     # target dataset size (variants fill the gap)
  holdout: 0.15                     # gate split (fraction of reals, or an int count)
  dev: 0.1                          # checkpoint-selection split
  consistency: 30                   # self-consistency probe: re-label this many
                                    # rows to measure the teacher's own ceiling
"""

_TAIL = """\
gate:
  agreement: 0.85                   # acceptance bar (int: within +/-1; enum: exact)
  must_beat_zeroshot: true
train:
  base: ibm-granite/granite-4.0-350m  # adapters inherit the base model's license (Apache-2.0)
  precision: auto
"""


def _spec(name: str, description: str, input_schema: dict, output: str, rubric: str) -> str:
    fields = "\n".join(f"  {k}: {v}" for k, v in input_schema.items())
    return (
        f"name: {name}\n"
        f"description: >\n  {description}\n"
        f"input_schema:\n{fields}\n"
        f"{output}"
        f"rubric: |\n{rubric}"
        f"{_TEACHER_BLOCK}{_TAIL}"
    )


def _items(input_schema: dict, n: int = 3) -> str:
    return json.dumps(
        [{k: f"TODO example {i + 1}" for k in input_schema} for i in range(n)],
        indent=1,
    )


TEMPLATES = {
    "classifier": dict(
        description="TODO: one sentence on what this function decides.",
        input_schema={"title": "str", "body": "str"},
        output=(
            "output:\n"
            "  type: enum\n"
            "  labels: [urgent, normal, low]   # TODO: your classes\n"
        ),
        rubric=(
            "  TODO: when does each label apply? Write it like instructions to a\n"
            "  careful new hire — concrete criteria and tie-breakers, not vibes.\n"
            "  urgent = ...\n  normal = ...\n  low = ...\n"
        ),
    ),
    "scorer": dict(
        description="TODO: one sentence on what this function scores.",
        input_schema={"title": "str", "summary": "str"},
        output="output:\n  type: int\n  range: [0, 10]\n",
        rubric=(
            "  Score 0-10.\n"
            "  9-10 — TODO: what makes a top score\n"
            "  7-8  — TODO\n  5-6  — TODO\n  3-4  — TODO\n"
            "  0-2  — TODO: what makes a bottom score\n"
        ),
    ),
    "structured": dict(
        description="TODO: one sentence on what this function extracts/decides.",
        input_schema={"title": "str", "body": "str"},
        output=(
            "# Each field is independently constrained; every field gates.\n"
            "output:\n"
            "  priority:\n"
            "    labels: [urgent, normal, low]        # TODO: your classes\n"
            "  reason:\n"
            "    labels: [outage, billing, question]  # TODO: controlled reason codes\n"
            "  confidence:\n"
            "    range: [1, 5]\n"
        ),
        rubric=(
            "  TODO: criteria for every field, including how the fields relate\n"
            "  (e.g. which reason codes justify which priorities).\n"
        ),
    ),
}


def init(template: str, name: str, directory: str | None = None) -> Path:
    if template not in TEMPLATES:
        raise ValueError(f"template must be one of {sorted(TEMPLATES)}")
    t = TEMPLATES[template]
    out = Path(directory or name)
    out.mkdir(parents=True, exist_ok=False)
    (out / "spec.yaml").write_text(
        _spec(name, t["description"], t["input_schema"], t["output"], t["rubric"])
    )
    (out / "items.json").write_text(_items(t["input_schema"]))
    return out
