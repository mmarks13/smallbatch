import textwrap

from smallbatch import prompts
from smallbatch.spec import FunctionSpec

SPEC = FunctionSpec(
    name="toy",
    description="Score a thing.",
    input_schema={"title": "str", "tags": "list[str]"},
    output={"type": "int", "range": [0, 10]},
    rubric="10 great, 0 junk",
    teacher={"backend": "claude-cli", "model": "sonnet"},
)

ENUM_SPEC = FunctionSpec(
    name="kind",
    description="Classify.",
    input_schema={"title": "str"},
    output={"type": "enum", "labels": ["paper", "release", "drama"]},
    rubric="-",
    teacher={"backend": "claude-cli", "model": "sonnet"},
)


def test_render_input_order_and_lists():
    text = prompts.render_input({"tags": ["a", "b"], "title": "T"}, SPEC.input_schema)
    assert text == "title: T\ntags: a, b"


def test_student_prompt_excludes_rubric():
    p = prompts.student_prompt(SPEC, {"title": "T", "tags": []})
    assert "junk" not in p and p.endswith("output:")


def test_parse_int_variants():
    assert prompts.parse_output(SPEC, " 7") == 7
    assert prompts.parse_output(SPEC, "reason: has 3 caveats\nscore: 9") == 9
    assert prompts.parse_output(SPEC, "Score: 10") == 10
    assert prompts.parse_output(SPEC, "42") is None  # out of range
    assert prompts.parse_output(SPEC, "no number") is None


def test_parse_enum():
    assert prompts.parse_output(ENUM_SPEC, " release\n") == "release"
    assert prompts.parse_output(ENUM_SPEC, "This is a Paper about X") == "paper"
    assert prompts.parse_output(ENUM_SPEC, "dunno") is None


def test_extract_json_with_fences():
    fenced = textwrap.dedent(
        """
        Here you go:
        ```json
        [{"id": 0, "score": 5, "reason": "meh"}]
        ```
        """
    )
    assert prompts.extract_json(fenced) == [{"id": 0, "score": 5, "reason": "meh"}]
