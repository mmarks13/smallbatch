from conftest import make_spec
from smallbatch import prompts


def test_prompt_is_visible_only_to_teacher_and_zeroshot():
    spec = make_spec(prompt="PRIVATE DECISION INSTRUCTIONS")
    item = {"title": "x", "body": "y"}
    assert "PRIVATE" in prompts.teacher_label_prompt(spec, [item])
    assert "PRIVATE" in prompts.zeroshot_prompt(spec, item)
    assert "PRIVATE" not in prompts.student_prompt(spec, item)


def test_canonical_input_serialization_preserves_order_and_types():
    spec = make_spec(input_schema={"enabled": "boolean", "count": "integer", "text": "string"})
    text = prompts.render_input(
        {"enabled": True, "count": 2, "text": "hello"}, spec.input_schema
    )
    assert text == 'enabled: true\ncount: 2\ntext: "hello"'


def test_scalar_and_structured_parsing_and_completions():
    integer = make_spec(output={"type": "int", "range": [0, 2]})
    assert prompts.parse_output(integer, "score: 2") == 2
    assert prompts.parse_output(integer, "9") is None
    assert prompts.allowed_completions(integer) == [" 0", " 1", " 2"]
    overlapping = make_spec(output={"type": "enum", "labels": ["no", "normal"]})
    assert prompts.parse_output(overlapping, " normal") == "normal"
    assert prompts.parse_output(overlapping, "reason: no outage\nscore: normal") == "normal"
    structured = make_spec(
        output={"priority": {"labels": ["high", "low"]}, "score": {"range": [0, 2]}}
    )
    assert prompts.parse_output(structured, "priority: high\nscore: 1") == {
        "priority": "high",
        "score": 1,
    }
    assert prompts.incomplete_fields(structured, {"priority": "high", "score": None}) == ["score"]


def test_extract_json_accepts_fences():
    assert prompts.extract_json('```json\n[{"id": 0}]\n```') == [{"id": 0}]
