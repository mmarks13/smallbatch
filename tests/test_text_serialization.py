"""Strict-JSON canonical serialization for text-bearing functions (v0.3)."""

from __future__ import annotations

import json

from conftest import make_spec
from smallbatch import prompts


def text_spec(**overrides):
    defaults = {
        "output": {"type": "text", "max_chars": 100},
        "candidates": {"granite": {"type": "lora"}},
    }
    defaults.update(overrides)
    return make_spec(**defaults)


def mixed_spec():
    return make_spec(
        output={
            "priority": {"range": [0, 4]},
            "explanation": {"type": "text", "max_chars": 100},
        },
        candidates={"granite": {"type": "lora"}},
    )


def test_scalar_text_completion_is_one_json_string():
    completion = prompts.student_completion(text_spec(), "California residency rules")
    assert completion == ' "California residency rules"'
    assert json.loads(completion) == "California residency rules"


def test_structured_completion_is_one_json_object_in_declared_order():
    completion = prompts.student_completion(
        mixed_spec(), {"priority": 3, "explanation": "The dates conflict."}
    )
    assert completion == ' {"priority": 3, "explanation": "The dates conflict."}'
    assert list(json.loads(completion)) == ["priority", "explanation"]


def test_bounded_only_functions_keep_the_compact_format():
    assert prompts.student_completion(make_spec(), "urgent") == " urgent"
    structured = make_spec(
        output={"priority": {"range": [0, 4]}, "reason": {"labels": ["outage", "other"]}}
    )
    assert (
        prompts.student_completion(structured, {"priority": 2, "reason": "outage"})
        == " priority: 2\nreason: outage"
    )


def test_declared_field_order_is_generation_order():
    """rationale-before-decision and decision-before-explanation serialize in
    exactly the declared order — there is no separate generation_order."""
    rationale_first = make_spec(
        output={"rationale": {"type": "text"}, "priority": {"range": [0, 4]}},
        candidates={"granite": {"type": "lora"}},
    )
    completion = prompts.student_completion(
        rationale_first, {"rationale": "Dates conflict.", "priority": 3}
    )
    assert completion.index("rationale") < completion.index("priority")


def test_strict_parse_rejects_wrappers_and_shape_violations():
    spec = mixed_spec()
    good = '{"priority": 3, "explanation": "ok"}'
    assert prompts.parse_generated(spec, good) == (
        {"priority": 3, "explanation": "ok"},
        None,
    )
    # outer whitespace is the only tolerated decoration
    assert prompts.parse_generated(spec, f"\n  {good}  \n")[1] is None

    rejected = {
        "```json\n" + good + "\n```": "malformed_json",
        "Here you go: " + good: "malformed_json",
        good + " hope that helps!": "malformed_json",
        '{"priority": 3, "explanation": ': "malformed_json",  # partial JSON
        '{"explanation": "ok", "priority": 3}': "contract",  # wrong key order
        '{"priority": 3}': "contract",  # missing key
        '{"priority": 3, "explanation": "ok", "extra": 1}': "contract",
        '{"priority": 3, "priority": 4, "explanation": "ok"}': "malformed_json",
        '{"priority": 9, "explanation": "ok"}': "contract",  # out of range
        '{"priority": 3, "explanation": ""}': "empty_text",
        json.dumps({"priority": 3, "explanation": "x" * 101}): "char_limit",
        "[1, 2]": "contract",
    }
    for text, category in rejected.items():
        output, got = prompts.parse_generated(spec, text)
        assert output is None, text
        assert got == category, text


def test_budget_exhaustion_owns_the_malformed_tail():
    spec = text_spec()
    output, category = prompts.parse_generated(spec, '"cut off mid-str', exhausted=True)
    assert output is None and category == "budget_exhausted"
    # a completion that ended on its own is plain malformed JSON
    assert prompts.parse_generated(spec, '"cut off mid-str')[1] == "malformed_json"


def test_json_escaping_and_multiline_text_round_trip():
    spec = text_spec()
    value = 'He said "wait"\n\tthen left\\now'
    completion = prompts.student_completion(spec, value)
    parsed, category = prompts.parse_generated(spec, completion.strip())
    assert category is None
    assert parsed == value


def test_teacher_prompt_carries_the_character_limit():
    prompt = prompts.teacher_label_prompt(mixed_spec(), [{"title": "t", "body": "b"}])
    assert "at most 100 characters" in prompt
    # and the reason channel is gone: the output is the whole answer
    assert '"reason"' not in prompt


def test_zeroshot_prompt_names_the_json_form():
    instruction = prompts.output_instruction(mixed_spec())
    assert instruction.startswith("a JSON object with exactly these keys in this order")
    assert prompts.output_instruction(text_spec()).startswith("a JSON string")


def test_segments_concatenate_to_the_exact_completion():
    """Training supervises the same bytes inference must produce: the
    attribution segments must reassemble every completion exactly."""
    from smallbatch import objective

    cases = [
        (text_spec(), 'He said "hi"\nbye'),
        (mixed_spec(), {"priority": 0, "explanation": "why: because\nnew line"}),
        (make_spec(), "urgent"),
        (
            make_spec(output={"priority": {"range": [0, 4]}, "kind": {"labels": ["a", "b"]}}),
            {"priority": 4, "kind": "b"},
        ),
    ]
    for spec, output in cases:
        segments = objective.completion_segments(spec, output)
        assert "".join(text for _, text in segments) == prompts.student_completion(
            spec, output
        )
