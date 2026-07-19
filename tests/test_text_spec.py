"""Schema contract for length-bounded text output fields (v0.3)."""

from __future__ import annotations

import pytest

from conftest import make_spec
from smallbatch.spec import (
    TEXT_MAX_CHARS_DEFAULT,
    effective_loss_weights,
    validate_output,
)


def text_spec(**overrides):
    defaults = {
        "output": {"type": "text", "max_chars": 100},
        "candidates": {"granite": {"type": "lora"}},
    }
    defaults.update(overrides)
    return make_spec(**defaults)


def mixed_spec(**overrides):
    defaults = {
        "output": {
            "priority": {"range": [0, 4]},
            "explanation": {"type": "text", "max_chars": 100},
        },
        "candidates": {"granite": {"type": "lora"}},
    }
    defaults.update(overrides)
    return make_spec(**defaults)


def test_scalar_text_shorthand_and_named_field_are_accepted():
    scalar = text_spec()
    assert scalar.output.is_text_only and scalar.output.scalar.type == "text"
    mixed = mixed_spec()
    assert mixed.output.text_field == "explanation"
    assert list(mixed.output.bounded_fields) == ["priority"]


def test_missing_max_chars_resolves_to_the_default():
    assert (
        text_spec(output={"type": "text"}).output.scalar.max_chars
        == TEXT_MAX_CHARS_DEFAULT
        == 300
    )
    named = make_spec(
        output={"rationale": {"type": "text"}, "priority": {"range": [0, 4]}},
        candidates={"granite": {"type": "lora"}},
    )
    assert named.output.fields["rationale"].max_chars == 300


def test_max_chars_bounds_are_enforced():
    with pytest.raises(ValueError, match="1-2000"):
        text_spec(output={"type": "text", "max_chars": 0})
    with pytest.raises(ValueError, match="1-2000"):
        text_spec(output={"type": "text", "max_chars": 2001})
    assert text_spec(output={"type": "text", "max_chars": 2000}).output.scalar.max_chars == 2000


def test_more_than_one_text_field_is_rejected():
    with pytest.raises(ValueError, match="at most one"):
        make_spec(
            output={"summary": {"type": "text"}, "rewrite": {"type": "text"}},
            candidates={"granite": {"type": "lora"}},
        )


def test_text_values_validate_strictly():
    spec = text_spec()
    # empty and whitespace-only are invalid; there is no allow_empty override
    for bad in ("", "   ", "\n\t "):
        with pytest.raises(ValueError, match="non-whitespace"):
            validate_output(spec, bad)
    # leading/trailing whitespace strips; internal newlines survive
    assert validate_output(spec, "  First line\nSecond line  ") == (
        "First line\nSecond line"
    )
    # a newline counts as one character against the limit
    assert len(validate_output(text_spec(output={"type": "text", "max_chars": 21}),
                               "First line\nSecondline")) == 21
    # NUL and other unsafe control characters are rejected
    for bad in ("a\x00b", "a\x07b", "a\rb", "a\x7fb"):
        with pytest.raises(ValueError, match="control character"):
            validate_output(spec, bad)
    # over-limit text invalidates the whole output; nothing is truncated
    with pytest.raises(ValueError, match="never truncates"):
        validate_output(spec, "x" * 101)
    # unicode code points are what count
    assert validate_output(text_spec(output={"type": "text", "max_chars": 3}), "日本語") == "日本語"


def test_mixed_output_is_atomic_in_validation():
    spec = mixed_spec()
    with pytest.raises(ValueError, match="never truncates"):
        validate_output(spec, {"priority": 3, "explanation": "x" * 101})
    good = validate_output(spec, {"priority": 3, "explanation": " ok "})
    assert good == {"priority": 3, "explanation": "ok"}


def test_classifier_candidates_are_rejected_with_text():
    for candidate in ({"type": "tfidf"}, {"type": "setfit", "model": "m"}):
        with pytest.raises(ValueError, match="requires generation"):
            text_spec(candidates={"c": candidate, "granite": {"type": "lora"}})


def test_augmentation_is_rejected_with_text():
    with pytest.raises(ValueError, match="augmentation is not supported"):
        text_spec(augmentation={"paraphrase": {"cap": 5}})


def test_passes_two_is_rejected_for_text_only_functions():
    teacher = {"backend": "openai-compatible", "model": "m", "passes": 2}
    with pytest.raises(ValueError, match="text-only"):
        text_spec(teacher=teacher)
    # beside bounded fields it is fine: flips compare the bounded projection
    assert mixed_spec(teacher=teacher).teacher.passes == 2


@pytest.mark.parametrize(
    "option, correction",
    [
        ({"decode": "median"}, "argmax"),
        ({"objective": "ordinal"}, "declared type"),
        ({"rationale_distillation": True}, "text output field"),
        ({"loss_type": "nll"}, "per-field objective"),
    ],
)
def test_removed_candidate_options_fail_with_the_correction(option, correction):
    with pytest.raises(ValueError, match=correction):
        make_spec(candidates={"granite": {"type": "lora", **option}})


def test_removed_decode_fails_on_every_candidate_family():
    for candidate in ({"type": "tfidf"}, {"type": "setfit", "model": "m"}):
        with pytest.raises(ValueError, match="argmax"):
            make_spec(candidates={"c": {**candidate, "decode": "median"}})


def test_loss_weights_validate_names_values_and_single_field():
    with pytest.raises(ValueError, match="unknown output fields"):
        mixed_spec(training={"loss_weights": {"missing": 1.0}})
    for bad in (0, -1, float("nan"), float("inf"), True, "2"):
        with pytest.raises(ValueError, match="finite number greater than zero"):
            mixed_spec(training={"loss_weights": {"priority": bad}})
    with pytest.raises(ValueError, match="no effect"):
        text_spec(training={"loss_weights": {"score": 1.0}})


def test_effective_weights_defaults_and_partial_overrides():
    # text beside bounded fields defaults low; alone it is the whole objective
    assert effective_loss_weights(mixed_spec()) == {
        "priority": 1.0,
        "explanation": 0.25,
    }
    assert effective_loss_weights(text_spec()) == {"score": 1.0}
    partial = mixed_spec(training={"loss_weights": {"priority": 2.0}})
    assert effective_loss_weights(partial) == {"priority": 2.0, "explanation": 0.25}


def test_removed_reason_channel_is_rejected_in_item_envelopes():
    from smallbatch.labeling import normalize_item_records

    spec = make_spec()
    with pytest.raises(ValueError, match="unexpected envelope keys"):
        normalize_item_records(
            spec,
            [
                {
                    "input": {"title": "t", "body": "b"},
                    "output": "urgent",
                    "reason": "because",
                }
            ],
        )
