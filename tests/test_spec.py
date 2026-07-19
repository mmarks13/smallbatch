import pytest

from conftest import make_spec
from smallbatch.spec import load_spec, validate_input, validate_output


def test_prompt_first_spec_and_identities(tmp_path):
    spec = make_spec()
    assert spec.name == "ticket-priority"
    assert spec.decision_hash() != spec.build_hash()
    changed = make_spec(candidates={"other": {"type": "tfidf"}})
    assert changed.decision_hash() == spec.decision_hash()
    assert changed.build_hash() != spec.build_hash()


def test_legacy_spec_rejected_with_migration_help(tmp_path):
    path = tmp_path / "spec.yaml"
    path.write_text("name: old\nrubric: old instructions\ngate: {}\n")
    with pytest.raises(ValueError, match="legacy pre-v0.2"):
        load_spec(path)


def test_ids_are_lowercase_kebab():
    with pytest.raises(ValueError, match="lowercase kebab-case"):
        make_spec(name="Ticket_Priority")
    with pytest.raises(ValueError, match="candidate id"):
        make_spec(candidates={"BGE_small": {"type": "tfidf"}})


def test_strict_input_contract():
    spec = make_spec()
    assert validate_input(spec, {"title": "x", "body": "y"}) == {"title": "x", "body": "y"}
    with pytest.raises(ValueError, match="missing fields"):
        validate_input(spec, {"title": "x"})
    with pytest.raises(ValueError, match="unexpected fields"):
        validate_input(spec, {"title": "x", "body": "y", "extra": "z"})
    with pytest.raises(ValueError, match="must be string"):
        validate_input(spec, {"title": 1, "body": "y"})


def test_strict_output_contract_scalar_and_structured():
    spec = make_spec()
    assert validate_output(spec, "urgent") == "urgent"
    with pytest.raises(ValueError, match="outside the contract"):
        validate_output(spec, "other")
    structured = make_spec(
        output={"priority": {"labels": ["urgent", "normal"]}, "score": {"range": [0, 4]}}
    )
    assert validate_output(structured, {"priority": "urgent", "score": 3}) == {
        "priority": "urgent",
        "score": 3,
    }
    with pytest.raises(ValueError, match="structured output mismatch"):
        validate_output(structured, {"priority": "urgent"})


def test_setfit_managed_training_args_rejected():
    with pytest.raises(ValueError, match="managed by smallbatch"):
        make_spec(
            candidates={
                "bge-small": {
                    "type": "setfit",
                    "model": "BAAI/bge-small-en-v1.5",
                    "training_args": {"output_dir": "/tmp/no"},
                }
            }
        )
    with pytest.raises(ValueError, match="managed by smallbatch"):
        make_spec(
            candidates={
                "bge-small": {
                    "type": "setfit",
                    "model": "BAAI/bge-small-en-v1.5",
                    "training_args": {"save_strategy": "steps"},
                }
            }
        )


def test_head_capacity_knob_parses_and_rejects_unknown_values():
    spec = make_spec(
        output={"type": "int", "range": [0, 4]},
        candidates={"words": {"type": "tfidf", "head": "linear"}},
    )
    assert spec.candidates["words"].head == "linear"
    assert make_spec().candidates["tfidf"].head == "auto"  # dev decides by default
    with pytest.raises(ValueError):
        make_spec(candidates={"words": {"type": "tfidf", "head": "wide"}})


def test_teacher_passes_validates_and_stays_out_of_decision_hash():
    from conftest import make_spec

    base = make_spec(teacher={"backend": "codex-cli", "model": "test"})
    measured = make_spec(teacher={"backend": "codex-cli", "model": "test", "passes": 2})
    # a measurement protocol, not a different decision: datasets stay valid
    assert base.decision_hash() == measured.decision_hash()
    # but builds must revision when the protocol changes
    assert base.build_hash() != measured.build_hash()
    with pytest.raises(ValueError):
        make_spec(teacher={"backend": "codex-cli", "model": "test", "passes": 3})
    with pytest.raises(ValueError):
        make_spec(teacher={"backend": "codex-cli", "model": "test", "passes": 0})
