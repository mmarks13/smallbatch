import textwrap

import pytest

from smallbatch.spec import load_spec

MINIMAL = """
name: toy
description: Score a thing 0-10.
input_schema: {title: str, summary: str}
output: {type: int, range: [0, 10]}
rubric: "10 = great, 0 = junk"
teacher: {backend: claude-cli, model: sonnet}
"""


def write_spec(tmp_path, body=MINIMAL):
    p = tmp_path / "toy.yaml"
    p.write_text(textwrap.dedent(body))
    return p


def test_defaults(tmp_path):
    spec = load_spec(write_spec(tmp_path))
    assert spec.teacher.backend == "claude-cli"
    assert spec.gate.agreement_pm1 == 0.85
    assert spec.train.alpha == 2 * spec.train.lora_r
    assert spec.output.range == (0, 10)


def test_codex_cli_teacher_parses(tmp_path):
    body = MINIMAL.replace(
        "teacher: {backend: claude-cli, model: sonnet}",
        "teacher: {backend: codex-cli, model: gpt-5.6-terra}",
    )
    spec = load_spec(write_spec(tmp_path, body))
    assert spec.teacher.backend == "codex-cli"
    assert spec.teacher.model == "gpt-5.6-terra"


def test_teacher_block_required(tmp_path):
    body = MINIMAL.replace("teacher: {backend: claude-cli, model: sonnet}\n", "")
    with pytest.raises(ValueError, match="responsible-use"):
        load_spec(write_spec(tmp_path, body))


def test_teacher_model_required(tmp_path):
    body = MINIMAL.replace(
        "teacher: {backend: claude-cli, model: sonnet}",
        "teacher: {backend: claude-cli}",
    )
    with pytest.raises(ValueError, match="backend: openai-compatible"):
        load_spec(write_spec(tmp_path, body))


def test_int_requires_range(tmp_path):
    bad = MINIMAL.replace("{type: int, range: [0, 10]}", "{type: int}")
    with pytest.raises(ValueError):
        load_spec(write_spec(tmp_path, bad))


def test_hash_tracks_spec_files(tmp_path):
    ref = tmp_path / "prefs.yaml"
    ref.write_text("likes: cats")
    body = MINIMAL + "spec_files: [prefs.yaml]\n"
    spec = load_spec(write_spec(tmp_path, body))
    h1 = spec.spec_hash()
    ref.write_text("likes: dogs")
    assert load_spec(tmp_path / "toy.yaml").spec_hash() != h1


def test_extra_keys_rejected(tmp_path):
    with pytest.raises(ValueError):
        load_spec(write_spec(tmp_path, MINIMAL + "surprise: true\n"))


def test_early_stopping_defaults(tmp_path):
    spec = load_spec(write_spec(tmp_path))
    assert spec.train.max_epochs == 12
    assert spec.train.patience == 2
    assert spec.teacher.dev == 0.1


def test_epochs_is_alias_for_max_epochs(tmp_path):
    spec = load_spec(write_spec(tmp_path, MINIMAL + "train: {epochs: 4}\n"))
    assert spec.train.max_epochs == 4


def test_split_counts_accept_ints(tmp_path):
    body = MINIMAL.replace(
        "teacher: {backend: claude-cli, model: sonnet}",
        "teacher: {backend: claude-cli, model: sonnet, holdout: 60, dev: 40}",
    )
    spec = load_spec(write_spec(tmp_path, body))
    assert spec.teacher.holdout == 60 and spec.teacher.dev == 40


def test_split_fraction_validated(tmp_path):
    body = MINIMAL.replace(
        "teacher: {backend: claude-cli, model: sonnet}",
        "teacher: {backend: claude-cli, model: sonnet, dev: 1.5}",
    )
    with pytest.raises(ValueError):
        load_spec(write_spec(tmp_path, body))


def test_gate_agreement_alias(tmp_path):
    spec = load_spec(write_spec(tmp_path))
    assert spec.gate.threshold == 0.85  # legacy agreement_pm1 default
    spec = load_spec(write_spec(tmp_path, MINIMAL + "gate: {agreement: 0.7}\n"))
    assert spec.gate.threshold == 0.7  # preferred name wins


def test_augment_block_parses_and_hashes(tmp_path):
    body = MINIMAL + textwrap.dedent("""\
        augment:
          paraphrase: {cap: 40}
          field_dropout: {fields: [summary], cap: 10}
          counterfactual: {cap: 20}
        """)
    spec = load_spec(write_spec(tmp_path, body))
    assert spec.augment.paraphrase.cap == 40
    assert spec.augment.field_dropout.fields == ["summary"]
    assert spec.augment.counterfactual.cap == 20
    plain = load_spec(write_spec(tmp_path))
    assert spec.spec_hash() != plain.spec_hash()  # augment is part of the recipe


def test_augment_dropout_unknown_field_rejected(tmp_path):
    body = MINIMAL + "augment: {field_dropout: {fields: [upvotes]}}\n"
    with pytest.raises(ValueError, match="unknown input fields"):
        load_spec(write_spec(tmp_path, body))


def test_augment_typo_rejected(tmp_path):
    body = MINIMAL + "augment: {paraphrse: {cap: 40}}\n"
    with pytest.raises(ValueError):
        load_spec(write_spec(tmp_path, body))


def test_teacher_consistency_validated(tmp_path):
    body = MINIMAL.replace(
        "teacher: {backend: claude-cli, model: sonnet}",
        "teacher: {backend: claude-cli, model: sonnet, consistency: -1}",
    )
    with pytest.raises(ValueError):
        load_spec(write_spec(tmp_path, body))
