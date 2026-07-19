"""The universal per-field objective: attribution, losses, normalization."""

from __future__ import annotations

import math

import pytest
import torch
import torch.nn.functional as F

from conftest import make_spec
from smallbatch import objective, prompts

EOS = 99
VOCAB = 200


class CharTokenizer:
    """One token per character (id = codepoint) with offset mapping, like a
    fast tokenizer. Digits after a space therefore share a one-token prefix
    (the space) and decide on the digit token."""

    eos_token = "<eos>"
    eos_token_id = EOS
    pad_token_id = EOS

    def __call__(self, text, add_special_tokens=False, return_offsets_mapping=False):
        encoded = {"input_ids": [ord(char) for char in text]}
        if return_offsets_mapping:
            encoded["offset_mapping"] = [(i, i + 1) for i in range(len(text))]
        return encoded


class MergingTokenizer(CharTokenizer):
    """Char tokenizer that merges chosen substrings into one token, to model
    BPE merges that cross a segment boundary."""

    def __init__(self, merges: list[str]):
        self.merges = merges

    def __call__(self, text, add_special_tokens=False, return_offsets_mapping=False):
        ids, offsets = [], []
        position = 0
        while position < len(text):
            for merge in self.merges:
                if text.startswith(merge, position):
                    ids.append(1000 + sum(ord(c) for c in merge))
                    offsets.append((position, position + len(merge)))
                    position += len(merge)
                    break
            else:
                ids.append(ord(text[position]))
                offsets.append((position, position + 1))
                position += 1
        encoded = {"input_ids": ids}
        if return_offsets_mapping:
            encoded["offset_mapping"] = offsets
        return encoded


def mixed_spec():
    return make_spec(
        output={
            "priority": {"range": [0, 4]},
            "explanation": {"type": "text", "max_chars": 100},
        },
        candidates={"granite": {"type": "lora"}},
    )


def enum_spec(labels=("aa", "ab")):
    return make_spec(
        output={"kind": {"labels": list(labels)}, "note": {"type": "text"}},
        candidates={"granite": {"type": "lora"}},
    )


def encode(spec, output, prompt="P:", tokenizer=None):
    tokenizer = tokenizer or CharTokenizer()
    codecs = objective.field_codecs(spec, tokenizer)
    return codecs, objective.encode_row(spec, tokenizer, codecs, prompt, output)


def test_token_attribution_majority_and_tie_rules():
    """A token spanning a boundary goes to the segment holding most of its
    characters; a tie goes to the later segment, so a value's merged leading
    space stays with the value."""
    segments = [("", ' {"a":'), ("a", " 3"), ("", "}")]
    completion = ' {"a": 3}'
    # '": ' spans two format chars and one field char: majority -> format,
    # and the digit stays its own field-owned token
    majority = MergingTokenizer(['": '])
    _, owners = objective.attribute_tokens(majority, segments)
    assert len(owners) == len(majority(completion)["input_ids"])
    assert owners[-3] == 0  # '": ' owned by the format segment (index 0)
    assert owners[-2] == 1  # '3' owned by the field segment (index 1)
    # '": 3' spans two format and two field chars: tie -> later (the field)
    tie = MergingTokenizer(['": 3'])
    _, owners = objective.attribute_tokens(tie, segments)
    assert owners[-2] == 1  # owned by the field segment (index 1)


def test_encode_row_masks_prompt_owns_eos_as_format_and_is_complete():
    spec = mixed_spec()
    output = {"priority": 3, "explanation": "ok fine"}
    _, row = encode(spec, output, prompt="P:")
    completion = prompts.student_completion(spec, output)
    assert len(row["input_ids"]) == 2 + len(completion) + 1  # prompt + eos
    assert row["labels"][:2] == [-100, -100]
    assert row["owners"][:2] == [-1, -1]
    assert row["input_ids"][-1] == EOS and row["owners"][-1] == 0
    # every supervised token is owned by exactly one objective: none dropped
    supervised = [owner for owner in row["owners"] if owner != -1]
    assert len(supervised) == len(completion) + 1
    assert set(supervised) <= {0, 1, 2}
    # the int deciding token is the digit; its shared space prefix is format
    digit_positions = [
        index for index, owner in enumerate(row["owners"]) if owner == 1
    ]
    assert [chr(row["input_ids"][p]) for p in digit_positions] == ["3"]


def test_encode_row_rejects_over_length_rows_instead_of_truncating():
    spec = mixed_spec()
    with pytest.raises(ValueError, match="never truncated"):
        codecs = objective.field_codecs(spec, CharTokenizer())
        objective.encode_row(
            spec,
            CharTokenizer(),
            codecs,
            "P:",
            {"priority": 3, "explanation": "long enough text"},
            max_seq_len=10,
        )


def batch_of(rows):
    return objective.collate(rows, pad_token_id=EOS)


class StaticModel:
    """Returns preset full logits; records the ids it was asked to score."""

    device = "cpu"

    def __init__(self, logits):
        self._logits = logits
        self.seen = None

    def __call__(self, input_ids, attention_mask=None, logits_to_keep=None):
        self.seen = input_ids
        logits = self._logits[:, : input_ids.shape[1], :]
        if logits_to_keep is not None:
            logits = logits[:, -logits_to_keep:, :]
        return type("Out", (), {"logits": logits})()


def uniform_logits(batch, length):
    return torch.zeros(batch, length, VOCAB)


def test_text_loss_is_a_mean_so_length_earns_no_extra_influence():
    spec = mixed_spec()
    short = {"priority": 3, "explanation": "abab"}
    long = {"priority": 3, "explanation": "abababab"}
    codecs = objective.field_codecs(spec, CharTokenizer())
    rows = [
        objective.encode_row(spec, CharTokenizer(), codecs, "P:", output)
        for output in (short, long)
    ]
    width = max(len(row["input_ids"]) for row in rows)
    batch = batch_of(rows)
    model = StaticModel(uniform_logits(2, width))
    logits, start = objective.completion_logits(model, batch)
    losses = objective.row_losses(
        logits, batch["input_ids"], batch["owners"], spec, codecs, start
    )
    # uniform logits: every text token costs log(VOCAB); the mean makes the
    # long row's text loss equal, not double
    assert losses["explanation"][0] == pytest.approx(math.log(VOCAB), abs=1e-5)
    assert losses["explanation"][1] == pytest.approx(
        float(losses["explanation"][0]), abs=1e-5
    )


def test_enum_loss_renormalizes_over_legal_labels_and_shared_tokens_cancel():
    spec = enum_spec(labels=("aa", "ab"))
    output = {"kind": "ab", "note": "x"}
    codecs, row = encode(spec, output)
    batch = batch_of([row])
    width = len(row["input_ids"])
    logits = uniform_logits(1, width)
    # find the deciding position: the second character of the label, where
    # 'a' and 'b' diverge; give 'b' twice the logit mass of 'a'
    label_positions = [
        index for index, owner in enumerate(row["owners"]) if owner == 1
    ]
    deciding = label_positions[-1]
    logits[0, deciding - 1, ord("b")] = 1.0
    logits[0, deciding - 1, ord("a")] = 0.0
    # noise on illegal vocabulary must not matter: legal-value NLL renormalizes
    logits[0, deciding - 1, ord("z")] = 9.0

    model = StaticModel(logits)
    got, start = objective.completion_logits(model, batch)
    losses = objective.row_losses(
        got, batch["input_ids"], batch["owners"], spec, codecs, start
    )
    expected = -float(
        F.log_softmax(torch.tensor([0.0, 1.0]), dim=-1)[1]
    )  # P(b | {a, b})
    assert float(losses["kind"][0]) == pytest.approx(expected, abs=1e-5)


def test_format_tokens_never_enter_semantic_losses():
    """Perturbing logits at format positions moves only the format loss."""
    spec = mixed_spec()
    output = {"priority": 3, "explanation": "ok"}
    codecs, row = encode(spec, output)
    batch = batch_of([row])
    width = len(row["input_ids"])

    def losses_with(extra):
        logits = uniform_logits(1, width)
        for position in range(1, width):
            if int(batch["owners"][0][position]) == 0:
                logits[0, position - 1, batch["input_ids"][0][position]] += extra
        model = StaticModel(logits)
        got, start = objective.completion_logits(model, batch)
        return objective.row_losses(
            got, batch["input_ids"], batch["owners"], spec, codecs, start
        )

    plain, boosted = losses_with(0.0), losses_with(5.0)
    assert float(boosted[objective.FORMAT_KEY][0]) < float(plain[objective.FORMAT_KEY][0])
    for name in ("priority", "explanation"):
        assert float(boosted[name][0]) == pytest.approx(float(plain[name][0]), abs=1e-5)


def test_combined_objective_is_a_weighted_mean_plus_fixed_format_term():
    weights = {"priority": 2.0, "explanation": 0.25}
    baselines = {"priority": 2.0, "explanation": 4.0, objective.FORMAT_KEY: 0.5}
    losses = {"priority": 1.0, "explanation": 2.0, objective.FORMAT_KEY: 0.25}
    semantic = (2.0 * (1.0 / 2.0) + 0.25 * (2.0 / 4.0)) / 2.25
    assert objective.checkpoint_score(losses, weights, baselines) == pytest.approx(
        semantic
    )
    assert objective.combine_losses(losses, weights, baselines) == pytest.approx(
        semantic + 0.1 * (0.25 / 0.5)
    )
    # changing semantic weights must not change the format contribution
    other = objective.combine_losses(
        losses, {"priority": 5.0, "explanation": 1.0}, baselines
    ) - objective.checkpoint_score(losses, {"priority": 5.0, "explanation": 1.0}, baselines)
    assert other == pytest.approx(0.1 * (0.25 / 0.5))
    # a single field reduces to that field's normalized loss
    assert objective.checkpoint_score(
        losses, {"priority": 1.0}, baselines
    ) == pytest.approx(0.5)


def test_unsafe_baselines_fail_before_training_never_clamp():
    for value in (0.0, 1e-9, float("nan"), float("inf"), None):
        with pytest.raises(ValueError, match="normalization"):
            objective.check_baselines(
                {"priority": 2.0, objective.FORMAT_KEY: value}
            )
    objective.check_baselines({"priority": 2.0, objective.FORMAT_KEY: 0.5})


def test_dev_losses_are_teacher_forced_on_reference_fields():
    """Development scoring conditions every field on the correct preceding
    reference tokens: the model is fed the reference completion itself."""
    spec = mixed_spec()
    output = {"priority": 3, "explanation": "why"}
    codecs, row = encode(spec, output)
    model = StaticModel(uniform_logits(1, len(row["input_ids"])))
    values = objective.evaluate_field_losses(
        model, spec, codecs, [row], pad_token_id=EOS, batch_size=4
    )
    reference = prompts.student_completion(spec, output)
    fed = "".join(chr(i) for i in model.seen[0].tolist()[2:-1])
    assert fed == reference
    assert set(values) == {"priority", "explanation", objective.FORMAT_KEY}
    assert values["explanation"] == pytest.approx(math.log(VOCAB), abs=1e-4)


def test_text_fidelity_reports_reference_prediction_not_correctness():
    spec = mixed_spec()
    rows = [
        {"input": {"title": "t", "body": "b"}, "output": {"priority": 3, "explanation": "ab"}},
        {"input": {"title": "u", "body": "c"}, "output": {"priority": 1, "explanation": "abab"}},
    ]
    codecs = objective.field_codecs(spec, CharTokenizer())
    encoded = [
        objective.encode_row(
            spec,
            CharTokenizer(),
            codecs,
            prompts.student_prompt(spec, row["input"]),
            row["output"],
        )
        for row in rows
    ]
    width = max(len(row["input_ids"]) for row in encoded)

    class BatchModel(StaticModel):
        def __init__(self):
            super().__init__(uniform_logits(len(rows), width))

    monkey_tokenizer = CharTokenizer()
    fidelity = objective.text_fidelity(
        BatchModel(), monkey_tokenizer, spec, codecs, rows, batch_size=2
    )
    assert fidelity["rows"] == 2
    # uniform logits: token NLL is log(VOCAB) everywhere
    assert fidelity["token_nll"] == pytest.approx(round(math.log(VOCAB), 4), abs=1e-3)
    assert fidelity["perplexity"] == pytest.approx(VOCAB, rel=1e-3)
    # bpb = total nats / ln2 / total reference bytes ("ab" + "abab" = 6 bytes)
    assert fidelity["bits_per_byte"] == pytest.approx(
        round(6 * math.log(VOCAB) / math.log(2) / 6, 4), abs=1e-3
    )
    assert 0.0 <= fidelity["top1_accuracy"] <= 1.0
    assert fidelity["median_example_bpb"] <= fidelity["p90_example_bpb"]
    assert "not correctness" in fidelity["measures"]


def test_train_requires_a_development_split():
    from smallbatch.spec import LoraCandidateSpec
    from smallbatch.training import train

    with pytest.raises(ValueError, match="development split"):
        train(mixed_spec(), LoraCandidateSpec(type="lora"), [], None, dev_rows=[])


def test_int_field_keeps_ordinal_loss_beside_a_text_field():
    """Adding a text field must not demote an integer scale to token-only
    supervision: its loss stays class NLL + RPS at the deciding token."""
    from smallbatch.heads import ordinal_loss

    spec = mixed_spec()
    output = {"priority": 3, "explanation": "ok"}
    codecs, row = encode(spec, output)
    assert codecs["priority"]["kind"] == "int"
    batch = batch_of([row])
    width = len(row["input_ids"])
    logits = uniform_logits(1, width)
    deciding = next(
        index for index, owner in enumerate(row["owners"]) if owner == 1
    )
    legal = codecs["priority"]["legal_ids"]
    logits[0, deciding - 1, legal] = torch.tensor([0.5, 0.0, 1.0, 4.0, 0.0])

    model = StaticModel(logits)
    got, start = objective.completion_logits(model, batch)
    losses = objective.row_losses(
        got, batch["input_ids"], batch["owners"], spec, codecs, start
    )
    expected = ordinal_loss(
        logits[0, deciding - 1, legal].unsqueeze(0), torch.tensor([3])
    )
    assert float(losses["priority"][0]) == pytest.approx(float(expected), abs=1e-5)
    # distance sensitivity survives: mass far from the decision costs more
    far = uniform_logits(1, width)
    far[0, deciding - 1, legal] = torch.tensor([4.0, 0.0, 1.0, 0.5, 0.0])
    got, start = objective.completion_logits(StaticModel(far), batch)
    far_losses = objective.row_losses(
        got, batch["input_ids"], batch["owners"], spec, codecs, start
    )
    assert float(far_losses["priority"][0]) > float(losses["priority"][0])
