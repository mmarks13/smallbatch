from __future__ import annotations

import pytest
import torch
import torch.nn.functional as F

from conftest import make_spec
from smallbatch.spec import LoraCandidateSpec
from smallbatch.training import ordinal_objective_applies


def lora(**overrides) -> LoraCandidateSpec:
    return LoraCandidateSpec(type="lora", **overrides)


def int_spec(**overrides):
    return make_spec(output={"type": "int", "range": [0, 4]}, **overrides)


def test_auto_uses_ordinal_only_for_scalar_int_decisions():
    assert ordinal_objective_applies(int_spec(), lora())
    # enum labels have no order to exploit
    assert not ordinal_objective_applies(make_spec(), lora())
    # a free-text rationale is not one of the legal completions
    assert not ordinal_objective_applies(int_spec(), lora(rationale_distillation=True))


def test_explicit_objective_overrides_and_fails_loudly():
    assert not ordinal_objective_applies(int_spec(), lora(objective="token"))
    assert ordinal_objective_applies(int_spec(), lora(objective="ordinal"))
    with pytest.raises(ValueError, match="scalar integer output"):
        ordinal_objective_applies(make_spec(), lora(objective="ordinal"))
    with pytest.raises(ValueError, match="rationale"):
        ordinal_objective_applies(
            int_spec(), lora(objective="ordinal", rationale_distillation=True)
        )


def _rps_loss(class_logits: torch.Tensor, target: torch.Tensor) -> torch.Tensor:
    """The loss implemented by OrdinalSFTTrainer, over given class scores."""
    classes = class_logits.size(-1)
    nll = F.cross_entropy(class_logits, target)
    probabilities = F.softmax(class_logits, dim=-1)
    cumulative = probabilities.cumsum(dim=-1)
    steps = (
        torch.arange(classes).unsqueeze(0) >= target.unsqueeze(1)
    ).to(cumulative.dtype)
    rps = ((cumulative - steps) ** 2).sum(dim=-1).mean() / max(classes - 1, 1)
    return nll + rps


def test_loss_penalizes_distant_predictions_more_than_adjacent_ones():
    """The point of the ordinal objective: a 0 predicted as 4 must cost more
    than a 0 predicted as 1, which token cross-entropy cannot express."""
    target = torch.tensor([0])
    adjacent = torch.tensor([[2.0, 3.0, 0.0, 0.0, 0.0]])  # mass on class 1
    distant = torch.tensor([[2.0, 0.0, 0.0, 0.0, 3.0]])  # same mass on class 4
    assert _rps_loss(distant, target) > _rps_loss(adjacent, target)

    # cross-entropy alone is blind to the distinction
    assert F.cross_entropy(distant, target) == pytest.approx(
        float(F.cross_entropy(adjacent, target)), abs=1e-6
    )


def test_loss_is_minimized_by_confident_correct_prediction():
    target = torch.tensor([2])
    confident = torch.tensor([[0.0, 0.0, 8.0, 0.0, 0.0]])
    uncertain = torch.tensor([[1.0, 1.0, 1.0, 1.0, 1.0]])
    assert _rps_loss(confident, target) < _rps_loss(uncertain, target)
    assert float(_rps_loss(confident, target)) < 0.05


class FakeTokenizer:
    """Single-token answers (" 0".." 4") plus a two-token answer for " 10"."""

    eos_token = "<eos>"

    def __call__(self, text, add_special_tokens=False):
        pieces = []
        for chunk in text.replace("<eos>", " <eos>").split():
            if chunk == "<eos>":
                pieces.append(99)
            elif len(chunk) == 1:
                pieces.append(100 + int(chunk))
            else:  # "10" splits into digits, as Qwen-style tokenizers do
                pieces.extend(100 + int(digit) for digit in chunk)
        return {"input_ids": pieces}


def test_decision_tokens_find_the_one_token_that_decides():
    """Completions share the leading space; only the digit decides, so the
    whole distribution is readable from the logits at that one position."""
    from smallbatch.training import ordinal_decision_tokens

    prefix, ids = ordinal_decision_tokens(int_spec(), FakeTokenizer())
    assert prefix == 0
    assert ids == [100, 101, 102, 103, 104]


class _Output:
    def __init__(self, logits):
        self.logits = logits


class FullLogitsModel:
    """A model without the logits_to_keep contract: unknown kwargs raise."""

    def __init__(self, logits):
        self._logits = logits

    def __call__(self, input_ids, attention_mask=None):
        return _Output(self._logits)


class KeepLogitsModel:
    """Projects only the requested positions, like transformers causal LMs."""

    def __init__(self, logits):
        self._logits = logits
        self.kept = None

    def __call__(self, input_ids, attention_mask=None, logits_to_keep=None):
        self.kept = logits_to_keep
        if logits_to_keep is None:
            return _Output(self._logits)
        return _Output(self._logits[:, logits_to_keep, :])


class IgnoresKwargModel:
    """Accepts arbitrary kwargs but returns full logits regardless."""

    def __init__(self, logits):
        self._logits = logits

    def __call__(self, input_ids, attention_mask=None, **kwargs):
        return _Output(self._logits)


def test_compute_loss_reads_only_deciding_positions_however_logits_arrive():
    """The trainer asks the model to project only the deciding positions
    (the full sequence-by-vocabulary logits are what OOM small cards), and
    must produce the identical loss when a model returns full logits instead,
    whether by raising on the kwarg or by silently ignoring it."""
    from smallbatch.training import _ordinal_trainer_class

    trainer_cls = _ordinal_trainer_class(int_spec(), FakeTokenizer())
    # two rows with different deciding positions, right-padded to length 5:
    # row 0 answers at position 2 with " 3", row 1 at position 3 with " 1"
    inputs = {
        "input_ids": torch.tensor([[1, 2, 103, 99, 0], [1, 2, 3, 101, 99]]),
        "labels": torch.tensor(
            [[-100, -100, 103, 99, -100], [-100, -100, -100, 101, 99]]
        ),
    }
    logits = torch.zeros(2, 5, 200)
    logits[0, 1, 100:105] = torch.tensor([0.5, 0.0, 1.0, 4.0, 0.0])
    logits[1, 2, 100:105] = torch.tensor([1.0, 3.0, 0.0, 0.0, 2.0])
    expected = _rps_loss(
        torch.stack([logits[0, 1, 100:105], logits[1, 2, 100:105]]),
        torch.tensor([3, 1]),
    )

    keep_model = KeepLogitsModel(logits)
    for model in (keep_model, FullLogitsModel(logits), IgnoresKwargModel(logits)):
        loss = trainer_cls.compute_loss(None, model, inputs)
        assert float(loss) == pytest.approx(float(expected), abs=1e-6)
    # the supporting model really was asked for just the deciding positions
    assert keep_model.kept is not None
    assert keep_model.kept.tolist() == [1, 2]


def test_scales_wider_than_one_digit_are_refused_by_the_spec():
    """A level must be one token to be trained and scored as an ordered choice,
    so an integer range has to fit in 0-9: a 0-10 scale would split "10" into
    two digits and no single position would carry the decision."""
    from smallbatch.spec import FunctionSpec

    with pytest.raises(ValueError, match="must lie within 0-9"):
        make_spec(output={"type": "int", "range": [0, 10]})
    with pytest.raises(ValueError, match="must lie within 0-9"):
        FunctionSpec(
            name="wide",
            description="d",
            input_schema={"text": "string"},
            output={"type": "int", "range": [-1, 5]},
            prompt="p",
            candidates={"c": {"type": "tfidf"}},
        )
    make_spec(output={"type": "int", "range": [0, 9]})  # the widest legal scale


def test_shared_completion_tokens_cancel_in_the_class_distribution():
    """Both paths normalize over the legal completions, so anything the
    completions share — the prompt, the leading space — cancels. Only the
    deciding tokens may move the distribution, which is why the shared trailing
    token is excluded from the scored span in both paths."""
    deciding = torch.tensor([[0.4, 2.1, -1.0, 0.3, 1.7]])
    shared = -3.7  # identical for every completion

    assert torch.allclose(
        torch.log_softmax(deciding, dim=-1),
        torch.log_softmax(deciding + shared, dim=-1),
        atol=1e-6,
    )
    target = torch.tensor([3])
    assert _rps_loss(deciding, target) == pytest.approx(
        float(_rps_loss(deciding + shared, target)), abs=1e-6
    )
