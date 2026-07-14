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
