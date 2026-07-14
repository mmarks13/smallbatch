from __future__ import annotations

import pytest
import torch

from conftest import make_spec
from smallbatch import decode


def test_scale_levels_only_for_integer_scales():
    assert decode.scale_levels(make_spec(output={"type": "int", "range": [0, 4]})) == [
        0,
        1,
        2,
        3,
        4,
    ]
    assert decode.scale_levels(make_spec()) is None  # enum labels are unordered


def test_argmax_reports_the_mode_and_median_the_middle_of_the_mass():
    levels = [0, 1, 2, 3, 4]
    # the mode is 0, but half the mass only accumulates by level 2: a decoder
    # that cares how far off it lands should not report 0 here
    distributions = torch.tensor([[0.40, 0.05, 0.30, 0.20, 0.05]])

    assert decode.decode_levels(distributions, levels, "argmax") == [0]
    assert decode.decode_levels(distributions, levels, "median") == [2]


def test_decoders_agree_when_the_distribution_is_peaked():
    levels = [0, 1, 2, 3, 4]
    distributions = torch.tensor([[0.01, 0.02, 0.90, 0.04, 0.03]])
    assert decode.decode_levels(distributions, levels, "argmax") == [2]
    assert decode.decode_levels(distributions, levels, "median") == [2]


def test_unknown_decoder_is_refused():
    with pytest.raises(ValueError, match="unknown decoder"):
        decode.decode_levels(torch.tensor([[0.5, 0.5]]), [0, 1], "mean")


def test_median_lands_on_the_first_level_reaching_half_the_mass():
    levels = [0, 1, 2, 3, 4]
    distributions = torch.tensor(
        [
            [0.6, 0.1, 0.1, 0.1, 0.1],  # half the mass by level 0
            [0.3, 0.3, 0.2, 0.1, 0.1],  # crosses at level 1
            [0.1, 0.1, 0.1, 0.1, 0.6],  # only the last level reaches it
        ]
    )
    assert decode.decode_levels(distributions, levels, "median") == [0, 1, 4]
