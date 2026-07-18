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


def test_within_one_maximizes_the_windowed_mass():
    levels = [0, 1, 2, 3, 4]
    # the mode is 0, but the window centered on 1 spans {0,1,2} = 0.70 of the
    # mass: the three decoders each read a different level out of one row
    distributions = torch.tensor([[0.40, 0.05, 0.25, 0.30, 0.00]])

    assert decode.decode_levels(distributions, levels, "argmax") == [0]
    assert decode.decode_levels(distributions, levels, "median") == [2]
    assert decode.decode_levels(distributions, levels, "within_one") == [1]


def test_within_one_ties_resolve_to_the_lower_level():
    """An edge window is a subset of its interior neighbor's window, so an edge
    level can only win by tie — and ties go to the lower level."""
    levels = [0, 1, 2, 3]
    distributions = torch.tensor(
        [
            [0.6, 0.0, 0.0, 0.4],  # {0,1}=0.6 ties {0,1,2}=0.6 -> 0
            [0.4, 0.0, 0.0, 0.6],  # {1,2,3}=0.6 ties {2,3}=0.6 -> 2
        ]
    )
    assert decode.decode_levels(distributions, levels, "within_one") == [0, 2]


def test_decode_levels_accepts_plain_arrays():
    import numpy as np

    levels = [0, 1, 2]
    distributions = np.array([[0.2, 0.5, 0.3]])
    assert decode.decode_levels(distributions, levels, "argmax") == [1]
    assert decode.decode_levels(distributions, levels, "median") == [1]
    assert decode.decode_levels(distributions, levels, "within_one") == [1]


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
