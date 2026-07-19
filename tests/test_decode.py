from __future__ import annotations

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


def test_decode_levels_reports_the_mode():
    """v0.3: every ordinal field decodes by argmax — the level constrained
    greedy decoding would emit — with no configurable alternatives."""
    levels = [0, 1, 2, 3, 4]
    distributions = torch.tensor(
        [
            [0.40, 0.05, 0.30, 0.20, 0.05],
            [0.01, 0.02, 0.90, 0.04, 0.03],
        ]
    )
    assert decode.decode_levels(distributions, levels) == [0, 2]


def test_decode_levels_ties_resolve_to_the_lower_level():
    assert decode.decode_levels(torch.tensor([[0.5, 0.5]]), [0, 1]) == [0]


def test_decode_levels_accepts_plain_arrays():
    import numpy as np

    assert decode.decode_levels(np.array([[0.2, 0.5, 0.3]]), [0, 1, 2]) == [1]
