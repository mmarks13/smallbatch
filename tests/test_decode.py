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


def test_score_levels_preserves_prompt_special_tokens():
    class BosTokenizer:
        eos_token_id = 98
        pad_token_id = 98
        padding_side = "right"

        def __init__(self):
            self.batch_add_special_tokens = None

        def __call__(
            self,
            text,
            add_special_tokens=True,
            return_tensors=None,
            padding=False,
        ):
            if isinstance(text, str):
                assert add_special_tokens is False
                return {"input_ids": [32, 10 if text.endswith("0") else 11]}
            self.batch_add_special_tokens = add_special_tokens
            rows = [[99, *[ord(char) % 40 for char in value]] for value in text]
            width = max(map(len, rows))
            padded = [[98] * (width - len(row)) + row for row in rows]
            return {
                "input_ids": torch.tensor(padded),
                "attention_mask": torch.ones((len(rows), width), dtype=torch.long),
            }

        def decode(self, ids):
            return " " if ids else ""

    class BosModel:
        device = "cpu"

        def __call__(self, input_ids, attention_mask):
            assert torch.all(input_ids[:, 0] == 99), "BOS was omitted from inference"
            logits = torch.zeros(input_ids.shape[0], input_ids.shape[1], 120)
            logits[:, -1, 11] = 2.0
            return type("Out", (), {"logits": logits})()

    tokenizer = BosTokenizer()
    spec = make_spec(output={"type": "int", "range": [0, 1]})
    distributions = decode.score_levels(BosModel(), tokenizer, spec, ["prompt"])
    assert tokenizer.batch_add_special_tokens is True
    assert decode.decode_levels(distributions, [0, 1]) == [1]
