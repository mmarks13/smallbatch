"""The lora standalone template's constrained-decoding hot path.

The token-prefix trie depends only on the fixed FUNCTION spec, so it must be
built once per process — retokenizing every legal completion on each
classify() call would dominate the packaged function's latency.
"""

import importlib.util
import sys
from pathlib import Path

import torch

import smallbatch

_COMMON_STUB = """\
from pathlib import Path

FUNCTION = {function!r}
ROOT = Path(".")


def metadata():
    return {{}}


def render_input(item):
    return str(item)


def validate_input(item):
    return item


def validate_output(output):
    return output


def parse_strict_json(text, exhausted=False):
    return text


class InvalidOutputError(ValueError):
    pass
"""


def load_lora_template(tmp_path, function):
    package_dir = tmp_path / "loratpl"
    package_dir.mkdir()
    template = Path(smallbatch.__file__).parent / "standalone_templates" / "lora.py.tmpl"
    (package_dir / "__init__.py").write_text(template.read_text())
    (package_dir / "_common.py").write_text(_COMMON_STUB.format(function=function))
    spec = importlib.util.spec_from_file_location(
        "loratpl",
        package_dir / "__init__.py",
        submodule_search_locations=[str(package_dir)],
    )
    module = importlib.util.module_from_spec(spec)
    sys.modules["loratpl"] = module
    try:
        spec.loader.exec_module(module)
    finally:
        sys.modules.pop("loratpl._common", None)
        sys.modules.pop("loratpl", None)
    return module


class CountingTokenizer:
    eos_token_id = 7
    pad_token_id = 7

    def __init__(self):
        self.calls = 0

    def __call__(self, text, add_special_tokens=False):
        self.calls += 1
        return {"input_ids": [ord(char) for char in text]}


class FakeIds(list):
    def tolist(self):
        return list(self)

    def __getitem__(self, key):
        value = super().__getitem__(key)
        return FakeIds(value) if isinstance(key, slice) else value


def test_completion_trie_is_built_once_per_process(tmp_path):
    module = load_lora_template(
        tmp_path,
        {
            "name": "fn",
            "output": {"fields": {"score": {"labels": ["a", "b"], "range": None}}},
            "runtime": {"decoding": {"max_new_tokens": 8}},
        },
    )
    tokenizer = CountingTokenizer()
    first = module._constraint(tokenizer, 3)
    assert tokenizer.calls == 2  # one tokenization per legal completion
    second = module._constraint(tokenizer, 5)
    assert tokenizer.calls == 2  # cached tables: no retokenization per call
    assert first is not None and second is not None

    # the per-call closure still honors its own prompt length: nothing has
    # been generated yet, so only the completions' shared first token is legal
    assert first(0, FakeIds([1, 2, 3])) == [ord(" ")]
    assert second(0, FakeIds([1, 2, 3, 4, 5])) == [ord(" ")]


def test_unconstrained_specs_cache_the_absence_too(tmp_path):
    """A text-bearing output has no finite completion set: the template must
    cache that absence instead of re-deriving it per call."""
    module = load_lora_template(
        tmp_path,
        {
            "name": "fn",
            "output": {
                "fields": {
                    "score": {"labels": None, "range": None, "max_chars": 300}
                }
            },
            "runtime": {"decoding": {"max_new_tokens": 8}},
        },
    )
    tokenizer = CountingTokenizer()
    assert module._constraint(tokenizer, 3) is None
    assert module._constraint(tokenizer, 3) is None
    assert tokenizer.calls == 0


def test_ordinal_hot_path_preserves_prompt_special_tokens(tmp_path):
    module = load_lora_template(
        tmp_path,
        {
            "name": "fn",
            "output": {"fields": {"score": {"labels": None, "range": [0, 1]}}},
            "runtime": {"decoding": {"max_new_tokens": 8}},
        },
    )

    class BosTokenizer:
        padding_side = "right"

        def __init__(self):
            self.batch_add_special_tokens = None

        def __call__(
            self,
            text,
            add_special_tokens=True,
            return_tensors=None,
            padding=False,
            truncation=False,
            max_length=None,
            return_token_type_ids=None,
        ):
            if isinstance(text, str):
                assert add_special_tokens is False
                return {"input_ids": [32, 10 if text.endswith("0") else 11]}
            self.batch_add_special_tokens = add_special_tokens
            rows = [[99, *[ord(char) % 40 for char in value]] for value in text]
            return {
                "input_ids": torch.tensor(rows),
                "attention_mask": torch.ones((len(rows), len(rows[0])), dtype=torch.long),
            }

        def decode(self, ids):
            return " " if ids else ""

    class BosModel:
        def __call__(self, input_ids, attention_mask):
            assert torch.all(input_ids[:, 0] == 99), "BOS was omitted from standalone inference"
            logits = torch.zeros(input_ids.shape[0], input_ids.shape[1], 120)
            logits[:, -1, 11] = 2.0
            return type("Out", (), {"logits": logits})()

    tokenizer = BosTokenizer()
    assert module._run_levels(tokenizer, BosModel(), ["prompt"], [0, 1]) == [1]
    assert tokenizer.batch_add_special_tokens is True
