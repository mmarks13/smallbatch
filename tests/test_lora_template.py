"""The lora standalone template's constrained-decoding hot path.

The token-prefix trie depends only on the fixed FUNCTION spec, so it must be
built once per process — retokenizing every legal completion on each
classify() call would dominate the packaged function's latency.
"""

import importlib.util
import sys
from pathlib import Path

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
            "runtime": {},
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
    module = load_lora_template(
        tmp_path,
        {
            "name": "fn",
            "output": {"fields": {"score": {"labels": ["a"], "range": None}}},
            "runtime": {"rationale_distillation": True},
        },
    )
    tokenizer = CountingTokenizer()
    assert module._constraint(tokenizer, 3) is None
    assert module._constraint(tokenizer, 3) is None
    assert tokenizer.calls == 0
