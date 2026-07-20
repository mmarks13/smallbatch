import json
import sys
import types

import pytest

from smallbatch.training import _load_tokenizer


def install_transformers(monkeypatch, tmp_path, auto_error, extra_special_tokens=None):
    module = types.ModuleType("transformers")
    utils = types.ModuleType("transformers.utils")
    config = tmp_path / "tokenizer_config.json"
    config.write_text(json.dumps({"extra_special_tokens": extra_special_tokens}))

    class AutoTokenizer:
        @classmethod
        def from_pretrained(cls, base):
            raise auto_error

    class PreTrainedTokenizerFast:
        @classmethod
        def from_pretrained(cls, base, **kwargs):
            return {"base": base, "kwargs": kwargs}

    module.AutoTokenizer = AutoTokenizer
    module.PreTrainedTokenizerFast = PreTrainedTokenizerFast
    utils.cached_file = lambda *args: config
    monkeypatch.setitem(sys.modules, "transformers", module)
    monkeypatch.setitem(sys.modules, "transformers.utils", utils)


def test_load_tokenizer_bridges_transformers_five_backend(monkeypatch, tmp_path):
    install_transformers(
        monkeypatch,
        tmp_path,
        ValueError("Tokenizer class TokenizersBackend does not exist or is not imported"),
        [],
    )

    tokenizer = _load_tokenizer("example/model")

    assert tokenizer == {
        "base": "example/model",
        "kwargs": {"extra_special_tokens": {}},
    }


def test_load_tokenizer_rejects_lossy_transformers_five_bridge(monkeypatch, tmp_path):
    install_transformers(
        monkeypatch,
        tmp_path,
        ValueError("Tokenizer class TokenizersBackend does not exist or is not imported"),
        ["<image>"],
    )

    with pytest.raises(ValueError, match="cannot translate them safely"):
        _load_tokenizer("example/model")


def test_load_tokenizer_does_not_hide_other_configuration_errors(monkeypatch, tmp_path):
    install_transformers(
        monkeypatch,
        tmp_path,
        ValueError("unsupported custom tokenizer"),
    )

    with pytest.raises(ValueError, match="unsupported custom tokenizer"):
        _load_tokenizer("example/model")


def test_latest_checkpoint_ignores_incomplete_dirs_and_compares_steps_numerically(tmp_path):
    from smallbatch.training import _latest_checkpoint

    trainer_dir = tmp_path / "trainer"
    for name in ("checkpoint-200", "checkpoint-1000", "checkpoint-partial"):
        (trainer_dir / name).mkdir(parents=True)
    (trainer_dir / "checkpoint-200" / "trainer_state.json").write_text("{}")
    (trainer_dir / "checkpoint-1000" / "trainer_state.json").write_text("{}")
    (trainer_dir / "checkpoint-2000").mkdir()
    assert _latest_checkpoint(trainer_dir).name == "checkpoint-1000"
    assert _latest_checkpoint(tmp_path / "missing") is None
