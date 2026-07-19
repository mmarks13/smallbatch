"""Constrained decoding: legal-completion enumeration and the token trie
(torch-free — the tokenizer is faked)."""

from conftest import make_spec
from smallbatch.evaluate import _prefix_allowed_fn, completion_trie
from smallbatch.prompts import allowed_completions


class FakeTokenizer:
    """One token per character, id = codepoint."""

    def __call__(self, text, add_special_tokens=False):
        return {"input_ids": [ord(c) for c in text]}


def test_allowed_completions_int_and_enum():
    assert allowed_completions(make_spec(output={"type": "int", "range": [0, 9]}))[:3] == [" 0", " 1", " 2"]
    spec = make_spec(output={"type": "enum", "labels": ["urgent", "low"]})
    assert allowed_completions(spec) == [" urgent", " low"]


def test_allowed_completions_none_when_output_has_text():
    """Generated text has no finite completion set, so text-bearing functions
    never use constrained decoding."""
    scalar_text = make_spec(
        output={"type": "text", "max_chars": 100},
        candidates={"g": {"type": "lora"}},
    )
    assert allowed_completions(scalar_text) is None
    mixed = make_spec(
        output={"priority": {"range": [0, 4]}, "explanation": {"type": "text"}},
        candidates={"g": {"type": "lora"}},
    )
    assert allowed_completions(mixed) is None


def test_trie_prefix_and_terminals():
    trie, terminals = completion_trie(FakeTokenizer(), [" 1", " 10"])
    # from the start, only " " is legal
    assert trie[()] == {ord(" ")}
    # after " ", only "1"
    assert trie[(ord(" "),)] == {ord("1")}
    # " 1" is complete but extendable by "0"
    assert (ord(" "), ord("1")) in terminals
    assert trie[(ord(" "), ord("1"))] == {ord("0")}
    assert (ord(" "), ord("1"), ord("0")) in terminals


def test_prefix_fn_allows_eos_at_terminal():
    trie, terminals = completion_trie(FakeTokenizer(), [" 1", " 10"])
    eos = 999

    class Ids:
        def __init__(self, ids):
            self._ids = ids

        def __getitem__(self, sl):
            return Ids(self._ids[sl])

        def tolist(self):
            return self._ids

    fn = _prefix_allowed_fn(trie, terminals, prompt_len=2, eos_id=eos)
    # prompt only: must start with " "
    assert fn(0, Ids([7, 7])) == [ord(" ")]
    # at " 1": may emit "0" or stop
    assert set(fn(0, Ids([7, 7, ord(" "), ord("1")]))) == {ord("0"), eos}
    # at " 10": only stop
    assert fn(0, Ids([7, 7, ord(" "), ord("1"), ord("0")])) == [eos]
