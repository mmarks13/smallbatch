"""Adaptive eval batching: the pure OOM-halving loop (no torch, no GPU).

Spec: local/adaptive-eval-batching.md — eval-time OOM is data-dependent, so
generate_batch halves its batch size and retries the same slice instead of
killing a compile whose adapter already trained.
"""

import pytest

from smallbatch.evaluate import _oom_backoff


class FakeOOM(Exception):
    pass


def make_process(fails_above: int, log: list):
    """process() that OOMs whenever the slice is larger than `fails_above`."""

    def process(chunk):
        log.append(len(chunk))
        if len(chunk) > fails_above:
            raise FakeOOM()
        return [f"out-{x}" for x in chunk]

    return process


def is_oom(e):
    return isinstance(e, FakeOOM)


def test_halves_and_retries_same_slice_results_complete():
    log = []
    items = list(range(10))
    results, final = _oom_backoff(make_process(2, log), items, 8, is_oom)
    assert results == [f"out-{x}" for x in items]  # ordered and complete
    assert final == 2
    assert log[:3] == [8, 4, 2]  # same slice retried at 8 -> 4 -> 2


def test_reduction_is_sticky():
    log = []
    _oom_backoff(make_process(4, log), list(range(16)), 8, is_oom)
    assert log == [8, 4, 4, 4, 4]  # never grows back after the first halving


def test_oom_at_size_one_reraises():
    with pytest.raises(FakeOOM):
        _oom_backoff(make_process(0, []), [1, 2], 1, is_oom)


def test_non_oom_exceptions_propagate_immediately():
    def process(chunk):
        raise RuntimeError("not an oom")

    with pytest.raises(RuntimeError):
        _oom_backoff(process, [1, 2, 3], 8, is_oom)


def test_on_oom_hook_called_per_reduction():
    hooks = []
    _oom_backoff(
        make_process(2, []), list(range(8)), 8, is_oom, on_oom=lambda: hooks.append(1)
    )
    assert len(hooks) == 2  # 8 -> 4 -> 2


def test_no_oom_returns_requested_size():
    results, final = _oom_backoff(make_process(99, []), list(range(5)), 4, is_oom)
    assert final == 4 and len(results) == 5
