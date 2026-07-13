import pytest

from conftest import make_spec
from smallbatch.evaluate import _oom_backoff, compute_metrics, train_fitted_constant


def test_train_fitted_constant_uses_only_training_rows():
    spec = make_spec(output={"type": "int", "range": [0, 4]})
    train = [{"output": value} for value in [0, 0, 1, 4]]
    assert train_fitted_constant(spec, train) == 0


def test_compute_metrics_delegates_shared_shape():
    spec = make_spec()
    metrics = compute_metrics(spec, ["urgent", "normal"], ["urgent", "urgent"])
    assert metrics["decision_agreement"] == 0.5
    assert "constant_baseline" not in metrics


def test_oom_backoff_retries_same_slice_and_sticks():
    class OOM(Exception):
        pass

    calls = []

    def process(items):
        calls.append(list(items))
        if len(items) > 2:
            raise OOM()
        return items

    output, size = _oom_backoff(process, list(range(5)), 4, lambda exc: isinstance(exc, OOM))
    assert output == list(range(5)) and size == 2
    assert calls[:2] == [[0, 1, 2, 3], [0, 1]]


def test_oom_at_one_is_real_failure():
    with pytest.raises(RuntimeError):
        _oom_backoff(lambda _: (_ for _ in ()).throw(RuntimeError()), [1], 1, lambda _: True)
