from smallbatch.profiling import _percentile, _runtime_dependencies


def test_profile_percentiles_are_nearest_rank():
    assert _percentile([5, 1, 3, 2, 4], 0.5) == 3
    assert _percentile([5, 1, 3, 2, 4], 0.95) == 5


def test_runtime_dependencies_are_candidate_specific():
    tfidf = _runtime_dependencies("tfidf")
    assert "scikit-learn" in tfidf and "torch" not in tfidf
    assert "torch" in _runtime_dependencies("lora")
