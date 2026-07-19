from __future__ import annotations

import numpy as np
import pytest
from scipy import sparse

from conftest import make_spec
from smallbatch import heads


@pytest.fixture(autouse=True)
def fast_training(monkeypatch):
    """Unit tests exercise selection logic, not convergence budgets."""
    monkeypatch.setattr(heads, "MAX_EPOCHS", 200)
    monkeypatch.setattr(heads, "NO_DEV_EPOCHS", 200)
    monkeypatch.setattr(heads, "PATIENCE", 30)
    monkeypatch.setattr(heads, "SEEDS", (0,))


def ordered_data(levels=4, rows_per_level=12):
    """One feature that increases with the level, plus a noise column."""
    rng = np.random.default_rng(0)
    features, labels = [], []
    for level in range(levels):
        for _ in range(rows_per_level):
            features.append([float(level) + rng.normal(0, 0.1), rng.normal()])
            labels.append(level)
    return np.array(features), labels


def test_applies_only_to_integer_scales():
    assert heads.applies(make_spec(output={"type": "int", "range": [0, 4]}), "score")
    assert not heads.applies(make_spec(), "score")  # enum labels have no order


def test_ordinal_loss_matches_the_lora_trainer_formula():
    """One definition of the objective: the shared helper must compute exactly
    what the LoRA trainer's inline loss computed — NLL plus RPS over the
    cumulative distribution, normalized by the span."""
    import torch
    import torch.nn.functional as F

    torch.manual_seed(0)
    class_logits = torch.randn(5, 4)
    target = torch.tensor([0, 3, 1, 2, 3])

    nll = F.cross_entropy(class_logits, target)
    probabilities = F.softmax(class_logits, dim=-1)
    cumulative = probabilities.cumsum(dim=-1)
    steps = (torch.arange(4).unsqueeze(0) >= target.unsqueeze(1)).to(cumulative.dtype)
    rps = ((cumulative - steps) ** 2).sum(dim=-1).mean() / 3

    assert torch.isclose(heads.ordinal_loss(class_logits, target), nll + rps)


def test_fit_head_recovers_the_ordered_levels():
    features, labels = ordered_data()
    head, _ = heads.fit_head([0, 1, 2, 3], features, labels, setting="linear")

    probe = np.array([[0.0, 0.0], [1.0, 0.0], [3.0, 0.0]])
    assert heads.predict(head, probe) == [0, 1, 3]
    distribution = heads.class_distribution(head, probe)
    assert distribution.shape == (3, 4)
    assert np.allclose(distribution.sum(axis=1), 1.0)


def test_predictions_live_in_the_value_space_not_index_space():
    """SetFit trains on indices of the real scale; whatever the values list
    holds is what predict must return."""
    features, labels = ordered_data(levels=3)
    head, _ = heads.fit_head(
        [10, 20, 30], features, [10 * (label + 1) for label in labels]
    )
    assert set(heads.predict(head, features)) <= {10, 20, 30}


def test_dev_rows_drive_capacity_selection_and_the_trial_record():
    features, labels = ordered_data()
    head, tuning = heads.fit_head(
        [0, 1, 2, 3], features, labels, features, labels
    )
    assert tuning["metric"] == "within_one"
    assert len(tuning["trials"]) == len(heads.HEAD_GRID) * len(heads.SEEDS)
    assert {"hidden", "dropout", "weight_decay", "seed"} <= set(tuning["selected"])
    assert head["hidden"] == tuning["selected"]["hidden"]


def test_without_dev_rows_the_default_entry_stands():
    features, labels = ordered_data()
    head, tuning = heads.fit_head([0, 1, 2, 3], features, labels)
    assert tuning is None
    assert head["hidden"] == heads.HEAD_GRID[0]["hidden"] == 0
    assert len(head["layers"]) == 1


def test_setting_pins_the_capacity_family():
    features, labels = ordered_data()
    linear, _ = heads.fit_head(
        [0, 1, 2, 3], features, labels, features, labels, setting="linear"
    )
    mlp, _ = heads.fit_head(
        [0, 1, 2, 3], features, labels, features, labels, setting="mlp"
    )
    assert linear["hidden"] == 0 and len(linear["layers"]) == 1
    assert mlp["hidden"] > 0 and len(mlp["layers"]) == 2
    with pytest.raises(ValueError, match="unknown head setting"):
        heads.fit_head([0, 1, 2, 3], features, labels, setting="argmax")


def test_sparse_features_train_and_predict_without_densifying():
    """TF-IDF features over an open bigram vocabulary must stay sparse: no
    scaler is fit, torch multiplies in CSR form, and inference multiplies
    through scipy directly."""
    dense, labels = ordered_data(levels=3)
    features = sparse.csr_matrix(np.hstack([dense, np.zeros((dense.shape[0], 40))]))
    head, _ = heads.fit_head([0, 1, 2], features, labels, features, labels)

    assert head["scaler"] is None
    predictions = heads.predict(head, features)
    matches = sum(p == r for p, r in zip(predictions, labels))
    assert matches / len(labels) > 0.8


def test_dense_features_are_standardized_and_the_scaler_persists():
    features, labels = ordered_data()
    head, _ = heads.fit_head([0, 1, 2, 3], features, labels)
    assert head["scaler"] is not None
    assert head["scaler"]["mean"].shape == (2,)
    assert (head["scaler"]["scale"] > 0).all()


def test_predict_reads_the_argmax_level():
    """v0.3: one decode rule for every candidate family — the head's argmax,
    with no per-head decoder state persisted or selectable."""
    features, labels = ordered_data()
    head, _ = heads.fit_head([0, 1, 2, 3], features, labels, features, labels)
    assert "decoder" not in head
    predictions = heads.predict(head, features)
    matches = sum(p == r for p, r in zip(predictions, labels))
    assert matches / len(labels) > 0.8


def test_head_diagnostics_report_per_level_calibration_only():
    """Public evidence: level-mass summaries, never inputs or per-row values."""
    features, labels = ordered_data(levels=3)
    head, _ = heads.fit_head([0, 1, 2], features, labels, features, labels)

    diagnostics = heads.head_diagnostics(
        head, features, labels, levels=["low", "mid", "high"]
    )
    assert diagnostics["rows"] == len(labels)
    assert [entry["level"] for entry in diagnostics["levels"]] == ["low", "mid", "high"]
    for entry in diagnostics["levels"]:
        assert 0.0 <= entry["mean_probability"] <= 1.0
        assert 0.0 <= entry["observed_rate"] <= 1.0
    assert 0.0 <= diagnostics["mean_confidence"] <= 1.0


def test_a_persisted_head_is_skops_loadable_under_the_strict_trust_policy(tmp_path):
    """Generated packages refuse artifacts with untrusted types; a head of
    plain numpy arrays in plain containers must load with trusted=[]."""
    import skops.io as sio

    features, labels = ordered_data(levels=3)
    head, _ = heads.fit_head([0, 1, 2], features, labels)

    path = tmp_path / "head.skops"
    sio.dump(head, path)
    assert sio.get_untrusted_types(file=path) == []
    loaded = sio.load(path, trusted=[])
    assert heads.predict(loaded, features) == heads.predict(head, features)
