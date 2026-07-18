"""TF-IDF candidate integration with the shared softmax ordinal head.

Unit coverage for the head itself lives in test_heads.py; these tests exercise
the candidate path — training, persistence, decoding, diagnostics, and the
generated package's reimplementation.
"""

from __future__ import annotations

import json

import numpy as np
import pytest

from conftest import make_spec
from smallbatch import heads


@pytest.fixture(autouse=True)
def fast_training(monkeypatch):
    """Unit tests exercise selection logic, not convergence budgets."""
    monkeypatch.setattr(heads, "MAX_EPOCHS", 200)
    monkeypatch.setattr(heads, "NO_DEV_EPOCHS", 200)
    monkeypatch.setattr(heads, "PATIENCE", 30)
    monkeypatch.setattr(heads, "SEEDS", (0,))


def scale_rows(levels=3):
    texts = ("routine question", "delayed response", "money withheld")
    return [
        {
            "id": f"{level}-{index}",
            "input": {"title": f"case {index}", "body": texts[level]},
            "output": level,
            "origin": "real",
        }
        for level in range(levels)
        for index in range(8)
    ]


def test_tfidf_trains_an_ordered_head_and_round_trips(tmp_path):
    from smallbatch.candidates import predict_tfidf, train_tfidf

    spec = make_spec(output={"type": "int", "range": [0, 2]})
    record = train_tfidf(spec, scale_rows(), tmp_path)
    assert record["objective"] == {"score": "ordinal"}

    predictions = predict_tfidf(
        tmp_path, spec, [{"title": "case", "body": "money withheld"}]
    )
    assert predictions == [2]


def test_enum_output_keeps_the_multinomial_head(tmp_path):
    from smallbatch.candidates import train_tfidf

    spec = make_spec()  # urgent / normal labels
    rows = [
        {
            "id": str(index),
            "input": {"title": f"t{index}", "body": "outage" if index % 2 else "question"},
            "output": "urgent" if index % 2 else "normal",
            "origin": "real",
        }
        for index in range(10)
    ]
    record = train_tfidf(spec, rows, tmp_path)
    assert record["objective"] == {"score": "multinomial"}


def test_ordered_head_persists_as_plain_numpy_only(tmp_path):
    """Artifacts must load under skops' strict policy and carry no Smallbatch
    class — and no torch — or generated packages could not unpickle them."""
    import skops.io as sio

    from smallbatch.candidates import train_tfidf

    spec = make_spec(output={"type": "int", "range": [0, 1]})
    rows = [
        {
            "id": str(index),
            "input": {"title": f"t{index}", "body": "withheld" if index % 2 else "asked"},
            "output": index % 2,
            "origin": "real",
        }
        for index in range(10)
    ]
    train_tfidf(spec, rows, tmp_path)
    path = tmp_path / "model.skops"
    assert sio.get_untrusted_types(file=path) == []
    loaded = sio.load(path, trusted=[])
    head = loaded["score"]["head"]
    assert head["kind"] == heads.KIND
    assert all(isinstance(layer["weight"], np.ndarray) for layer in head["layers"])


def test_a_cumulative_head_from_an_old_build_fails_closed():
    """v0.2 builds persisted a boundary-chain head; running one must refuse
    with direction, not decode garbage or crash mid-arithmetic."""
    from smallbatch.candidates import predict_tfidf_pipelines

    class IdentityVectorizer:
        def transform(self, texts):
            return np.zeros((len(texts), 1))

    spec = make_spec(output={"type": "int", "range": [0, 2]})
    old_head = {"kind": "ordinal-cumulative", "values": [0, 1, 2], "steps": []}
    pipelines = {"score": {"vectorizer": IdentityVectorizer(), "head": old_head}}
    with pytest.raises(ValueError, match="predates the v0.3 softmax head"):
        predict_tfidf_pipelines(pipelines, spec, [{"title": "t", "body": "b"}])


@pytest.mark.parametrize("decoder", [None, "argmax", "median", "within_one"])
def test_predict_matches_the_standalone_template_arithmetic(decoder):
    """The generated package reimplements the head in a few lines of numpy;
    the two must agree for every decode rule and for heads persisted before
    the decoder key existed."""
    template = (
        __import__("pathlib")
        .Path("src/smallbatch/standalone_templates/common.py.tmpl")
        .read_text()
    )
    namespace: dict = {"json": __import__("json")}
    body = template.split("def ordinal_predict", 1)[1]
    exec("def ordinal_predict" + body.split("\ndef metadata", 1)[0], namespace)

    rng = np.random.default_rng(0)
    features = np.array(
        [[float(level) + rng.normal(0, 0.1)] for level in (0, 1, 2, 3) for _ in range(8)]
    )
    labels = [level for level in (0, 1, 2, 3) for _ in range(8)]
    head, _ = heads.fit_head([0, 1, 2, 3], features, labels)
    if decoder is not None:
        head["decoder"] = decoder

    probe = np.array([[0.3], [1.6], [3.2]])
    assert namespace["ordinal_predict"](head, probe) == heads.predict(head, probe)


def test_tfidf_selects_and_persists_a_decoder(tmp_path):
    import skops.io as sio

    from smallbatch.candidates import train_tfidf

    spec = make_spec(output={"type": "int", "range": [0, 2]})
    rows = scale_rows()
    record = train_tfidf(spec, rows, tmp_path, dev_rows=rows)
    assert record["decode"]["score"] in ("argmax", "median", "within_one")
    assert record["dev_decode_comparison"]["score"]["selected"] == record["decode"]["score"]

    # the decoder key persists inside the head under skops' strict trust policy
    loaded = sio.load(tmp_path / "model.skops", trusted=[])
    assert loaded["score"]["head"]["decoder"] == record["decode"]["score"]


def test_tfidf_pinned_decoder_needs_no_dev_rows(tmp_path):
    import skops.io as sio

    from smallbatch.candidates import train_tfidf

    spec = make_spec(output={"type": "int", "range": [0, 1]})
    rows = [
        {
            "id": str(index),
            "input": {"title": f"t{index}", "body": "withheld" if index % 2 else "asked"},
            "output": index % 2,
            "origin": "real",
        }
        for index in range(10)
    ]
    record = train_tfidf(spec, rows, tmp_path, decode="median")
    assert record["decode"] == {"score": "median"}
    assert record["dev_decode_comparison"] == {}
    loaded = sio.load(tmp_path / "model.skops", trusted=[])
    assert loaded["score"]["head"]["decoder"] == "median"


def test_tfidf_records_head_tuning_diagnostics_and_local_distributions(tmp_path):
    from smallbatch.candidates import DEV_DISTRIBUTIONS_FILE, train_tfidf

    spec = make_spec(output={"type": "int", "range": [0, 2]})
    rows = scale_rows()
    record = train_tfidf(spec, rows, tmp_path, dev_rows=rows)

    tuning = record["head_tuning"]["score"]
    assert {"hidden", "dropout", "weight_decay", "seed"} <= set(tuning["selected"])
    assert len(tuning["trials"]) == len(heads.HEAD_GRID) * len(heads.SEEDS)

    diagnostics = record["head_diagnostics"]["score"]
    assert diagnostics["rows"] == len(rows)
    assert [entry["level"] for entry in diagnostics["levels"]] == [0, 1, 2]
    assert 0.0 <= diagnostics["mean_confidence"] <= 1.0

    local = json.loads((tmp_path / DEV_DISTRIBUTIONS_FILE).read_text())
    assert local["score"]["levels"] == [0, 1, 2]
    first = local["score"]["rows"][0]
    assert first["id"] == "0-0"
    assert first["reference"] == 0
    assert len(first["distribution"]) == 3
    # per-row distributions are decision evidence: local file only, never the record
    assert "dev_distributions" not in json.dumps(record)


def test_deployable_copy_refuses_local_files(tmp_path):
    from smallbatch.artifacts import copy_deployable_model, deployable_size

    source = tmp_path / "model"
    source.mkdir()
    (source / "model.skops").write_text("weights")
    (source / "dev_distributions.local.json").write_text("{}")

    copy_deployable_model(source, tmp_path / "deployed", "tfidf")
    assert (tmp_path / "deployed" / "model.skops").exists()
    assert not (tmp_path / "deployed" / "dev_distributions.local.json").exists()
    assert deployable_size(source, "tfidf") == len("weights")
