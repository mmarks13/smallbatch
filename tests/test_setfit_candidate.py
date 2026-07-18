import json
import sys
import types
from pathlib import Path

from conftest import make_spec
from smallbatch.setfit_candidate import predict_setfit, train_setfit


class Dataset:
    @classmethod
    def from_dict(cls, value):
        return value


class TrainingArguments:
    def __init__(self, **kwargs):
        self.kwargs = kwargs

    def to_dict(self):
        return self.kwargs


class Model:
    labels = None

    @classmethod
    def from_pretrained(cls, path, **kwargs):
        model = cls()
        model.path = path
        return model

    def save_pretrained(self, path):
        Path(path).mkdir(parents=True)
        (Path(path) / "model.txt").write_text("fake")

    def predict(self, texts, use_labels=False):
        return [0 if "outage" in text else 1 for text in texts]

    def encode(self, texts, show_progress_bar=False):
        import numpy as np

        # one feature that counts severity words, so an ordered head can
        # recover the level from it deterministically
        return np.array(
            [[float(sum(word in text for word in ("slow", "down")))] for text in texts]
        )


class Trainer:
    def __init__(self, **kwargs):
        self.kwargs = kwargs

    def train_embeddings(self, *args, **kwargs):
        self.embedding_args = args
        output_dir = Path(kwargs["args"].kwargs["output_dir"])
        checkpoint = output_dir / "checkpoint-1"
        checkpoint.mkdir(parents=True)
        (checkpoint / "optimizer.pt").write_text("training only")

    def train_classifier(self, *args, **kwargs):
        self.classifier_args = args
        return None


def install_fake_setfit(monkeypatch):
    setfit = types.ModuleType("setfit")
    setfit.__version__ = "1.1.3"
    setfit.SetFitModel = Model
    setfit.Trainer = Trainer
    setfit.TrainingArguments = TrainingArguments
    datasets = types.ModuleType("datasets")
    datasets.Dataset = Dataset
    monkeypatch.setitem(sys.modules, "setfit", setfit)
    monkeypatch.setitem(sys.modules, "datasets", datasets)


def test_setfit_trains_and_predicts_each_field(tmp_path, monkeypatch):
    install_fake_setfit(monkeypatch)
    spec = make_spec()
    config = spec.candidates.setdefault(
        "bge-small", {"type": "setfit", "model": "BAAI/bge-small-en-v1.5"}
    )
    # Re-parse so the inserted dictionary becomes the discriminated config.
    spec = make_spec(candidates={"bge-small": config})
    rows = [
        {"input": {"title": "routine", "body": "question"}, "output": "normal"},
        {"input": {"title": "outage", "body": "server down"}, "output": "urgent"},
    ]
    metadata = train_setfit(spec, spec.candidates["bge-small"], rows, rows, tmp_path)
    assert metadata["setfit_version"] == "1.1.3"
    assert metadata["field_training"]["score"]["embedding_train_rows"] == 0
    assert (
        metadata["field_training"]["score"]["embedding_status"]
        == "skipped-no-positive-pair"
    )
    assert metadata["field_training"]["score"]["classifier_train_rows"] == 2
    assert metadata["field_training"]["score"]["resolved_args"]["num_iterations"] == 20
    assert metadata["field_training"]["score"]["resolved_args"]["save_strategy"] == "no"
    predictions = predict_setfit(
        tmp_path,
        spec,
        [rows[0]["input"], rows[1]["input"]],
    )
    assert predictions == ["normal", "urgent"]


def test_setfit_ordinal_head_gets_a_dev_selected_decoder(tmp_path, monkeypatch):
    install_fake_setfit(monkeypatch)
    spec = make_spec(
        output={"type": "int", "range": [0, 2]},
        candidates={"bge-small": {"type": "setfit", "model": "BAAI/bge-small-en-v1.5"}},
    )
    bodies = {0: "calm question", 1: "slow response", 2: "slow down failure"}
    rows = [
        {"input": {"title": f"case {index}", "body": bodies[level]}, "output": level}
        for level in bodies
        for index in range(8)
    ]

    metadata = train_setfit(spec, spec.candidates["bge-small"], rows, rows, tmp_path)

    training = metadata["field_training"]["score"]
    assert training["objective"] == "ordinal"
    assert training["decode"] in ("argmax", "median", "within_one")
    assert training["dev_decode_comparison"]["selected"] == training["decode"]
    assert training["dev_decode_comparison"]["metric"] == "within_one"

    # aggregate head diagnostics in the record; per-row evidence local-only
    diagnostics = training["head_diagnostics"]
    assert diagnostics["rows"] == len(rows)
    assert diagnostics["boundaries"][0]["boundary"] == 0
    local = json.loads((tmp_path / "score" / "dev_distributions.local.json").read_text())
    assert local["score"]["levels"] == [0, 1, 2]
    assert local["score"]["rows"][0]["reference"] == 0
    assert len(local["score"]["rows"]) == len(rows)

    # the persisted head carries the selected decoder and predictions round-trip
    predictions = predict_setfit(
        tmp_path,
        spec,
        [{"title": "new", "body": bodies[0]}, {"title": "new", "body": bodies[2]}],
    )
    assert predictions == [0, 2]


def test_setfit_ordinal_pinned_decoder_skips_the_comparison(tmp_path, monkeypatch):
    install_fake_setfit(monkeypatch)
    spec = make_spec(
        output={"type": "int", "range": [0, 2]},
        candidates={
            "bge-small": {
                "type": "setfit",
                "model": "BAAI/bge-small-en-v1.5",
                "decode": "median",
            }
        },
    )
    bodies = {0: "calm question", 1: "slow response", 2: "slow down failure"}
    rows = [
        {"input": {"title": f"case {index}", "body": bodies[level]}, "output": level}
        for level in bodies
        for index in range(8)
    ]

    metadata = train_setfit(spec, spec.candidates["bge-small"], rows, [], tmp_path)

    training = metadata["field_training"]["score"]
    assert training["decode"] == "median"
    assert training["dev_decode_comparison"] is None


def test_setfit_default_embeds_every_training_row(tmp_path, monkeypatch):
    """The few-shot per-class cap is opt-in: by default the contrastive phase
    sees the whole train split, with the pair budget bounding compute."""
    install_fake_setfit(monkeypatch)
    spec = make_spec(
        candidates={"bge-small": {"type": "setfit", "model": "BAAI/bge-small-en-v1.5"}}
    )
    rows = [
        {
            "input": {"title": f"ticket {index}", "body": "body"},
            "output": "urgent" if index % 2 else "normal",
        }
        for index in range(24)
    ]

    metadata = train_setfit(spec, spec.candidates["bge-small"], rows, rows, tmp_path)

    training = metadata["field_training"]["score"]
    assert training["embedding_train_rows"] == 24
    assert training["embedding_eval_rows"] == 24
    assert metadata["embedding_samples_per_class"] is None


def test_pair_budget_backs_off_iterations_as_embedding_rows_grow(tmp_path, monkeypatch):
    install_fake_setfit(monkeypatch)
    from smallbatch.setfit_candidate import _resolved_args
    from smallbatch.spec import SetFitCandidateSpec

    config = SetFitCandidateSpec(type="setfit", model="m")
    assert _resolved_args(config, tmp_path, 8).kwargs["num_iterations"] == 20
    assert _resolved_args(config, tmp_path, 512).kwargs["num_iterations"] == 4
    assert _resolved_args(config, tmp_path, 4096).kwargs["num_iterations"] == 1


def test_setfit_bounds_embedding_rows_but_trains_head_on_all_rows(tmp_path, monkeypatch):
    install_fake_setfit(monkeypatch)
    spec = make_spec(
        candidates={
            "bge-small": {
                "type": "setfit",
                "model": "BAAI/bge-small-en-v1.5",
                "embedding_samples_per_class": 3,
                "training_args": {"num_iterations": 2, "seed": 17},
            }
        }
    )
    rows = [
        {
            "input": {"title": f"ticket {index}", "body": "body"},
            "output": "urgent" if index % 2 else "normal",
        }
        for index in range(24)
    ]

    metadata = train_setfit(spec, spec.candidates["bge-small"], rows, rows, tmp_path)

    training = metadata["field_training"]["score"]
    assert training["embedding_train_rows"] == 6
    assert training["embedding_eval_rows"] == 6
    assert training["classifier_train_rows"] == 24
    assert training["resolved_args"]["num_iterations"] == 2
    assert training["resolved_args"]["save_strategy"] == "no"
    assert not (tmp_path / "score" / "checkpoints").exists()
