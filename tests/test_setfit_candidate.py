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


class Trainer:
    def __init__(self, **kwargs):
        self.kwargs = kwargs

    def train(self):
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
    predictions = predict_setfit(
        tmp_path,
        spec,
        [rows[0]["input"], rows[1]["input"]],
    )
    assert predictions == ["normal", "urgent"]
