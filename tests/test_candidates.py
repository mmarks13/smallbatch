import pytest

from conftest import make_spec
from smallbatch.candidates import predict_tfidf, train_tfidf


def rows():
    return [
        {"input": {"title": "outage", "body": "server down production"}, "output": "urgent"},
        {"input": {"title": "broken", "body": "service unavailable"}, "output": "urgent"},
        {"input": {"title": "question", "body": "how do I update profile"}, "output": "normal"},
        {"input": {"title": "help", "body": "routine account question"}, "output": "normal"},
    ]


def test_tfidf_train_save_and_predict_without_torch(tmp_path):
    spec = make_spec()
    metadata = train_tfidf(spec, rows(), tmp_path)
    assert metadata["format"] == "skops"
    predictions = predict_tfidf(
        tmp_path,
        spec,
        [
            {"title": "outage now", "body": "production server down"},
            {"title": "question", "body": "account help"},
        ],
    )
    assert predictions == ["urgent", "normal"]


def test_tfidf_requires_two_observed_classes(tmp_path):
    spec = make_spec()
    with pytest.raises(ValueError, match="single observed class"):
        train_tfidf(spec, rows()[:2], tmp_path)


def test_structured_trains_one_pipeline_per_field(tmp_path):
    spec = make_spec(
        output={"priority": {"labels": ["urgent", "normal"]}, "score": {"range": [0, 1]}}
    )
    structured = [
        {**row, "output": {"priority": row["output"], "score": index % 2}}
        for index, row in enumerate(rows())
    ]
    train_tfidf(spec, structured, tmp_path)
    output = predict_tfidf(tmp_path, spec, [rows()[0]["input"]])[0]
    assert set(output) == {"priority", "score"}
