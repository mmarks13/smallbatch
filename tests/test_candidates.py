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


def test_per_row_weight_shifts_the_enum_decision_boundary(tmp_path):
    """Rows from a passes=2 labeling run carry a `weight` (agreement-derived);
    the enum head's LogisticRegression must actually consume it as
    sample_weight, not just carry it as inert metadata."""
    ambiguous = {"title": "ambiguous", "body": "case text repeated"}
    other = [
        {"input": {"title": "outage", "body": "clear production outage"}, "output": "urgent"},
        {"input": {"title": "help", "body": "clear routine account question"}, "output": "normal"},
    ]

    def rows(urgent_weight):
        return [
            {"input": ambiguous, "output": "normal", "weight": 1.0},
            {"input": ambiguous, "output": "normal", "weight": 1.0},
            {"input": ambiguous, "output": "urgent", "weight": urgent_weight},
            *other,
        ]

    spec = make_spec()
    train_tfidf(spec, rows(urgent_weight=1.0), tmp_path)
    assert predict_tfidf(tmp_path, spec, [ambiguous])[0] == "normal"  # 2:1 majority

    train_tfidf(spec, rows(urgent_weight=20.0), tmp_path)
    assert predict_tfidf(tmp_path, spec, [ambiguous])[0] == "urgent"  # weight overrides it


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
