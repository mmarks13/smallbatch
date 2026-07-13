from conftest import make_spec
from smallbatch import prompts
from smallbatch.metrics import compare
from smallbatch.spec import validate_output


def structured_spec():
    return make_spec(
        output={
            "priority": {"labels": ["urgent", "normal"]},
            "reason": {"labels": ["outage", "question"]},
            "score": {"range": [0, 4]},
        }
    )


def test_partial_structured_output_is_invalid():
    spec = structured_spec()
    parsed = prompts.parse_output(spec, "priority: urgent\nreason: outage")
    assert prompts.incomplete_fields(spec, parsed) == ["score"]


def test_structured_contract_rejects_extra_and_missing_fields():
    spec = structured_spec()
    complete = {"priority": "urgent", "reason": "outage", "score": 4}
    assert validate_output(spec, complete) == complete
    for invalid in (
        {"priority": "urgent", "reason": "outage"},
        {**complete, "extra": "x"},
    ):
        try:
            validate_output(spec, invalid)
        except ValueError:
            pass
        else:
            raise AssertionError("invalid structured output was accepted")


def test_structured_metrics_do_not_hide_worst_field():
    spec = structured_spec()
    refs = [
        {"priority": "urgent", "reason": "outage", "score": 4},
        {"priority": "normal", "reason": "question", "score": 1},
    ]
    preds = [
        {"priority": "urgent", "reason": "question", "score": 3},
        {"priority": "normal", "reason": "outage", "score": 4},
    ]
    metrics = compare(spec, preds, refs)
    assert metrics["fields"]["priority"]["decision_agreement"] == 1
    assert metrics["fields"]["reason"]["decision_agreement"] == 0
    assert metrics["fields"]["score"]["max_absolute_error"] == 3
