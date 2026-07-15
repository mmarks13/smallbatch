from __future__ import annotations

from smallbatch.spec import FunctionSpec


def make_spec(**overrides) -> FunctionSpec:
    value = {
        "name": "ticket-priority",
        "description": "Decide ticket priority.",
        "input_schema": {"title": "string", "body": "string"},
        "output": {"type": "enum", "labels": ["urgent", "normal"]},
        "prompt": "urgent for outages; normal otherwise",
        "candidates": {"tfidf": {"type": "tfidf"}},
    }
    value.update(overrides)
    return FunctionSpec(**value)


def imported_records(count: int = 30) -> list[dict]:
    return [
        {
            "input": {
                "title": f"ticket {index}",
                "body": "production outage server down" if index % 2 else "routine question account",
            },
            "output": "urgent" if index % 2 else "normal",
        }
        for index in range(count)
    ]
