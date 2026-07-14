from __future__ import annotations

import io
import json
import urllib.request

import pytest

from smallbatch.spec import TeacherSpec
from smallbatch.teacher import make_teacher
from smallbatch.teacher.openai_compat import OpenAICompatTeacher


def _response(payload: dict):
    class FakeResponse(io.BytesIO):
        def __enter__(self):
            return self

        def __exit__(self, *args):
            return False

    return FakeResponse(json.dumps(payload).encode())


def test_factory_requires_base_url():
    with pytest.raises(ValueError, match="base_url"):
        make_teacher(TeacherSpec(backend="openai-compatible", model="m"))


def test_complete_returns_message_content(monkeypatch):
    payload = {
        "choices": [
            {"message": {"content": '[{"id": 0, "output": 3}]'}, "finish_reason": "stop"}
        ]
    }
    monkeypatch.setattr(
        urllib.request, "urlopen", lambda req, timeout: _response(payload)
    )
    teacher = OpenAICompatTeacher("http://127.0.0.1:1/v1", "m", retries=0)
    assert teacher.complete("p") == '[{"id": 0, "output": 3}]'


def test_null_content_is_a_failed_attempt_not_none(monkeypatch):
    """A reasoning model that truncates before its final channel returns
    content: null; the backend must fail the attempt instead of returning
    None into the labeling parser."""
    calls = []
    payload = {
        "choices": [{"message": {"content": None}, "finish_reason": "length"}]
    }

    def fake_urlopen(req, timeout):
        calls.append(req)
        return _response(payload)

    monkeypatch.setattr(urllib.request, "urlopen", fake_urlopen)
    monkeypatch.setattr("time.sleep", lambda seconds: None)
    teacher = OpenAICompatTeacher("http://127.0.0.1:1/v1", "m", retries=1)
    with pytest.raises(RuntimeError, match="no text content.*length"):
        teacher.complete("p")
    assert len(calls) == 2  # the null response was retried before failing


def test_empty_content_is_a_failed_attempt(monkeypatch):
    payload = {"choices": [{"message": {"content": "  "}, "finish_reason": "stop"}]}
    monkeypatch.setattr(
        urllib.request, "urlopen", lambda req, timeout: _response(payload)
    )
    teacher = OpenAICompatTeacher("http://127.0.0.1:1/v1", "m", retries=0)
    with pytest.raises(RuntimeError, match="no text content"):
        teacher.complete("p")
