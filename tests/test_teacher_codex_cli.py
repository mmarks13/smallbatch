from __future__ import annotations

import os
import subprocess
from types import SimpleNamespace

import pytest

from smallbatch.spec import TeacherSpec
from smallbatch.teacher import make_teacher
from smallbatch.teacher.codex_cli import CodexCLITeacher


def test_factory_builds_codex_teacher():
    teacher = make_teacher(TeacherSpec(backend="codex-cli", model="gpt-test"))
    assert isinstance(teacher, CodexCLITeacher)
    assert teacher.model == "gpt-test"


def test_complete_invokes_ephemeral_read_only_codex(monkeypatch):
    calls = []

    def fake_run(command, **kwargs):
        calls.append((command, kwargs))
        return SimpleNamespace(returncode=0, stdout='[{"score": 7}]\n', stderr="")

    monkeypatch.setattr(subprocess, "run", fake_run)
    monkeypatch.setenv("OPENAI_API_KEY", "must-not-leak")
    monkeypatch.setenv("CODEX_THREAD_ID", "parent-thread")
    teacher = CodexCLITeacher(model="gpt-test", retries=0)

    assert teacher.complete("label this") == '[{"score": 7}]'
    command, kwargs = calls[0]
    assert command == [
        "codex",
        "exec",
        "--ephemeral",
        "--ignore-user-config",
        "--sandbox",
        "read-only",
        "--skip-git-repo-check",
        "--color",
        "never",
        "--model",
        "gpt-test",
        "-",
    ]
    assert kwargs["input"] == "label this"
    assert kwargs["capture_output"] is True
    assert kwargs["env"].get("OPENAI_API_KEY") is None
    assert kwargs["env"].get("CODEX_THREAD_ID") is None


def test_complete_retries_then_succeeds(monkeypatch):
    replies = iter(
        [
            SimpleNamespace(returncode=1, stdout="", stderr="temporary failure"),
            SimpleNamespace(returncode=0, stdout="[]", stderr=""),
        ]
    )
    monkeypatch.setattr(subprocess, "run", lambda *args, **kwargs: next(replies))
    monkeypatch.setattr("time.sleep", lambda _: None)
    assert CodexCLITeacher("gpt-test", retries=1).complete("x") == "[]"


def test_complete_reports_timeout(monkeypatch):
    def timeout(*args, **kwargs):
        raise subprocess.TimeoutExpired(args[0], 1)

    monkeypatch.setattr(subprocess, "run", timeout)
    with pytest.raises(RuntimeError, match="codex-cli teacher failed"):
        CodexCLITeacher("gpt-test", retries=0).complete("x")


def test_codex_environment_keeps_unrelated_values(monkeypatch):
    monkeypatch.setenv("SMALLBATCH_TEST_VALUE", "kept")
    env = CodexCLITeacher("gpt-test")._env()
    assert env["SMALLBATCH_TEST_VALUE"] == "kept"
    assert os.environ["SMALLBATCH_TEST_VALUE"] == "kept"


def test_usage_parses_codex_token_report(monkeypatch):
    monkeypatch.setattr(
        subprocess,
        "run",
        lambda *args, **kwargs: SimpleNamespace(
            returncode=0,
            stdout="[]",
            stderr="tokens used\n2,115\n",
        ),
    )
    teacher = CodexCLITeacher("gpt-test", retries=0)
    teacher.complete("x")
    assert teacher.usage == {
        "attempts": 1,
        "successful_calls": 1,
        "reported_tokens": 2115,
    }
