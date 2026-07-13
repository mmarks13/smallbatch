"""Teacher backends: anything that can complete a prompt can teach."""

from __future__ import annotations

from typing import Protocol

from ..spec import TeacherSpec


class Teacher(Protocol):
    def complete(self, prompt: str) -> str: ...


def make_teacher(cfg: TeacherSpec) -> Teacher:
    if cfg.backend == "claude-cli":
        from .claude_cli import ClaudeCLITeacher

        return ClaudeCLITeacher(model=cfg.model)
    if cfg.backend == "codex-cli":
        from .codex_cli import CodexCLITeacher

        return CodexCLITeacher(model=cfg.model, reasoning_effort=cfg.reasoning_effort)
    if cfg.backend == "openai-compatible":
        from .openai_compat import OpenAICompatTeacher

        if not cfg.base_url:
            raise ValueError("teacher.base_url is required for openai-compatible")
        return OpenAICompatTeacher(
            base_url=cfg.base_url, model=cfg.model, api_key_env=cfg.api_key_env
        )
    raise ValueError(f"unknown teacher backend: {cfg.backend}")
