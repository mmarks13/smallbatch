"""Teacher backed by the Codex CLI's non-interactive mode (`codex exec`).

Requires the `codex` CLI installed and logged in. Invocations are ephemeral,
run from the system temporary directory, ignore user config, and use a
read-only sandbox so a labeling prompt cannot mutate the caller's repository.
Before using any provider as a labeling teacher, confirm your use fits its
terms — see docs/responsible-use.md.
"""

from __future__ import annotations

import os
import re
import subprocess
import tempfile
import time


class CodexCLITeacher:
    def __init__(
        self,
        model: str,
        timeout: int = 600,
        retries: int = 2,
        binary: str = "codex",
        reasoning_effort: str | None = None,
    ):
        self.model = model
        self.timeout = timeout
        self.retries = retries
        self.binary = binary
        self.reasoning_effort = reasoning_effort
        self.attempts = 0
        self.successful_calls = 0
        self.reported_tokens = 0

    def _env(self) -> dict[str, str]:
        # Use the account authenticated by `codex login`, not an inherited API
        # key with potentially different billing. Transient variables injected
        # by a parent Codex session should not define the child invocation.
        transient = {
            "CODEX_CI",
            "CODEX_PERMISSION_PROFILE",
            "CODEX_SANDBOX_NETWORK_DISABLED",
            "CODEX_THREAD_ID",
            "OPENAI_API_KEY",
        }
        return {k: v for k, v in os.environ.items() if k not in transient}

    def _command(self) -> list[str]:
        cmd = [
            self.binary,
            "exec",
            "--ephemeral",
            "--ignore-user-config",
            "--sandbox",
            "read-only",
            "--skip-git-repo-check",
            "--color",
            "never",
            "--model",
            self.model,
        ]
        if self.reasoning_effort:
            cmd += ["-c", f'model_reasoning_effort="{self.reasoning_effort}"']
        return [*cmd, "-"]

    def complete(self, prompt: str) -> str:
        last_err: Exception | None = None
        for attempt in range(self.retries + 1):
            self.attempts += 1
            try:
                res = subprocess.run(
                    self._command(),
                    input=prompt,
                    capture_output=True,
                    text=True,
                    timeout=self.timeout,
                    cwd=tempfile.gettempdir(),
                    env=self._env(),
                )
                if res.returncode == 0 and res.stdout.strip():
                    self.successful_calls += 1
                    matches = re.findall(
                        r"tokens used\s*\n\s*([\d,]+)", res.stderr, re.IGNORECASE
                    )
                    tokens = int(matches[-1].replace(",", "")) if matches else 0
                    self.reported_tokens += tokens
                    suffix = f", {tokens:,} tokens" if tokens else ""
                    print(
                        f"codex-cli call {self.successful_calls} complete{suffix}",
                        flush=True,
                    )
                    return res.stdout.strip()
                last_err = RuntimeError(
                    f"codex CLI exit {res.returncode}: {res.stderr.strip()[:500]}"
                )
            except subprocess.TimeoutExpired as e:
                last_err = e
            if attempt < self.retries:
                time.sleep(5 * (attempt + 1))
        raise RuntimeError(
            f"codex-cli teacher failed after {self.retries + 1} tries: {last_err}"
        )

    @property
    def usage(self) -> dict[str, int]:
        return {
            "attempts": self.attempts,
            "successful_calls": self.successful_calls,
            "reported_tokens": self.reported_tokens,
        }
