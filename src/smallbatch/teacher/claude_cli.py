"""Teacher backed by the Claude Code CLI's headless mode (`claude -p`).

Requires the `claude` CLI installed and logged in. Before using any provider
as a labeling teacher, confirm your use fits its terms — see
docs/responsible-use.md."""

from __future__ import annotations

import os
import subprocess
import time


class ClaudeCLITeacher:
    def __init__(self, model: str = "sonnet", timeout: int = 600, retries: int = 2):
        self.model = model
        self.timeout = timeout
        self.retries = retries

    def _env(self) -> dict[str, str]:
        # isolate the nested invocation: CLAUDE* vars inherited from a parent
        # Claude Code session confuse a child `claude -p`, and an inherited
        # ANTHROPIC_API_KEY silently reroutes billing away from the account
        # the user logged the CLI into. Strip both so the call behaves exactly
        # like running `claude -p` in a fresh shell.
        return {
            k: v
            for k, v in os.environ.items()
            if k != "ANTHROPIC_API_KEY" and not k.startswith("CLAUDE")
        }

    def complete(self, prompt: str) -> str:
        last_err: Exception | None = None
        for attempt in range(self.retries + 1):
            try:
                res = subprocess.run(
                    ["claude", "-p", "--model", self.model, "--output-format", "text"],
                    input=prompt,
                    capture_output=True,
                    text=True,
                    timeout=self.timeout,
                    env=self._env(),
                )
                if res.returncode == 0 and res.stdout.strip():
                    return res.stdout
                last_err = RuntimeError(
                    f"claude CLI exit {res.returncode}: {res.stderr.strip()[:500]}"
                )
            except subprocess.TimeoutExpired as e:
                last_err = e
            if attempt < self.retries:
                time.sleep(5 * (attempt + 1))
        raise RuntimeError(f"claude-cli teacher failed after {self.retries + 1} tries: {last_err}")
