"""Teacher backed by any /chat/completions endpoint (OpenAI, Gemini, Ollama,
vLLM, LM Studio...). Stdlib-only on purpose."""

from __future__ import annotations

import json
import os
import time
import urllib.request


class OpenAICompatTeacher:
    def __init__(
        self,
        base_url: str,
        model: str,
        api_key_env: str = "OPENAI_API_KEY",
        timeout: int = 300,
        retries: int = 2,
    ):
        self.url = base_url.rstrip("/") + "/chat/completions"
        self.model = model
        self.api_key = os.environ.get(api_key_env, "")
        self.timeout = timeout
        self.retries = retries

    def complete(self, prompt: str) -> str:
        body = json.dumps(
            {
                "model": self.model,
                "messages": [{"role": "user", "content": prompt}],
                "temperature": 0,
            }
        ).encode()
        headers = {"Content-Type": "application/json"}
        if self.api_key:
            headers["Authorization"] = f"Bearer {self.api_key}"
        last_err: Exception | None = None
        for attempt in range(self.retries + 1):
            try:
                req = urllib.request.Request(self.url, data=body, headers=headers)
                with urllib.request.urlopen(req, timeout=self.timeout) as resp:
                    data = json.loads(resp.read())
                return data["choices"][0]["message"]["content"]
            except Exception as e:  # noqa: BLE001 - retry any transport error
                last_err = e
                if attempt < self.retries:
                    time.sleep(5 * (attempt + 1))
        raise RuntimeError(f"openai-compatible teacher failed: {last_err}")
