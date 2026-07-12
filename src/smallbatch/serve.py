"""`smallbatch serve <fn>`: a local HTTP endpoint for a compiled function.

Runs llama.cpp's llama-server on the exported GGUF (grammar enforced per
request) behind a tiny stdlib front end that builds the student prompt,
validates the output against the contract, and returns JSON:

    POST /call {"title": "..."}  ->  {"output": ..., "raw": " 7"}

The request/response logic is pure (`handle_call`) so it unit-tests without
any server; only the wiring at the bottom touches sockets/subprocesses.
"""

from __future__ import annotations

import json
import os
import shutil
import subprocess
import sys
import time
import urllib.error
import urllib.request
from pathlib import Path
from typing import Any, Callable

from . import artifacts, prompts
from .spec import FunctionSpec, load_spec


def handle_call(
    spec: FunctionSpec, item: Any, complete_fn: Callable[[str], str]
) -> tuple[int, dict]:
    """Validate input, generate via complete_fn, parse against the contract.
    Returns (http_status, response_payload)."""
    if not isinstance(item, dict):
        return 400, {"error": "body must be a JSON object of input fields"}
    missing = [k for k in spec.input_schema if k not in item]
    if missing:
        return 400, {"error": f"missing input fields: {missing}"}
    try:
        raw = complete_fn(prompts.student_prompt(spec, item))
    except Exception as e:  # noqa: BLE001 - surface backend failures as 502
        return 502, {"error": f"generation backend failed: {e}"}
    output = prompts.parse_output(spec, raw)
    bad = prompts.incomplete_fields(spec, output)
    if bad:  # a partially-parsed structured output is a violation, not a 200
        return 422, {
            "error": f"output failed contract validation (fields: {', '.join(bad)})",
            "raw": raw,
        }
    return 200, {"output": output, "raw": raw}


def handle_call_direct(
    spec: FunctionSpec, item: Any, predict_fn: Callable[[dict], Any]
) -> tuple[int, dict]:
    """handle_call for backends that map an item straight to an output (the
    tfidf candidate) — same input checks, same contract enforcement."""
    if not isinstance(item, dict):
        return 400, {"error": "body must be a JSON object of input fields"}
    missing = [k for k in spec.input_schema if k not in item]
    if missing:
        return 400, {"error": f"missing input fields: {missing}"}
    try:
        output = predict_fn(item)
    except Exception as e:  # noqa: BLE001 - surface backend failures as 502
        return 502, {"error": f"prediction backend failed: {e}"}
    bad = prompts.incomplete_fields(spec, output)
    if bad:
        return 422, {"error": f"output failed contract validation (fields: {', '.join(bad)})"}
    return 200, {"output": output}


def find_export(version_dir: Path, name: str) -> tuple[Path, Path]:
    """(gguf, grammar) from the version's export bundle."""
    export_dir = version_dir / "export"
    ggufs = sorted(p for p in export_dir.glob("*.gguf") if ".lora." not in p.name)
    grammar = export_dir / f"{name}.gbnf"
    if not ggufs or not grammar.exists():
        raise FileNotFoundError(
            f"no export bundle under {export_dir} — run `smallbatch export {name}` first"
        )
    return ggufs[0], grammar


def find_llama_server(explicit: str | None = None) -> Path:
    candidates = [
        explicit,
        shutil.which("llama-server"),
        os.path.join(os.environ.get("LLAMA_CPP_DIR", ""), "build", "bin", "llama-server"),
    ]
    for c in candidates:
        if c and Path(c).exists():
            return Path(c)
    raise FileNotFoundError(
        "llama-server not found: put it on PATH, set LLAMA_CPP_DIR, or pass --llama-server"
    )


def _llama_complete(backend_url: str, grammar: str, max_new: int) -> Callable[[str], str]:
    def complete(prompt: str) -> str:
        req = urllib.request.Request(
            f"{backend_url}/completion",
            data=json.dumps({
                "prompt": prompt,
                "n_predict": max_new,
                "temperature": 0,
                "grammar": grammar,
            }).encode(),
            headers={"Content-Type": "application/json"},
        )
        with urllib.request.urlopen(req, timeout=120) as resp:
            return json.loads(resp.read())["content"]

    return complete


def serve(
    name: str,
    artifacts_root: str | Path = artifacts.DEFAULT_ROOT,
    version: str | None = None,
    port: int = 8080,
    llama_server: str | None = None,
    allow_failed: bool = False,
    candidate: str | None = None,
) -> int:
    root = Path(artifacts_root)
    version_dir = artifacts.resolve_version(
        root, name, version, allow_failed, candidate=candidate
    )
    manifest = artifacts.read_manifest(version_dir)
    rec = artifacts.candidate_record(manifest, candidate)
    spec = load_spec(version_dir / "spec.yaml")

    if rec["backend"] == "tfidf":
        # a tfidf winner serves straight from the Python runtime: CPU-only,
        # no GGUF export, no llama.cpp
        from . import candidates as cand

        model_dir = version_dir / (rec.get("artifact_path") or cand.TFIDF_DIR)
        predict = lambda item: cand.predict_tfidf(model_dir, spec, [item])[0]  # noqa: E731
        return _http_serve(
            spec, port, lambda item: handle_call_direct(spec, item, predict)
        )

    gguf, grammar_path = find_export(version_dir, spec.name)
    server_bin = find_llama_server(llama_server)
    grammar = grammar_path.read_text()
    max_new = prompts.completion_budget(spec) + 8
    backend_port = port + 1
    backend_url = f"http://127.0.0.1:{backend_port}"

    print(f"starting llama-server on :{backend_port} ({gguf.name})")
    proc = subprocess.Popen(
        [str(server_bin), "-m", str(gguf), "--port", str(backend_port),
         "--host", "127.0.0.1", "--log-disable"],
        stdout=subprocess.DEVNULL,
        stderr=subprocess.DEVNULL,
    )
    try:
        for _ in range(120):  # wait for the model to load
            try:
                urllib.request.urlopen(f"{backend_url}/health", timeout=2)
                break
            except (urllib.error.URLError, ConnectionError):
                if proc.poll() is not None:
                    print("llama-server exited during startup", file=sys.stderr)
                    return 1
                time.sleep(1)
        complete_fn = _llama_complete(backend_url, grammar, max_new)
        return _http_serve(
            spec, port, lambda item: handle_call(spec, item, complete_fn)
        )
    except KeyboardInterrupt:
        return 0
    finally:
        proc.terminate()
        try:
            proc.wait(timeout=10)
        except subprocess.TimeoutExpired:
            proc.kill()


def _http_serve(spec: FunctionSpec, port: int, call: Callable[[Any], tuple[int, dict]]) -> int:
    """The stdlib front end shared by both backends: `call` maps a parsed
    JSON item to (status, payload)."""
    from http.server import BaseHTTPRequestHandler, HTTPServer

    class Handler(BaseHTTPRequestHandler):
        def _send(self, status: int, payload: dict) -> None:
            body = json.dumps(payload, ensure_ascii=False).encode()
            self.send_response(status)
            self.send_header("Content-Type", "application/json")
            self.send_header("Content-Length", str(len(body)))
            self.end_headers()
            self.wfile.write(body)

        def do_GET(self):  # noqa: N802 - http.server API
            if self.path == "/health":
                self._send(200, {"status": "ok", "function": spec.name})
            else:
                self._send(200, {
                    "function": spec.name,
                    "description": spec.description.strip(),
                    "input_fields": list(spec.input_schema),
                    "usage": "POST /call with a JSON object of the input fields",
                })

        def do_POST(self):  # noqa: N802 - http.server API
            if self.path != "/call":
                self._send(404, {"error": "POST /call"})
                return
            try:
                length = int(self.headers.get("Content-Length", 0))
                item = json.loads(self.rfile.read(length) or b"{}")
            except (ValueError, json.JSONDecodeError) as e:
                self._send(400, {"error": f"bad JSON body: {e}"})
                return
            self._send(*call(item))

        def log_message(self, *args):  # quiet
            pass

    print(f"serving '{spec.name}' on http://127.0.0.1:{port}")
    print(f"  curl -s http://127.0.0.1:{port}/call -d "
          f"'{json.dumps({k: '...' for k in spec.input_schema})}'")
    try:
        HTTPServer(("127.0.0.1", port), Handler).serve_forever()
    except KeyboardInterrupt:
        pass
    return 0
