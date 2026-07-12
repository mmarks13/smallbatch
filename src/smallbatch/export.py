"""Export a compiled function to a zero-PyTorch runtime artifact.

Produces, under `<version_dir>/export/`:
  <fn>.<quant>.gguf   merged base+adapter, converted and quantized (llama.cpp)
  <fn>.gbnf           a grammar generated from the spec's output contract, so
                      llama.cpp can only emit valid outputs
  Modelfile           an Ollama Modelfile wrapping the gguf

Requires a llama.cpp checkout (for convert_hf_to_gguf.py / llama-quantize):
pass --llama-cpp, or set LLAMA_CPP_DIR. The `gguf` pip package must be
importable by the current Python (`pip install gguf`).

The grammar file works with llama.cpp directly:
    llama-cli -m <fn>.q4_k_m.gguf --grammar-file <fn>.gbnf -p "<prompt>"
Ollama does not support grammar files; the Modelfile relies on the fine-tune
plus `parse_output`-style validation by the caller.
"""

from __future__ import annotations

import json
import os
import shutil
import subprocess
import sys
from dataclasses import dataclass
from pathlib import Path

from . import artifacts
from .spec import FunctionSpec, load_spec

QUANTS = ("f16", "q8_0", "q4_k_m")  # presets; llama-quantize knows the names
_MAX_INT_RANGE = 1000  # enumerating alternatives is exact; refuse absurd ranges


# --- grammar + modelfile (pure text, unit-tested) ---------------------------


def _gbnf_str(s: str) -> str:
    return '"' + s.replace("\\", "\\\\").replace('"', '\\"') + '"'


def _gbnf_values(field) -> str:
    """Alternation of a field's exact legal values."""
    if field.type == "int":
        lo, hi = field.range
        if hi - lo + 1 > _MAX_INT_RANGE:
            raise ValueError(
                f"output range {lo}..{hi} too large to enumerate in a grammar"
            )
    return " | ".join(_gbnf_str(str(v)) for v in field.values())


def gbnf_grammar(spec: FunctionSpec) -> str:
    """A GBNF grammar accepting exactly the completions the student was
    trained to emit (see prompts.student_completion): a leading space, then
    the bare value (scalar), fixed-order `name: value` lines (multi-field),
    with the free-text rationale line prefixed in rationale mode."""
    if spec.output.is_scalar:
        value = _gbnf_values(spec.output.scalar)
        if spec.train.rationale_distillation:
            return (
                f'root ::= " "? "reason: " reason "\\nscore: " value\n'
                f"reason ::= [^\\n]+\n"
                f"value ::= {value}\n"
            )
        return f'root ::= " "? value\nvalue ::= {value}\n'

    names = list(spec.output.fields)
    seq = ' "\\n" '.join(f'"{name}: " f{i}' for i, name in enumerate(names))
    rules = "".join(
        f"f{i} ::= {_gbnf_values(field)}\n"
        for i, field in enumerate(spec.output.fields.values())
    )
    if spec.train.rationale_distillation:
        return (
            f'root ::= " "? "rationale: " rationale "\\n" {seq}\n'
            f"rationale ::= [^\\n]+\n{rules}"
        )
    return f'root ::= " "? {seq}\n{rules}'


def bundle_readme(spec: FunctionSpec, manifest: dict, gguf_name: str) -> str:
    """README.md for the export bundle: what this artifact is and exactly how
    to call it outside Python."""
    from .prompts import completion_budget, output_instruction

    input_lines = "\\n".join(f"{k}: <{t}>" for k, t in spec.input_schema.items())
    example_prompt = f"[{spec.name}]\\n{input_lines}\\noutput:"
    gate = manifest.get("gate", {})
    adapter = (manifest.get("metrics") or {}).get("adapter") or {}
    ci = adapter.get("agreement_ci")
    ci_txt = f" (95% CI {ci[0]:.0%}-{ci[1]:.0%})" if ci else ""
    max_new = completion_budget(spec) + 8
    return f"""# {spec.name} — compiled function (GGUF bundle)

{spec.description.strip()}

This is a **narrow compiled function**, not a chat model: it answers exactly
one prompt shape with a constrained output — {output_instruction(spec)}.

- base model: `{manifest.get("base_model")}` (the adapter inherits its license)
- teacher: `{manifest.get("data", {}).get("teacher_model")}` via {manifest.get("data", {}).get("teacher_backend")}
- gate: **{"PASS" if gate.get("passed") else "FAIL"}** — agreement {adapter.get("agreement", 0):.1%}{ci_txt} on {adapter.get("n", "?")} held-out items
- files: `{gguf_name}` (weights), `{spec.name}.gbnf` (grammar), `Modelfile` (Ollama), `spec.yaml`, `manifest.json`, `report.md`

## Prompt format

```
{example_prompt}
```

(One `field: value` line per input field, then the literal line `output:`.)

## llama.cpp

```bash
llama-cli -m {gguf_name} --grammar-file {spec.name}.gbnf -n {max_new} --temp 0 \\
  -p "{example_prompt}"
```

Or serve it:

```bash
llama-server -m {gguf_name} --port 8080
curl -s http://localhost:8080/completion -d '{{
  "prompt": "{example_prompt}",
  "n_predict": {max_new}, "temperature": 0,
  "grammar": {json.dumps(gbnf_grammar(spec))}
}}'
```

## Ollama

```bash
ollama create {spec.name} -f Modelfile
ollama run {spec.name} "{example_prompt}"
```

Note: Ollama does not apply GBNF grammars; outputs are usually well-formed
(temperature 0, short `num_predict`) but validate them against the contract
in `spec.yaml`. `smallbatch serve` and llama.cpp enforce the grammar exactly.

## smallbatch

```bash
smallbatch run {spec.name} --json '{{...input fields...}}'
smallbatch serve {spec.name}   # local HTTP endpoint with output validation
```
"""


def modelfile(spec: FunctionSpec, gguf_name: str) -> str:
    from .prompts import completion_budget

    # headroom over the trained format (more in rationale mode: free text)
    max_new = completion_budget(spec) + (16 if spec.train.rationale_distillation else 8)
    return (
        f"# Ollama Modelfile for the compiled function '{spec.name}'\n"
        f"#   ollama create {spec.name} -f Modelfile\n"
        f'#   ollama run {spec.name} "[{spec.name}]\\n<input fields>\\noutput:"\n'
        f"# Prompts are raw (see smallbatch docs); the model expects the\n"
        f"# student prompt format and answers with only the output value.\n"
        f"FROM ./{gguf_name}\n"
        f'TEMPLATE """{{{{ .Prompt }}}}"""\n'
        f"PARAMETER temperature 0\n"
        f"PARAMETER num_predict {max_new}\n"
    )


# --- llama.cpp tool discovery ------------------------------------------------


@dataclass
class LlamaCpp:
    root: Path
    convert_hf: Path
    convert_lora: Path
    quantize: Path | None  # None is fine for --quant f16


_INSTALL_HINT = (
    "smallbatch export needs a llama.cpp checkout: pass --llama-cpp DIR or set "
    "LLAMA_CPP_DIR. Get one with:\n"
    "  git clone https://github.com/ggml-org/llama.cpp\n"
    "  cmake -B llama.cpp/build llama.cpp && cmake --build llama.cpp/build "
    "--target llama-quantize\n"
    "  pip install gguf   # used by the conversion scripts"
)


def find_llama_cpp(explicit: str | Path | None = None) -> LlamaCpp:
    root = Path(explicit or os.environ.get("LLAMA_CPP_DIR", "")).expanduser()
    if not root or not root.is_dir():
        raise FileNotFoundError(_INSTALL_HINT)
    convert_hf = root / "convert_hf_to_gguf.py"
    convert_lora = root / "convert_lora_to_gguf.py"
    if not convert_hf.exists():
        raise FileNotFoundError(f"{convert_hf} not found.\n{_INSTALL_HINT}")
    quantize = next(
        (
            p
            for p in (
                root / "build" / "bin" / "llama-quantize",
                root / "llama-quantize",
                Path(shutil.which("llama-quantize") or "/nonexistent"),
            )
            if p.exists()
        ),
        None,
    )
    return LlamaCpp(root, convert_hf, convert_lora, quantize)


def _run(argv: list[str | Path], what: str) -> None:
    # errors="replace": llama.cpp tools can emit non-UTF-8 bytes in progress
    # output, which must not crash the export
    res = subprocess.run(
        [str(a) for a in argv], capture_output=True, text=True, errors="replace"
    )
    if res.returncode != 0:
        tail = (res.stderr or res.stdout).strip()[-2000:]
        raise RuntimeError(f"{what} failed (exit {res.returncode}):\n{tail}")


# --- the export pipeline -----------------------------------------------------


@dataclass
class ExportResult:
    export_dir: Path
    gguf: Path
    grammar: Path
    modelfile: Path | None  # None for adapter-only exports
    quant: str


def export(
    name: str,
    artifacts_root: str | Path = artifacts.DEFAULT_ROOT,
    version: str | None = None,
    quant: str = "q4_k_m",
    adapter_only: bool = False,
    allow_failed: bool = False,
    llama_cpp: str | Path | None = None,
    keep_merged: bool = False,
) -> ExportResult:
    """Merge, convert, and quantize a compiled function into GGUF (+ grammar).

    `quant` is a llama-quantize type name (f16 skips quantization).
    `adapter_only` converts just the LoRA to GGUF for `llama-server --lora`
    over a shared base (no Modelfile; merged is the reliable default).
    """
    if quant not in QUANTS:
        raise ValueError(f"quant must be one of {QUANTS}")
    tools = find_llama_cpp(llama_cpp)
    if quant != "f16" and tools.quantize is None:
        raise FileNotFoundError(
            f"llama-quantize not found under {tools.root} (needed for "
            f"--quant {quant}; --quant f16 works without it).\n{_INSTALL_HINT}"
        )

    root = Path(artifacts_root)
    # GGUF export covers the LoRA candidate only, so usability is checked for
    # THAT candidate — a tfidf winner neither blocks exporting a passing
    # adapter nor lets a failed, unaccepted adapter slip out
    version_dir = artifacts.resolve_version(
        root, name, version, allow_failed, candidate="lora"
    )
    spec = load_spec(version_dir / "spec.yaml")
    manifest = artifacts.read_manifest(version_dir)
    if artifacts.candidate_record(manifest, "lora") is None:
        raise ValueError(
            f"{version_dir} has no LoRA candidate — GGUF export supports the "
            "LoRA backend only (a tfidf winner is served/run directly: "
            "`smallbatch serve` / `smallbatch run`)"
        )
    adapter_dir = version_dir / "adapter"

    out = version_dir / "export"
    out.mkdir(exist_ok=True)
    grammar_path = out / f"{spec.name}.gbnf"
    grammar_path.write_text(gbnf_grammar(spec))

    if adapter_only:
        gguf_path = out / f"{spec.name}.lora.{quant}.gguf"
        _run(
            [sys.executable, tools.convert_lora, adapter_dir,
             "--outfile", gguf_path, "--outtype", "f16"],
            "convert_lora_to_gguf.py",
        )
        print(f"exported adapter-only gguf: {gguf_path}")
        return ExportResult(out, gguf_path, grammar_path, None, quant)

    # 1. merge adapter into the base on CPU (fp32 to match fp32-trained
    #    adapters exactly; conversion emits f16 regardless)
    merged_dir = out / "merged"
    if not (merged_dir / "config.json").exists():
        import torch  # noqa: F401 - heavy import deferred to here on purpose
        from peft import PeftModel
        from transformers import AutoModelForCausalLM, AutoTokenizer

        print(f"merging {manifest['base_model']} + {adapter_dir} (cpu)")
        base = AutoModelForCausalLM.from_pretrained(
            manifest["base_model"], device_map="cpu"
        )
        merged = PeftModel.from_pretrained(base, str(adapter_dir)).merge_and_unload()
        merged.save_pretrained(merged_dir)
        AutoTokenizer.from_pretrained(str(adapter_dir)).save_pretrained(merged_dir)
        del base, merged

    # 2. convert to f16 gguf
    f16_path = out / f"{spec.name}.f16.gguf"
    if not f16_path.exists():
        print(f"converting -> {f16_path}")
        _run(
            [sys.executable, tools.convert_hf, merged_dir,
             "--outfile", f16_path, "--outtype", "f16"],
            "convert_hf_to_gguf.py",
        )

    # 3. quantize
    if quant == "f16":
        gguf_path = f16_path
    else:
        gguf_path = out / f"{spec.name}.{quant}.gguf"
        print(f"quantizing -> {gguf_path}")
        _run([tools.quantize, f16_path, gguf_path, quant.upper()], "llama-quantize")
        f16_path.unlink()

    if not keep_merged:
        shutil.rmtree(merged_dir, ignore_errors=True)

    modelfile_path = out / "Modelfile"
    modelfile_path.write_text(modelfile(spec, gguf_path.name))

    # make the bundle self-describing: spec + manifest + report + usage README
    (out / "README.md").write_text(bundle_readme(spec, manifest, gguf_path.name))
    for extra in ("spec.yaml", "manifest.json", "report.md"):
        src = version_dir / extra
        if src.exists():
            shutil.copy(src, out / extra)

    size_mb = gguf_path.stat().st_size / 1e6
    print(
        f"exported {gguf_path} ({size_mb:.0f}MB)\n"
        f"  ollama:    ollama create {spec.name} -f {modelfile_path}\n"
        f"  llama.cpp: llama-cli -m {gguf_path} --grammar-file {grammar_path} "
        f'-p "<student prompt>"'
    )
    return ExportResult(out, gguf_path, grammar_path, modelfile_path, quant)
