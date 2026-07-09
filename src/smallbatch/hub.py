"""Push a compiled artifact to the Hugging Face Hub (opt-in).

Local-first stays the default: nothing is ever uploaded unless you call
`smallbatch push`. Repos are created private unless --public. The uploaded
repo is the artifact as-is (adapter/, spec.yaml, manifest.json, any export/
files) plus a model card rendered from the manifest.

Auth comes from the standard huggingface_hub login (`hf auth login` or
HF_TOKEN).
"""

from __future__ import annotations

import io
import json
from pathlib import Path

from . import artifacts
from .spec import FunctionSpec, load_spec


def model_card(spec: FunctionSpec, manifest: dict, repo_id: str) -> str:
    """Render manifest facts into a Hub model card (torch-free, unit-tested).

    The card must make sense to someone who did not run the compile: what the
    function does, exactly what it emits, how it was trained and measured,
    and how to run it — with the license/provider caveats stated."""
    from .prompts import output_instruction

    m = manifest["metrics"]
    adapter = m["adapter"]
    zeroshot = m.get("zeroshot")
    gate = manifest["gate"]
    data = manifest.get("data", {})
    contract = output_instruction(spec)
    agreement_kind = (
        "per-field (int fields within ±1, enum fields exact); the headline "
        "number is the joint all-fields rate"
        if not spec.output.is_scalar
        else ("within ±1 of the teacher label" if spec.output.type == "int"
              else "exact match with the teacher label")
    )
    ci = adapter.get("agreement_ci")
    ci_txt = f" (95% CI {ci[0]:.0%}–{ci[1]:.0%})" if ci else ""

    rows = [f"| adapter | {adapter['agreement']:.1%}{ci_txt} | {adapter.get('invalid_rate', 0):.1%} |"]
    if zeroshot:
        rows.append(
            f"| zero-shot base | {zeroshot['agreement']:.1%} | "
            f"{zeroshot.get('invalid_rate', 0):.1%} |"
        )
    field_rows = ""
    if not spec.output.is_scalar and adapter.get("fields"):
        field_rows = "\n### Per-field agreement\n\n| field | agreement | invalid |\n|---|---|---|\n" + "\n".join(
            f"| {name} | {fm['agreement']:.1%} | {fm.get('invalid_rate', 0):.1%} |"
            for name, fm in adapter["fields"].items()
        ) + "\n"

    if data.get("gate") is not None:
        split_summary = (
            f"{data.get('train', '?')} train / {data.get('dev', '?')} dev / "
            f"{data.get('gate', '?')} gate rows "
            f"({data.get('real', '?')} real items + {data.get('variants', '?')} "
            "teacher-generated variants; gate and dev are real items only)"
        )
    else:  # pre-v0.2 manifest
        split_summary = (
            f"{data.get('real', '?')} real items + {data.get('variants', '?')} "
            f"teacher-generated variants; holdout n={adapter.get('n', '?')}"
        )
    training_line = ""
    if manifest.get("best_epoch") is not None:
        training_line = (
            f"- **Training:** best of {manifest.get('epochs_run', '?')} epochs by dev "
            f"agreement (epoch {manifest['best_epoch']}; {manifest.get('stopped_reason', '')})\n"
        )
    example_input = json.dumps({k: f"<{t}>" for k, t in spec.input_schema.items()})

    return f"""---
base_model: {manifest["base_model"]}
library_name: peft
tags:
- smallbatch
- lora
- text-classification
---

# {spec.name}

{spec.description.strip()}

A narrow "fuzzy function" compiled with
[smallbatch](https://github.com/mmarks13/smallbatch): a teacher model labeled
real examples against a rubric, and this LoRA adapter was fine-tuned on those
labels. It does exactly one job — given the input fields below, it emits
{contract} — and nothing else. **It is not a chat model.**

- **Base model:** `{manifest["base_model"]}` — the adapter inherits its license
- **Input fields:** {", ".join(f"`{k}`" for k in spec.input_schema)}
- **Output contract:** {contract}
- **Teacher:** {data.get("teacher_model", "n/a")} ({data.get("teacher_backend", "n/a")}) — check the
  provider's terms before redistributing artifacts trained on its outputs
- **Data:** {split_summary}
{training_line}- **Gate:** {"PASS" if gate["passed"] else "FAIL"} — agreement is {agreement_kind}, bar {spec.gate.threshold:.0%}{"" if gate["passed"] else "; reasons: " + "; ".join(gate["reasons"])}

| model | agreement | invalid rate |
|---|---|---|
{chr(10).join(rows)}
{field_rows}
Full metrics (per-band tables, confusion, training curve) are in `report.md`
and `report.json`; complete provenance in `manifest.json`; the exact function
definition in `spec.yaml`.

## Example

Input: `{example_input}`
Prompt format (raw text, no chat template):

```
[{spec.name}]
<field>: <value>
...
output:
```

## Usage

```python
from peft import PeftModel
from transformers import AutoModelForCausalLM, AutoTokenizer

base = AutoModelForCausalLM.from_pretrained("{manifest["base_model"]}")
model = PeftModel.from_pretrained(base, "{repo_id}")
tok = AutoTokenizer.from_pretrained("{repo_id}")
```

With smallbatch: place this repo's contents under
`artifacts/{spec.name}/<version>/` and call `smallbatch.load_fn("{spec.name}")`,
or `smallbatch run {spec.name} --json '...'`.

If an `export/` directory is included, it holds a quantized GGUF plus a GBNF
grammar enforcing the output contract exactly — see `export/README.md` for
llama.cpp, llama-server, and Ollama invocations.

## Limitations

Trained to imitate one teacher on one rubric over one input distribution;
scores reflect agreement with that teacher on held-out real items, not ground
truth. Inputs far from the training distribution degrade silently — re-run
the eval (`smallbatch compile`) after any rubric or distribution change.

Compiled with smallbatch {manifest.get("smallbatch_version", "")}.
"""


def push(
    name: str,
    repo_id: str,
    artifacts_root: str | Path = artifacts.DEFAULT_ROOT,
    version: str | None = None,
    private: bool = True,
    allow_failed: bool = False,
) -> str:
    """Upload an artifact version to the Hub; returns the repo URL."""
    root = Path(artifacts_root)
    version_dir = artifacts.resolve_version(root, name, version, allow_failed)
    manifest = artifacts.read_manifest(version_dir)
    spec = load_spec(version_dir / "spec.yaml")
    card = model_card(spec, manifest, repo_id)

    try:
        from huggingface_hub import HfApi
    except ImportError as e:  # transformers brings it in, but be explicit
        raise RuntimeError("pip install huggingface_hub") from e

    api = HfApi()
    url = api.create_repo(repo_id, private=private, exist_ok=True).repo_url
    api.upload_folder(
        repo_id=repo_id,
        folder_path=version_dir,
        commit_message=f"smallbatch push: {name}/{manifest.get('version', version_dir.name)}",
        ignore_patterns=["export/merged/**"],
    )
    api.upload_file(
        path_or_fileobj=io.BytesIO(card.encode()),
        path_in_repo="README.md",
        repo_id=repo_id,
        commit_message="model card",
    )
    print(f"pushed {version_dir} -> {url} ({'private' if private else 'public'})")
    return str(url)
