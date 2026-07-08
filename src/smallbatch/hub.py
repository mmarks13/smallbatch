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
    """Render manifest facts into a Hub model card (torch-free, unit-tested)."""
    m = manifest["metrics"]
    adapter = m["adapter"]
    zeroshot = m.get("zeroshot")
    if spec.output.type == "int":
        lo, hi = spec.output.range
        contract = f"an integer from {lo} to {hi}"
        agreement_kind = "within ±1 of the teacher label"
    else:
        contract = "one of: " + ", ".join(spec.output.labels)
        agreement_kind = "exact match with the teacher label"
    gate = manifest["gate"]
    rows = [f"| adapter | {adapter['agreement']:.1%} | {adapter.get('invalid_rate', 0):.1%} |"]
    if zeroshot:
        rows.append(
            f"| zero-shot base | {zeroshot['agreement']:.1%} | "
            f"{zeroshot.get('invalid_rate', 0):.1%} |"
        )
    data = manifest.get("data", {})
    provenance = (
        f"{data.get('real', '?')} real items + {data.get('variants', '?')} "
        f"teacher-generated variants; holdout n={adapter.get('n', '?')} (real items only)"
        if data
        else f"holdout n={adapter.get('n', '?')}"
    )
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
{contract} — and nothing else.

- **Base model:** `{manifest["base_model"]}` (the adapter inherits its license)
- **Input fields:** {", ".join(f"`{k}`" for k in spec.input_schema)}
- **Teacher:** {data.get("teacher_model", "n/a")} ({data.get("teacher_backend", "n/a")})
- **Data:** {provenance}
- **Gate:** {"PASS" if gate["passed"] else "FAIL"} — agreement is {agreement_kind}, bar {spec.gate.agreement_pm1:.0%}{"" if gate["passed"] else "; reasons: " + "; ".join(gate["reasons"])}

| model | agreement | invalid rate |
|---|---|---|
{chr(10).join(rows)}

## Usage

```python
from peft import PeftModel
from transformers import AutoModelForCausalLM, AutoTokenizer

base = AutoModelForCausalLM.from_pretrained("{manifest["base_model"]}")
model = PeftModel.from_pretrained(base, "{repo_id}")
tok = AutoTokenizer.from_pretrained("{repo_id}")
# prompt format (raw text, no chat template):
# [{spec.name}]\\n<field>: <value>...\\noutput:
```

Or with smallbatch itself, place this repo's contents under
`artifacts/{spec.name}/<version>/` and call `smallbatch.load_fn("{spec.name}")`.

Compiled with smallbatch {manifest.get("smallbatch_version", "")}; full
metrics and provenance in `manifest.json`, the exact function definition in
`spec.yaml`.
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
