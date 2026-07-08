"""Load and call compiled functions locally."""

from __future__ import annotations

from pathlib import Path
from typing import Any

from . import artifacts, prompts
from .evaluate import generate_batch
from .spec import load_spec


class CompiledFunction:
    def __init__(self, spec, model, tokenizer, manifest: dict):
        self.spec = spec
        self.model = model
        self.tokenizer = tokenizer
        self.manifest = manifest
        self._max_new = 80 if spec.train.rationale_distillation else 8

    def __call__(self, item: dict[str, Any]) -> Any:
        return self.batch([item])[0]

    def batch(self, items: list[dict[str, Any]]) -> list[Any]:
        for it in items:
            missing = [k for k in self.spec.input_schema if k not in it]
            if missing:
                raise ValueError(f"input missing fields {missing} (have {list(it)})")
        texts = [prompts.student_prompt(self.spec, it) for it in items]
        raw = generate_batch(
            self.model, self.tokenizer, texts, self._max_new,
            batch_size=self.spec.train.eval_batch_size,
            allowed_completions=prompts.allowed_completions(self.spec),
        )
        return [prompts.parse_output(self.spec, t) for t in raw]


def load_fn(
    name: str,
    artifacts_root: str | Path = artifacts.DEFAULT_ROOT,
    allow_failed: bool = False,
) -> CompiledFunction:
    root = Path(artifacts_root)
    version = artifacts.latest(root, name, passing_only=not allow_failed)
    if version is None:
        raise FileNotFoundError(
            f"no {'passing ' if not allow_failed else ''}artifact for '{name}' under {root}"
        )
    manifest = artifacts.read_manifest(version)
    stale = artifacts.staleness(version)
    if stale:
        print(f"warning: '{name}' artifact is stale — {stale}; consider recompiling")

    from peft import PeftModel

    from .training import load_base_model

    spec = load_spec(version / "spec.yaml")
    tokenizer, model = load_base_model(manifest["base_model"], manifest["inference_precision"])
    model = PeftModel.from_pretrained(model, str(version / "adapter"))
    model.eval()
    return CompiledFunction(spec, model, tokenizer, manifest)
