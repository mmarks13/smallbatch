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
        self._max_new = prompts.completion_budget(spec)

    def __call__(self, item: dict[str, Any]) -> Any:
        out = self.batch([item])[0]
        if out is None:
            raise ValueError("model output failed contract validation")
        return out

    def batch(self, items: list[dict[str, Any]]) -> list[Any]:
        """Outputs aligned with `items`; an output that fails the contract
        (including a partially-parsed structured output) is None, never a
        partial object."""
        for it in items:
            missing = [k for k in self.spec.input_schema if k not in it]
            if missing:
                raise ValueError(f"input missing fields {missing} (have {list(it)})")
        texts = [prompts.student_prompt(self.spec, it) for it in items]
        raw, _ = generate_batch(
            self.model, self.tokenizer, texts, self._max_new,
            batch_size=self.spec.train.eval_batch_size,
            allowed_completions=prompts.allowed_completions(self.spec),
        )
        parsed = [prompts.parse_output(self.spec, t) for t in raw]
        return [
            None if prompts.incomplete_fields(self.spec, p) else p for p in parsed
        ]


class TfidfFunction:
    """A compiled tfidf candidate: same call surface as CompiledFunction,
    no base model, no GPU, no torch import."""

    def __init__(self, spec, model_dir: Path, manifest: dict):
        self.spec = spec
        self.model_dir = model_dir
        self.manifest = manifest

    def __call__(self, item: dict[str, Any]) -> Any:
        out = self.batch([item])[0]
        if out is None:
            raise ValueError("model output failed contract validation")
        return out

    def batch(self, items: list[dict[str, Any]]) -> list[Any]:
        from . import candidates as cand

        for it in items:
            missing = [k for k in self.spec.input_schema if k not in it]
            if missing:
                raise ValueError(f"input missing fields {missing} (have {list(it)})")
        outs = cand.predict_tfidf(self.model_dir, self.spec, items)
        return [
            None if prompts.incomplete_fields(self.spec, o) else o for o in outs
        ]


def load_fn(
    name: str,
    artifacts_root: str | Path = artifacts.DEFAULT_ROOT,
    allow_failed: bool = False,
    version: str | None = None,
    candidate: str | None = None,
) -> CompiledFunction | TfidfFunction:
    """Load a compiled function: the latest usable version by default, or an
    explicit `version` dir name. `candidate` selects a retained non-winner
    candidate — its own gate/acceptance state applies (an accepted winner
    never unlocks an unaccepted secondary)."""
    root = Path(artifacts_root)
    version_dir = artifacts.resolve_version(
        root, name, version, allow_failed=allow_failed, candidate=candidate
    )
    manifest = artifacts.read_manifest(version_dir)
    stale = artifacts.staleness(version_dir)
    if stale:
        print(f"warning: '{name}' artifact is stale — {stale}; consider recompiling")
    rec = artifacts.candidate_record(manifest, candidate)
    chosen = candidate or artifacts.winner(manifest)
    if not (rec.get("gate") or {}).get("passed"):
        print(
            f"warning: '{name}' candidate '{chosen}' FAILED its gate "
            f"({'; '.join((rec.get('gate') or {}).get('reasons', [])) or 'no reasons recorded'}) — "
            "loaded anyway"
        )
    spec = load_spec(version_dir / "spec.yaml")
    if rec["backend"] == "tfidf":
        # CPU-only sklearn pipeline: never imports torch
        return TfidfFunction(spec, version_dir / (rec.get("artifact_path") or "tfidf"), manifest)
    if rec["backend"] != "lora":
        raise ValueError(
            f"candidate '{chosen}' uses backend '{rec['backend']}', which this "
            "runtime cannot load"
        )

    from peft import PeftModel

    from .training import load_base_model

    tokenizer, model = load_base_model(
        rec.get("base_model") or manifest["base_model"],
        rec.get("inference_precision") or manifest["inference_precision"],
    )
    adapter = version_dir / (rec.get("artifact_path") or "adapter")
    model = PeftModel.from_pretrained(model, str(adapter))
    model.eval()
    return CompiledFunction(spec, model, tokenizer, manifest)
