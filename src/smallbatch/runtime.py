"""Internal runtime used to compare candidates before standalone packaging."""

from __future__ import annotations

import hashlib
import importlib
import json
import sys
import tempfile
import zipfile
from collections.abc import Callable
from pathlib import Path
from typing import Any

from . import artifacts, decode, prompts
from .spec import FunctionSpec, load_spec, validate_input, validate_output


class CandidateFunction:
    def __init__(self, spec: FunctionSpec, predict: Callable[[list[dict]], list[Any]], manifest: dict):
        self.spec = spec
        self._predict = predict
        self.manifest = manifest

    def __call__(self, item: dict[str, Any]) -> Any:
        return self.batch([item])[0]

    def batch(self, items: list[dict[str, Any]]) -> list[Any]:
        normalized = [validate_input(self.spec, item) for item in items]
        outputs = self._predict(normalized)
        if len(outputs) != len(normalized):
            raise ValueError("candidate returned the wrong number of outputs")
        return [validate_output(self.spec, output) for output in outputs]


class PackagedFunction:
    """Callable facade over the selected, integrity-checked standalone wheel."""

    def __init__(self, module, temporary: tempfile.TemporaryDirectory):
        self._module = module
        self._temporary = temporary
        self.manifest = module.metadata()

    def __call__(self, item: dict[str, Any]) -> Any:
        return self._module.classify(item)

    def batch(self, items: list[dict[str, Any]]) -> list[Any]:
        return self._module.classify_batch(items)


def _load_active_package(root: Path, name: str) -> PackagedFunction:
    active = artifacts.read_active(root, name)
    if active is None:
        raise FileNotFoundError(
            f"{name!r} has no active candidate; run `smallbatch select {name} <candidate>`"
        )
    package = root / name / active["package"]
    package_manifest_path = package / "package.json"
    if not package_manifest_path.exists():
        raise FileNotFoundError(f"active standalone package is missing at {package}")
    package_manifest = json.loads(package_manifest_path.read_text())
    wheel = package / package_manifest["wheel"]
    if not wheel.exists():
        raise FileNotFoundError(f"active standalone wheel is missing at {wheel}")
    digest = hashlib.sha256(wheel.read_bytes()).hexdigest()
    expected = {active.get("wheel_sha256"), package_manifest.get("wheel_sha256")}
    if None in expected or expected != {digest}:
        raise ValueError("active standalone wheel failed its integrity check")

    temporary = tempfile.TemporaryDirectory(prefix=f"smallbatch-{name}-")
    extracted = Path(temporary.name)
    with zipfile.ZipFile(wheel) as archive:
        archive.extractall(extracted)
    import_root = str(extracted)
    if import_root not in sys.path:
        sys.path.insert(0, import_root)
    module_name = f"smallbatch_functions.{name.replace('-', '_')}"
    for loaded in [key for key in sys.modules if key == module_name or key.startswith(module_name + ".")]:
        del sys.modules[loaded]
    namespace = sys.modules.get("smallbatch_functions")
    if namespace is not None and import_root not in namespace.__path__:
        namespace.__path__ = [import_root, *list(namespace.__path__)]
    importlib.invalidate_caches()
    try:
        module = importlib.import_module(module_name)
    except BaseException:
        temporary.cleanup()
        raise
    return PackagedFunction(module, temporary)


def load_candidate(build: Path, candidate: str) -> CandidateFunction:
    manifest = artifacts.read_manifest(build)
    record = artifacts.candidate_record(manifest, candidate)
    if record.get("status") != "completed":
        raise ValueError(f"candidate {candidate!r} did not complete")
    spec = load_spec(build / "spec.yaml")
    model_dir = build / record["artifact_path"]
    backend = record["backend"]
    if backend == "tfidf":
        from .candidates import _load_pipelines, predict_tfidf_pipelines

        pipelines = _load_pipelines(model_dir)
        return CandidateFunction(
            spec,
            lambda items: predict_tfidf_pipelines(pipelines, spec, items),
            manifest,
        )
    if backend == "setfit":
        from .setfit_candidate import load_setfit_models, predict_setfit_models

        models = load_setfit_models(model_dir, spec, "cpu")
        return CandidateFunction(
            spec,
            lambda items: predict_setfit_models(models, spec, items),
            manifest,
        )
    if backend != "lora":
        raise ValueError(f"unsupported candidate backend {backend!r}")

    from peft import PeftModel

    from .evaluate import generate_batch
    from .training import load_base_model

    tokenizer, base = load_base_model(record["base_model"], record["inference_precision"])
    model = PeftModel.from_pretrained(base, str(model_dir))
    model.eval()
    rationale = bool(record.get("rationale_distillation"))
    levels = None if rationale else decode.scale_levels(spec)
    decoder = record.get("decode", "argmax")

    def predict(items: list[dict]) -> list[Any]:
        texts = [prompts.student_prompt(spec, item) for item in items]
        if levels is not None:
            # a level is one token, so the distribution over the scale is in the
            # logits at one position: no decoding loop, and the decoder chooses
            # which point of it to report
            distributions = decode.score_levels(
                model, tokenizer, spec, texts, record.get("eval_batch_size", 16)
            )
            return decode.decode_levels(distributions, levels, decoder)
        raw, _ = generate_batch(
            model,
            tokenizer,
            texts,
            prompts.completion_budget(spec, rationale),
            batch_size=record.get("eval_batch_size", 16),
            allowed_completions=prompts.allowed_completions(spec, rationale),
        )
        return [prompts.parse_output(spec, text) for text in raw]

    return CandidateFunction(spec, predict, manifest)


def load_fn(
    name: str,
    artifacts_root: str | Path = artifacts.DEFAULT_ROOT,
    version: str | None = None,
    candidate: str | None = None,
) -> CandidateFunction | PackagedFunction:
    root = Path(artifacts_root)
    if version is None and candidate is None:
        return _load_active_package(root, name)
    build, selected = artifacts.resolve_runtime(root, name, version, candidate)
    return load_candidate(build, selected)
