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
    """Internal evaluation facade over one trained candidate.

    Unlike packaged inference (which raises `InvalidOutputError`), evaluation
    records an invalid free-running result as None so the failing row is
    counted instead of aborting the run; `structural_failures` accumulates
    the categories. The output is still atomic — a partially valid record is
    never returned.
    """

    def __init__(self, spec: FunctionSpec, predict: Callable[[list[dict]], list[Any]], manifest: dict):
        self.spec = spec
        self._predict = predict
        self.manifest = manifest
        self.structural_failures: dict[str, int] = {}

    def __call__(self, item: dict[str, Any]) -> Any:
        return self.batch([item])[0]

    def batch(self, items: list[dict[str, Any]]) -> list[Any]:
        normalized = [validate_input(self.spec, item) for item in items]
        outputs = self._predict(normalized)
        if len(outputs) != len(normalized):
            raise ValueError("candidate returned the wrong number of outputs")
        return [
            None if output is None else validate_output(self.spec, output)
            for output in outputs
        ]


class PackagedFunction:
    """Callable facade over the selected, integrity-checked standalone wheel."""

    def __init__(self, module, temporary: tempfile.TemporaryDirectory):
        self._module = module
        self._temporary = temporary
        self.manifest = module.metadata()

    def __call__(self, item: dict[str, Any]) -> Any:
        return self._module.run(item)

    def batch(self, items: list[dict[str, Any]]) -> list[Any]:
        return self._module.run_batch(items)


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
    namespace_root = str(extracted / "smallbatch_functions")
    if namespace is not None and namespace_root not in namespace.__path__:
        namespace.__path__ = [namespace_root, *list(namespace.__path__)]
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

    from .evaluate import generate_outputs
    from .training import load_base_model

    tokenizer, base = load_base_model(record["base_model"], record["inference_precision"])
    model = PeftModel.from_pretrained(base, str(model_dir))
    model.eval()
    levels = None if spec.output.has_text else decode.scale_levels(spec)
    batch_size = record.get("eval_batch_size", 16)
    function: CandidateFunction

    def predict(items: list[dict]) -> list[Any]:
        texts = [prompts.student_prompt(spec, item) for item in items]
        if levels is not None:
            # a level is one token, so the distribution over the scale is in
            # the logits at one position: no decoding loop, argmax read
            distributions = decode.score_levels(
                model, tokenizer, spec, texts, batch_size
            )
            return decode.decode_levels(distributions, levels)
        outputs, failures = generate_outputs(
            spec, model, tokenizer, texts, batch_size
        )
        for category, count in failures.items():
            function.structural_failures[category] = (
                function.structural_failures.get(category, 0) + count
            )
        return outputs

    function = CandidateFunction(spec, predict, manifest)
    if spec.output.has_text:
        from . import objective

        codecs = objective.field_codecs(spec, tokenizer)

        def fidelity(rows: list[dict]) -> dict | None:
            return objective.text_fidelity(
                model, tokenizer, spec, codecs, rows, batch_size
            )

        function.text_fidelity = fidelity  # type: ignore[attr-defined]
    return function


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
