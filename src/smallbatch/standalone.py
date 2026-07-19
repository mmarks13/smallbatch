"""Generate and validate standalone Python function distributions."""

from __future__ import annotations

import base64
import datetime
import hashlib
import importlib.metadata
import json
import os
import shutil
import subprocess
import sys
import tempfile
import zipfile
from collections.abc import Callable
from pathlib import Path

from . import artifacts
from .api import SelectionResult
from .evaluate import compute_metrics
from .labeling import read_jsonl
from .spec import load_spec

_TEMPLATES = Path(__file__).parent / "standalone_templates"

# backends whose packaged function needs no network after pip install
OFFLINE_BACKENDS = frozenset({"tfidf", "setfit"})


def package_selection(
    root: Path,
    name: str,
    candidate: str,
    *,
    version: str | None,
    accept_drift: bool,
    interactive: bool | None,
    input_fn: Callable[[str], str],
) -> SelectionResult:
    build = artifacts.resolve_build(root, name, version)
    broken = artifacts.artifact_integrity(build)
    if broken:
        raise ValueError(f"cannot package damaged build: {broken}")
    manifest = artifacts.read_manifest(build)
    record = artifacts.candidate_record(manifest, candidate)
    if record.get("status") != "completed":
        raise ValueError(f"candidate {candidate!r} did not complete CPU evaluation")
    local_record = json.loads(
        (build / "candidates" / candidate / "result.local.json").read_text()
    )
    eval_rows = read_jsonl(build / "evaluation.local.jsonl")
    if not eval_rows:
        raise ValueError("build lacks its local evaluation rows; recompile")
    spec = load_spec(build / "spec.yaml")
    package = artifacts.package_dir(root, name, build.name, candidate)
    source = package / "source"
    dist = package / "dist"
    module = name.replace("-", "_")
    module_dir = source / "src" / "smallbatch_functions" / module
    module_dir.mkdir(parents=True)
    dist.mkdir(parents=True)
    artifacts.copy_deployable_model(
        build / record["artifact_path"], module_dir / "model", record["backend"]
    )
    shutil.copy(_TEMPLATES / "common.py.tmpl", module_dir / "_common.py")
    shutil.copy(_TEMPLATES / f"{record['backend']}.py.tmpl", module_dir / "__init__.py")

    from . import prompts

    function = spec.model_dump(mode="json")
    function.pop("teacher", None)
    function.pop("candidates", None)
    function.pop("augmentation", None)
    function.pop("training", None)
    function["runtime"] = {
        key: record.get(key)
        for key in (
            "backend",
            "base_model",
            "inference_precision",
            "eval_batch_size",
            "loss_weights",
        )
        if record.get(key) is not None
    }
    # the resolved output contract (function.json carries max_chars per field)
    # plus the exact deterministic decoding settings the package must use
    function["runtime"]["decoding"] = {
        "strategy": "greedy",
        "do_sample": False,
        "max_new_tokens": prompts.completion_budget(spec),
    }
    (module_dir / "function.json").write_text(
        json.dumps(function, indent=2, ensure_ascii=False)
    )
    (module_dir / "evidence.json").write_text(json.dumps({"status": "pending"}))
    dependencies = _dependencies(record["backend"])
    (source / "requirements-tested.txt").write_text(
        "\n".join(_tested_requirements(dependencies)) + "\n"
    )
    (source / "pyproject.toml").write_text(
        _pyproject(name, module, manifest["build_hash"], candidate, dependencies)
    )
    (source / "README.md").write_text(_package_readme(name, module, record))
    wheel = _build_wheel(
        source, dist, name, manifest["build_hash"], candidate, dependencies
    )

    items = [row["input"] for row in eval_rows]
    references = [row["output"] for row in eval_rows]
    candidate_profile = local_record["profile"]
    # verify under the thread count the candidate was evaluated with, or the
    # packaged profile silently describes different operating conditions
    threads = max(1, candidate_profile.get("threads") or min(4, os.cpu_count() or 1))
    provisional = _evaluate_wheel(wheel, module, items, threads)
    provisional["profile"].update(
        {
            "runtime": record["backend"],
            "candidate_owned_bytes": artifacts.dir_size(module_dir / "model"),
            "required_shared_bytes": candidate_profile.get("required_shared_bytes"),
            "required_base_model": record.get("base_model"),
            "dependencies": candidate_profile.get("dependencies", {}),
            "cpu": candidate_profile.get("cpu", "unknown"),
            "offline_after_install": record["backend"] in OFFLINE_BACKENDS,
        }
    )
    predictions = provisional["predictions"]
    invalid_rows = [
        index for index, prediction in enumerate(predictions) if prediction is None
    ]
    if invalid_rows:
        # §invariant: any invalid output on the complete held-out package
        # evaluation blocks selection. The candidate stays inspectable in the
        # report; it just cannot become active.
        raise ValueError(
            f"standalone package produced invalid outputs on "
            f"{len(invalid_rows)}/{len(predictions)} held-out evaluation rows "
            f"(rows {invalid_rows[:10]}); selection is blocked. Inspect the "
            "build report's structural failures and choose another candidate "
            "or improve the data"
        )
    package_metrics = compute_metrics(spec, predictions, references)
    original_predictions = local_record["predictions"]
    changed = [
        index
        for index, (original, packaged) in enumerate(zip(original_predictions, predictions))
        if original != packaged
    ]
    parity = {
        "n": len(predictions),
        "exact": round((len(predictions) - len(changed)) / len(predictions), 4),
        "changed": len(changed),
        "changed_rows": changed[:20],
    }
    drift = {
        "candidate_metrics": local_record["metrics"],
        "package_metrics": package_metrics,
        "numeric_deltas": _numeric_deltas(local_record["metrics"], package_metrics),
    }
    if changed and not accept_drift:
        if interactive is None:
            interactive = sys.stdin.isatty() and sys.stdout.isatty()
        summary = (
            f"standalone package changed {len(changed)}/{len(predictions)} decisions "
            f"({parity['exact']:.1%} exact parity)"
        )
        if not interactive:
            raise ValueError(summary + "; inspect package evidence and pass --accept-package-drift")
        print(summary)
        print(json.dumps(drift, indent=2))
        answer = input_fn("Activate this standalone package despite the drift? [y/N] ")
        if answer.strip().lower() not in {"y", "yes"}:
            raise ValueError("standalone package drift was not accepted")

    evidence = {
        "schema_version": 1,
        "function": name,
        "build": build.name,
        "candidate": candidate,
        "decision_correctness_validated": False,
        "package_metrics": package_metrics,
        "candidate_metrics": local_record["metrics"],
        "package_profile": provisional["profile"],
        "candidate_profile": candidate_profile,
        "parity": parity,
        "comparison": manifest.get("candidates"),
        "selection_bias_note": json.loads((build / "report.json").read_text()).get(
            "selection_bias_note"
        ),
        "package_drift_accepted": bool(changed),
    }
    (module_dir / "evidence.json").write_text(
        json.dumps(evidence, indent=2, ensure_ascii=False)
    )
    wheel = _build_wheel(
        source,
        dist,
        name,
        manifest["build_hash"],
        candidate,
        dependencies,
        clean=True,
    )
    final = _evaluate_wheel(wheel, module, items, threads)
    if final["predictions"] != predictions:
        raise ValueError("final wheel predictions changed after embedding evidence")
    wheel_hash = hashlib.sha256(wheel.read_bytes()).hexdigest()
    package_manifest = {
        "schema_version": 1,
        "function": name,
        "build": build.name,
        "candidate": candidate,
        "module": f"smallbatch_functions.{module}",
        "wheel": str(wheel.relative_to(package)),
        "wheel_sha256": wheel_hash,
        "smallbatch_runtime_dependency": False,
        "created_at": datetime.datetime.now(datetime.timezone.utc).isoformat(),
        "evidence": evidence,
    }
    (package / "package.json").write_text(
        json.dumps(package_manifest, indent=2, ensure_ascii=False)
    )
    active = {
        "function": name,
        "build": build.name,
        "candidate": candidate,
        "package": str(package.relative_to(root / name)),
        "wheel_sha256": wheel_hash,
        "selected_at": datetime.datetime.now(datetime.timezone.utc).isoformat(),
        "method": "select",
        "package_drift_accepted": bool(changed),
    }
    artifacts.activate(root, name, active)
    return SelectionResult(name, build.name, candidate, package, wheel, evidence, active)


def _dependencies(backend: str) -> list[str]:
    return {
        "tfidf": ["scikit-learn>=1.4,<2", "skops>=0.10"],
        "setfit": [
            "setfit>=1.1.3,<1.2",
            "sentence-transformers>=3,<6",
            "transformers>=4.56,<5",
            # an integer scale is served by the ordered head persisted with skops
            "scikit-learn>=1.4,<2",
            "skops>=0.10",
        ],
        "lora": ["torch>=2.4", "transformers>=4.56,<5", "peft>=0.14"],
    }[backend]


def _requirement_name(requirement: str) -> str:
    return requirement.split("<", 1)[0].split(">", 1)[0].split("=", 1)[0]


def runtime_dependency_names(backend: str) -> list[str]:
    """Package names behind a backend's runtime, for evidence reporting.

    Derived from the pinned packaging list so public evidence can never
    understate what the standalone package installs, plus the load-bearing
    transitive packages worth versioning in a profile.
    """
    if backend == "zeroshot":
        return ["torch", "transformers"]
    extras = {"tfidf": ["numpy", "scipy"], "setfit": ["torch"]}
    names = [_requirement_name(requirement) for requirement in _dependencies(backend)]
    return names + extras.get(backend, [])


def _tested_requirements(requirements: list[str]) -> list[str]:
    output = []
    for requirement in requirements:
        name = _requirement_name(requirement)
        try:
            output.append(f"{name}=={importlib.metadata.version(name)}")
        except importlib.metadata.PackageNotFoundError:
            output.append(requirement)
    return output


def _pyproject(
    name: str,
    module: str,
    build_hash: str,
    candidate: str,
    dependencies: list[str],
) -> str:
    deps = "\n".join(f'  "{dependency}",' for dependency in dependencies)
    return f'''[project]
name = "smallbatch-function-{name}"
version = "0.0.0+{build_hash[:12]}.{candidate.replace('-', '.')}"
description = "Standalone local function generated by Smallbatch"
requires-python = ">=3.10,<3.13"
dependencies = [
{deps}
]

[build-system]
requires = ["hatchling"]
build-backend = "hatchling.build"

[tool.hatch.build.targets.wheel]
packages = ["src/smallbatch_functions"]

[tool.hatch.build.targets.wheel.force-include]
"src/smallbatch_functions/{module}/function.json" = "smallbatch_functions/{module}/function.json"
"src/smallbatch_functions/{module}/evidence.json" = "smallbatch_functions/{module}/evidence.json"
"src/smallbatch_functions/{module}/model" = "smallbatch_functions/{module}/model"
'''


def _package_readme(name: str, module: str, record: dict) -> str:
    base = record.get("base_model")
    requirement = f" It downloads or reuses base model `{base}`." if base else ""
    return f"""# {name}

Standalone CPU function generated by Smallbatch.{requirement}

```python
from smallbatch_functions.{module} import run, run_batch, metadata
```

The package does not depend on Smallbatch. `metadata()` returns its contract
and evidence. `run` returns the declared output shape — a value, a string, or
a complete dictionary — and raises `InvalidOutputError` (importable from the
same module) instead of ever returning a partial or repaired result.
"""


def _build_wheel(
    source: Path,
    dist: Path,
    name: str,
    build_hash: str,
    candidate: str,
    dependencies: list[str],
    clean: bool = False,
) -> Path:
    """Write a deterministic pure-Python wheel without network/build isolation."""
    if clean:
        shutil.rmtree(dist, ignore_errors=True)
        dist.mkdir(parents=True)
    distribution = f"smallbatch_function_{name.replace('-', '_')}"
    version = f"0.0.0+{build_hash[:12]}.{candidate.replace('-', '.')}"
    dist_info = f"{distribution}-{version}.dist-info"
    wheel = dist / f"{distribution}-{version}-py3-none-any.whl"
    files: dict[str, bytes] = {}
    package_root = source / "src"
    for path in sorted(package_root.rglob("*")):
        if path.is_file():
            files[str(path.relative_to(package_root))] = path.read_bytes()
    requires = "".join(f"Requires-Dist: {dependency}\n" for dependency in dependencies)
    files[f"{dist_info}/METADATA"] = (
        "Metadata-Version: 2.4\n"
        f"Name: smallbatch-function-{name}\n"
        f"Version: {version}\n"
        "Summary: Standalone local function generated by Smallbatch\n"
        "Requires-Python: >=3.10,<3.13\n"
        f"{requires}\n"
    ).encode()
    files[f"{dist_info}/WHEEL"] = (
        b"Wheel-Version: 1.0\nGenerator: smallbatch\nRoot-Is-Purelib: true\n"
        b"Tag: py3-none-any\n"
    )
    records = []
    for path, content in files.items():
        digest = base64.urlsafe_b64encode(hashlib.sha256(content).digest()).decode().rstrip("=")
        records.append(f"{path},sha256={digest},{len(content)}")
    records.append(f"{dist_info}/RECORD,,")
    files[f"{dist_info}/RECORD"] = ("\n".join(records) + "\n").encode()
    with zipfile.ZipFile(wheel, "w", compression=zipfile.ZIP_DEFLATED) as archive:
        for path, content in sorted(files.items()):
            archive.writestr(path, content)
    return wheel


def _evaluate_wheel(wheel: Path, module: str, items: list[dict], threads: int) -> dict:
    runner = r'''
import builtins, importlib, json, os, platform, resource, sys, time
real_import = builtins.__import__
def guarded(name, *args, **kwargs):
    if name == "smallbatch" or name.startswith("smallbatch."):
        raise RuntimeError("standalone function attempted to import smallbatch")
    return real_import(name, *args, **kwargs)
builtins.__import__ = guarded
request = json.loads(open(sys.argv[1]).read())
started = time.perf_counter()
function = importlib.import_module(request["module"])
invalid_error = getattr(function, "InvalidOutputError", ())
def call(item):
    # packaged inference fails atomically on an invalid output; here each
    # failure is recorded as None so the full held-out run is measured and
    # the caller can block selection with the complete row list
    try:
        return function.run(item)
    except invalid_error:
        return None
if request["items"]:
    call(request["items"][0])
cold = time.perf_counter() - started
for item in request["items"][:3]:
    call(item)
outputs, latencies = [], []
for item in request["items"]:
    started = time.perf_counter()
    outputs.append(call(item))
    latencies.append((time.perf_counter() - started) * 1000)
ordered = sorted(latencies)
def percentile(q):
    if not ordered: return None
    return ordered[min(len(ordered)-1, int((len(ordered)-1)*q + .999999))]
peak = resource.getrusage(resource.RUSAGE_SELF).ru_maxrss
if sys.platform != "darwin": peak *= 1024
json.dump({"predictions": outputs, "profile": {
    "cold_load_seconds": round(cold, 4),
    "batch_one_latency_ms": {"p50": percentile(.5), "p95": percentile(.95), "n": len(latencies)},
    "peak_rss_bytes": int(peak), "threads": request["threads"],
    "os": platform.platform(), "python": platform.python_version()
}}, open(sys.argv[2], "w"))
'''
    with tempfile.TemporaryDirectory(prefix="smallbatch-standalone-") as tmp:
        temporary = Path(tmp)
        extracted = temporary / "wheel"
        extracted.mkdir()
        with zipfile.ZipFile(wheel) as archive:
            archive.extractall(extracted)
        request = temporary / "request.json"
        result = temporary / "result.json"
        script = temporary / "runner.py"
        request.write_text(
            json.dumps(
                {
                    "module": f"smallbatch_functions.{module}",
                    "items": items,
                    "threads": threads,
                }
            )
        )
        script.write_text(runner)
        env = {
            **os.environ,
            "PYTHONPATH": str(extracted),
            "CUDA_VISIBLE_DEVICES": "",
            "OMP_NUM_THREADS": str(threads),
            "MKL_NUM_THREADS": str(threads),
            "OPENBLAS_NUM_THREADS": str(threads),
            "TOKENIZERS_PARALLELISM": "false",
        }
        completed = subprocess.run(
            [sys.executable, str(script), str(request), str(result)],
            cwd=temporary,
            env=env,
            text=True,
            capture_output=True,
            check=False,
        )
        if completed.returncode:
            raise RuntimeError(
                "standalone wheel evaluation failed: "
                + (completed.stderr.strip() or completed.stdout.strip())
            )
        return json.loads(result.read_text())


def _numeric_deltas(first: dict, second: dict, prefix: str = "") -> dict[str, float]:
    output = {}
    for key in first.keys() & second.keys():
        path = f"{prefix}.{key}" if prefix else key
        if isinstance(first[key], dict) and isinstance(second[key], dict):
            output.update(_numeric_deltas(first[key], second[key], path))
        elif (
            isinstance(first[key], (int, float))
            and not isinstance(first[key], bool)
            and isinstance(second[key], (int, float))
            and not isinstance(second[key], bool)
        ):
            delta = second[key] - first[key]
            if delta:
                output[path] = round(delta, 6)
    return output
