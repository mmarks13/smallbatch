import hashlib
import json
import zipfile

import pytest

from conftest import imported_records, make_spec
from smallbatch import artifacts, standalone
from smallbatch.api import compile as compile_fn
from smallbatch.api import label, select
from smallbatch.runtime import load_fn


def _write_active_test_wheel(root, name):
    package = root / name / "packages" / "test"
    package.mkdir(parents=True)
    wheel = package / f"{name}.whl"
    module = name.replace("-", "_")
    with zipfile.ZipFile(wheel, "w") as archive:
        archive.writestr(
            f"smallbatch_functions/{module}/__init__.py",
            "def metadata(): return {'function': '" + name + "'}\n"
            "def run(item): return '" + name + "'\n"
            "def run_batch(items): return ['" + name + "' for _ in items]\n",
        )
    digest = hashlib.sha256(wheel.read_bytes()).hexdigest()
    (package / "package.json").write_text(
        json.dumps({"wheel": wheel.name, "wheel_sha256": digest})
    )
    (root / name / "active.json").write_text(
        json.dumps({"package": "packages/test", "wheel_sha256": digest})
    )


def test_load_fn_refreshes_cached_namespace_for_a_different_artifact_root(tmp_path):
    first_root = tmp_path / "first"
    second_root = tmp_path / "second"
    _write_active_test_wheel(first_root, "first-function")
    _write_active_test_wheel(second_root, "second-function")

    first = load_fn("first-function", artifacts_root=first_root)
    second = load_fn("second-function", artifacts_root=second_root)

    assert first({}) == "first-function"
    assert second({}) == "second-function"


def test_selected_wheel_has_no_smallbatch_runtime_dependency(tmp_path, monkeypatch):
    spec = make_spec()
    data = tmp_path / "data"
    root = tmp_path / "artifacts"
    label(spec, imported_records(30), out_dir=data)
    compiled = compile_fn(spec, data_dir=data, artifacts_root=root, cpu_threads=1)
    selected = select(
        spec.name,
        "tfidf",
        version=compiled.build_id,
        artifacts_root=root,
        interactive=False,
    )
    assert selected.wheel.exists()
    assert selected.evidence["parity"]["exact"] == 1
    assert (
        selected.evidence["package_profile"]["threads"]
        == selected.evidence["candidate_profile"]["threads"]
    )
    with zipfile.ZipFile(selected.wheel) as archive:
        names = archive.namelist()
        metadata_name = next(name for name in names if name.endswith(".dist-info/METADATA"))
        metadata = archive.read(metadata_name).decode()
        assert "Requires-Dist: smallbatch\n" not in metadata
        assert any(name.endswith("smallbatch_functions/ticket_priority/evidence.json") for name in names)
        assert not any("evaluation.local" in name for name in names)
    assert selected.active["candidate"] == "tfidf"
    function = load_fn(spec.name, artifacts_root=root)
    assert function({"title": "outage", "body": "server down"}) in {"urgent", "normal"}

    evaluate_wheel = standalone._evaluate_wheel

    def valid_drift(*args, **kwargs):
        result = evaluate_wheel(*args, **kwargs)
        result["predictions"][0] = (
            "normal" if result["predictions"][0] == "urgent" else "urgent"
        )
        return result

    monkeypatch.setattr(standalone, "_evaluate_wheel", valid_drift)
    active_before_drift = artifacts.read_active(root, spec.name)
    with pytest.raises(ValueError, match="--accept-package-drift"):
        select(
            spec.name,
            "tfidf",
            version=compiled.build_id,
            artifacts_root=root,
            interactive=False,
        )
    assert artifacts.read_active(root, spec.name) == active_before_drift

    previous_package = selected.package_dir
    selected_again = select(
        spec.name,
        "tfidf",
        version=compiled.build_id,
        artifacts_root=root,
        accept_package_drift=True,
        interactive=False,
    )
    assert selected_again.package_dir != previous_package
    assert previous_package.exists()
    assert selected_again.evidence["package_drift_accepted"] is True
    assert selected_again.evidence["parity"]["exact"] < 1

    selected_again.wheel.write_bytes(selected_again.wheel.read_bytes() + b"tampered")
    with pytest.raises(ValueError, match="integrity"):
        load_fn(spec.name, artifacts_root=root)
    assert hashlib.sha256(selected.wheel.read_bytes()).hexdigest() == selected.active["wheel_sha256"]


def test_reported_dependency_names_cover_the_packaged_pins():
    """profiling's public-evidence dependency list is derived from the pinned
    packaging list, so it can never understate what the wheel installs."""
    for backend in ("tfidf", "setfit", "lora"):
        pinned = {
            standalone._requirement_name(requirement)
            for requirement in standalone._dependencies(backend)
        }
        assert pinned <= set(standalone.runtime_dependency_names(backend))
    assert {"scikit-learn", "skops"} <= set(standalone.runtime_dependency_names("setfit"))
    assert standalone.runtime_dependency_names("zeroshot") == ["torch", "transformers"]
