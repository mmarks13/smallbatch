import zipfile
from importlib.metadata import version
from pathlib import Path

import pytest

from conftest import imported_records, make_spec
from smallbatch.api import compile as compile_fn
from smallbatch.api import label, select


def cached_bge() -> Path | None:
    root = Path.home() / ".cache" / "huggingface" / "hub" / "models--BAAI--bge-small-en-v1.5" / "snapshots"
    snapshots = sorted(root.glob("*")) if root.exists() else []
    return snapshots[-1] if snapshots else None


@pytest.mark.skipif(cached_bge() is None, reason="BGE-small is not cached; unit test stays offline")
def test_real_setfit_compile_and_standalone_wheel(tmp_path):
    model = str(cached_bge())
    spec = make_spec(
        candidates={
            "bge-small": {
                "type": "setfit",
                "model": model,
                "training_args": {"num_epochs": 1, "batch_size": 4, "seed": 17},
            }
        }
    )
    data = tmp_path / "data"
    root = tmp_path / "artifacts"
    label(spec, imported_records(30), out_dir=data)
    compiled = compile_fn(spec, data_dir=data, artifacts_root=root, cpu_threads=1)
    record = compiled.candidates["bge-small"]
    assert record["status"] == "completed"
    assert record["setfit_version"] == version("setfit")
    assert record["profile"]["runtime"] == "setfit"
    model_dir = compiled.version_dir / record["artifact_path"]
    assert not list(model_dir.rglob("checkpoints"))
    assert not list(model_dir.rglob("optimizer.pt"))

    selected = select(
        spec.name,
        "bge-small",
        version=compiled.build_id,
        artifacts_root=root,
        interactive=False,
    )
    assert selected.wheel.exists()
    assert selected.evidence["parity"]["exact"] == 1
    assert selected.evidence["package_profile"]["runtime"] == "setfit"
    assert selected.evidence["package_profile"]["offline_after_install"] is True
    with zipfile.ZipFile(selected.wheel) as archive:
        assert not any("/checkpoints/" in name for name in archive.namelist())

@pytest.mark.skipif(cached_bge() is None, reason="BGE-small is not cached; unit test stays offline")
def test_real_setfit_integer_scale_uses_the_ordered_head_end_to_end(tmp_path):
    """An integer scale must train, persist, and package the ordered head: the
    generated wheel reproduces it without importing smallbatch."""
    model = str(cached_bge())
    spec = make_spec(
        output={"type": "int", "range": [0, 2]},
        candidates={
            "bge-small": {
                "type": "setfit",
                "model": model,
                "training_args": {"num_epochs": 1, "batch_size": 4, "seed": 17},
            }
        },
    )
    records = [
        {
            "input": {"title": f"case {index}", "body": body},
            "output": level,
        }
        for level, body in enumerate(
            ("asked a routine question", "delayed and unanswered", "money withheld and frozen")
        )
        for index in range(10)
    ]
    data = tmp_path / "data"
    root = tmp_path / "artifacts"
    label(spec, records, out_dir=data)
    compiled = compile_fn(spec, data_dir=data, artifacts_root=root, cpu_threads=1)
    record = compiled.candidates["bge-small"]
    assert record["status"] == "completed"
    assert record["field_training"]["score"]["objective"] == "ordinal"

    model_dir = compiled.version_dir / record["artifact_path"]
    assert (model_dir / "score" / "ordinal_head.skops").exists()

    selected = select(
        spec.name,
        "bge-small",
        version=compiled.build_id,
        artifacts_root=root,
        interactive=False,
    )
    assert selected.evidence["parity"]["exact"] == 1
    with zipfile.ZipFile(selected.wheel) as archive:
        assert any(name.endswith("ordinal_head.skops") for name in archive.namelist())
