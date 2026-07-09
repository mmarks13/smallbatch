"""serve request handling (pure part) + init templates + export bundle README."""

import pytest

from smallbatch.export import bundle_readme
from smallbatch.init_cmd import TEMPLATES, init
from smallbatch.serve import handle_call
from smallbatch.spec import FunctionSpec, load_spec

SPEC = FunctionSpec(
    name="toy",
    description="Score.",
    input_schema={"title": "str"},
    output={"type": "int", "range": [0, 4]},
    rubric="-",
    teacher={"backend": "claude-cli", "model": "sonnet"},
)


def test_handle_call_happy_path():
    status, payload = handle_call(SPEC, {"title": "x"}, lambda prompt: " 3")
    assert status == 200 and payload["output"] == 3


def test_handle_call_input_validation():
    status, payload = handle_call(SPEC, {"nope": 1}, lambda p: " 3")
    assert status == 400 and "title" in payload["error"]
    status, _ = handle_call(SPEC, [1, 2], lambda p: " 3")
    assert status == 400


def test_handle_call_contract_violation_and_backend_error():
    status, payload = handle_call(SPEC, {"title": "x"}, lambda p: "banana")
    assert status == 422 and payload["raw"] == "banana"

    def boom(p):
        raise ConnectionError("down")

    status, _ = handle_call(SPEC, {"title": "x"}, boom)
    assert status == 502


def test_init_templates_produce_loadable_specs(tmp_path):
    for template in TEMPLATES:
        out = init(template, f"fn-{template}", directory=str(tmp_path / template))
        spec = load_spec(out / "spec.yaml")  # TODO placeholders still validate
        assert spec.name == f"fn-{template}"
        assert (out / "items.json").exists()
    with pytest.raises(FileExistsError):
        init("scorer", "again", directory=str(tmp_path / "classifier"))


def test_bundle_readme_mentions_all_runtimes():
    manifest = {
        "base_model": "org/base",
        "gate": {"passed": True, "reasons": []},
        "metrics": {"adapter": {"agreement": 0.9, "agreement_ci": [0.8, 0.95], "n": 60}},
        "data": {"teacher_model": "sonnet", "teacher_backend": "claude-cli"},
    }
    md = bundle_readme(SPEC, manifest, "toy.q4_k_m.gguf")
    for needle in ("llama-cli", "llama-server", "ollama create", "smallbatch serve",
                   "PASS", "95% CI", "grammar"):
        assert needle in md, f"missing {needle!r}"
