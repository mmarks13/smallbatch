"""Export: grammar/Modelfile generation and artifact resolution (torch-free)."""

import pytest

from smallbatch import artifacts
from smallbatch.artifacts import resolve_version
from smallbatch.export import find_llama_cpp, gbnf_grammar, modelfile
from smallbatch.spec import FunctionSpec


def make_spec(**over):
    base = dict(
        name="toy",
        description="Score.",
        input_schema={"title": "str"},
        output={"type": "int", "range": [0, 3]},
        rubric="-",
        teacher={"backend": "claude-cli", "model": "sonnet"},
    )
    base.update(over)
    return FunctionSpec(**base)


def test_int_grammar_enumerates_range():
    g = gbnf_grammar(make_spec())
    assert 'value ::= "0" | "1" | "2" | "3"' in g
    assert g.startswith('root ::= " "? value')


def test_enum_grammar_escapes():
    g = gbnf_grammar(
        make_spec(output={"type": "enum", "labels": ["a\"b", "plain"]})
    )
    assert '"a\\"b" | "plain"' in g


def test_rationale_grammar_shape():
    spec = make_spec(train={"rationale_distillation": True})
    g = gbnf_grammar(spec)
    assert '"reason: " reason "\\nscore: " value' in g
    assert "reason ::= [^\\n]+" in g


def test_huge_range_rejected():
    with pytest.raises(ValueError, match="too large"):
        gbnf_grammar(make_spec(output={"type": "int", "range": [0, 100000]}))


def test_modelfile_contents():
    m = modelfile(make_spec(), "toy.q4_k_m.gguf")
    assert "FROM ./toy.q4_k_m.gguf" in m
    assert 'TEMPLATE """{{ .Prompt }}"""' in m
    assert "PARAMETER temperature 0" in m
    assert "PARAMETER num_predict 16" in m
    assert "num_predict 96" in modelfile(
        make_spec(train={"rationale_distillation": True}), "x.gguf"
    )


def test_find_llama_cpp_missing():
    with pytest.raises(FileNotFoundError, match="LLAMA_CPP_DIR"):
        find_llama_cpp("/nonexistent")


def _artifact(tmp_path, passed):
    d = artifacts.new_version_dir(tmp_path, "toy")
    artifacts.write_manifest(d, {"gate": {"passed": passed}})
    return d


def test_resolve_refuses_failed_gate(tmp_path):
    _artifact(tmp_path, passed=False)
    with pytest.raises(FileNotFoundError):  # no passing artifact at all
        resolve_version(tmp_path, "toy", None, allow_failed=False)
    d = resolve_version(tmp_path, "toy", None, allow_failed=True)
    assert d.exists()


def test_resolve_explicit_version_still_gated(tmp_path):
    d = _artifact(tmp_path, passed=False)
    with pytest.raises(ValueError, match="allow-failed"):
        resolve_version(tmp_path, "toy", d.name, allow_failed=False)
