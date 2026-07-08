"""Hub push: model card rendering (torch-free, no network)."""

from smallbatch.hub import model_card
from smallbatch.spec import FunctionSpec

SPEC = FunctionSpec(
    name="toy",
    description="Score a thing.",
    input_schema={"title": "str", "body": "str"},
    output={"type": "int", "range": [0, 10]},
    rubric="-",
    teacher={"backend": "claude-cli", "model": "sonnet"},
)

MANIFEST = {
    "base_model": "org/Base-350M",
    "gate": {"passed": True, "reasons": []},
    "metrics": {
        "adapter": {"n": 22, "agreement": 0.9091, "invalid_rate": 0.0},
        "zeroshot": {"n": 22, "agreement": 0.1364, "invalid_rate": 0.2},
    },
    "data": {"real": 147, "variants": 458, "teacher_model": "sonnet",
             "teacher_backend": "claude-cli"},
    "smallbatch_version": "0.1.0",
}


def test_model_card_contents():
    card = model_card(SPEC, MANIFEST, "user/toy")
    assert card.startswith("---\nbase_model: org/Base-350M")
    assert "library_name: peft" in card
    assert "# toy" in card and "Score a thing." in card
    assert "`title`, `body`" in card
    assert "| adapter | 90.9% | 0.0% |" in card
    assert "| zero-shot base | 13.6% |" in card
    assert "PASS" in card
    assert 'PeftModel.from_pretrained(base, "user/toy")' in card
    assert "inherits its license" in card
    assert "147 real items + 458" in card


def test_model_card_failed_gate_and_no_zeroshot():
    manifest = dict(MANIFEST)
    manifest["gate"] = {"passed": False, "reasons": ["agreement 70% < required 85%"]}
    manifest["metrics"] = {"adapter": {"n": 22, "agreement": 0.7, "invalid_rate": 0.0},
                           "zeroshot": None}
    card = model_card(SPEC, manifest, "user/toy")
    assert "FAIL" in card and "agreement 70% < required 85%" in card
    assert "zero-shot" not in card.split("## Usage")[0].split("|---|")[1]
