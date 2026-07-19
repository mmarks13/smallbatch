"""The real HF stack on CPU: no fakes, no network, no GPU.

Every other LoRA test in the default suite fakes the model and tokenizer.
These tests build a ~1M-parameter randomly initialized Llama and a real BPE
tokenizer (trained in-process on a fixed corpus, byte-level pretokenizer like
production vocabularies) and run the genuine pipeline — transformers Trainer,
PEFT wrapping, offset-mapping attribution, deterministic generate, profiling
subprocess, wheel packaging — end to end. This is what catches transformers/
TRL/PEFT API drift and tokenizer-boundary assumptions on every `pytest -q`.

The student cannot learn anything meaningful (it is tiny and the data is
tiny); these tests assert the machinery's observable contract, never quality.
"""

from __future__ import annotations

import pytest

from conftest import make_spec
from smallbatch.api import compile as compile_fn
from smallbatch.api import label, select
from smallbatch.runtime import load_fn
from smallbatch.spec import FunctionSpec

CORPUS = [
    "priority explanation score urgent normal low outage question spam",
    ' {"priority": 0, "explanation": "text"} {"priority": 1} {"priority": 2}',
    ' {"priority": 3, "explanation": "words"} {"priority": 4}',
    " 0 1 2 3 4 5 6 7 8 9 0 1 2 3 4 0 1 2 3 4 0 1 2 3 4 0 1 2 3 4",
    "the server is down for one user and every customer report ticket",
    "slow broken failing blocked cosmetic minor spam question outage",
    "One user is blocked. Every customer is blocked. A minor typo only.",
]


@pytest.fixture(scope="session")
def tiny_base(tmp_path_factory) -> str:
    """A saved directory that loads through AutoTokenizer/AutoModel like any
    Hub model: real fast tokenizer (offsets included), random tiny weights."""
    from tokenizers import Tokenizer, decoders, models, pre_tokenizers, trainers
    from transformers import LlamaConfig, LlamaForCausalLM, PreTrainedTokenizerFast

    directory = tmp_path_factory.mktemp("tiny-base")
    bpe = Tokenizer(models.BPE(unk_token=None))
    bpe.pre_tokenizer = pre_tokenizers.ByteLevel(add_prefix_space=False)
    bpe.decoder = decoders.ByteLevel()
    trainer = trainers.BpeTrainer(
        vocab_size=420,
        min_frequency=1,
        special_tokens=["</s>"],
        initial_alphabet=pre_tokenizers.ByteLevel.alphabet(),
    )
    bpe.train_from_iterator(CORPUS * 8, trainer)
    tokenizer = PreTrainedTokenizerFast(
        tokenizer_object=bpe, eos_token="</s>", pad_token="</s>"
    )
    # the ordinal codec needs one token per level, as production BPE
    # vocabularies provide; the fixture fails loudly if training missed one
    for level in range(5):
        assert len(tokenizer(f" {level}", add_special_tokens=False)["input_ids"]) == 1
    tokenizer.save_pretrained(directory)

    config = LlamaConfig(
        vocab_size=512,
        hidden_size=64,
        intermediate_size=128,
        num_hidden_layers=2,
        num_attention_heads=2,
        num_key_value_heads=2,
        max_position_embeddings=512,
        eos_token_id=tokenizer.eos_token_id,
        pad_token_id=tokenizer.pad_token_id,
    )
    import torch

    torch.manual_seed(0)
    LlamaForCausalLM(config).save_pretrained(directory)
    return str(directory)


def tiny_lora(base: str, **overrides) -> dict:
    config = {
        "type": "lora",
        "model": base,
        "precision": "fp32",
        "lora_r": 4,
        "max_epochs": 2,
        "patience": None,
        "batch_size": 4,
        "eval_batch_size": 8,
        "max_seq_len": 256,
    }
    config.update(overrides)
    return config


def bounded_records(count: int = 30) -> list[dict]:
    bodies = {
        0: "spam offer click here",
        1: "minor typo cosmetic only",
        2: "one user is blocked",
        3: "slow for many customers",
        4: "every customer is down",
    }
    return [
        {
            "input": {"title": f"ticket {index}", "body": bodies[index % 5]},
            "output": index % 5,
        }
        for index in range(count)
    ]


def test_bounded_scalar_trains_selects_and_runs_on_the_real_stack(tmp_path, tiny_base):
    """Real Trainer -> real ordinal codec on a real BPE vocabulary -> real
    constrained scoring in the profiling subprocess -> real wheel, selected
    and called. Constrained decoding keeps every output valid, so the full
    select path must succeed even though the student is random."""
    spec = make_spec(
        name="tiny-scale",
        output={"type": "int", "range": [0, 4]},
        candidates={"tiny": tiny_lora(tiny_base)},
    )
    data = tmp_path / "data"
    root = tmp_path / "artifacts"
    label(spec, bounded_records(30), out_dir=data)
    compiled = compile_fn(spec, data_dir=data, artifacts_root=root, cpu_threads=1)
    record = compiled.candidates["tiny"]
    assert record["status"] == "completed", record.get("error")

    training = record["training"]
    assert set(training["untuned_baselines"]) == {"score", "__format__"}
    assert all(value > 0 for value in training["untuned_baselines"].values())
    assert [entry["epoch"] for entry in training["curve"]] == [1, 2]
    scores = {e["epoch"]: e["checkpoint_score"] for e in training["curve"]}
    assert training["best_epoch"] == min(scores, key=scores.get)
    assert record["loss_weights"] == {"score": 1.0}
    assert record["metrics"]["invalid_rate"] == 0.0  # constrained decode

    selected = select(
        spec.name, "tiny", version=compiled.build_id,
        artifacts_root=root, interactive=False,
    )
    assert selected.evidence["parity"]["exact"] == 1.0
    function = load_fn(spec.name, artifacts_root=root)
    output = function({"title": "t", "body": "every customer is down"})
    assert output in {0, 1, 2, 3, 4}


def test_mixed_text_function_compiles_with_full_evidence_on_the_real_stack(
    tmp_path, tiny_base
):
    """The text path end to end on real tokenization: offset attribution,
    per-field losses, teacher-forced baselines and dev scoring, free-running
    generation with strict atomic validation, structural-failure counting,
    and CPU text fidelity. A random student cannot emit valid JSON, so this
    asserts the evidence contract, not quality."""
    spec = FunctionSpec(
        name="tiny-note",
        description="d",
        input_schema={"title": "string", "body": "string"},
        output={
            "priority": {"range": [0, 4]},
            "explanation": {"type": "text", "max_chars": 60},
        },
        prompt="p",
        candidates={"tiny": tiny_lora(tiny_base)},
        training={"loss_weights": {"priority": 2.0}},
    )
    notes = {
        0: "Spam with no relevance.",
        1: "A minor cosmetic issue.",
        2: "One user is blocked.",
        3: "Slow for many customers.",
        4: "Every customer is down.",
    }
    records = [
        {
            "input": record["input"],
            "output": {"priority": record["output"], "explanation": notes[record["output"]]},
        }
        for record in bounded_records(30)
    ]
    data = tmp_path / "data"
    root = tmp_path / "artifacts"
    label(spec, records, out_dir=data)
    compiled = compile_fn(spec, data_dir=data, artifacts_root=root, cpu_threads=1)
    record = compiled.candidates["tiny"]
    assert record["status"] == "completed", record.get("error")

    training = record["training"]
    assert set(training["untuned_baselines"]) == {"priority", "explanation", "__format__"}
    assert record["loss_weights"] == {"priority": 2.0, "explanation": 0.25}
    for entry in training["curve"]:
        assert set(entry["normalized_dev_losses"]) == {"priority", "explanation"}
        assert entry["checkpoint_score"] > 0

    fidelity = record["text_fidelity"]
    assert fidelity["rows"] == record["metrics"]["n"]
    assert fidelity["bits_per_byte"] > 0
    assert "not correctness" in fidelity["measures"]

    # atomic outputs: every prediction is either None or a complete valid dict
    metrics = record["metrics"]
    failures = metrics.get("structural_failures") or {}
    assert sum(failures.values()) == metrics["n"] - metrics["valid_n"]
    # the constant diagnostic is honestly absent for text functions
    assert "train-fitted-constant" not in compiled.manifest["diagnostics"]
