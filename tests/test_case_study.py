import hashlib
import importlib.util
import json
from pathlib import Path

import pytest

from smallbatch.spec import load_spec

CASE = Path(__file__).parents[1] / "case-study" / "cfpb-complaint-priority"


def load_prepare():
    spec = importlib.util.spec_from_file_location("cfpb_prepare", CASE / "prepare.py")
    module = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(module)
    return module


def load_run():
    spec = importlib.util.spec_from_file_location("cfpb_run", CASE / "run.py")
    module = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(module)
    return module


def test_case_spec_is_prompt_first_and_frozen_scale():
    spec = load_spec(CASE / "spec.yaml")
    assert spec.name == "complaint-review-priority"
    assert spec.output.scalar.range == (0, 4)
    assert "Do not infer legal violations" in spec.prompt
    assert set(spec.candidates) == {"tfidf", "bge-small", "granite-350m"}


def test_prepare_normalizes_deduplicates_and_balances():
    module = load_prepare()
    narrative = "material unresolved complaint " * 12
    rows = [
        {
            "complaint_id": index,
            "complaint_what_happened": narrative + str(index),
            "product": f"product {index % 3}",
            "issue": "issue",
        }
        for index in range(650)
    ]
    rows.append(dict(rows[0]))
    normalized = module.normalize(rows)
    assert len(normalized) == 650
    selected = module.balanced_sample(normalized)
    assert len(selected) == 600
    counts = {}
    for row in selected:
        product = row["input"]["product"]
        counts[product] = counts.get(product, 0) + 1
    assert max(counts.values()) - min(counts.values()) <= 1


def test_case_requires_every_candidate_to_complete():
    module = load_run()
    complete = {
        "candidates": {
            name: {"status": "completed"} for name in module.EXPECTED_CANDIDATES
        }
    }
    module.require_all_candidates(complete)
    complete["candidates"]["bge-small"] = {"status": "error"}
    with pytest.raises(RuntimeError, match="bge-small=error"):
        module.require_all_candidates(complete)


def test_frozen_inputs_verify_provenance_and_content_hashes(tmp_path, monkeypatch):
    module = load_run()
    monkeypatch.setattr(module, "HERE", tmp_path)
    monkeypatch.setattr(module, "EXPECTED_COUNT", 2)
    (tmp_path / "spec.yaml").write_text("spec")
    complaints = []
    items = []
    for index in range(2):
        narrative = f"complaint narrative {index}"
        content_hash = hashlib.sha256(narrative.encode()).hexdigest()
        provenance = {"complaint_id": str(index), "content_sha256": content_hash}
        complaints.append(provenance)
        items.append(
            {
                "input": {"product": "x", "issue": "y", "narrative": narrative},
                "provenance": provenance,
            }
        )
    items_text = "".join(json.dumps(item) + "\n" for item in items)
    (tmp_path / "items.jsonl").write_text(items_text)
    frozen = {
        "count": 2,
        "spec_sha256": hashlib.sha256(b"spec").hexdigest(),
        "items_sha256": hashlib.sha256(items_text.encode()).hexdigest(),
        "complaints": complaints,
    }
    (tmp_path / "frozen_ids.json").write_text(json.dumps(frozen))
    assert module.verify_frozen() == frozen

    items[0]["input"]["narrative"] = "changed"
    tampered = "".join(json.dumps(item) + "\n" for item in items)
    (tmp_path / "items.jsonl").write_text(tampered)
    frozen["items_sha256"] = hashlib.sha256(tampered.encode()).hexdigest()
    (tmp_path / "frozen_ids.json").write_text(json.dumps(frozen))
    with pytest.raises(RuntimeError, match="content hash mismatch"):
        module.verify_frozen()
