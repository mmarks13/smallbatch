import importlib.util
from pathlib import Path

from smallbatch.spec import load_spec

CASE = Path(__file__).parents[1] / "case-study" / "cfpb-complaint-priority"


def load_prepare():
    spec = importlib.util.spec_from_file_location("cfpb_prepare", CASE / "prepare.py")
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
