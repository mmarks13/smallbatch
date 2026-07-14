import hashlib
import importlib.util
import io
import json
import urllib.error
import urllib.parse
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
    # the rubric is an ordinal ladder: cumulative levels, a constrained output,
    # and decision rules that keep uncertainty out of the scale
    assert "Do not infer facts, motives, or legal violations" in spec.prompt
    assert "Return exactly one of: 0, 1, 2, 3, 4." in spec.prompt
    for level in range(5):
        assert f"\n{level} - " in spec.prompt
    assert set(spec.candidates) == {
        "tfidf",
        "bge-small",
        "qwen3-06b",
        "qwen3-17b",
        "qwen3-4b",
    }


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


def test_prepare_requests_json_and_retries_transient_api_failures(monkeypatch):
    module = load_prepare()
    requests = []
    sleeps = []

    class Headers:
        @staticmethod
        def get_content_type():
            return "application/json"

    class Response(io.BytesIO):
        headers = Headers()

        def __enter__(self):
            return self

        def __exit__(self, *args):
            self.close()

    def urlopen(request, timeout):
        requests.append((request, timeout))
        if len(requests) == 1:
            raise urllib.error.HTTPError(request.full_url, 503, "temporary", {}, None)
        return Response(b'{"_meta": {"license": "CC0"}, "hits": {"hits": []}}')

    monkeypatch.setattr(module.urllib.request, "urlopen", urlopen)

    payload = module.fetch_json("https://example.test", sleep=sleeps.append)

    assert payload["_meta"]["license"] == "CC0"
    assert requests[0][0].get_header("Accept") == "application/json"
    assert "github.com/mmarks13/smallbatch" in requests[0][0].get_header("User-agent")
    assert requests[0][1] == 60
    assert sleeps == [1]


def test_prepare_follows_cfpb_search_after_breakpoints(monkeypatch):
    module = load_prepare()
    monkeypatch.setattr(module, "POOL_TARGET", 4)
    monkeypatch.setattr(module, "PAGE_SIZE", 2)
    urls = []
    responses = [
        {
            "_meta": {"license": "CC0", "break_points": {"2": [123, "2"]}},
            "hits": {
                "hits": [
                    {"_source": {"complaint_id": "1"}},
                    {"_source": {"complaint_id": "2"}},
                ]
            },
        },
        {
            "_meta": {"license": "CC0", "break_points": {"3": [456, "4"]}},
            "hits": {
                "hits": [
                    {"_source": {"complaint_id": "3"}},
                    {"_source": {"complaint_id": "4"}},
                ]
            },
        },
    ]

    def fetch_json(url):
        urls.append(url)
        return responses.pop(0)

    monkeypatch.setattr(module, "fetch_json", fetch_json)

    rows = module.fetch_pool()

    assert [row["complaint_id"] for row in rows] == ["1", "2", "3", "4"]
    first = urllib.parse.parse_qs(urllib.parse.urlsplit(urls[0]).query)
    second = urllib.parse.parse_qs(urllib.parse.urlsplit(urls[1]).query)
    assert first["frm"] == ["0"]
    assert "page" not in first
    assert "search_after" not in first
    assert second["frm"] == ["2"]
    assert second["page"] == ["2"]
    assert second["search_after"] == ["123_2"]


def test_prepare_rejects_repeated_api_pages(monkeypatch):
    module = load_prepare()
    monkeypatch.setattr(module, "POOL_TARGET", 4)
    monkeypatch.setattr(module, "PAGE_SIZE", 2)
    payload = {
        "_meta": {"license": "CC0", "break_points": {"2": [123, "2"]}},
        "hits": {
            "hits": [
                {"_source": {"complaint_id": "1"}},
                {"_source": {"complaint_id": "2"}},
            ]
        },
    }
    monkeypatch.setattr(module, "fetch_json", lambda url: payload)

    with pytest.raises(RuntimeError, match="repeated complaint IDs"):
        module.fetch_pool()


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
        narrative = (f"complaint narrative {index} " * 12).strip()
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
        "source": module.EXPECTED_SOURCE,
        "api_license": module.EXPECTED_API_LICENSE,
        "extracted_at": "2026-07-13T16:17:57+00:00",
        "selection": module.EXPECTED_SELECTION,
        "count": 2,
        "spec_sha256": hashlib.sha256(b"spec").hexdigest(),
        "items_sha256": hashlib.sha256(items_text.encode()).hexdigest(),
        "complaints": complaints,
    }
    (tmp_path / "frozen_ids.json").write_text(json.dumps(frozen))
    assert module.verify_frozen() == frozen

    frozen["source"] = "https://example.test/not-cfpb"
    (tmp_path / "frozen_ids.json").write_text(json.dumps(frozen))
    with pytest.raises(RuntimeError, match="frozen source mismatch"):
        module.verify_frozen()
    frozen["source"] = module.EXPECTED_SOURCE

    issue = items[0]["input"].pop("issue")
    invalid_schema = "".join(json.dumps(item) + "\n" for item in items)
    (tmp_path / "items.jsonl").write_text(invalid_schema)
    frozen["items_sha256"] = hashlib.sha256(invalid_schema.encode()).hexdigest()
    (tmp_path / "frozen_ids.json").write_text(json.dumps(frozen))
    with pytest.raises(RuntimeError, match="input schema mismatch"):
        module.verify_frozen()
    items[0]["input"]["issue"] = issue

    items[0]["input"]["narrative"] = ("changed complaint narrative " * 12).strip()
    tampered = "".join(json.dumps(item) + "\n" for item in items)
    (tmp_path / "items.jsonl").write_text(tampered)
    frozen["items_sha256"] = hashlib.sha256(tampered.encode()).hexdigest()
    (tmp_path / "frozen_ids.json").write_text(json.dumps(frozen))
    with pytest.raises(RuntimeError, match="content hash mismatch"):
        module.verify_frozen()
