"""Freeze 600 public CFPB complaint narratives for the v0.2 case study."""

from __future__ import annotations

import argparse
import datetime
import hashlib
import json
import re
import sys
import time
import urllib.error
import urllib.parse
import urllib.request
from collections import defaultdict
from pathlib import Path

HERE = Path(__file__).resolve().parent
API = "https://www.consumerfinance.gov/data-research/consumer-complaints/search/api/v1/"
POOL_TARGET = 3000
FINAL_COUNT = 600
PAGE_SIZE = 100
API_LICENSE = "CC0"
REQUEST_HEADERS = {
    "Accept": "application/json",
    "User-Agent": "smallbatch-case-study/0.2 (+https://github.com/mmarks13/smallbatch)",
}
RETRYABLE_HTTP_STATUS = {424, 429, 500, 502, 503, 504}


def atomic_write(path: Path, text: str) -> None:
    tmp = path.with_suffix(path.suffix + ".tmp")
    tmp.write_text(text)
    tmp.replace(path)


def fetch_json(url: str, *, attempts: int = 4, sleep=time.sleep) -> dict:
    for attempt in range(attempts):
        request = urllib.request.Request(url, headers=REQUEST_HEADERS)
        try:
            with urllib.request.urlopen(request, timeout=60) as response:
                content_type = response.headers.get_content_type()
                if content_type != "application/json":
                    raise RuntimeError(f"CFPB API returned unexpected content type {content_type!r}")
                return json.load(response)
        except urllib.error.HTTPError as exc:
            if exc.code not in RETRYABLE_HTTP_STATUS or attempt + 1 == attempts:
                raise RuntimeError(f"CFPB API request failed with HTTP {exc.code}") from exc
        except (TimeoutError, urllib.error.URLError) as exc:
            if attempt + 1 == attempts:
                raise RuntimeError(f"CFPB API request failed after {attempts} attempts") from exc
        sleep(2**attempt)
    raise AssertionError("unreachable")


def fetch_pool() -> list[dict]:
    rows = []
    seen_ids = set()
    page = 1
    search_after = None
    while len(rows) < POOL_TARGET:
        params = {
            "has_narrative": "true",
            "no_aggs": "true",
            "no_highlight": "true",
            "sort": "created_date_desc",
            "size": PAGE_SIZE,
            "frm": (page - 1) * PAGE_SIZE,
        }
        if search_after is not None:
            params.update({"page": page, "search_after": search_after})
        query = urllib.parse.urlencode(params)
        payload = fetch_json(API + "?" + query)
        license_name = payload.get("_meta", {}).get("license")
        if license_name != API_LICENSE:
            raise RuntimeError(
                f"CFPB API reported license {license_name!r}; expected {API_LICENSE!r}"
            )
        hits = payload.get("hits", {}).get("hits", [])
        if not hits:
            break
        sources = [hit.get("_source", {}) for hit in hits]
        page_ids = []
        for source in sources:
            if not isinstance(source, dict):
                raise RuntimeError("CFPB API returned a non-object complaint")
            raw_id = source.get("complaint_id")
            complaint_id = "" if raw_id is None else str(raw_id).strip()
            if not complaint_id:
                raise RuntimeError("CFPB API returned a complaint without an ID")
            page_ids.append(complaint_id)
        if len(set(page_ids)) != len(page_ids):
            raise RuntimeError("CFPB API returned duplicate complaint IDs within a page")
        if seen_ids.intersection(page_ids):
            raise RuntimeError("CFPB API pagination repeated complaint IDs from an earlier page")
        seen_ids.update(page_ids)
        rows.extend(sources)
        print(
            f"CFPB fetch page={page} hits={len(hits)} accumulated={len(rows)}",
            file=sys.stderr,
        )
        if len(rows) >= POOL_TARGET:
            break
        page += 1
        break_point = payload.get("_meta", {}).get("break_points", {}).get(str(page))
        if not isinstance(break_point, list) or len(break_point) != 2:
            raise RuntimeError(f"CFPB API omitted the pagination breakpoint for page {page}")
        search_after = "_".join(str(value) for value in break_point)
    return rows


def normalize(rows: list[dict]) -> list[dict]:
    unique = {}
    for row in rows:
        narrative = re.sub(r"\s+", " ", str(row.get("complaint_what_happened") or "")).strip()
        if not 200 <= len(narrative) <= 4000:
            continue
        product = re.sub(r"\s+", " ", str(row.get("product") or "")).strip()
        issue = re.sub(r"\s+", " ", str(row.get("issue") or "")).strip()
        raw_id = row.get("complaint_id")
        complaint_id = "" if raw_id is None else str(raw_id).strip()
        if not complaint_id or not product or not issue:
            continue
        content_hash = hashlib.sha256(narrative.encode()).hexdigest()
        unique.setdefault(
            content_hash,
            {
                "complaint_id": complaint_id,
                "content_sha256": content_hash,
                "input": {"product": product, "issue": issue, "narrative": narrative},
            },
        )
    return sorted(unique.values(), key=lambda row: (row["input"]["product"], row["complaint_id"]))


def balanced_sample(rows: list[dict]) -> list[dict]:
    groups = defaultdict(list)
    for row in rows:
        groups[row["input"]["product"]].append(row)
    products = sorted(groups, key=lambda name: (-len(groups[name]), name))
    selected = []
    index = 0
    while len(selected) < FINAL_COUNT and any(groups.values()):
        product = products[index % len(products)]
        if groups[product]:
            selected.append(groups[product].pop(0))
        index += 1
    if len(selected) != FINAL_COUNT:
        raise RuntimeError(f"only {len(selected)} usable narratives were available")
    return sorted(selected, key=lambda row: int(row["complaint_id"]))


def main() -> None:
    argparse.ArgumentParser().parse_args()
    frozen_path = HERE / "frozen_ids.json"
    if frozen_path.exists():
        raise SystemExit("frozen_ids.json already exists; the case-study freeze is immutable")
    selected = balanced_sample(normalize(fetch_pool()))
    items_path = HERE / "items.jsonl"
    atomic_write(
        items_path,
        "".join(
            json.dumps(
                {
                    "input": row["input"],
                    "provenance": {
                        "complaint_id": row["complaint_id"],
                        "content_sha256": row["content_sha256"],
                    },
                },
                ensure_ascii=False,
            )
            + "\n"
            for row in selected
        ),
    )
    spec_hash = hashlib.sha256((HERE / "spec.yaml").read_bytes()).hexdigest()
    frozen = {
        "source": API,
        "api_license": API_LICENSE,
        "extracted_at": datetime.datetime.now(datetime.timezone.utc).isoformat(),
        "selection": "normalize, narrative length 200-4000, content dedupe, product round-robin",
        "count": len(selected),
        "spec_sha256": spec_hash,
        "items_sha256": hashlib.sha256(items_path.read_bytes()).hexdigest(),
        "complaints": [
            {"complaint_id": row["complaint_id"], "content_sha256": row["content_sha256"]}
            for row in selected
        ],
    }
    atomic_write(frozen_path, json.dumps(frozen, indent=2) + "\n")
    print(f"froze {len(selected)} complaints -> {items_path}")


if __name__ == "__main__":
    main()
