# CFPB Complaint Priority Case Study

This case study exercises Smallbatch on a constructed prompt-driven decision:
assign a `review_priority` from 0 to 4 to a public CFPB consumer complaint.

It is not a benchmark of complaint correctness, legal severity, CFPB policy,
or representative consumer harm. The priority prompt is an illustrative first
draft authored for this case study. The CFPB states that complaint narratives
are consumers' descriptions, are not verified, and are not a representative
sample of consumer experience.

Sources:

- <https://www.consumerfinance.gov/data-research/consumer-complaints/>
- <https://cfpb.github.io/api/ccdb/api.html>
- <https://www.consumerfinance.gov/complaint/data-use/>

## Frozen Protocol

1. `prepare.py` fetches public narratives from the official API, normalizes and
   deduplicates them, balances across available product categories, and freezes
   600 complaint IDs and content hashes. The freeze is immutable; creating a
   different sample requires a separately versioned protocol.
2. `spec.yaml` and its prompt hash are frozen before any teacher access. The
   prompt is not revised after calibration; an unacceptable first draft stops
   the case study.
3. The open-weights `openai/gpt-oss-120b` (Apache 2.0), self-hosted behind an
   OpenAI-compatible endpoint on a rented single H100, labels the inputs after
   interactive calibration. Smallbatch creates the fixed 420/60/120 split.
4. TF-IDF, SetFit, and LoRA candidates run through full CPU evaluation. The
   report contains decision-fidelity and operating evidence only. All three
   configured candidates must complete, but there is no minimum fidelity
   threshold and no PASS/FAIL verdict.
5. A maintainer may explicitly select one candidate or select none. The script
   never chooses automatically.

## Commands

```bash
python case-study/cfpb-complaint-priority/prepare.py

smallbatch label case-study/cfpb-complaint-priority/spec.yaml \
  --items case-study/cfpb-complaint-priority/items.jsonl \
  --out case-study/cfpb-complaint-priority/work/data

set -o pipefail
python case-study/cfpb-complaint-priority/run.py --cpu-threads 4 \
  2>&1 | tee case-study/cfpb-complaint-priority/work/run.log
```

`prepare.py` requires network access. It requests JSON explicitly, follows the
API's search-after breakpoints, rejects repeated pages, validates the
API-reported `CC0` license, and retries bounded transient upstream errors.
Labeling requires the maintainer's own authorized Codex access. Model downloads
and LoRA training requirements remain candidate-specific.

## Publication

The frozen public inputs, IDs, hashes, prompt, scripts, and aggregate results
may be published after the checks in `TERMS.md`. Teacher decisions and trained
packages remain blocked until a dated rights review explicitly permits each.
