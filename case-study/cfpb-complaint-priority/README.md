# CFPB Complaint Priority Case Study

One constructed decision taken through the whole Smallbatch pipeline: freeze
public inputs, calibrate an open-weights teacher, distill its decisions into
five CPU candidates, and publish the aggregate evidence a maintainer would use
to select one candidate — or none.

It is not a benchmark of complaint correctness, legal severity, CFPB policy,
or representative consumer harm. The priority scale is constructed for this
case study and is not CFPB policy. The CFPB states that complaint narratives
are consumers' own descriptions, are not verified, and are not a
representative sample of consumer experience. Throughout, decision agreement
measures fidelity to the teacher's decisions, not correctness.

Sources:

- <https://www.consumerfinance.gov/data-research/consumer-complaints/>
- <https://cfpb.github.io/api/ccdb/api.html>
- <https://www.consumerfinance.gov/complaint/data-use/>

## The Decision

`spec.yaml` defines `review_priority`: an integer 0-4 for a public consumer
complaint, where each level is a cumulative ladder (level N requires
everything level N-1 requires, plus one new stated fact). The prompt contains
the full rubric, decision rules, and worked examples.

The ladder, with each rung's newly required fact illustrated by a real
complaint from the frozen inputs:

| level | adds the stated fact | a complaint that states it |
|---|---|---|
| 0 | a routine ask; nothing asserted wrong or unresolved | "Please provide a complete itemized accounting of all fees, costs, and expenses charged …" (checking, `22612794`) |
| 1 | something is asserted wrong; no money or access affected | "… all inaccurate, incomplete, or unverifiable information be deleted from my credit report." (debt collection, `22613257`) |
| 2 | money or account access is affected | "They are holding my funds and have frozen my accounts for no reason." (checking, `22627671`) |
| 3 | the affected amount is $1,000+, or the loss of a home, vehicle, or essential funds continues | "I deposited a Truist-issued check for {$11000.00} from a law firm 's …" (checking, `22610663`) |
| 4 | a date or deadline makes the loss permanent, or the person cannot now pay for basics | "… past due bill around XXXX dollars and a auction date of XX/XX/XXXX." (mortgage, `22652424`) |

Excerpts are verbatim from the published `items.jsonl` (CFPB scrubbing
replaces dates and identifiers with `XX..` and rounds amounts in braces).
Each excerpt was chosen because it states the fact its rung newly requires;
the placements illustrate the ladder and are not teacher decisions, which
remain unpublished per the protocol.

The trained students never see that rubric. At inference they receive only
`product`, `issue`, and `narrative` — a short prompt with no instructions in
it. The rubric exists only in training: it is what the teacher applies and
what distillation moves into the students' weights. The zero-shot diagnostics
below are the opposite arrangement (full rubric in context, no training),
which is what makes the two comparable.

## Frozen Inputs

`prepare.py` fetched 600 complaint narratives from the official CFPB API
(license reported as `CC0`), normalized and deduplicated them, kept narratives
of 200-4000 characters, and balanced across product categories round-robin.
The complaint IDs and content hashes were frozen in `frozen_ids.json`, and the
spec's prompt hash was frozen, before any teacher access. Narratives are
opt-in and scrubbed under the CFPB's scrubbing standard; the freeze was
additionally pattern-scanned for PII. The rights review is in `TERMS.md`
(signed 2026-07-14).

## Teacher

The teacher is the open-weights `openai/gpt-oss-120b` (Apache 2.0),
self-hosted with vLLM on a rented single H100 and reached through an SSH
tunnel as an OpenAI-compatible endpoint. No hosted-model provider holds
rights over its outputs.

Because there are no gold labels, the only measurable property of a rubric
draft is whether the teacher applies it the same way twice. Each draft was
probed by labeling the same 100 frozen complaints twice — the second pass
with shuffled row order and reversed input fields, at temperature 0, so any
disagreement is presentation sensitivity rather than sampling noise
(`work/rubric-probe/`, local). Draft r1 repeated itself on 77% of complaints
with flips concentrated at the 0/1 boundary; r2 fixed that boundary but
wobbled at 3/4 (76% repeat-exact); r3 tightened level 4 to require a stated
date, deadline, or present inability to pay for basics, and repeated itself
on 76% of the same 100 complaints with the 0/1 flips resolved. Interactive
calibration on the final rubric showed 90% repeat-exact on the reviewed
batch, and the maintainer approved. Calibration approves or declines teacher
behavior; it never edits decisions.

The shipped rubric's consistency was then measured on the exact 120
evaluation rows the candidates are scored against: labeling them twice, the
teacher repeated its own decision **82% of the time** (within one rung 97%,
MAE 0.22; aggregates in
[`results/teacher_consistency.json`](results/teacher_consistency.json)). That
82% is the reference ceiling for the report below — a student cannot reliably
reproduce the teacher's decisions more often than the teacher reproduces them
itself, so the 4B student's 69% exact agreement is 85% of what is achievable.

The approved teacher labeled all 600 complaints. The label distribution is
concentrated: 0:3, 1:269, 2:168, 3:148, 4:12. That concentration shapes what
the metrics below can and cannot say.

## Split and Candidates

Smallbatch created a proportional stratified split: 420 train / 60 dev / 120
evaluation. Evaluation rows are sticky and never feed training.

Five candidates were configured, all trained on the same 420 decisions and
evaluated on the same 120, in one build on one machine (a rented A10 VM with
30 vCPUs; the GPU is used for training only — every evaluation and profile
below is CPU). All ordinal machinery is engaged: the TF-IDF and SetFit heads
are cumulative-link ordinal classifiers, and the LoRA students train with a
class NLL + ranked-probability-score loss, decide in a single forward pass,
and select their decoder (argmax vs. median) on the dev split. Total rented
GPU cost for teacher labeling plus the build: roughly $5.

## Results

Evaluation: n=120, no invalid outputs from any candidate. Exact agreement is
with the teacher's decision; MAE and Spearman are over the 0-4 scale. Latency
is single-decision p50 at 4 threads on an Intel Xeon Platinum 8358
(`results/results.json` has CIs, error histograms, and full profiles).

| candidate | backend | exact (95% CI) | MAE | Spearman | p50 latency | peak RSS | trained state (+ shared base) |
|---|---|---|---:|---:|---:|---:|---|
| tfidf | TF-IDF, ordinal head | 0.54 (0.45-0.63) | 0.60 | 0.37 | 1.2 ms | 0.8 GB | 27 MB |
| bge-small | SetFit, ordinal head | 0.57 (0.48-0.65) | 0.55 | 0.47 | 42 ms | 1.8 GB | 135 MB |
| qwen3-06b | LoRA on Qwen3-0.6B | 0.63 (0.54-0.71) | 0.45 | 0.59 | 781 ms | 5.3 GB | 56 MB + 1.5 GB |
| qwen3-17b | LoRA on Qwen3-1.7B (qlora) | 0.64 (0.55-0.72) | 0.43 | 0.61 | 1.9 s | 11.1 GB | 51 MB + 4.1 GB |
| qwen3-4b | LoRA on Qwen3-4B (qlora) | 0.69 (0.60-0.77) | 0.38 | 0.64 | 4.3 s | 20.7 GB | 82 MB + 8.1 GB |

Diagnostics (never selectable), evaluated on the same 120 rows:

| diagnostic | exact | MAE | p50 latency |
|---|---:|---:|---:|
| train-fitted constant (always 2) | 0.28 | 0.74 | — |
| Qwen3-0.6B zero-shot, full rubric in context | 0.18 | 1.49 | 3.9 s |
| Qwen3-1.7B zero-shot, full rubric in context | 0.03 | 2.03 | 7.2 s |
| Qwen3-4B zero-shot, full rubric in context | 0.48 | 0.73 | 16.5 s |

## Reading the Results

**Distillation moved the rubric into the weights.** Each LoRA student beats
its own base model given the complete rubric in context, while running about
5x faster because its prompt carries no rubric: 0.6B goes from 0.18 exact at
3.9 s (prompted) to 0.63 at 781 ms (trained); 4B from 0.48 at 16.5 s to 0.69
at 4.3 s. The 420 teacher decisions taught these bases more of this rubric
than the rubric itself does in context.

**Each rung of fidelity costs about an order of magnitude of latency.**
1 ms buys 0.54 exact; 42 ms buys 0.57; 0.8 s buys 0.63; 4.3 s buys 0.69.
Whether any rung is worth it depends on the workload, which is why the report
declares no winner and `explicit_selection` is null. No candidate was
selected.

**Within-one is uninformative on this label distribution.** The train-fitted
constant scores 0.975 within-one because 88% of eval labels sit in levels
1-3. Exact agreement, MAE, and rank correlation carry the signal here;
within-one flatters everything.

**Adjacent candidates are not separated.** 120 evaluation rows separate the
ends of the table, not neighbors — the exact-agreement CIs of qwen3-06b and
qwen3-17b overlap almost completely. And as `results.json` states: these
candidates share one evaluation split, so selecting after comparison makes
the selected result optimistic; v0.2 does not provide an independent
confirmation set.

**Latency scales with cores for the LoRA candidates only.** The published
latencies are at 4 threads. `results/thread_scaling.json` re-profiles every
candidate at 4/8/16/30 threads on the same machine: the LoRA candidates reach
2.4-2.9x at 16 threads (then regress at 30), SetFit gains 1.5x, and TF-IDF is
flat at ~1.2 ms regardless.

## Reproducing

```bash
python case-study/cfpb-complaint-priority/prepare.py

smallbatch label case-study/cfpb-complaint-priority/spec.yaml \
  --items case-study/cfpb-complaint-priority/items.jsonl \
  --out case-study/cfpb-complaint-priority/work/data

set -o pipefail
python case-study/cfpb-complaint-priority/run.py --cpu-threads 4 \
  2>&1 | tee case-study/cfpb-complaint-priority/work/run.log

python case-study/cfpb-complaint-priority/thread_scaling.py --threads 4 8 16 30
```

`prepare.py` requires network access. It requests JSON explicitly, follows
the API's search-after breakpoints, rejects repeated pages, validates the
API-reported `CC0` license, and retries bounded transient upstream errors.
Labeling requires an OpenAI-compatible endpoint serving
`openai/gpt-oss-120b` at the spec's `base_url` (this run used self-hosted
vLLM on a rented H100, tunneled to `127.0.0.1:8000`). Model downloads and
LoRA training requirements remain candidate-specific; the LoRA candidates
were trained on a rented 24 GB GPU.

Teacher decisions are not deterministic across serving stacks, so a
reproduction yields its own labeled dataset and its own calibration decision,
not byte-identical results.

## Publication Scope

Published here: the frozen inputs (public CC0 complaint text), IDs and
hashes, the prompt, the scripts, and aggregate results
(`results/results.json`, `results/report.md`, `results/protocol.json`,
`results/thread_scaling.json`). Not published, by protocol: per-complaint
teacher decisions, journals, per-row disagreements, and trained artifacts.
`TERMS.md` records the rights review and required disclosures.

- The priority scale is constructed and is not CFPB policy.
- Narratives are not verified and are not representative of all consumers.
- Decision agreement is not correctness, fairness, or legal validation.
- CPU latency, memory, and footprint are not energy measurements.
