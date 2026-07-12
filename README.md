<p align="center">
  <img src="https://raw.githubusercontent.com/mmarks13/smallbatch/main/assets/smallbatch_banner.png" alt="smallbatch — distill once, own the function" width="100%">
</p>

*Turn a rubric and representative examples into a tested local classifier.*

> **Project status: experimental.** The CPU-only unit suite covers the spec,
> data, metric, report, and orchestration layers; the complete pipeline has
> been exercised end-to-end on a small number of tasks. A passing
> teacher-agreement gate is **not** ground-truth accuracy — see
> [What the gate means](#what-the-gate-means). Interfaces and the artifact
> format may still change before 1.0.

Lots of useful functions are fuzzy: *score this item 0–10 against a rubric*,
*classify this ticket as urgent/normal/low*. A frontier model does these well
from a prompt — so teams end up renting one forever for a task that never
changes.

smallbatch is a **classifier compiler** for exactly that shape of function.
You write the rubric as a YAML spec and provide representative inputs; a
teacher model you choose labels them **during dataset creation** (labeling is
batched — it's "a teacher phase," not literally one call); then the compiler
trains **two candidates on the same labels** — a TF-IDF + logistic-regression
pipeline (KBs, CPU-only) and a LoRA adapter on a small base model — scores
both against a held-out gate, and selects the winner (highest teacher
agreement; within a small margin, the smaller artifact wins). After that, the
function runs locally with **no hosted-model API fee** at runtime (local
compute is still yours to pay for).

Outputs are deliberately constrained — an integer in a range, one label from
a fixed list, or several such fields at once. That narrowness is the point:
it's the regime where a small student can genuinely match its teacher, it
makes quality measurable per field, and it keeps compiled functions squarely
in "specialized classifier" territory (see
[Responsible use](#responsible-use)).

## When to use it — and when not to

Use smallbatch when the output is a stable set of labels or a bounded score,
the decision runs often enough to be worth compiling, and you can provide
representative inputs (ideally some with trusted `gold` answers to judge
against). Keep the API call when volume is low, the rubric changes weekly, or
the task needs current knowledge, long reasoning, or open-ended text. If a
regex or a SQL expression already solves it, use that.

## Install

```bash
pip install smallbatch            # + [qlora] for 4-bit training of 3-9B bases
```

Compilation trains a LoRA candidate, so **a CUDA GPU is required to compile**
(minutes for the default 350M base — [docs/local-gpu.md](docs/local-gpu.md),
or rent one per compile: [docs/cloud.md](docs/cloud.md)). Match your torch
build to your GPU — old and very new cards both need specific wheels. A
compiled function whose winner is the TF-IDF candidate then runs CPU-only.

## Quickstart

A complete runnable example ships in
[`examples/ticket-priority/`](examples/ticket-priority/) — a support-ticket
priority classifier with 71 bundled synthetic tickets and a zero-API-key
teacher config (local Ollama):

```bash
# 0. preflight: teacher reachable? GPU/precision sane? splits viable?
smallbatch doctor examples/ticket-priority/spec.yaml \
    --items examples/ticket-priority/items.json

# 1. the teacher labels the items into a train/dev/gate dataset
#    (journaled: a crash resumes without re-paying completed calls)
smallbatch label examples/ticket-priority/spec.yaml \
    --items examples/ticket-priority/items.json

# 2. inspect/correct the teacher's labels before spending GPU time
smallbatch review examples/ticket-priority/spec.yaml

# 3. train both candidates + eval report + acceptance gate
smallbatch compile examples/ticket-priority/spec.yaml
#    -> artifacts/ticket-priority/<date>/  (candidates + report.md + manifest)

# 4. call it
smallbatch run ticket-priority --json '{"subject": "Site down", "body": "...", "product_area": "auth", "customer_tier": "pro"}'
smallbatch status        # functions, verdicts, winners, integrity/drift
```

(Starting from scratch instead? `smallbatch init classifier my-fn` writes a
working spec skeleton.)

Or from Python:

```python
import json, smallbatch

items = json.load(open("examples/ticket-priority/items.json"))
smallbatch.label("examples/ticket-priority/spec.yaml", items)
result = smallbatch.compile("examples/ticket-priority/spec.yaml")   # CompileResult
fn = smallbatch.load_fn("ticket-priority")
fn({"subject": "Site down", "body": "...", "product_area": "auth", "customer_tier": "pro"})
# -> "urgent"
```

## What the gate means

The acceptance gate measures **agreement with the teacher's held-out labels**
— imitation, not ground truth. A teacher that misreads your rubric produces a
student that faithfully misreads it too, and the gate cannot see that. Two
things keep this honest:

- **Gold labels.** Any item may carry a `"gold"` value — an answer you trust
  independently (a human decision, a historical outcome). Gold rows are
  routed to the gate, never trained on, and the report then shows three
  numbers side by side: teacher-vs-gold (is the teacher right?),
  student-vs-gold (is the function right?), and student-vs-teacher. A
  teacher that scores poorly against your gold gets called out loudly.
- **Baselines.** Every candidate must beat the zero-shot base model and the
  best constant prediction (an *oracle* constant picked on the gate labels —
  a hurdle, not a deployable model) before it can pass.

Exit codes are load-bearing: **0** gate pass, **2** honest fail, **1** error.
When every candidate fails, compile shows a full decision table (both
candidates, baselines, gold breakdown, error margins) and lets you *explicitly*
accept the best one for deployment — recorded in the manifest, surfaced by
`status` as `IN USE - GATE FAIL`, exit code still 2.

## What you get

- **A spec, not a script.** One diffable YAML file defines the function:
  input fields, output contract, and the rubric the teacher labels by.
  Build-setting changes (base model, precision) never invalidate your labeled
  data; rubric/contract/teacher changes make compile refuse stale labels.
- **Two candidates, one honest comparison.** Every compile fits the TF-IDF
  candidate (seconds, CPU) and the LoRA adapter on the same labels and gates
  both. When the linear model wins, that's your artifact — hundreds of KB,
  no base model needed. Both are scored on the same gate, so the selected
  number is optimistically biased by selection; the report says so.
- **A real eval report, not just a verdict.** `report.md`/`report.json`:
  agreement **with a 95% confidence interval**, macro F1 / balanced accuracy /
  per-class breakdown (enums) or MAE / severe-miss / correlation (scores),
  confusion matrix, training curve, shortcut audit, and the gold three-way
  when gold exists. Shipped reports are **redacted by default** — raw inputs
  and teacher rationales stay in a local-only `report_details.json`.
- **Labeling that can crash.** Every paid teacher call is journaled the
  moment it completes; rerunning `label` resumes instead of re-spending.
- **Small adapter artifacts.** LoRA adapters are tens of MB layered on a
  frozen base, so ten functions share one base model **on disk** (the Python
  runtime currently loads a base per loaded function — a shared-base
  multi-adapter runtime is on the roadmap).
- **Constrained decoding end to end.** The LoRA path decodes under the output
  contract, so the function always returns a value *inside the contract*;
  whether it's the right value is what the gate and report measure.

## Experimental commands

These work and are tested to the level noted, but haven't had enough
end-to-end mileage to call production-ready. Expect rough edges:

| command | state |
|---|---|
| `smallbatch export <fn>` | GGUF + grammar + Modelfile for the **LoRA candidate only** (llama.cpp checkout required). Unit-tested generation; conversion exercised manually per release. |
| `smallbatch serve <fn>` | Local HTTP endpoint. TF-IDF winners serve straight from Python (CPU); LoRA winners need an export bundle + `llama-server`. Single-threaded stdlib server — not a production stack. |
| `smallbatch push <fn> --repo you/name` | Hub upload of a **whitelisted** file set (redacted reports only; exact upload list printed first; `--dry-run` available). Note: trained model state is *not* redacted — a fitted TF-IDF vectorizer stores raw training-text tokens in its vocabulary, and push warns about this. |
| `smallbatch sweep <sweep.yaml>` | Model × technique grid for research. Cells are honest results, but selection over many cells on one small gate overfits it — treat the table as exploration, not evidence. |

## Evidence

*A reproducible public-dataset benchmark (teacher-labeled train/dev/gate, a
locked external gold test scored exactly once, both candidates + baselines,
three LoRA seeds) is the release gate for v0.2.0 and will be published here
with its scripts and raw per-seed results.*

## Responsible use

smallbatch trains on teacher outputs, so **your teacher provider's terms
govern what you may build**. Constrained scorers/classifiers like these fit
the "specialized, non-competing tool" category that major providers expressly
allow (e.g. content categorization, sentiment) — but general chatbots or
open-ended generators trained on provider outputs are prohibited, and some
providers restrict distribution of models trained on their outputs. Using a
self-hosted open-weights teacher (Ollama/vLLM) sidesteps the question for
labeling. Read [docs/responsible-use.md](docs/responsible-use.md) before
pointing a hosted teacher at a dataset.

An adapter inherits its **base model's** license. The default student
(`ibm-granite/granite-4.0-350m`) is Apache-2.0. Remember also that
`spec.yaml`, your rubric, and any `spec_files` ship inside the artifact —
review them like code before sharing.

## Docs

| | |
|---|---|
| [docs/how-it-works.md](docs/how-it-works.md) | Pipeline, spec reference, teacher protocol, training/eval design, artifacts, the Program-as-Weights lineage. |
| [docs/local-gpu.md](docs/local-gpu.md) | Running on your own GPU: precision auto-select, VRAM sizing, old-GPU (Pascal) pins, OOM knobs. |
| [docs/cloud.md](docs/cloud.md) | Renting a GPU per compile with SkyPilot; the label-locally/compile-remotely split. |
| [docs/responsible-use.md](docs/responsible-use.md) | Provider-terms guidance for choosing a teacher. |
| [examples/ticket-priority/](examples/ticket-priority/) | Complete runnable example (synthetic data — a smoke fixture, not evidence). |
| [ROADMAP.md](ROADMAP.md) | Outcomes we're working toward — and what's deliberately out of scope. |
| [CHANGELOG.md](CHANGELOG.md) | What shipped, per release. |

## Development

```bash
uv venv && source .venv/bin/activate
uv pip install -e .[dev]
pytest -q          # CPU-only; no GPU or network needed
ruff check src tests
```

MIT licensed. Inspired by
[Program-as-Weights (arXiv:2607.02512)](https://arxiv.org/abs/2607.02512) —
see [docs/how-it-works.md](docs/how-it-works.md#relationship-to-program-as-weights)
for what smallbatch borrows and what it deliberately doesn't.
