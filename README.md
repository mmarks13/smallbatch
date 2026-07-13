# smallbatch

**Smallbatch distills prompt-driven LLM decisions into small, tested local
functions that run on a CPU.**

Replace repeated LLM inference with a local CPU function you control, with
clear evidence about the quality and operating tradeoffs.

Smallbatch is an alpha release. It measures fidelity to supplied decisions;
it does not establish that those decisions are correct or validate the prompt,
imported decisions, or teacher behavior.

You provide a constrained decision prompt and representative inputs, either
with existing decisions or with a callable LLM teacher whose behavior you
approve on a sample. Smallbatch builds and compares CPU-runnable candidates;
you explicitly select one or none.

## When It Fits

Use Smallbatch when the same constrained prompt is repeatedly producing an
integer, enum, or structured combination of those outputs, and the prompt,
output meaning, and input distribution are stable enough to compile.

Keep the original LLM call when the task is open-ended, the prompt changes
frequently, examples are not representative, or a local function cannot
express the output.

## Install

```bash
pip install smallbatch
```

The v0.2 package is intentionally a full installation containing TF-IDF,
SetFit, and LoRA training support. The final function wheel contains only the
selected candidate's runtime dependencies and never depends on Smallbatch.

## Workflow

```bash
smallbatch init classifier ticket-priority

# Inspect/edit the generated prompt, contract, teacher, and candidates.
smallbatch doctor ticket-priority/spec.yaml \
  --items ticket-priority/items.json

# Imported decisions skip the teacher. Unlabeled inputs start with calibration.
smallbatch label ticket-priority/spec.yaml \
  --items ticket-priority/items.json

# Build every configured candidate and evaluate each through its CPU runtime.
smallbatch compile ticket-priority/spec.yaml

# Review artifacts/ticket-priority/builds/<build>/report.md, then choose or stop.
smallbatch select ticket-priority tfidf

smallbatch run ticket-priority \
  --json '{"title":"Production down","body":"All requests return 503"}'
```

`compile` never chooses a candidate. `select` builds an inspectable source
package and wheel, evaluates that standalone package on the complete evaluation
split, and only then updates the active selection.

```python
from smallbatch_functions.ticket_priority import classify, classify_batch, metadata

priority = classify({"title": "Production down", "body": "All requests return 503"})
```

The generated package does not import Smallbatch. For LoRA candidates it still
requires the recorded base model plus the Python PEFT runtime.

## Item Format

Unlabeled JSONL records use one envelope:

```json
{"input":{"title":"Production down","body":"All requests return 503"}}
```

To reuse existing production decisions, add `output` to every record:

```json
{"input":{"title":"Production down","body":"All requests return 503"},"output":"urgent"}
```

Files must be entirely labeled or entirely unlabeled. Input fields are required
and strictly typed as `string`, `integer`, `number`, or `boolean`.

## Candidates

- **TF-IDF + logistic regression:** fast, small, interpretable operating cost,
  and often the right answer for lexical decisions.
- **SetFit:** a sentence-transformer body with a classifier head, useful when
  TF-IDF misses semantic similarity but an adapted language model is excessive.
- **LoRA:** a PEFT adapter over a small foundation model. It is the heaviest
  option and may require a GPU to train, but the selected function is evaluated
  and packaged for CPU inference.

Candidate failures are isolated. A build succeeds when at least one candidate
fully trains and completes CPU evaluation; unavailable or failed candidates
remain visible in the report.

## Evidence

Every completed candidate is run over the same held-out evaluation decisions on
CPU. Reports include:

- Exact and within-one behavior, MAE, error distribution, p90/max error,
  signed error, and correlations for bounded integers.
- Decision agreement, macro/weighted F1, balanced accuracy, per-class behavior,
  worst-class recall, and confusion for enums.
- Joint and per-field results for structured outputs.
- Cold load, p50/p95 batch-one latency, peak RSS, candidate-owned bytes,
  required shared/base bytes, runtime dependencies, CPU, OS, and thread count.
- A train-fitted constant diagnostic and one zero-shot diagnostic per LoRA base.

Smallbatch may report observed strict dominance, but it never declares a winner
or PASS/FAIL. Comparing candidates on one evaluation split introduces selection
bias; v0.2 reports that limitation and does not claim independent confirmation.
CPU time, memory, and footprint are operating proxies, not energy measurements.

## Responsible use

Smallbatch trains candidate functions on teacher outputs, so **your teacher
provider's terms govern what you may build**. Constrained classifiers and
scorers like these fit the "specialized, non-competing tool" category that
major providers expressly allow (e.g. content categorization, sentiment) —
but general chatbots or open-ended generators trained on provider outputs are
prohibited, and some providers restrict distribution of models trained on
their outputs. Using a self-hosted open-weights teacher (Ollama/vLLM)
sidesteps the question for labeling. Read
[docs/responsible-use.md](docs/responsible-use.md) before pointing a hosted
teacher at a dataset.

A LoRA adapter remains subject to its **base model's** license. The default
LoRA student (`ibm-granite/granite-4.0-350m`) is Apache-2.0. The selected
function's prompt, contract, generated source, and trained state ship in its
standalone package — review them like code before sharing.

## Commands

| Command | Purpose |
|---|---|
| `smallbatch init` | Generate a prompt-first starter project. |
| `smallbatch doctor` | Validate the contract, data mode, teacher, candidates, and environment. |
| `smallbatch label` | Import complete decisions or generate them after teacher calibration. |
| `smallbatch compile` | Train, CPU-evaluate, and compare every configured candidate. |
| `smallbatch select` | Package and activate one candidate, or clear the active selection. |
| `smallbatch run` | Call the active function or an explicit build/candidate. |
| `smallbatch status` | Show builds, failures, integrity, drift, and active selection. |

## Development

```bash
pip install -e '.[dev]'
pytest -q
ruff check src tests case-study
python -m build
```

See [docs/how-it-works.md](docs/how-it-works.md) for identities and artifacts,
[docs/responsible-use.md](docs/responsible-use.md) for decision limitations,
and [ROADMAP.md](ROADMAP.md) for deliberately deferred features.
