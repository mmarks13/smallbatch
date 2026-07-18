# How Smallbatch Works

This document describes Smallbatch's mechanism and evidence model. The README
provides the public product overview.

## 1. Define A Decision

A spec contains a self-contained `prompt`, a strict flat JSON input contract,
a constrained output contract, an optional teacher, and an ID-keyed candidate
map. Smallbatch adds the batch protocol, canonical input serialization, legal
output instructions, and parsing rules.

Changing the prompt, contracts, teacher identity, augmentation recipe, or
prompt protocol changes the `decision_hash`. Candidate configuration changes
only the `build_hash`.

## 2. Obtain Decisions

Items use `{"input": {...}}`. Either every item also has `output` or none do.

- **Imported:** Smallbatch validates and uses existing production decisions.
- **Teacher-generated:** Smallbatch shows ten diverse inputs labeled twice in
  shuffled field order. The user approves, declines, or asks for another ten.
  Approval continues the same journaled labeling run.

Calibration is teacher-behavior inspection, not correctness validation. Its
rows are training-only because the user saw them before approval.

Real rows split deterministically into 70% train, 10% dev, and 20% evaluation.
Every decision class takes a proportional share of each split, so a rare class
is present in training and development rather than being concentrated in
evaluation. Evaluation membership remains sticky when data is appended.
Optional augmentation uses training rows only and requires a callable teacher.

## 3. Build Candidates

`compile` attempts every configured candidate independently:

- TF-IDF (term frequency-inverse document frequency) plus logistic regression,
  a small word-and-phrase baseline persisted with `skops`.
- SetFit using a Sentence Transformer body and classifier head per field, the
  semantic middle tier between sparse text features and a language model. By
  default the embedding phase fine-tunes on every training row under a bounded
  pair budget; `embedding_samples_per_class` restricts it to a deterministic
  per-class sample only when set. On ordered scales the embedding trains with
  graded cosine targets that decay with level distance, every epoch — including
  the frozen starting point — is scored on the development split, and the
  best-scoring weights are what survives, so fine-tuning that hurts is
  discarded rather than shipped. The report records the row counts, the pair
  budget, the per-epoch curve, the frozen-versus-tuned delta, and the resolved
  SetFit arguments.
- LoRA (low-rank adaptation) over each configured foundation model. LoRA is a
  parameter-efficient fine-tuning method: it trains a small adapter while the
  base model remains frozen.

An integer output is an **ordered scale**, not a set of unrelated categories,
and every candidate is trained the same way: a softmax distribution over the
levels, optimized with class likelihood plus the ranked probability score, so
predicting 4 when the decision was 0 costs more than predicting 1. A LoRA
student reads that distribution from the logits of the legal levels; TF-IDF
and SetFit train the shared softmax head on their own features — a linear
layer, or one hidden layer when the development split says the extra capacity
earns its keep. The head tries a small capacity grid, with the training seed
searched alongside, and keeps the best development within-one agreement;
`head: linear` or `head: mlp` pins the family when the search should not
decide. Enum labels have no order to exploit and keep an ordinary multinomial
head.

Integer ranges must lie within **0-9**, so each level is a single token and the
decision is one ordered choice a student can be trained and scored on. A wider
scale is refused: rescale the decision (a 0-10 scale becomes 0-9), or use
`labels` when the values are unordered. Structured functions train one TF-IDF
or SetFit head per field and return one validated object.

Every ordered scale therefore ends in a distribution over the levels — a LoRA
student reads it from a single forward pass with no decoding loop, and the
softmax heads compute it with a few lines of numpy. That exposes a choice
shared by all three candidates: `decode: argmax` reports the most likely
level, which maximizes exact agreement; `decode: median` reports the first
level whose cumulative probability reaches one half, which trades exact hits
for smaller misses; `decode: within_one` reports the level whose ±1
neighborhood holds the most probability. The default, `auto`, measures all
three on the development split and keeps the best within-one agreement, ties
resolving to argmax; the evaluation split never decides it. The selected
decoder and the full comparison are recorded with the candidate, and the
softmax heads persist the selection inside the head artifact. The report also
carries head diagnostics: the selected capacity, the head's mean confidence,
and each level's mean predicted probability against its observed rate — the
view that shows a rare extreme level being starved of probability mass on the
development split, before it surfaces as tail bias in evaluation.

Completed stages are durable. An interrupted build resumes in place, including
LoRA trainer checkpoints and completed zero-shot diagnostics. Re-running a
completed build with candidate or diagnostic errors creates a new `-rN`
revision: completed candidate files and diagnostics are reused, failed stages
run again, and the prior manifest remains unchanged. During a run, progress
identifies the current candidate and stage and prints a compact evidence
summary as each CPU evaluation completes. An error remains visible without
discarding other candidates. A profile that runs longer than 30 seconds also
reports completed rows and elapsed time every 30 seconds. Progress writes occur
between timed calls and are excluded from batch-one latency measurements.

## 4. Evaluate On CPU

Every completed candidate is loaded in an isolated CPU subprocess and called
once per evaluation item. The subprocess uses up to four threads by default
and records the exact CPU, OS, Python, dependencies, and thread count.

Quality predictions come from that CPU runtime. Operating evidence includes
cold load, p50/p95 batch-one latency, peak RSS, candidate-owned bytes, required
shared/base bytes, and offline requirements. Candidate-owned bytes count only
files required for inference; transient trainer checkpoints and optimizer state
are excluded from both the reported footprint and standalone package.

Integer evidence includes exact, within-one, MAE, the absolute-error
histogram, p90/max error, mean signed error, Pearson, Spearman, invalid
rate, and a per-level breakdown — support against predicted counts and signed
error per rubric level, so a candidate that leans on the scale shows where.
Enum evidence includes decision agreement, F1 variants, balanced
accuracy, per-class results, worst-class recall, confusion, and invalid rate.
Structured outputs expose joint and every field result.

A train-fitted constant and LoRA zero-shot runs are diagnostics, never
selectable candidates. Smallbatch reports strict observed dominance only when
one candidate is no worse on every evaluated decision and operating measure
and strictly better somewhere.

## 5. Select And Package

`compile` never activates a candidate. `select` takes an explicit candidate,
generates an inspectable Python project and wheel, and runs the standalone
wheel over the full evaluation split.

Exact output parity is expected on the same machine. Valid differences are
reported with changed rows and metric deltas and require explicit acceptance.
Invalid output, load failure, or missing output prevents activation.

The standalone package is the generated source project and wheel that runs
without Smallbatch. It exposes:

```python
from smallbatch_functions.ticket_priority import classify, classify_batch, metadata
```

It does not depend on Smallbatch. TF-IDF packages depend only on their sklearn
runtime, SetFit packages on SetFit's inference stack, and LoRA packages on
Torch, Transformers, PEFT, and the recorded base model.

The build manifest remains immutable. Successful selection atomically updates
`active.json` and appends selection history. Clearing the selection removes
only the pointer, not prior builds or packages.

## What The Evidence Means

Decision agreement measures reproduction of the supplied outputs. It cannot
show that the prompt or decisions are correct, fair, lawful, or useful.
Candidate comparison on one evaluation split also introduces selection bias;
v0.2 does not claim an independent confirmation result.

Latency, memory, and bytes are measured operating characteristics. They are
not direct energy measurements.
