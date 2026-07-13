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
Evaluation membership remains sticky when data is appended. Optional
augmentation uses training rows only and requires a callable teacher.

## 3. Build Candidates

`compile` attempts every configured candidate independently:

- TF-IDF plus logistic regression, persisted with `skops`.
- SetFit using a Sentence Transformer body and classifier head per field.
- LoRA/PEFT over each configured foundation model.

Bounded integers are learned as discrete classes. Structured functions train
one TF-IDF or SetFit head per field and return one validated object.

Completed stages are durable. Re-running a matching decision, dataset, and
build resumes candidate work; LoRA uses trainer checkpoints. An error remains
visible without discarding other candidates.

## 4. Evaluate On CPU

Every completed candidate is loaded in an isolated CPU subprocess and called
once per evaluation item. The subprocess uses up to four threads by default
and records the exact CPU, OS, Python, dependencies, and thread count.

Quality predictions come from that CPU runtime. Operating evidence includes
cold load, p50/p95 batch-one latency, peak RSS, candidate-owned bytes, required
shared/base bytes, and offline requirements.

Integer evidence includes exact, within-one, MAE, the absolute-error
histogram, p90/max error, mean signed error, Pearson, Spearman, and invalid
rate. Enum evidence includes decision agreement, F1 variants, balanced
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

The standalone package exposes:

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
