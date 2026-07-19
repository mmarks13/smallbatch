# Smallbatch Contributor Guide

## Product Contract

This is the authoritative, exact product contract. `README.md` is its concise
public-facing expression. Product and implementation decisions must remain
consistent with this section.

### Tagline

**Smallbatch distills prompt-driven LLM tasks into small, tested local
functions that run on a CPU.**

### Value Proposition

**Replace repeated LLM inference with a local CPU function you control, with
clear evidence about the quality and operating tradeoffs.**

### Contract

**You provide:** A constrained prompt and representative inputs, either with
existing outputs or with a callable LLM teacher whose behavior you approve on
a sample.

**A function returns one of three shapes:**

1. A bounded decision — one enum, one bounded integer scale (0-9), or a fixed
   object of enum/integer fields.
2. A bounded decision plus one length-bounded text field.
3. One length-bounded text field (a rationale, explanation, rewrite,
   normalized message, title, summary, or another stable, narrowly defined
   transformation; `max_chars` 1-2000 Unicode code points, default 300).

**Two evidence contracts:** bounded fields are evaluated with behavioral
decision metrics; text fields with held-out reference fidelity (bits per byte
primary) and structural diagnostics. Text evaluation never establishes
semantic correctness, factual accuracy, usefulness, or downstream success.
Reasoning is not a separate concept: a rationale is an ordinary text field,
and Smallbatch does not claim it faithfully represents internal reasoning.

**Smallbatch provides:** CPU-runnable candidates compared on that evidence
plus speed, resource use, and portability. You select one or none; Smallbatch
packages your selection as a shareable Python function that does not require
Smallbatch at runtime.

**Smallbatch does not:** Validate whether the prompt, imported outputs, or
teacher behavior is correct.

### Suitability

Smallbatch is for repeated tasks whose prompt, output meaning, and input
distribution remain consistent long enough to justify compilation. Functions
with a text field are LoRA-only, reject augmentation, and use strict JSON as
their canonical completion; the full output is atomic (never truncated,
repaired, or partially returned).

### Feature Test

A feature belongs in Smallbatch only when it directly helps a user:

1. Define a constrained prompt-driven task.
2. Obtain outputs from an approved teacher or reuse existing outputs.
3. Build and compare CPU-runnable candidates.
4. Understand behavioral and operating tradeoffs.
5. Explicitly select one candidate or none.
6. Package and run the selected local function.

### Explicit Non-Goals

- Gold-label or ground-truth infrastructure.
- Rubric, policy, prompt, or teacher correctness validation.
- General-purpose data annotation.
- Universal acceptance gates or automatic winners.
- Open-ended or long-form generation, general chat, multi-turn behavior,
  multiple or optional text fields, empty text as a valid result, dynamic
  output schemas, or task-specific free-text evaluators (no LLM-as-judge).
- GPU-only runtime artifacts.
- A general AutoML, experiment-tracking, serving, or deployment platform.

## Product Flow

```text
prompt + typed inputs
  -> imported outputs OR calibrated teacher outputs
  -> deterministic train/dev/eval data
  -> TF-IDF + SetFit + LoRA candidates (LoRA-only when a text field exists)
  -> full CPU evaluation and operating profiles
  -> user selects one or none
  -> standalone source package + wheel
```

The evaluation reference is a supplied output, not ground truth. There are
no gold labels, acceptance gates, PASS/FAIL verdicts, or automatic winners.

## Commands

```bash
smallbatch init <classifier|scorer|structured|rewriter> <name>
smallbatch doctor <spec> [--items items.json]
smallbatch label <spec> --items items.json [--append] [--skip-calibration]
smallbatch compile <spec> [--cpu-threads N]
smallbatch select <function> <candidate> [--version BUILD]
smallbatch select <function> --clear
smallbatch run <function> --json '{...}'
smallbatch run <function> --version BUILD --candidate ID --json '{...}'
smallbatch status
```

Compile exits `0` when at least one candidate completes and `1` on operational
failure. It never selects. Selection packages and validates the standalone
runtime before atomically changing `active.json`.

## Module Ownership

- `spec.py`: prompt-first schema, strict input/output validation, identities.
- `labeling.py`, `calibration.py`, `journal.py`: decision acquisition and data.
- `candidates.py`, `setfit_candidate.py`, `training.py`: candidate training.
- `heads.py`, `decode.py`: the shared softmax ordinal head, its single CE+RPS
  objective, and argmax distribution decoding, shared across candidates;
  capacity and seed tuning use dev, never eval.
- `objective.py`: the universal LoRA per-field objective — completion segment
  attribution, per-field losses, untuned-baseline normalization, the
  checkpoint score, and text-fidelity scoring.
- `evaluate.py`, `metrics.py`, `profiling.py`, `report.py`: evidence.
- `artifacts.py`, `runtime.py`: immutable builds and internal candidate loading.
- `standalone.py`, `standalone_templates/`: generated no-Smallbatch packages.
- `api.py`, `cli.py`, `doctor.py`, `init_cmd.py`: public orchestration.
- `teacher/`: callable LLM backends; keep provider imports lazy.

## Invariants

1. Specs, items, datasets, manifests, and old artifacts fail closed across
   the v0.2 and v0.3 schema breaks. Do not restore legacy aliases; removed
   options (`decode`, `objective`, `rationale_distillation`, `loss_type`,
   the `reason` channel) fail validation with their correction.
2. A file contains either imported decisions for every item or no decisions.
3. Calibration never edits decisions; it approves, declines, or reviews more.
4. Evaluation rows remain sticky on append and never feed training or
   augmentation.
5. Candidate and diagnostic errors are recorded and isolated. One failed stage
   must not discard completed work.
6. Every selectable candidate has completed full CPU evaluation.
7. Build manifests are immutable evidence. Selection is a separate atomic
   pointer and history.
   Re-running a completed build that contains candidate or diagnostic errors
   creates a new retry revision, reuses completed work, and never repairs the
   prior build in place.
8. Generated wheels must not import or depend on `smallbatch` and must reproduce
   the evaluated candidate or disclose and explicitly accept package drift.
9. Public evidence contains no inputs, rationales, or raw disagreements.
10. CPU time, memory, and bytes are proxies; never describe them as measured
    energy use.
11. A generated output is atomic. Any invalid field invalidates the whole
    result at every stage — import, teacher labeling, training data,
    evaluation, packaged inference (`InvalidOutputError`) — with no
    truncation, repair, partial results, or automatic packaged-inference
    retry. Any invalid held-out package output blocks selection.
12. Text evidence is reference fidelity and structure only. Never present
    it as correctness, and never blend bounded and text metrics into one
    score or rank candidates by the normalized checkpoint score.

## Artifact Layout

```text
data/<function>/
  train.jsonl dev.jsonl eval.jsonl labeled.jsonl meta.json calibration.json
  unresolved.jsonl   # only at teacher.passes: 2 — three-way splits awaiting
  journal/           # optional user resolution; never blocks a run

artifacts/<function>/
  builds/<build-id>/
    spec.yaml manifest.json build_state.json report.json report.md
    report_details.local.json evaluation.local.jsonl candidates/<candidate>/
  packages/<build-id>--<candidate>/
    source/ dist/*.whl package.json
  active.json selection-history.jsonl
```

Files suffixed `.local` never enter standalone packages.

## Verification

```bash
pytest -q                    # CPU-only, offline; every PR (hosted CI)
ruff check src tests tests_gpu case-study
python -m build

# real-GPU tiers (self-hosted runner; collected only with the env var):
SMALLBATCH_GPU_TESTS=1 pytest tests_gpu -m gpu_smoke -q   # ~15 min PR gate
SMALLBATCH_GPU_TESTS=1 pytest tests_gpu -q                # ~45 min, manual;
                                                          # gates release
```

CPU-only, offline unit tests are the default and include the tiny-random-model
integration tests (`tests/test_lora_integration.py`), which run the real
transformers/PEFT/tokenizers stack end to end without a GPU or network — keep
them offline. The `tests_gpu` tier runs real students on real hardware:
`gpu-smoke` is a required PR check on the self-hosted runner, `gpu-release` is
manually dispatched per release candidate and the release workflow refuses to
publish without a successful run on the tagged commit. Do not make the normal
test suite download models or call a teacher.

The repository may be dirty. Preserve user changes and work with compatible
in-progress edits rather than reverting them.
