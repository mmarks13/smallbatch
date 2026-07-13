# Smallbatch Contributor Guide

## Product Contract

This is the authoritative, exact product contract. `README.md` is its concise
public-facing expression. Product and implementation decisions must remain
consistent with this section.

### Tagline

**Smallbatch distills prompt-driven LLM decisions into small, tested local
functions that run on a CPU.**

### Value Proposition

**Replace repeated LLM inference with a local CPU function you control, with
clear evidence about the quality and operating tradeoffs.**

### Contract

**You provide:** A constrained decision prompt and representative inputs,
either with existing decisions or with a callable LLM teacher whose behavior
you approve on a sample.

**Smallbatch provides:** CPU-runnable candidates compared on decision
agreement, error profile, speed, resource use, and portability. You select one
or none; Smallbatch packages your selection as a shareable Python function
that does not require Smallbatch at runtime.

**Smallbatch does not:** Validate whether the prompt, imported decisions, or
teacher behavior is correct.

### Suitability

Smallbatch is for repeated decisions whose prompt, output meaning, and input
distribution remain consistent long enough to justify compilation.

### Feature Test

A feature belongs in Smallbatch only when it directly helps a user:

1. Define a constrained prompt-driven decision.
2. Obtain decisions from an approved teacher or reuse existing decisions.
3. Build and compare CPU-runnable candidates.
4. Understand behavioral and operating tradeoffs.
5. Explicitly select one candidate or none.
6. Package and run the selected local function.

### Explicit Non-Goals

- Gold-label or ground-truth infrastructure.
- Rubric, policy, prompt, or teacher correctness validation.
- General-purpose data annotation.
- Universal acceptance gates or automatic winners.
- Open-ended text generation.
- GPU-only runtime artifacts.
- A general AutoML, experiment-tracking, serving, or deployment platform.

## Product Flow

```text
prompt + typed inputs
  -> imported decisions OR calibrated teacher decisions
  -> deterministic train/dev/eval data
  -> TF-IDF + SetFit + LoRA candidates
  -> full CPU evaluation and operating profiles
  -> user selects one or none
  -> standalone source package + wheel
```

The evaluation reference is a supplied decision, not ground truth. There are
no gold labels, acceptance gates, PASS/FAIL verdicts, or automatic winners.

## Commands

```bash
smallbatch init <classifier|scorer|structured> <name>
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
- `evaluate.py`, `metrics.py`, `profiling.py`, `report.py`: evidence.
- `artifacts.py`, `runtime.py`: immutable builds and internal candidate loading.
- `standalone.py`, `standalone_templates/`: generated no-Smallbatch packages.
- `api.py`, `cli.py`, `doctor.py`, `init_cmd.py`: public orchestration.
- `teacher/`: callable LLM backends; keep provider imports lazy.

## Invariants

1. Specs, items, datasets, manifests, and old artifacts fail closed across the
   v0.2 schema break. Do not restore legacy aliases.
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

## Artifact Layout

```text
data/<function>/
  train.jsonl dev.jsonl eval.jsonl labeled.jsonl meta.json calibration.json
  journal/

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
pytest -q
ruff check src tests case-study
python -m build
```

CPU-only, offline unit tests are the default. Real SetFit and LoRA release
checks use cached models; do not make the normal test suite download models or
call a teacher.

The repository may be dirty. Preserve user changes and work with compatible
in-progress edits rather than reverting them.
