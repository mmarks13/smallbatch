# Changelog

## 0.2.0 (2026-07-14)

### Added

- Prompt-first decision specs with required string, integer, number, and
  boolean inputs and constrained integer/enum outputs.
- Imported production decisions and interactive teacher calibration before a
  resumable full labeling run.
- First-class TF-IDF, SetFit, and LoRA candidates with per-candidate settings,
  isolated failures, automatic stage resume, and LoRA checkpoint resume.
- Full held-out CPU evaluation for every completed candidate, including error
  distributions, per-class behavior, latency, memory, footprint, runtime
  dependencies, and portability facts.
- Train-fitted constant and per-LoRA-base zero-shot diagnostics.
- Explicit `select` command, separate active-selection history, and standalone
  inspectable source packages and wheels with no Smallbatch runtime dependency.
- Full standalone-package evaluation, parity reporting, and explicit valid
  drift acceptance.
- Stage-by-stage labeling and compile progress with immediate quality and
  operating summaries for each completed CPU evaluation.
- Thirty-second row and elapsed-time heartbeats during long isolated CPU
  evaluations, shared by selectable candidates and zero-shot diagnostics.
- A release case-study protocol with 600 hash-verified public CFPB inputs,
  aggregate-only publication boundaries, and an all-candidate completion gate.
- Ordinal-aware training for bounded integer scales: cumulative-link ordinal
  heads for TF-IDF and SetFit, and a LoRA objective of class NLL plus ranked
  probability score read from a single forward pass over the legal
  completions.
- Per-field decoder selection (argmax vs. median cumulative mass) measured on
  the dev split and recorded in the build manifest.
- Optional per-candidate LoRA gradient checkpointing for memory-bound
  training.
- A completed CFPB case-study run with published aggregate evidence, a
  narrative write-up, and a dated rights review.

### Changed

- Reframed Smallbatch around distilling prompt-driven LLM decisions into tested
  local CPU functions with explicit user selection.
- Replaced `rubric` with a self-contained `prompt` and candidate booleans with
  an ID-keyed mapping.
- Replaced train/dev/gate data with deterministic train/dev/eval decisions.
- Replaced verdict-oriented reports with descriptive decision-fidelity and
  operating evidence. Candidate choice is always explicit.
- Introduced clean spec, dataset, manifest, and package schema v3; old projects
  and artifacts must be regenerated.
- Completed zero-shot diagnostics now resume from their durable local records
  after an interrupted build.
- Completed builds with candidate or zero-shot diagnostic errors now retry in
  a new immutable revision while reusing successful artifacts and diagnostics.
- Bounded SetFit's embedding phase with deterministic per-class sampling and a
  recorded contrastive-pair budget while fitting its classifier on all rows.
- SetFit artifacts now discard trainer checkpoints and optimizer state; reported
  and packaged footprint counts only files required for inference.
- Added compatibility for tokenizer.json-only model repositories that use the
  Transformers 5 `TokenizersBackend` name and for models that reject generated
  `token_type_ids`.
- Integer scale outputs must lie within 0-9; specs outside that range fail
  closed.
- Train/dev/eval splits are proportionally stratified per class with
  largest-remainder allocation, and `meta.json` records per-split label
  histograms.
- The OpenAI-compatible teacher now treats null or empty completion content
  as a retryable error instead of a silently missing decision.

### Removed

- Gold-label infrastructure, acceptance gates, PASS/FAIL, automatic winners,
  label editing, and all-failure deployment acceptance.
- `review`, `sweep`, `export`, `serve`, and `push` commands. Their underlying
  behavior may return only as evidence-preserving post-selection adapters.

## 0.1.0

Initial development snapshot: prompt/rubric, teacher labels, LoRA fine-tuning,
experimental sweeps, GGUF export, and Hub publishing. Not published to PyPI.
