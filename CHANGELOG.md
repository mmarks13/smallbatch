# Changelog

## 0.3.0 (2026-07-19)

### Added

- **Length-bounded text outputs.** A function may declare one text field
  (`type: text`, `max_chars` 1-2000 Unicode code points, default 300), alone
  or beside bounded fields — a rationale, explanation, rewrite, title, or
  another stable narrow transformation. Text functions are LoRA-only and
  reject augmentation. Declared field order is generation order.
- Strict-JSON canonical completions for text-bearing functions: the entire
  completion must parse as exactly the declared JSON value, in declared key
  order; fences, preambles, trailing content, and duplicate/missing/extra/
  misordered keys are rejected. Bounded-only functions keep their compact
  format.
- One universal LoRA objective for every function shape: per-field semantic
  losses (int: class NLL + ranked probability score; enum: legal-value NLL;
  text: mean next-token NLL), each normalized by the candidate's untuned-base
  development loss, combined as a weighted mean via function-level
  `training.loss_weights` (partial overrides allowed; defaults 1.0 bounded,
  0.25 text-beside-bounded, 1.0 text-alone), plus an internal JSON-format
  loss at fixed weight 0.1. The same weighted normalized teacher-forced dev
  score selects checkpoints and drives early stopping; behavioral metrics no
  longer do. Unsafe normalization baselines fail the candidate before
  training instead of being clamped.
- Atomic invalid-output behavior at every stage: invalid imported rows are
  rejected with the row and field, invalid teacher labels are rejected and
  retried within the existing limited-retry protocol, invalid free-running
  evaluation results count as whole-row failures with structural categories
  (malformed JSON, empty text, character limit, budget exhaustion), and any
  invalid output on the held-out package evaluation blocks selection.
- Report-only text evidence: bits per byte (the primary cross-model
  reference-fidelity metric), token NLL, perplexity, top-1 token accuracy,
  and median/p90 per-example bits per byte — all described as held-out
  teacher-text prediction, never correctness, factuality, or usefulness —
  plus structural failure rates and per-candidate training-configuration and
  checkpoint-curve evidence.
- `smallbatch doctor` warns when the p95 reference text length exceeds 80% of
  `max_chars`, and labeling appends fail closed when the existing dataset was
  produced under a different decision identity.

### Changed

- **Clean schema break — no aliases, no migration.** Pre-v0.3 specs and
  datasets fail validation; historical builds stay inspectable but never
  resume under the new semantics.
- Standalone packages export `run` / `run_batch` / `metadata` (previously
  `classify` / `classify_batch`) and raise a structured `InvalidOutputError`
  — no partial dictionaries, no silent repair, no automatic retry. Package
  metadata now records the resolved output contract, character limits,
  deterministic greedy decoding settings, and the selected candidate's loss
  weights.
- Ordinal fields always decode by argmax across every candidate family.
- `teacher.passes: 2` compares draws on bounded fields only — text wording
  variation is never a decision flip — and is rejected for text-only
  functions, where self-agreement is undefined.
- Splitting never treats unique text values as classes: mixed outputs
  stratify on the first bounded field and text-only functions use
  deterministic seeded splits; evaluation membership stays sticky on append.
- The tagline is now "Smallbatch distills prompt-driven LLM **tasks** into
  small, tested local functions that run on a CPU."

### Removed

- `rationale_distillation` (declare an ordinary text output field instead),
  the labeled-item `reason` channel, the user-configurable LoRA `objective`,
  the user-configurable ordinal `decode` (and its automatic dev-time
  argmax/median selection), and the TRL `loss_type` passthrough. Each removed
  option fails validation with its correction.

### Fixed

- Text validation now preserves leading and trailing whitespace exactly,
  text-fidelity bpb uses decoded reference bytes, and completion budgets use
  conservative UTF-8 bounds for multilingual text and long enum labels.
- LoRA inference preserves tokenizer special tokens, and interrupted training
  restores development curves, best-checkpoint selection, and patience state.
- Appended user resolutions retain teacher provenance, report a mixed decision
  source, and no longer count as synthetic variants.
- Free-text wording no longer creates a false observed-dominance claim, and
  teacher self-agreement is presented as stability evidence rather than an
  upper bound on candidate agreement.
- The release gate now requires every GPU precision, resume, and OOM regression
  to execute successfully; skipped GPU jobs or required tests cannot publish.

### Roadmap

- Enforced structured decoding (grammar-constrained JSON generation) is
  planned; v0.3 validates generated output strictly but does not constrain
  text-bearing generation token-by-token.

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
