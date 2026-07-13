# Changelog

## 0.2.0 (unreleased)

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

### Removed

- Gold-label infrastructure, acceptance gates, PASS/FAIL, automatic winners,
  label editing, and all-failure deployment acceptance.
- `review`, `sweep`, `export`, `serve`, and `push` commands. Their underlying
  behavior may return only as evidence-preserving post-selection adapters.

## 0.1.0

Initial development snapshot: prompt/rubric, teacher labels, LoRA fine-tuning,
experimental sweeps, GGUF export, and Hub publishing. Not published to PyPI.
