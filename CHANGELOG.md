# Changelog

## 0.2.0 (unreleased)

The "classifier compiler" release: every compile now compares a conventional
TF-IDF + logistic-regression candidate against the LoRA adapter and ships the
winner, with an honesty-first evaluation story.

### Added
- **TF-IDF candidate**: trained on the same teacher labels, gated on the same
  gate, persisted with skops (audited loading — unexpected types refused),
  runnable and servable CPU-only. Selection: highest gate agreement; within
  `gate.tie_margin` (default 0.02) the smaller artifact wins.
- **Gold labels**: optional `"gold"` on items — validated before any paid
  call, routed to the gate (never trained on), never sent to the teacher.
  Reports show teacher-vs-gold / student-vs-gold / student-vs-teacher.
- **Labeling journal**: every paid teacher call (real labels, generated
  variants with provenance, consistency probes) is durably journaled;
  rerunning `label` after a crash resumes instead of re-spending. Dataset
  writes are atomic; a lock prevents concurrent labeling.
- **All-fail acceptance**: when every candidate misses the gate, compile
  shows a full decision table and lets you explicitly accept the best
  candidate (`--use-best-anyway` for automation). Candidate-scoped and
  persistent; the gate verdict and exit code 2 never change.
- **Shared metric layer** (`metrics.py`): integers get MAE/p90/max/signed
  error/Pearson/Spearman/severe/invalid; enums get macro+weighted F1,
  balanced accuracy, per-class P/R/F1/support, worst-class recall. One
  implementation drives reports, manifests, sweeps, and decision tables.
- **Adaptive eval batching**: eval-time OOM halves the batch and retries with
  a warning; the effective size is recorded in report + manifest.
- `load_fn(version=…, candidate=…)`; `run`/`serve --candidate`;
  `push --dry-run`; `compile --allow-stale-labels`.

### Changed
- **Identity split**: `labeling_hash` (rubric/contract/teacher/references)
  vs `spec_hash` (full build identity) vs `dataset_hash` (exact rows).
  Compile fails closed on stale labels; `--base`/`--precision` changes no
  longer invalidate datasets.
- **Self-contained artifacts**: the resolved (post-override) spec plus
  content-addressed copies of every spec_file are archived; integrity
  (archive vs manifest) is reported separately from source drift; a moved or
  deleted project no longer marks an artifact stale.
- **Privacy**: shipped reports are redacted by default (raw inputs, teacher
  rationales, input-derived slice values live only in the local
  `report_details.json`); `push` uploads a positive whitelist and prints the
  exact file list; model cards describe every candidate's state.
- **Default base model**: `ibm-granite/granite-4.0-350m` (Apache-2.0)
  replaces `LiquidAI/LFM2.5-350M-Base` (revenue-conditioned license).
- `--max-variants` is now a true global budget across all augmentation
  stages (0 = no synthetic work); it previously acted as an override that
  could *increase* generation.
- Doctor fails (not warns) on a planned gate/dev split of zero, and on a
  labeling-identity mismatch.
- Version ordering is numeric-aware (`-r10` no longer sorts before `-r9`).
- Manifest schema v2: per-candidate records, selection, deployment;
  usability is candidate-scoped.

### Fixed
- Partially-parsed structured outputs are rejected at both runtime
  boundaries (HTTP 422 / None) instead of returned as partial objects.
- Spec validation: reversed ranges, duplicate/newline/case-colliding labels,
  unbounded thresholds/counts, YAML `yes`/`no` label trap.
- Function/sweep/arm names and run tags validated as filesystem-safe slugs.
- README banner referenced by absolute URL so the PyPI page renders it.

### Removed
- The README's private-pilot "Does it work?" section (non-reproducible; to
  be replaced by the public benchmark).

## 0.1.0

Initial development snapshot: spec → teacher labels → LoRA fine-tune → gate →
callable local function; sweeps; GGUF export; Hub push. Never published to
PyPI.
