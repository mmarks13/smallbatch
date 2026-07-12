# Release checklist

The operational gate for tagging a release. Every box must be checked in the
release PR; a clean clone must be able to verify each item. (Consolidated
from the v0.2 external plan reviews; narrative review documents are not
tracked.)

## Correctness & integrity

- [ ] `pytest -q` green and `ruff check src tests` clean on every supported
      Python (CI matrix).
- [ ] Labeling, build, and dataset identities are separate and tested:
      `--base`/`--precision`/gate-threshold changes do NOT invalidate labels;
      rubric/contract/teacher/spec_file changes fail compile closed.
- [ ] Archived specs are self-contained and hash-consistent from a foreign
      cwd with the source project deleted.
- [ ] Candidate failure isolation: a completed candidate survives the other
      candidate's crash; all-error compiles exit 1 and never offer acceptance.
- [ ] The manifest is versioned (manifest_schema_version) and backend-neutral;
      old LoRA-only manifests still read.
- [ ] Both TF-IDF and LoRA runtime paths pass an end-to-end call
      (`run --candidate tfidf` / `--candidate lora`).
- [ ] Gold cannot leak into teacher prompts or training rows (sentinel tests).
- [ ] Journal crash recovery covers real labels, generated variants (with
      provenance), and probe results; a truncated final line is tolerated;
      the dataset fold is atomic.

## Metrics & honesty

- [ ] Every completed candidate and sweep cell records the complete
      type-appropriate metric set (integers: MAE/p90/max/signed error/
      correlations/severe/invalid with valid_n; enums: macro+weighted F1/
      balanced accuracy/per-class/worst-class recall/invalid).
- [ ] All-failure acceptance preserves FAIL + exit 2 while making explicit
      acceptance persistent, candidate-scoped, and visible in `status`.
- [ ] The benchmark used a locked external gold test scored exactly once
      after candidate selection; all seeds/candidates/baselines published.
- [ ] README claims are limited to measured results and tested paths;
      experimental commands labeled.

## Privacy & rights

- [ ] Public artifacts (push/export) contain no source examples, teacher
      rationales, or absolute local paths by default (sentinel scan over the
      exact ship list).
- [ ] `push --dry-run` prints the exact upload list + TF-IDF vocabulary
      preflight.
- [ ] Teacher-output distribution rights are resolved and dated for anything
      published (see the benchmark's TERMS file); aggregate-only fallback
      applied if ambiguous.

## Packaging & release mechanics

- [ ] pyproject version == `smallbatch.__version__` == the tag.
- [ ] Clean-venv wheel install passes the smoke (CLI --help, init,
      torch-free tfidf train/predict).
- [ ] Local GPU e2e smoke on the ticket example (label with a forced crash +
      resume, review, compile both candidates, run both via --candidate,
      status, export LoRA, serve tfidf, push --dry-run).
- [ ] PTY all-failure smoke: decline path, accept path, exit 2 both ways,
      default resolution after acceptance, unaccepted-secondary refusal,
      older-PASS displacement message.
- [ ] Tag pushed only after CI is green on the release commit; PyPI publish
      via the protected trusted-publishing workflow; `pip install smallbatch`
      verified from a clean venv afterward, PyPI page renders (banner, links).
