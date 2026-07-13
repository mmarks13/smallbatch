# v0.2 Release Checklist

This is an operational publication checklist, not a model-quality gate.

## Product And Documentation

- [ ] `CLAUDE.md` contains the exact product contract and README presents its
      public-facing version.
- [ ] README, contributor guide, docs, examples, metadata, and CLI help agree.
- [ ] No public gold, gate, PASS/FAIL, automatic-winner, review, sweep, export,
      serve, or push behavior remains.
- [ ] Limitations distinguish decision fidelity from correctness and operating
      proxies from energy measurement.

## Behavior And Evidence

- [ ] Imported and calibrated-teacher workflows pass end to end.
- [ ] TF-IDF, real SetFit, and LoRA candidate paths have been exercised.
- [ ] Every selectable candidate completes full CPU evaluation.
- [ ] Reports contain every required metric and no private examples.
- [ ] Candidate errors remain visible and do not discard survivors.
- [ ] No candidate becomes active during compile.

## Standalone Package

- [ ] Generated source is inspectable and excludes decision data and journals.
- [ ] Wheel metadata has no Smallbatch runtime dependency.
- [ ] Wheel installs/imports outside the repository and exposes
      `classify`, `classify_batch`, and `metadata`.
- [ ] Full package parity, valid-drift confirmation, invalid-output refusal,
      integrity, and atomic selection rollback are tested.

## Case Study

- [ ] CFPB IDs, inputs, hashes, extraction date, prompt, teacher, and scripts
      are frozen before full labeling.
- [ ] The first-draft priority prompt was not revised after teacher access.
- [ ] Results are described as decision-distillation evidence, not correctness.
- [ ] Rights review is dated before publishing teacher decisions or packages.

## Packaging And Publication

- [ ] `pytest -q` and `ruff check src tests case-study` pass.
- [ ] Smallbatch sdist/wheel and generated-function wheel smoke tests pass.
- [ ] CI passes on every supported Python version.
- [ ] Clean-environment install from the built Smallbatch wheel succeeds.
- [ ] Version, changelog, package metadata, tag, and PyPI page agree.
- [ ] Trusted publishing runs only after explicit maintainer approval.
