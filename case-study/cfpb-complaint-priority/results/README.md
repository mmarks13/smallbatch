# Aggregate Results

Evidence from the frozen protocol run of 2026-07-14 (build
`2026-07-14-538d9418`). See the case-study `README.md` for the narrative and
`TERMS.md` for the rights review.

- `results.json` — per-candidate metrics with CIs, error histograms,
  training curves, CPU profiles, and the zero-shot and constant diagnostics.
- `report.md` — the generated Smallbatch report for the same build.
- `protocol.json` — frozen input identities, spec/items hashes, versions,
  candidate statuses, and `explicit_selection` (null: no candidate selected).
- `thread_scaling.json` — every candidate re-profiled at 4/8/16/30 threads
  on the same machine.

These files contain aggregate evidence only: no complaint text, no teacher
decisions, no per-row disagreements.
