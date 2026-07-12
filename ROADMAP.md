# Roadmap

Where smallbatch is headed. The philosophy stays fixed: **narrow functions
with checkable output contracts, compiled and run local-first** — and the
compiler proves when weights are useful rather than assuming it. Features
that dilute that (chat task types, hosted endpoints, multi-GPU rigs, GUIs)
are non-goals — see the bottom of this page.

Shipped work lives in [CHANGELOG.md](CHANGELOG.md); this page holds outcomes
we haven't reached yet, each with the evidence that would count as "done."

## Next: broaden the candidate ladder

- **SetFit / small-encoder candidate.** A third compile candidate between
  TF-IDF and the causal adapter. Done when: a public benchmark task selects
  it over both neighbours, or the data shows it's dominated.
- **Vocabulary-free TF-IDF variant.** HashingVectorizer pipeline for
  privacy-critical sharing (no raw training tokens stored in the artifact).
  Done when: a shared artifact passes a token-leakage scan with equivalent
  quality on the benchmark.
- **Dev-based candidate selection.** Select the winner on dev, spend the
  gate exactly once on it (today both candidates are scored on the gate with
  a recorded selection-bias caveat). Done when: the benchmark quantifies the
  bias and the dev split is routinely large enough to select on.

## Runtime & artifacts

- **A runtime-only install.** A recipient of a compiled function shouldn't
  install the training stack: split extras (`smallbatch[train]`, a slim
  runtime core). Done when: a clean venv calls a TF-IDF artifact with no
  torch and an adapter with torch-only.
- **Shared-base multi-adapter runtime.** Load one base model, route calls to
  many adapters (until then, functions share a base on *disk*, not in RAM).
  Done when: ten compatible functions serve from one loaded base with
  measured memory.
- **Version promotion & rollback.** A stable alias (`production`) per
  function, one-command rollback, garbage collection. Done when: promotion
  survives recompiles and a rollback is a single command.
- **Build lock / reproducibility metadata.** Immutable model+tokenizer
  revisions, package/CUDA/hardware facts in every manifest. Done when: a
  rebuild either reproduces the candidate or names every changed input.
- **Transactional builds.** Compile into a temp dir, atomically promote on
  completion. Done when: a killed compile can't leave a half-populated
  version dir.

## Evaluation & data

- **Gate-on-gold verdict.** When the gate holds enough gold rows, the
  PASS/FAIL verdict itself uses gold accuracy (today gold is independent
  evidence beside a teacher-agreement verdict).
- **Bootstrap CIs for macro F1** (declared resampling, recorded seed).
- **Typed input contracts.** A small JSON-Schema subset validated identically
  at labeling and runtime (types, lengths, unknown-field policy) — today's
  `input_schema` is informational.
- **Review-to-gold.** Mark human-reviewed labels as gold from `smallbatch
  review`; regenerate variants from review notes.
- **Variant diversity axes.** Band-targeting balances labels but not input
  space; let the spec declare variation dimensions (length, formality,
  domain).
- **Production capture and drift.** Opt-in `run()` capture with redaction,
  scheduled gold spot-checks, drift surfaced in `status`, one-command
  recompile on accumulated real data.

## Training conveniences

- **Checkpoint resume + training telemetry** (`resume_from_checkpoint`,
  tensorboard pass-through). Long consumer-GPU runs fail for boring reasons;
  resuming beats restarting.
- **Sweep redesign.** Dev-based selection inside sweeps and first-class
  per-cell metric retention (today's sweep compares cells on the same gate —
  fine for exploration, biased for selection, and documented as such).

## Parked

Free-position span extraction (until it can be done without open-ended text
output); `--include-examples` opt-in disclosure of report excerpts.

## Non-goals

Open-ended generation task types, general chatbots, preference tuning (DPO),
multi-GPU/distributed training, hosted inference, a GUI, a public function
marketplace. Other tools do these well; smallbatch stays a compiler for
narrow constrained functions that run on the hardware you already have.
