# CLAUDE.md

smallbatch is a classifier compiler: YAML spec → teacher labels → two
candidates trained on the same data (tfidf+logreg and a LoRA adapter) →
eval gate against the teacher's holdout (+ gold labels when provided) →
winner selected → callable local function.

If a `CLAUDE.local.md` exists in this checkout, it describes THIS machine's
hardware constraints — read it before running any training command; it
overrides the general guidance here.

## Which doc answers what

| You need to... | Read |
|---|---|
| Understand the pipeline, spec fields, identities, gold, gate semantics, artifacts | `docs/how-it-works.md` |
| Run a compile on a local GPU (precision, VRAM, old-GPU pins, OOM knobs) | `docs/local-gpu.md` |
| Run a compile on a rented cloud GPU | `docs/cloud.md` |
| Judge whether a teacher provider may be used | `docs/responsible-use.md` |
| See a complete working spec + walkthrough | `examples/ticket-priority/README.md` |
| The operational release gate | `RELEASE_CHECKLIST.md` |

## Commands

```bash
pytest -q                                   # unit tests: CPU-only, no network, ~5s (sklearn import)
ruff check src tests                        # must be clean before any commit
smallbatch init <template> <name>           # starter spec.yaml + items.json (instant)
smallbatch doctor <spec> [--items X]        # preflight; 1 live teacher probe unless --no-probe
                                            #   plan-stage gate/dev of 0 = FAIL, not warn
smallbatch label <spec> --items items.json  # teacher-label a dataset (network, no GPU)
                                            #   journaled: crashes resume without re-spending;
                                            #   --append keeps rows + sticky gate; --max-variants N
                                            #   = GLOBAL synthetic budget (0 = none); items may
                                            #   carry "gold": routed to gate, never to the teacher
smallbatch review <spec>                    # step through labels: accept/reject/edit (interactive)
smallbatch compile <spec>                   # tfidf + LoRA candidates + eval + gate (GPU, minutes)
                                            #   fails closed on stale labels (--allow-stale-labels);
                                            #   all-fail -> interactive accept prompt (TTY only) or
                                            #   --use-best-anyway; exit stays 2 either way
smallbatch sweep <sweep.yaml>               # grid of model x arm compiles (GPU, long; experimental)
smallbatch run <fn> --json '{...}'          # call a compiled function
                                            #   --version <dir> --candidate lora|tfidf
smallbatch export <fn>                      # GGUF for the LoRA candidate only (experimental)
smallbatch serve <fn> [--port 8080]         # HTTP endpoint; tfidf winners serve CPU-only,
                                            #   LoRA needs export + llama-server (experimental)
smallbatch push <fn> --repo you/name        # whitelist upload to HF Hub; --dry-run previews
                                            #   (NEVER run a real push without the user asking)
smallbatch status                           # functions, verdicts, winners, integrity/drift (instant)
```

Python API mirrors the CLI: `smallbatch.label(...)`, `smallbatch.compile(...)`,
`smallbatch.load_fn(name, version=..., candidate=...)` (see `src/smallbatch/api.py`).

## Exit codes (load-bearing — sweeps and CI depend on them)

`smallbatch compile` returns **0** = gate PASS, **2** = honest gate FAIL, **1**
= real error (including every-candidate-errored). The sweep runner treats 0/2
as honest results and records anything else as an error, continuing. A user
ACCEPTING a failed candidate never changes the gate result or the exit code —
it only sets `deployment.accepted_candidate` in the manifest. Never "fix" a 2
by weakening the gate.

## Module map

Torch-free (import-cheap, unit-tested on CPU): `spec.py` (pydantic
`FunctionSpec`/`TrainSpec`, `extra="forbid"`; `labeling_hash` = label-meaning
identity vs `spec_hash` = build identity; `validate_slug` for path-safe
names), `metrics.py` (THE shared metric layer — every comparison everywhere
computes here; stdlib-only), `artifacts.py` (versioned dirs, manifest v2 with
per-candidate records, `candidate_is_usable` — the ONE usability predicate,
`select_winner`, integrity vs source-drift), `candidates.py` (tfidf+logreg
via sklearn/skops; strict skops trust policy), `journal.py` (durable labeling
journal + lock), `decision.py` (all-fail decision tables — values from
manifest/report, never recomputed), `sweep.py`, `hardware.py`, `prompts.py`
(incl. `incomplete_fields` — the one output-completeness check), `labeling.py`
(three-way sticky splits, gold routing, global variant budget, atomic writes),
`report.py` (build/render + `redact_report`; `report_details.json` is
local-only), `doctor.py`, `review.py`, `init_cmd.py`, `serve.py`'s handlers,
`hub.py` (whitelist `ship_list`), `cli.py`, `api.py` (heavy imports live
*inside* the functions — keep them lazy so `label`/`status`/`sweep` never
import torch; sklearn imports stay inside `candidates.py` functions too).

Torch-heavy (imported only when a compile actually runs): `training.py`
(LoRA/qlora fine-tune), `evaluate.py` (holdout scoring + gate + OOM-backoff
eval batching), `runtime.py`'s LoRA path (`TfidfFunction` is torch-free),
`export.py`'s merge step. `teacher/` holds the labeling backends.

**Subprocess isolation invariant:** `sweep.py` runs each cell as its own
`smallbatch compile` process. This is deliberate — it guarantees VRAM is
released between runs and isolates crashes/OOMs. Do not in-process the loop.

**Candidate isolation invariant:** each compile candidate's failure is
captured into its manifest record; a crash in one candidate must never
discard another's completed work. All-error = exit 1.

**Privacy invariant:** anything that leaves the machine (push/export) ships
the redacted report only, via a positive whitelist (`hub.SHIP_PATTERNS`).
`report_details.json` and `provenance.local.json` never ship. Trained model
state is NOT redacted (tfidf vocabularies contain raw training tokens) — the
push preflight says so.

## Artifact layout

```
artifacts/<fn>/<date>[-rN]/          # a compile version: tfidf/ + adapter/ +
                                     #   spec.yaml (resolved) + spec_files/ +
                                     #   manifest.json + report.md/json +
                                     #   report_details.json (local-only) +
                                     #   provenance.local.json (local-only)
artifacts/<fn>/<sweep>/<tag>/        # a sweep run (never the deployed version)
data/<fn>/                           # labeled train/dev/gate JSONL + journal/
                                     #   (gitignored)
```

`versions()`/`latest()` only look one level under `<fn>` (numeric-aware -rN
sort), so sweep runs never get deployed by `smallbatch run`. Usability is
candidate-scoped: an accepted winner never unlocks an unaccepted secondary.

## Conventions

- Defaults must stay generic: no personal paths, no machine-specific pins in
  tracked files (that's what `CLAUDE.local.md`/`local/` are for — both
  gitignored).
- The default base model is `ibm-granite/granite-4.0-350m` (Apache-2.0, dense
  transformer — deliberately NOT the hybrid `-h-` variant); adapters inherit
  the base model's license.
- `pytest -q` and `ruff check src tests` must pass on CPU with no network
  before any commit.
- The three plan-review documents (PROJECT_EVALUATION.md, V0.2_RELEASE_PLAN_*)
  are gitignored working papers — never track or ship them.
