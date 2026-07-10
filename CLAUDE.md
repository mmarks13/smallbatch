# CLAUDE.md

smallbatch compiles a fuzzy function (score/classify per a rubric) into a
small local model: YAML spec → teacher labels → fine-tune a small base →
eval gate against the teacher's holdout → callable local function.

If a `CLAUDE.local.md` exists in this checkout, it describes THIS machine's
hardware constraints — read it before running any training command; it
overrides the general guidance here.

## Which doc answers what

| You need to... | Read |
|---|---|
| Understand the pipeline, spec fields, gate semantics, artifacts | `docs/how-it-works.md` |
| Run a compile on a local GPU (precision, VRAM, old-GPU pins, OOM knobs) | `docs/local-gpu.md` |
| Run a compile on a rented cloud GPU | `docs/cloud.md` |
| Judge whether a teacher provider may be used | `docs/responsible-use.md` |
| See a complete working spec + walkthrough | `examples/ticket-priority/README.md` |

## Commands

```bash
pytest -q                                   # unit tests: CPU-only, no network, <1s
smallbatch init <template> <name>           # starter spec.yaml + items.json (instant)
smallbatch doctor <spec> [--items X]        # preflight; 1 live teacher probe unless --no-probe
smallbatch label <spec> --items items.json  # teacher-label a dataset (network, no GPU)
                                            #   --append keeps rows + sticky gate; --max-variants N
                                            #   spec `augment:` block: paraphrase/field_dropout/
                                            #   counterfactual; `teacher.consistency: N` self-probe
smallbatch review <spec>                    # step through labels: accept/reject/edit (interactive)
                                            #   --unstable: rows the teacher probe disagreed on
smallbatch compile <spec>                   # train + eval + gate (GPU, minutes); writes report.md
smallbatch sweep <sweep.yaml>               # grid of model x arm compiles (GPU, long)
smallbatch run <fn> --json '{...}'          # call a compiled function (GPU/CPU)
smallbatch export <fn>                      # GGUF + grammar + Modelfile + bundle README
                                            #   (needs LLAMA_CPP_DIR + `pip install gguf`)
smallbatch serve <fn> [--port 8080]         # llama-server + validating HTTP endpoint
smallbatch push <fn> --repo you/name        # upload artifact to HF Hub (private default;
                                            #   NEVER run without the user asking)
smallbatch status                           # list artifacts, gates, staleness (instant)
```

Python API mirrors the CLI: `smallbatch.label(...)`, `smallbatch.compile(...)`,
`smallbatch.load_fn(...)` (see `src/smallbatch/api.py`).

## Exit codes (load-bearing — sweeps and CI depend on them)

`smallbatch compile` returns **0** = gate PASS, **2** = honest gate FAIL (the
adapter trained fine but didn't clear the bar), **1** = real error. The sweep
runner treats 0/2 as honest results and records anything else as an error,
continuing. Never "fix" a 2 by weakening the gate.

## Module map

Torch-free (import-cheap, unit-tested on CPU): `spec.py` (pydantic
`FunctionSpec`/`TrainSpec`, `extra="forbid"` so typos fail loudly; the
`teacher` block is required, no defaults; `output` is scalar or a flat
multi-field map normalized into `OutputSpec.fields`; optional `augment:`
block for paraphrase/field_dropout/counterfactual data), `artifacts.py`
(versioned dirs + manifests), `sweep.py` (grid math + orchestration),
`hardware.py` (precision auto-select), `prompts.py`, `labeling.py`
(three-way sticky splits, append, variant source links), `report.py` (eval
report build/render), `doctor.py` (preflight findings), `review.py` (label
stepper), `init_cmd.py` (templates), `serve.py`'s `handle_call`, `cli.py`,
`api.py` (public `label`/`compile`; heavy imports live *inside* the
functions — keep them lazy so `label`/`status`/`sweep` never import torch).

Torch-heavy (imported only when a compile actually runs): `training.py`
(LoRA/qlora fine-tune), `evaluate.py` (holdout scoring + gate), `runtime.py`
(loading a compiled adapter for `run`), `export.py`'s merge step (grammar and
Modelfile generation are torch-free and unit-tested). `teacher/` holds the
two labeling backends (claude-cli subprocess, openai-compatible stdlib HTTP).

**Subprocess isolation invariant:** `sweep.py` runs each cell as its own
`smallbatch compile` process. This is deliberate — it guarantees VRAM is
released between runs and isolates crashes/OOMs. Do not in-process the loop.

## Artifact layout

```
artifacts/<fn>/<date>[-rN]/          # a normal `compile` version (+ report.md/json)
artifacts/<fn>/<sweep>/<tag>/        # a sweep run (never the deployed version)
artifacts/<fn>/<sweep>/results.json  # + summary.md, written by the sweep
data/<fn>/                           # labeled train/dev/gate JSONL (gitignored;
                                     #   pre-v0.2 dirs had holdout.jsonl = gate)
```

`versions()`/`latest()` only look one level under `<fn>`, so sweep runs never
get deployed by `smallbatch run`. Manifests carry `spec_hash` (spec + all
`spec_files` contents) for staleness detection.

## Conventions

- Defaults must stay generic: no personal paths, no machine-specific pins in
  tracked files (that's what `CLAUDE.local.md`/`local/` are for — both
  gitignored).
- The default base model is `LiquidAI/LFM2.5-350M-Base`; docs note that
  adapters inherit the base model's license.
- `pytest -q` must pass on CPU with no network before any commit.
