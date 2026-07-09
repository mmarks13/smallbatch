# Roadmap

Where smallbatch is headed. The philosophy stays fixed: **narrow functions
with checkable output contracts, compiled and run local-first**. Features
that dilute that (chat task types, hosted endpoints, multi-GPU rigs, GUIs)
are non-goals — see the bottom of this page.

## Near-term: table stakes

- **Richer eval reports.** ✅ Shipped: every compile writes
  `report.md`/`report.json` — agreement with a Wilson 95% CI, per-label
  agreement, confusion matrix, severe-miss rate, training curve + chosen
  epoch, and the largest disagreements with the teacher's rationale.
- **Checkpoint selection + early stopping.** ✅ Shipped: a dev split is
  scored each epoch (same constrained decoding as the final eval), the best
  checkpoint is kept, and training stops on `patience` stale epochs. The
  gate is a final acceptance check, not the discovery mechanism.
- **Constrained decoding.** ✅ Shipped: the PyTorch runtime and eval mask the
  vocabulary token-by-token to the output contract (adapter, zero-shot
  baseline, and `run` alike; rationale mode stays unconstrained), and the
  GGUF export ships a GBNF grammar. Invalid outputs are structurally
  impossible.
- **Structured outputs.** ✅ Shipped: a flat multi-field output map (enum /
  int-range fields, e.g. label + reason code + confidence) with per-field
  metrics and gates, fixed-order line emission, and per-field grammar.
- **Preflight.** ✅ Shipped as `smallbatch doctor`: contract complexity,
  teacher reachability probe, split/coverage checks, CUDA/precision/qlora
  readiness, disk, export prerequisites.
- **Checkpoint resume + training telemetry.** `resume_from_checkpoint` and
  `report_to=[tensorboard]` pass-throughs from TRL. Long runs on consumer
  GPUs fail for boring reasons; resuming beats restarting.

## The headline: a zero-PyTorch runtime artifact

**Mostly shipped** as `smallbatch export`
([docs](docs/how-it-works.md#exporting-to-a-zero-pytorch-runtime)):
merged + quantized GGUF with quant presets ✅, GBNF grammar generated from
the output contract ✅, Ollama Modelfile ✅, `--adapter-only` GGUF for
`llama-server --lora` over a shared base ✅.

- **`smallbatch serve`** ✅ Shipped: wraps llama-server on the exported GGUF
  behind a validating `POST /call` endpoint. The export bundle is now
  self-describing (README with exact commands, spec/manifest/report copies).

## After that

- **Production capture and drift.** Optionally log `run()` inputs/outputs to
  a per-function dataset; periodically spot-check a sample against the
  teacher; surface drift in `smallbatch status` the same way spec staleness
  is surfaced today, and make "recompile on accumulated real data" a
  one-command loop.
- **HF Hub push (opt-in).** ✅ Shipped as `smallbatch push` — uploads a
  passing artifact with the manifest rendered as the model card, private by
  default.
- **Label review loop.** ✅ Shipped as `smallbatch review`:
  accept/reject/edit/annotate with split/origin/label/field filters, audit
  trail kept, splits rebuilt. Still to come from this line: regenerating
  variants from review notes.
- **Variant diversity axes.** Band-targeting balances labels but not input
  space. Let the spec declare variation dimensions (e.g. length, formality,
  domain) and steer variant generation across them.
- **Extraction task type.** Largely covered by structured outputs (multiple
  typed fields per item, per-field gates). What remains from this line:
  free-position span extraction — deliberately parked until it can be done
  without open-ended text output.

## Non-goals

Open-ended generation task types, general chatbots, preference tuning (DPO),
multi-GPU/distributed training, hosted inference, a GUI. Other tools do these
well; smallbatch stays a compiler for narrow fuzzy functions that run on the
hardware you already have.
