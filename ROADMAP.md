# Roadmap

Where smallbatch is headed. The philosophy stays fixed: **narrow functions
with checkable output contracts, compiled and run local-first**. Features
that dilute that (chat task types, hosted endpoints, multi-GPU rigs, GUIs)
are non-goals — see the bottom of this page.

## Near-term: table stakes

- **Richer eval reports.** The stratified holdout already exists; the report
  should use it: confusion matrix and per-label precision/recall for enums,
  per-band agreement for int scores, in both the manifest and `summary.md`.
  A gate FAIL should show *where* it failed — which also feeds the existing
  band-targeted variant generation.
- **Constrained decoding.** ✅ Shipped: the PyTorch runtime and eval mask the
  vocabulary token-by-token to the output contract (adapter, zero-shot
  baseline, and `run` alike; rationale mode stays unconstrained), and the
  GGUF export ships a GBNF grammar. Invalid outputs are structurally
  impossible.
- **Checkpoint resume + training telemetry.** `resume_from_checkpoint` and
  `report_to=[tensorboard]` pass-throughs from TRL. Long runs on consumer
  GPUs fail for boring reasons; resuming beats restarting.

## The headline: a zero-PyTorch runtime artifact

**Mostly shipped** as `smallbatch export`
([docs](docs/how-it-works.md#exporting-to-a-zero-pytorch-runtime)):
merged + quantized GGUF with quant presets ✅, GBNF grammar generated from
the output contract ✅, Ollama Modelfile ✅, `--adapter-only` GGUF for
`llama-server --lora` over a shared base ✅.

Still to come from this line:

- A thin **`smallbatch serve`**: one command that stands up an
  OpenAI-compatible endpoint for a compiled function (wrapping llama-server
  or the PyTorch runtime) so non-Python callers get the function too.

## After that

- **Production capture and drift.** Optionally log `run()` inputs/outputs to
  a per-function dataset; periodically spot-check a sample against the
  teacher; surface drift in `smallbatch status` the same way spec staleness
  is surfaced today, and make "recompile on accumulated real data" a
  one-command loop.
- **HF Hub push (opt-in).** ✅ Shipped as `smallbatch push` — uploads a
  passing artifact with the manifest rendered as the model card, private by
  default.
- **Label review loop.** A CLI pass over teacher labels and generated
  variants — accept/reject/annotate, regenerate from the notes. Trusting
  unseen synthetic rows is the reasonable objection to teacher labeling;
  make inspection cheap.
- **Variant diversity axes.** Band-targeting balances labels but not input
  space. Let the spec declare variation dimensions (e.g. length, formality,
  domain) and steer variant generation across them.
- **Extraction task type.** Multiple typed fields per item — keeps the
  "narrow contract, checkable gate" property, unlike open QA. The `output`
  schema was designed to grow; this is the first real test of that.

## Non-goals

Open-ended generation task types, general chatbots, preference tuning (DPO),
multi-GPU/distributed training, hosted inference, a GUI. Other tools do these
well; smallbatch stays a compiler for narrow fuzzy functions that run on the
hardware you already have.
