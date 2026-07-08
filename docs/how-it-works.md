# How smallbatch works

## The idea

Lots of useful "functions" are fuzzy: *score this item's relevance 0–10
against an editorial rubric*, *classify this ticket's priority*, *is this
review about shipping, quality, or support?* A frontier model does these well
from a prompt, but calling one per item is slow, costly, and couples your
pipeline to an external service. smallbatch **compiles** such a function into
a small local model:

```
spec.yaml
   │
   ▼
[1] teacher labels your real items (+ generated variants)
   │        -> data/<fn>/{train,holdout}.jsonl        (stratified holdout)
   ▼
[2] LoRA fine-tune of a small base model (HF PEFT/TRL)
   │
   ▼
[3] eval gate: adapter vs teacher holdout, adapter vs zero-shot base
   │
   ▼
artifacts/<fn>/<version>/    adapter (~tens of MB) + manifest
```

The sweet spot is a student small enough to run on almost any hardware
(typically under 2B parameters), but any open-weights causal LM works as the
base — tested to 9B via 4-bit training on 12GB cards. What a compiled small
specialist **can** do: genuinely match the teacher on narrow tasks with
constrained outputs — scoring, classification, extraction. What it **cannot**
do: open-ended generation, and it inherits the teacher's mistakes (a student
can't exceed its labels). Task choice is the biggest quality lever.

## Function specs

One YAML file per function — explicit, diffable, the single source of truth.
See [`examples/ticket-priority/spec.yaml`](../examples/ticket-priority/spec.yaml)
for a complete working spec.

| Field | Meaning |
|---|---|
| `name`, `description` | Identity; `description` is shown to the teacher. |
| `input_schema` | Ordered `field: type-hint` map. Fields are serialized into prompts in this order. Hints are informational (not validated). |
| `output` | `{type: int, range: [lo, hi]}` or `{type: enum, labels: [...]}`. That's the whole output contract — parsed, validated, out-of-range → `None`. |
| `rubric` | The instructions the teacher labels by; also embedded in the student's training prompt. Anchor it with concrete examples per band — calibration lives here. |
| `spec_files` | External files whose *content* is part of the spec (embedded into teacher prompts, hashed for staleness). Paths resolve relative to the spec file. |
| `teacher` | **Required, no defaults**: `backend` (`openai-compatible` \| `claude-cli`), `model`, plus `examples` (target dataset size), `holdout` fraction, `batch_size`, and for openai-compatible `base_url`/`api_key_env`. |
| `gate` | `agreement_pm1` (default 0.85) and `must_beat_zeroshot` (default true). |
| `train` | `base` model id (default `LiquidAI/LFM2.5-350M-Base`; note an adapter inherits its base model's license — LFM's conditions commercial use above $10M revenue), `precision` (`auto|fp32|bf16|qlora`), LoRA knobs (`lora_r`, `lora_alpha`, `use_dora`), `rationale_distillation`, epochs/lr/batch sizes, `loss_type`. Defaults are sensible; a spec can omit the whole block. |

Both the spec and `train` overrides reject unknown keys (`extra="forbid"`), so
a typo fails at load time, not silently.

**Staleness:** the artifact manifest records `spec_hash` — a SHA-256 of the
spec plus the contents of every `spec_files` entry. `smallbatch status` and
`load_fn` warn when a deployed adapter was compiled from a stale spec.
Recompilation is explicit, never automatic.

## Teacher backends

A teacher is anything that completes a prompt:

```python
class Teacher(Protocol):
    def complete(self, prompt: str) -> str: ...
```

Two shipped implementations, selected by `teacher.backend`:

- **`openai-compatible`** — base URL + key + model against any
  `/chat/completions` endpoint. One backend covers OpenAI, Gemini, Ollama,
  vLLM, LM Studio — teacher-agnosticism without a plugin system. Stdlib-only.
- **`claude-cli`** — shells out to the Claude Code CLI's headless mode
  (`claude -p`); requires the CLI installed and logged in.

Before pointing either at a hosted provider, read
[responsible-use.md](responsible-use.md).

## Data generation

The general pattern is **real inputs, teacher labels** — synthetic inputs only
to fill gaps, because distribution mismatch between synthetic and real inputs
is the classic silent failure of this kind of training.

1. Your real items are labeled in batches (rubric-guided, temperature 0). The
   teacher also emits a one-line rationale per label, stored in provenance for
   auditing (not trained on unless `rationale_distillation` is set).
2. Augmentation to reach `teacher.examples`: the teacher generates *realistic
   variants* of real items, targeted at underrepresented label bands, then the
   variants are relabeled through the same path as real items.
3. A stratified holdout (default 15%) is drawn **from real rows only**, so the
   gate is judged on the true input distribution, never on synthetic variants.
4. Every row records provenance: `real | variant`, teacher model/backend,
   prompt version, label date.

## Training

- **Stack:** HF Transformers + PEFT + TRL `SFTTrainer`. LoRA (r=16 default)
  on a causal LM; the task renders as a short prompt → single constrained
  completion.
- **Why LoRA, not full fine-tuning:** full FT of even a ~0.5B model in fp32 needs
  ~10GB for weights+grads+Adam before activations, ships ~2GB per function,
  and risks catastrophic forgetting — for no quality gain on narrow
  constrained tasks. LoRA trains ~1% of params and ships tens-of-MB adapters
  over one shared frozen base.
- **Precision** (`train.precision`, auto-selected by default):
  - `bf16` on Ampere+ GPUs;
  - `fp32` on pre-Ampere cards (no bf16 there; fp16 is slow and fragile) —
    fine for sub-2B students;
  - `qlora`: 4-bit NF4 frozen base + fp16 LoRA — fits 3–8B bases on an 11GB
    card. Needs the `qlora` extra (`pip install smallbatch[qlora]`).
  See [local-gpu.md](local-gpu.md) for the hardware details.
- **Optional arms** (A/B-testable via [sweeps](#sweeps)): `use_dora: true`
  (DoRA, one PEFT boolean) and `rationale_distillation: true` (student learns
  to emit reason-then-score using the stored teacher rationales; the runtime
  still returns only the parsed score). In the experiments run so far, neither
  beat plain LoRA on constrained scoring — plain stays the default.

## Evaluation and the gate

Per compile, recorded in the manifest:

- **agreement** with teacher holdout labels ≥ `gate.agreement_pm1` — within ±1
  for `int` outputs, exact match for `enum`;
- **must beat the zero-shot base model** on the same holdout (otherwise the
  fine-tune added nothing);
- plus diagnostics: exact-match rate, invalid-output rate, Pearson r (int
  outputs).

Generation is **constrained to the output contract**: decoding masks the
vocabulary token-by-token so the model can only emit one of the legal values
(an enumerated int or one of the labels), making invalid outputs impossible.
This applies identically to the adapter, the zero-shot baseline, and
`load_fn`/`run` at inference time, so the gate comparison stays
apples-to-apples. The exception is `rationale_distillation` mode, where the
free-text reason can't be enumerated — that path decodes unconstrained and
relies on output parsing (expect a nonzero invalid rate there).

Failing adapters are kept but marked `failed`; `load_fn` refuses them unless
`allow_failed=True`. Exit codes are load-bearing for automation: `compile`
returns **0** = gate PASS, **2** = honest FAIL (trained fine, didn't clear the
bar), **1** = real error.

Mind your holdout size: with n=22, one item is ~4.5 points of agreement, and a
pass/fail verdict near the bar is noise. Accumulate real items and re-gate on
a bigger holdout before trusting a marginal result.

## Sweeps

`smallbatch sweep <sweep.yaml>` answers "which base model and which technique
arm?" by running a `(models × arms)` grid of compiles:

```yaml
name: pick-a-base
spec: ../examples/ticket-priority/spec.yaml
timeout_minutes: 180
models:
  LiquidAI/LFM2.5-350M-Base: {}
  ibm-granite/granite-4.1-3b: {precision: qlora, batch_size: 2}
arms:
  plain: {}
  rationale: {rationale_distillation: true}
```

Each cell runs as its **own `smallbatch compile` subprocess** — deliberate, so
VRAM is fully released between runs and one crash/OOM can't poison the rest.
Merged train config precedence: base spec < model override < arm override;
every merged spec is validated up front so an override typo fails before any
GPU time is spent, and a preflight checks all model ids are reachable.
Results: `artifacts/<fn>/<sweep>/<tag>/` per run, plus `results.json` and a
models×arms `summary.md`. Sweep runs are never picked up as the deployed
version. Each finished run also prints a `MANIFEST::<json>` line so results
survive in the log even if a rented instance dies before artifacts sync.

## Artifacts and runtime

```
artifacts/<fn>/<date>[-rN]/     # normal compile versions
  adapter/                      # PEFT adapter weights (~10–50MB)
  spec.yaml                     # copy of the spec that produced it
  manifest.json                 # spec_hash, base model, data provenance,
                                # metrics, gate verdict, library version
artifacts/<fn>/<sweep>/<tag>/   # sweep runs (never auto-deployed)
```

`load_fn("name")` loads the base + newest **passing** adapter and returns a
callable that renders the input and parses/validates the output.
`smallbatch run` and `smallbatch status` mirror this on the CLI.

## Exporting to a zero-PyTorch runtime

`smallbatch export <fn>` turns a gate-passing artifact into files that run
without Python or a GPU, written to `<version>/export/`:

- **`<fn>.q4_k_m.gguf`** — the adapter merged into its base, converted and
  quantized with llama.cpp. The default 350M base becomes a ~230MB file that
  runs on a laptop CPU at >100 tokens/s (and each call generates only a few
  tokens).
- **`<fn>.gbnf`** — a grammar generated from the spec's output contract, so
  llama.cpp *cannot* emit anything but a valid output:
  `llama-cli -m <fn>.q4_k_m.gguf --grammar-file <fn>.gbnf -p "<prompt>"`
- **`Modelfile`** — `ollama create <fn> -f Modelfile` and the function is
  servable over Ollama's API. (Ollama doesn't support grammar files, so
  callers on this path should still validate outputs.)

Requirements: a llama.cpp checkout (`--llama-cpp` or `LLAMA_CPP_DIR`), its
`llama-quantize` binary for quantized outputs (`--quant f16` works without
it), and `pip install gguf`. Quant presets: `q4_k_m` (default), `q8_0`,
`f16`. `--adapter-only` instead converts just the LoRA for
`llama-server --lora` over one shared base — useful when running many
functions; merged is the default because separate-adapter serving on a
quantized base is still the flaky path in the ecosystem. Exports refuse
gate-failed artifacts unless `--allow-failed`.

Student prompts are raw text (no chat template): render them exactly as
training did — `[<fn-name>]\n<field>: <value>...\noutput:` — which is what
`prompts.student_prompt` produces.

## Sharing: pushing to the Hugging Face Hub

Local-first is the default — nothing is ever uploaded unless you ask.
`smallbatch push <fn> --repo you/fn-name` uploads an artifact version
(adapter, spec.yaml, manifest.json, and any export/ files) with a model card
rendered from the manifest: base model, gate verdict, metrics table, data
provenance, and a usage snippet. Repos are created **private** unless you
pass `--public`; gate-failed artifacts are refused without `--allow-failed`.
Auth is the standard `hf auth login` / `HF_TOKEN`. Remember the adapter
inherits its base model's license, and your teacher provider's terms govern
distribution — see [responsible-use.md](responsible-use.md).

## Relationship to Program-as-Weights

smallbatch is inspired by **Program-as-Weights**
([arXiv:2607.02512](https://arxiv.org/abs/2607.02512)): natural-language
function specs compiled into LoRA adapters executed by a frozen small
interpreter. PAW's core contribution is a 4B *hypernetwork compiler* (trained
on ~10M synthetic examples) that emits an adapter in one forward pass — a
trade that pays at thousands of functions and welds you to one frozen
interpreter model.

smallbatch deliberately does **not** replicate the hypernetwork. It implements
per-function distillation (one of the paper's own baselines): teacher labels →
gradient-descent LoRA on *any* small HF base you point at.

| | PAW compiler | smallbatch |
|---|---|---|
| Per-function cost | ~seconds, zero examples | teacher calls + minutes of GPU |
| One-time cost | 10M examples + cluster-trained 4B | none |
| Target models | one frozen interpreter (baked in) | any HF causal LM |

What smallbatch borrows is the goal and the product shape — specs, an explicit
compile step, a quality gate, cheap local execution: *distillation wearing a
compiler's UX*. Note the regime difference: PAW beats per-function LoRA in the
paper's low-data setting; smallbatch generates as much teacher data as budget
allows, exactly the regime where gradient-descent LoRA is strong.

A related metaphor collision: **DSPy** also "compiles," but it optimizes the
prompts (and sometimes weights) of a running Python program against a metric.
smallbatch produces a versioned, self-contained local artifact for one
function — closer to a build step than a runtime optimizer.
