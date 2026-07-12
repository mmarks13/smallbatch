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
   │        -> data/<fn>/{train,dev,gate}.jsonl      (stratified; gate is sticky;
   │           every paid call journaled — crashes resume, never re-spend)
   ▼
[2] two candidates trained on the same labels:
   │        a. tfidf + logistic regression (seconds, CPU, ~KBs)
   │        b. LoRA fine-tune of a small base model (HF PEFT/TRL);
   │           dev split scored each epoch -> keep the best checkpoint,
   │           stop early when it plateaus
   ▼
[3] eval report + acceptance gate: BOTH candidates vs the untouched gate split
   │        and vs the zero-shot base; winner = highest teacher agreement,
   │        ties (within gate.tie_margin) go to the smaller artifact
   │                                              (report.md / report.json)
   ▼
artifacts/<fn>/<version>/    candidates (tfidf/ + adapter/) + manifest + report
```

Both candidates are scored on the same gate, so the winner's number is
optimistically biased by selection — the report carries that caveat verbatim.
A candidate that passes every required field always beats a field-failing one
regardless of joint headlines, and an errored candidate never competes (a
LoRA OOM can't discard a completed tfidf candidate, and vice versa).

The sweet spot is a student small enough to run on almost any hardware
(typically under 2B parameters). Other open-weights causal LMs may work as
the base — the tested set spans four architectures from 350M to 9B (the
larger via 4-bit training on 12GB cards); treat anything outside that as
best-effort. What a compiled small specialist **can** do: genuinely match
the teacher on narrow tasks with constrained outputs — scoring and
classification (bounded structured decisions, not free-text extraction).
What it **cannot** do: open-ended generation, and it inherits the teacher's
mistakes (a student can't exceed its labels — supply `gold` items so the
report can show you both). Task choice is the biggest quality lever.

## Function specs

One YAML file per function — explicit, diffable, the single source of truth.
See [`examples/ticket-priority/spec.yaml`](../examples/ticket-priority/spec.yaml)
for a complete working spec.

| Field | Meaning |
|---|---|
| `name`, `description` | Identity; `description` is shown to the teacher. |
| `input_schema` | Ordered `field: type-hint` map. Fields are serialized into prompts in this order. Hints are informational (not validated). |
| `output` | The output contract. Scalar form: `{type: int, range: [lo, hi]}` or `{type: enum, labels: [...]}`. Structured form: a flat map of field name → `{labels: [...]}` (enum) or `{range: [lo, hi]}` (int) — see [Structured outputs](#structured-outputs). Parsed, validated, out-of-contract → `None`. |
| `rubric` | The instructions the teacher labels by; also embedded in the student's training prompt. Anchor it with concrete examples per band — calibration lives here. |
| `spec_files` | External files whose *content* is part of the spec (embedded into teacher prompts, hashed for staleness). Paths resolve relative to the spec file. |
| `teacher` | **Required, no defaults**: `backend` (`openai-compatible` \| `claude-cli` \| `codex-cli`), `model`, plus `examples` (target dataset size), `holdout` (gate split) and `dev` (checkpoint-selection split) — each a fraction of the real rows or an absolute int count — `batch_size`, and for openai-compatible `base_url`/`api_key_env`. |
| `gate` | `agreement` (default 0.85; `agreement_pm1` is the legacy alias), `must_beat_zeroshot` (default true), `must_beat_constant` (default true — the oracle constant on the split), `tie_margin` (default 0.02 — candidate agreements within it tie, smaller artifact wins; a pragmatic margin, not a CI test), `severe_delta` (default 3 — int misses at/beyond it count severe), and for structured outputs optional per-field overrides under `gate.fields`. |
| `train` | `base` model id (default `ibm-granite/granite-4.0-350m`, Apache-2.0; an adapter inherits its base model's license), `precision` (`auto|fp32|bf16|qlora`), LoRA knobs (`lora_r`, `lora_alpha`, `use_dora`), `rationale_distillation`, `max_epochs` (default 12; `epochs` is the legacy alias) with `patience` (default 2, `null` disables early stopping), lr/batch sizes, `loss_type`. Defaults are sensible; a spec can omit the whole block. |

Both the spec and `train` overrides reject unknown keys (`extra="forbid"`), so
a typo fails at load time, not silently.

**Identities:** the manifest records three hashes. `labeling_hash` covers
everything that gives labels their meaning — contract, description, rubric,
`spec_files` *contents*, teacher identity, prompt version, split/augment
recipe — and deliberately excludes build knobs, so `compile --base`/
`--precision` never invalidates a dataset while a rubric edit makes compile
**refuse** stale labels (override: `--allow-stale-labels`, recorded in the
manifest). `spec_hash` is the full build identity; `dataset_hash` fingerprints
the exact rows consumed (review edits change it). Artifacts archive the
*resolved* spec plus content-addressed copies of every spec_file, so they
verify from any machine: `status` reports **integrity** (archive vs manifest)
separately from **source drift** (your live spec vs the snapshot) — a moved
or deleted project is "source comparison unavailable," never stale.

**Gold labels:** any item may carry `"gold": <answer>` — a trusted reference
(human decision, historical outcome, a public dataset's ground truth). Gold
is validated against the contract before any paid call, routed to the gate
(never trained on, the gate grows to hold it), and never sent to the teacher.
With gold present the report shows teacher-vs-gold, student-vs-gold, and
student-vs-teacher side by side; the PASS/FAIL verdict stays **teacher
agreement** and is named as such.

**Starting out:** `smallbatch init classifier|scorer|structured <name>` writes
a working spec skeleton + starter `items.json`, and `smallbatch doctor
<spec>` preflights everything — contract complexity, teacher reachability
(one live probe call), data splits and label coverage (a planned gate or dev
of zero is a hard failure), CUDA/precision/qlora readiness, free disk, export
prerequisites — before you spend teacher calls or GPU time.

## Structured outputs

A function can emit several constrained fields at once — a label plus a
controlled reason code plus a confidence — while keeping every smallbatch
property (narrow contract, constrained decoding, per-field measurability):

```yaml
output:
  priority:
    labels: [urgent, normal, low]
  reason:
    labels: [outage, billing, question, bug]
  confidence:
    range: [1, 5]
```

Type is inferred: `labels` → enum, `range` → int (a reason code is just an
enum). No free-text fields — the useful boundary is structured, checkable
outputs. Field names `type`/`range`/`labels` are reserved (they signal the
scalar form, which keeps working unchanged).

The student emits one `name: value` line per field in declaration order;
decoding is constrained to the enumerated legal completions (when the cross
product is small enough — otherwise outputs are parse-validated), and the
exported GBNF grammar enforces the exact line format. Metrics and the gate
are per-field — int fields ±1, enum fields exact, each against
`gate.agreement` or its `gate.fields.<name>` override — with the joint
all-fields-correct rate reported alongside. The first declared field is the
"primary" one used for stratified splits and variant band-targeting.

## Teacher backends

A teacher is anything that completes a prompt:

```python
class Teacher(Protocol):
    def complete(self, prompt: str) -> str: ...
```

Three shipped implementations, selected by `teacher.backend`:

- **`openai-compatible`** — base URL + key + model against any
  `/chat/completions` endpoint. One backend covers OpenAI, Gemini, Ollama,
  vLLM, LM Studio — teacher-agnosticism without a plugin system. Stdlib-only.
- **`claude-cli`** — shells out to the Claude Code CLI's headless mode
  (`claude -p`); requires the CLI installed and logged in.
- **`codex-cli`** — shells out to the OpenAI Codex CLI's ephemeral headless
  mode (`codex exec`); requires the CLI installed and logged in. Labeling runs
  ignore user config in a read-only sandbox outside the caller's repository.

Before pointing either at a hosted provider, read
[responsible-use.md](responsible-use.md).

## Data generation

The general pattern is **real inputs, teacher labels** — synthetic inputs only
to fill gaps, because distribution mismatch between synthetic and real inputs
is the classic silent failure of this kind of training.

1. Your real items are labeled in batches (rubric-guided, temperature 0). The
   teacher also emits a one-line rationale per label, stored in provenance for
   auditing (not trained on unless `rationale_distillation` is set).
2. The real rows are split **three ways, stratified by label**: `train`,
   `dev` (checkpoint selection during training), and `gate` (the untouched
   acceptance set). Dev and gate are real rows only.
3. Augmentation toward `teacher.examples`: the teacher generates *realistic
   variants* targeted at underrepresented label bands, then the variants are
   relabeled through the same path as real items. Variants are generated
   **only from train-split reals** (the style examples shown to the teacher
   leak into the variants) and always land in train; each records the
   `source_ids` of the reals that seeded it.
4. Every row records provenance: a stable `id`, `real | variant`, its split,
   teacher model/backend, prompt version, label date.

**The gate is sticky.** `smallbatch label --items new.json --append` labels
only unseen items and re-splits without ever moving a row out of the gate —
once anything has trained against the rest of the data, reshuffling the gate
would quietly leak. Top-ups to gate/dev come only from newly labeled reals.
`--max-variants N` is a **global cost budget**: the maximum total new
synthetic rows one `label` invocation may generate across paraphrase,
field-dropout, and counterfactual stages (`0` = no synthetic generation or
labeling calls at all; retained rows never count).

**Every paid call is journaled.** Labeled rows, teacher-generated variant
inputs (with their provenance), and consistency-probe results are appended
durably to `data/<fn>/journal/` the moment they complete. A crash — transport
failure, OOM, ^C — loses at most the in-flight batch; rerunning the same
command replays the journal instead of re-spending, then folds everything
into the dataset files atomically and archives itself. A lock file prevents
two labeling processes from fighting over one dataset. Journals are keyed by
the labeling identity, so a rubric change never replays stale work.

### The `augment:` block — making each label teach more

With a few hundred labels, any surface feature that co-occurs with a label
band a few times becomes a learnable shortcut ("this row had `upvotes: 17`
and scored 8"). The optional top-level `augment:` block configures targeted
augmentation against that; when present it *replaces* the legacy
fill-toward-`teacher.examples` behavior — only the kinds listed run:

```yaml
augment:
  paraphrase: {cap: 50}                        # the classic variants, now declarative
  field_dropout: {fields: [signals], cap: 30}  # per field
  counterfactual: {cap: 30}
```

- **paraphrase** — the legacy name for target-band synthetic variation: the
  teacher sees three train examples as style references, writes new plausible
  inputs for thin score bands, and relabels them independently. These are not
  paired rewrites of one source row.
- **field_dropout** — copies of train reals with one input field blanked,
  **relabeled by the teacher**. If the field mattered, the label honestly
  moves; if not, the pair teaches invariance. Either outcome is real data —
  copying the source label would teach exactly the wrong thing when the
  field is load-bearing. This is the direct fix for optional-field shortcuts.
- **counterfactual** — the teacher makes the *smallest realistic edit* to a
  train real aimed at an underrepresented label band, then the edit is
  relabeled independently. Minimal label-moving pairs trace the rubric's
  decision boundary — the highest-information examples per teacher call. An
  edit whose independently judged label does not move closer to the intended
  band gets one stronger retry; the miss is kept anyway (it's a paid-for
  invariance example). `label` prints the target-progress hit rate. Note
  `cap` bounds the first-pass edits — kept retries land on top, so a run can
  produce more than `cap` counterfactual rows (up to 2× in the worst case).

All augmented rows are generated from **train-split reals only**, land in
train, and record `source_ids` (origin: `variant`, `dropout`, or
`counterfactual` — filterable in `review --origin`).

### Teacher self-consistency (`teacher.consistency: N`)

The student can't agree with the teacher more consistently than the teacher
agrees with itself. With `consistency: 30`, `label` re-sends a stratified
sample of 30 real rows (input fields shuffled) after the main pass and
reports **teacher self-agreement** — the ceiling every gate number should be
read against. It lands in `meta.json` and the eval report ("student is at
94% of the teacher's own ceiling"). Rows where the teacher disagreed with
itself keep their original label but gain `probe_output`; step through them
with `smallbatch review --unstable` (a warning fires if one sits in the
gate split).

**Reviewing labels:** `smallbatch review <spec>` steps through the labeled
rows (filter by split/origin/label/field/review-status) to accept, reject,
edit, or annotate teacher labels before training. Rejected rows stay in
`labeled.jsonl` for audit but are excluded from the split files; edits are
contract-validated and keep the original value in the audit trail.

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
- **Checkpoint selection, not a fixed run:** after each epoch the dev split is
  scored with the same constrained decoding as the final eval, the adapter is
  snapshotted whenever dev agreement improves, and training stops after
  `patience` epochs without improvement (ceiling `max_epochs`). The artifact
  is the *best* checkpoint, not the last one; the manifest and report record
  the full curve, the chosen epoch, and why training stopped. Discovering
  whether training worked is this loop's job — the gate is a final trust
  check, not the discovery mechanism.
- **Optional arms** (A/B-testable via [sweeps](#sweeps)): `use_dora: true`
  (DoRA, one PEFT boolean) and `rationale_distillation: true` (student learns
  to emit reason-then-score using the stored teacher rationales; the runtime
  still returns only the parsed score). In the experiments run so far, neither
  beat plain LoRA on constrained scoring — plain stays the default.

## Evaluation, the report, and the gate

Every compile writes `report.md` + `report.json` next to the manifest — the
report is the headline output, the verdict one line inside it:

- **agreement** with the gate split's teacher labels, always with a **Wilson
  95% confidence interval** (small gates get a loud noise warning below 50
  items) — within ±1 for `int` outputs, exact for `enum`, per-field for
  structured outputs (joint rate reported alongside); int fields also report
  **MAE**, which the ±1 tolerance can't hide distance errors from;
- **must beat the zero-shot base model** on the same gate split (otherwise
  the fine-tune added nothing);
- **must beat the best constant predictor** (`gate.must_beat_constant`,
  default on): on concentrated labels, always answering the modal band can
  score deceptively well under ±1 tolerance — a model that can't strictly
  beat that has learned the label prior, not the task. The report shows the
  constant and its score (majority class for enums);
- diagnostics that make failures actionable: per-label agreement table,
  gold×pred confusion matrix, severe-miss rate (|Δ|≥3), the training curve
  with the chosen epoch, and the largest disagreements with the teacher's own
  rationale for each.

The acceptance check: agreement ≥ `gate.agreement` (per field for structured
outputs, with `gate.fields` overrides).

### Shortcut audit

Every report also carries a `## Shortcut audit` section, auto-derived from
the input schema (no configuration):

- **slices** — agreement/MAE on gate subsets: each field present vs empty,
  per-value slices for low-cardinality fields, length terciles of the
  longest text field. A slice where agreement collapses tells you *where*
  the model is weak.
- **surface-feature correlations** — for input length, field presence, and
  numeric tokens found in the inputs (`upvotes: 17`), the spearman ρ of the
  feature against the **teacher's labels** and against the **student's
  predictions**, side by side. Some surface correlation is legitimate (big
  stories have long summaries); the smoking gun is the student tracking a
  feature notably harder than the teacher does — a gap above 0.25 warns
  (in `report.json`'s `warnings` and after the compile summary) and usually
  means the model learned the feature, not the task. `augment.field_dropout`
  on the implicated field is the standard fix.

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

Mind your gate size: with n=22, one item is ~4.5 points of agreement, and a
pass/fail verdict near the bar is noise — that's why the CI is always shown.
Accumulate real items (`label --append` grows the gate without reshuffling
it) before trusting a marginal verdict either way.

## Sweeps

`smallbatch sweep <sweep.yaml>` answers "which base model and which technique
arm?" by running a `(models × arms)` grid of compiles:

```yaml
name: pick-a-base
spec: ../examples/ticket-priority/spec.yaml
timeout_minutes: 180
models:
  ibm-granite/granite-4.0-350m: {}
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
  adapter/                      # PEFT adapter weights (~10–50MB), best epoch
  spec.yaml                     # copy of the spec that produced it
  manifest.json                 # spec_hash, base model, data provenance,
                                # metrics (+CI), gate verdict, best epoch,
                                # stopping reason, library version
  report.md / report.json       # the full eval report
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
- **`README.md`** + copies of `spec.yaml`, `manifest.json`, `report.md` — the
  bundle is self-describing, with exact llama-cli / llama-server / Ollama /
  curl invocations for this specific function.

`smallbatch serve <fn>` turns the bundle into a local HTTP endpoint: it
launches llama.cpp's `llama-server` on the exported GGUF and fronts it with a
tiny validator — `POST /call` with a JSON object of the input fields builds
the student prompt, enforces the grammar per request, parses the completion
against the contract, and returns `{"output": ..., "raw": ...}` (HTTP 422 if
the output failed validation, which the grammar makes practically
impossible).

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
rendered from the manifest: base model, gate verdict, metrics table, every
candidate's independent quality + deployment state, data provenance, and a
usage snippet. Repos are created **private** unless you pass `--public`;
gate-failed artifacts are refused without `--allow-failed`. Auth is the
standard `hf auth login` / `HF_TOKEN`.

What ships is a **positive whitelist** — manifest, spec + spec_files copies,
the redacted `report.json`/`report.md`, and the candidate model files — and
the exact upload list is printed before any transfer (`--dry-run` stops
there). The local-only `report_details.json` (raw input excerpts, teacher
rationales) and `provenance.local.json` (absolute local paths) never ship.
Two things are NOT redacted and are warned about at push time: the spec/
rubric/spec_files themselves, and trained model state — a fitted TF-IDF
vectorizer stores a `vocabulary_` of raw tokens from your training text.
Remember the adapter inherits its base model's license, and your teacher
provider's terms govern distribution — see
[responsible-use.md](responsible-use.md).

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
