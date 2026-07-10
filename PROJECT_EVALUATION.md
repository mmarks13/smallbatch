# smallbatch project evaluation

Date: 2026-07-09

This evaluation is based on the `v0.2-rework` branch at commit `94490ba`, the
alignment answers provided by the maintainer, local source and documentation
inspection, CPU test execution, CLI smoke tests, package inspection, and a
review of the current adjacent-tool landscape. Market links were checked on
the date above. No product code or existing documentation was changed as part
of this review.

## Executive summary

smallbatch has a useful core idea and unusually good implementation instincts
for an unvalidated alpha. It draws a sensible boundary around constrained
classifiers and scorers, separates development and gate data, attempts to
avoid synthetic-data leakage, uses constrained decoding, writes inspectable
artifacts, and has 95 fast unit tests. The repository is more coherent than
many first releases.

It is not ready to support its current claims or a broad open-source launch.
The primary problem is not missing polish. The project has built a wide
feature surface before demonstrating that its central method is the best way
to solve its target problem. The only experiment is a small, private pilot
that did not clear its own gate. There is no human-grounded evaluation, no
classical or encoder-classifier baseline, no cost or energy accounting, no
end-to-end CI test, no published package, and no external user validation.

The most important product correction is this:

> smallbatch should be a classifier compiler, not a LoRA compiler.

Given a rubric and representative examples, it should try cheap conventional
classifiers, small encoder models, and a small generative adapter, then package
the smallest candidate that meets an independently defined quality and runtime
budget. A 350M autoregressive language model may win on some tasks, but it is
an implausible universal default for a product whose stated priorities are
cost and energy. The tool should prove when weights are useful, not assume the
answer in advance.

The recommended value proposition is:

> **Turn a rubric and real examples into a tested local classifier.**
>
> smallbatch labels with the teacher you choose, compares small candidate
> models, and packages the smallest one that meets your quality bar. Runtime
> calls stay local and incur no hosted-model API fee.

For the software that exists today, use the narrower, fully accurate version:

> **Build a constrained local classifier from a rubric and representative
> examples, and measure how closely it matches your chosen teacher.**

Do not lead with "Program-as-Weights," "fuzzy functions," energy reduction,
"runs anywhere," or "a model you own" until the project has evidence and
license semantics to support those phrases. ProgramAsWeights now occupies the
spec-only fuzzy-function compiler position directly. smallbatch can be
meaningfully different: slower to build, but grounded in private real-world
data, independently evaluated, auditable, and controlled by the user.

### Release recommendation

Do not publish the current branch as a feature-complete v0.2. First make a
smaller "truthful alpha" release that:

1. fixes data/spec integrity, artifact privacy, artifact reproducibility, and
   the install path;
2. makes independent gold labels an explicit concept;
3. adds at least one non-generative baseline;
4. publishes a reproducible benchmark and removes the current empirical
   claims until that benchmark exists;
5. narrows the public workflow to the few commands needed to build, inspect,
   and call one classifier; and
6. clearly labels every unexercised integration as experimental.

## Goals used for this evaluation

The intended outcomes are:

- a widely adopted open-source tool;
- a tool the maintainer personally relies on;
- a first audience of data scientists familiar with Hugging Face, followed by
  application developers replacing recurring hosted-LLM calls;
- lower inference cost and, when measured, a lower energy footprint;
- approachable fine-tuning;
- shareable local classifier functions; and
- eventually, a company-managed library of such functions.

These goals imply a stronger standard than "training completed." A successful
build must answer four questions:

1. Is the task suitable for compilation at all?
2. Is the resulting function correct enough against evidence independent of
   the teacher response used for training?
3. Is it materially cheaper, faster, smaller, or more private than the
   alternative?
4. Can another person install and call the artifact without reconstructing
   the training environment?

The current product answers parts of question 2 and question 4, but not yet
strongly enough to rely on.

## Current-state scorecard

These scores are directional, not scientific. They make the major imbalance
visible.

| Dimension | Score | Rationale |
|---|---:|---|
| Product usefulness | 4/10 | The target job is real, but no production or external use has been validated and no break-even result exists. |
| Robustness | 4/10 | Pure helpers are well tested, but expensive operations are not resumable, builds are not atomic, and no real end-to-end path is tested. |
| Simplicity | 4/10 | The core idea is narrow; the public surface has already grown to 11 commands and several experimental technique arms. |
| Usability | 3/10 | The quickstart requires Ollama, a teacher model, CUDA, a large Python environment, and later a llama.cpp checkout. The documented PyPI install currently returns 404. |
| Evaluation rigor | 2/10 | Dev/gate separation and confidence intervals are good starts, but the gate is another draw from the same teacher and model selection has touched the small gate. |
| Documentation structure | 6/10 | The writing is clear and the conceptual docs are strong, but prominent claims exceed the evidence and important caveats appear too late. |
| Engineering foundation | 6/10 | 4,076 source lines, 1,426 test lines, 95 passing tests, lazy heavy imports, and clear modules are positive. There is no CI, lint/type configuration, release automation, or integration test. |
| Open-source readiness | 2/10 | No published package, tag, CI, contributing guide, security policy, changelog, issue templates, or independently runnable benchmark. |

The repository contains 54 tracked files across only eight commits. That does
not make the work poor, but it means the apparent breadth has not yet had time
to be exercised by users, dependency changes, failed runs, or multiple release
cycles.

## What is already strong

The following ideas should survive a ruthless simplification.

### 1. A deliberately narrow output contract

Enum and bounded-integer outputs make evaluation, parsing, serving, and safety
tractable. The project is right to reject general chat and open-ended
generation. This is the clearest product boundary in the repository.

### 2. A spec as the semantic source of truth

A diffable function definition is useful for a catalog of organizational
decisions. Pydantic's rejection of unknown keys is a good choice. The spec
needs stronger validation and a separation between semantic contract and build
configuration, but the concept is sound.

### 3. Real examples before synthetic augmentation

The effort to keep generated variants out of dev and gate splits is thoughtful.
Sticky gate assignment is also directionally correct. The terminology should
be "user-provided" rather than "real," because the bundled example inputs are
themselves synthetic, but the leakage concern is valid.

### 4. Development selection separate from final acceptance

Scoring each epoch on a dev split and preserving the best adapter is much
better than evaluating only the final epoch. The distinction between training
selection and final acceptance should remain.

### 5. Contract-constrained inference

Applying the same constrained output space to the adapter and zero-shot base is
a useful invariant. It eliminates format noise from the comparison. It does
not make the semantic answer correct, but it makes failure easier to measure.

### 6. Inspectable reports and failure-preserving artifacts

Keeping a failed build, returning exit code 2 for a quality failure, and
producing confusion and disagreement details are good operational choices.
The report needs privacy controls and stronger metrics, not removal.

### 7. Local-first ownership of the workflow

Plain files, no required control plane, and an opt-in hosted teacher are
valuable. This is more defensible than claiming all data is local: a hosted
teacher necessarily receives labeling inputs. The accurate advantage is that
smallbatch itself requires no hosted platform and runtime inference can be
fully local.

## Product definition and positioning

### The job to be done

The primary job is not "fine-tune a model." It is:

> I have a stable, repeated decision currently implemented by a costly LLM
> call. Help me determine whether it can become a reliable local classifier,
> build the cheapest acceptable implementation, and give me an artifact I can
> share and operate.

This framing changes product priorities:

- Quality must be evaluated against an independent target, not only imitation.
- Cost, latency, memory, and energy are outputs of the compile, not marketing
  assumptions.
- Conventional classifiers are candidates, not competitors to exclude.
- Packaging and repeatable loading matter as much as training.
- A failed compile that explains why the API should not be replaced is a
  valuable product result.

### Recommended category

Use **local classifier compiler** or **decision compiler**. "Small-batch
distillation" accurately names one technique, but it does not clearly describe
the user outcome. "Fuzzy function compiler" now invites a direct comparison
with ProgramAsWeights, whose compilation is much faster and requires no
dataset or GPU from the user.

### The differentiated promise

smallbatch should own these words:

- **grounded:** built and evaluated on the user's representative inputs;
- **measured:** quality, cost, latency, size, and energy are reported;
- **smallest acceptable:** the compiler selects a method under explicit
  constraints;
- **local runtime:** after build, inputs need not leave the user's environment;
- **auditable:** spec, data lineage, model revisions, metrics, and failure cases
  are preserved; and
- **portable:** a recipient can install a function artifact without the
  training stack.

### When to use smallbatch

- The output is a stable enum, boolean, bounded ordinal score, or a small set of
  such fields.
- The same decision runs often enough for compilation cost to amortize.
- Representative historical inputs exist or can be collected.
- The team can define a meaningful gold set or review labels.
- Privacy, latency, offline use, or platform independence matters.
- An occasional retraining cycle is operationally acceptable.

### When not to use it

- Call volume is low enough that a hosted API is simpler and cheaper.
- The rubric or label set changes frequently.
- The task requires current world knowledge, long reasoning, citations,
  explanations, or open-ended generation.
- Inputs differ radically from anything available during development.
- The task is safety-critical and no independent expert-labeled test set is
  available.
- A regex, rules engine, SQL expression, embedding similarity threshold, or
  conventional classifier already meets the requirement.

The last point is important. A tool earns trust by declining needless model
training.

## Market landscape

There is no single incumbent with exactly the recommended combination, but
every layer of the current implementation has mature alternatives. The market
opportunity is orchestration plus evidence, not novel LoRA mechanics.

| Tool/category | What it already does well | Implication for smallbatch |
|---|---|---|
| [ProgramAsWeights](https://programasweights.com/docs) | Compiles a natural-language spec in seconds into a small local program; offers shared base runtimes, Python/browser SDKs, versioned community programs, and a public hub. | This is the direct competitor for "describe a fuzzy function and run it locally." Do not compete on zero-data speed. Differentiate on private representative data, independent evaluation, user-controlled training, constrained business classifiers, and artifact ownership. |
| [Prompt2Model](https://arxiv.org/abs/2308.12261) | The 2023 research system retrieved data/models, generated data, fine-tuned a deployable model, and reported reliability from a natural-language task description. | The overall concept predates this repository. Cite it and explain that smallbatch is narrower, local-first, contract-constrained, and operational rather than presenting prompt-to-model as new. |
| [DSPy optimizers](https://github.com/stanfordnlp/dspy/blob/main/docs/docs/learn/optimization/optimizers.md) | Optimize prompts and examples, bootstrap traces, and use `BootstrapFinetune` to distill a prompted program into weights against a user metric. | DSPy is broader and code-centric. smallbatch can win on a packaged single-function lifecycle and local artifact, but must have a stronger evaluation contract and simpler first run. |
| [Kiln](https://github.com/Kiln-AI/Kiln) | A local-first workbench spanning tasks, datasets, human ratings, synthetic data, evals, prompt optimization, fine-tuning, and deployment, with a desktop UI and Python library. | Kiln is a serious integrated-workflow alternative with far more collaboration UX. smallbatch must stay much narrower and be substantially easier to reason about. "No GUI" is viable only if files/notebooks and CLI review are excellent. |
| [OpenPipe](https://docs.openpipe.ai/overview) | Captures production requests, curates datasets, fine-tunes, evaluates, and hosts cheaper model replacements behind an OpenAI-compatible API. | This validates the replacement-model job. smallbatch's counter-position is open-source, local, provider-independent runtime and no hosted control plane. OpenPipe is ahead on data capture and production iteration. |
| [SetFit](https://huggingface.co/docs/setfit/index) | Trains prompt-free, low-latency few-shot text classifiers using sentence transformers; supports knowledge distillation and ONNX deployment. | This should be a baseline and likely a candidate backend. Excluding it undermines the cost and energy proposition. |
| [Distilabel](https://distilabel.argilla.io/) | Provides scalable, fault-tolerant synthetic-data and AI-feedback pipelines across providers, with structured-generation integrations. | Do not try to become a general data-generation framework. Keep one opinionated labeling path or integrate/export to mature dataset tools. |
| [Hugging Face AutoTrain](https://huggingface.co/docs/autotrain/en/quickstart), [Axolotl](https://docs.axolotl.ai/docs/getting-started.html), and [LLaMA-Factory](https://llamafactory.readthedocs.io/en/latest/getting_started/webui.html) | Mature configurable local/cloud training, LoRA/QLoRA, evaluation, checkpointing, and export across many architectures. | smallbatch should not compete on training knobs or model coverage. Hide the engine behind task-level decisions and a tested compatibility set. |
| [OpenAI model distillation](https://openai.com/index/api-model-distillation/) and managed fine-tuning | Integrated stored completions, evals, fine-tuning, and hosted inference historically made distillation convenient. [OpenAI's current fine-tuning page](https://openai.com/index/gpt-4o-fine-tuning/) states that its self-serve fine-tuning platform is being wound down for new users. | Managed platforms can change or disappear, which strengthens the portability argument. Do not depend on one provider's output rights or product lifecycle. |

### Most defensible market position

The strongest position is between SetFit and OpenPipe:

- more end-to-end and decision-oriented than SetFit;
- more local, open, portable, and constrained than OpenPipe;
- more evidence-driven and distribution-specific than ProgramAsWeights;
- far narrower and quieter than Kiln; and
- far easier than assembling Distilabel, TRL/PEFT, evaluation code, and
  llama.cpp manually.

The product is not "fine-tuning for everyone." It is "an evidence-driven exit
ramp from recurring LLM classification calls."

### A time-sensitive opportunity

ProgramAsWeights makes the broad compiler metaphor legible to the market, but
its [hosted service terms](https://programasweights.com/terms) grant broad
rights over submitted content and state that compiled programs are not the
user's intellectual property. smallbatch
can offer a clean alternative for proprietary functions: local files, a
teacher chosen by the user, no mandatory compilation service, and artifacts
whose redistribution is governed transparently by the chosen teacher and base
licenses. That advantage is only credible if the default base license is
company-friendly and export/push cannot leak evaluation inputs.

## Ruthless scope decisions

The current 11-command surface is too large for a project with no validated
user. It makes the README look mature while distributing attention across
training research, dataset curation, serving, cloud operation, and publishing.

### The public happy path

The README should teach only this conceptual flow:

```text
init -> build -> inspect -> run
```

`build` can remain named `compile` if the compiler metaphor is important. The
key simplification is behavioral: a normal compile should accept the source
items, automatically run preflight checks, checkpoint labeling, train the
default candidate set, evaluate, and produce one report. Expert users can
still invoke data and model stages separately.

The public mental model should contain five nouns:

1. **function spec:** what decision is being implemented;
2. **examples:** representative inputs, some optionally gold-labeled;
3. **build:** the attempt to find a suitable local implementation;
4. **report:** evidence and resource tradeoffs; and
5. **artifact:** the installable function selected by the build.

Teacher, model, LoRA, split, grammar, and quantization details belong in the
advanced path unless they require a user decision.

### Capability disposition

| Current capability | Decision | Reason |
|---|---|---|
| `init` | Keep and improve | A guided starting point is core. Generate enough explicit examples to explain the schema, warn that a meaningful build needs many more, and validate names and YAML label traps. |
| `doctor` | Keep as an expert command, run automatically | Preflight is valuable but users should not have to remember it. Hard failures such as a zero-sized gate must stop before paid labeling. |
| `label` | Keep under an advanced data workflow | Label generation, append, provenance, and resumability are core internals. It need not be a top-level concept for an application developer. |
| `review` | Keep the concept, redesign the interface | Human correction is more important than technique sweeps. A terminal row stepper will not scale. Add CSV/JSONL round-trip and notebook/DataFrame workflows before considering a custom GUI. |
| `compile` | Keep as the product center | It should select among task-appropriate candidates, enforce data integrity, evaluate against gold data, benchmark runtime, and emit a portable artifact. |
| `run` | Keep, but separate runtime from training | A recipient should not install Torch, TRL, PEFT, and Datasets to call an exported function. Support explicit version selection and machine-readable batch output. |
| `status` | Keep, refocus on a function catalog | Show one promoted version per function, available versions, runtime/backend, license, quality basis, freshness, and last use. Do not dump every sweep cell as if it were deployable. |
| `export` | Strategically core, operationally experimental | A shareable CPU artifact is central to the promise. The current llama.cpp checkout workflow has not been validated end to end and only covers the causal-LM backend. Stabilize the artifact contract before promoting it. |
| `sweep` | Remove from the primary surface | It is a maintainer research tool today and reuses the final gate for selection. Candidate comparison should eventually happen inside `compile` using dev data, with one final untouched test. |
| `push` | Remove from the release surface for now | It can upload reports containing excerpts of proprietary gate inputs and teacher rationales. Generic Hub upload adds little before a safe portable artifact exists. |
| `serve` | Defer | A single-threaded stdlib proxy around a separately exported GGUF is not yet a production serving story. First make the artifact easy to call; document standard serving engines later. |
| Structured multi-field output | Freeze, do not lead with it | It is useful eventually, but it multiplies cross-products, metrics, grammar, and partial-validation cases before scalar classification is proven. It is not free-text extraction. |
| Integer scoring | Keep after enum classification | Ordinal scoring is a plausible second task, but it needs ordinal baselines and metrics such as MAE and weighted kappa, not only +/-1 agreement. |
| DoRA | Remove from user-facing configuration | The repository's only experiment found no benefit. It is training-engine detail without demonstrated product value. |
| Rationale distillation | Remove or quarantine as experimental | It disables constrained decoding, expands output and parsing complexity, and did not improve the pilot. It conflicts with the strongest product boundary. |
| `claude-cli` teacher | De-emphasize | It is operationally unusual, difficult to reproduce, and easy to misunderstand under provider terms. Keep a Python teacher protocol and one well-tested HTTP backend first. |
| SkyPilot guide | Move to an advanced recipe | Cloud training is useful, but it should not be part of the core product until the local workflow and artifact transfer are proven. |
| Production capture and drift | Defer until single builds are trusted | Feedback and drift are necessary for long-lived functions, but building them now would compound an unvalidated artifact and evaluation model. |

### Features that are actually missing

The most important missing features are not more training techniques.

1. **Independent gold evaluation.** A user-reviewed or ground-truth final set
   must be a first-class input and must never be generated by the teacher being
   evaluated.
2. **Cheap candidate baselines.** At minimum: majority/class-frequency,
   TF-IDF plus logistic regression, and a small embedding or SetFit classifier.
3. **Build economics.** Record teacher tokens and cost, training time and
   energy, artifact size, cold start, memory, throughput, latency, and the
   call-volume break-even point.
4. **Resumable labeling.** Persist each successful batch atomically, cache by
   prompt/model/input hash, and resume after transport or process failure.
5. **A portable runtime package.** Installing and calling a finished function
   should be a small dependency path independent of the training stack.
6. **Safe artifact sharing.** Reports and manifests must default to no raw or
   excerpted user data. Sharing should run a privacy preflight.
7. **Function version promotion.** Build versions, select one, give it a stable
   alias such as `production`, and support rollback and explicit loading.
8. **A shared-base library runtime.** Multiple compatible adapters should share
   one loaded base model, or the documentation must stop claiming shared RAM.
9. **Typed input contracts and length policy.** Validate values, extra fields,
   nesting, encoding, maximum lengths, and truncation behavior consistently.
10. **Reproducible model identity.** Pin base/tokenizer revisions and capture
    dependency, hardware, dataset, prompt, and code versions.

## Technical and robustness audit

Severity definitions:

- **P0:** blocks a trustworthy public release or risks wasted money/data;
- **P1:** should be fixed before recommending the tool for repeated use; and
- **P2:** important hardening or usability work after the core is proven.

### P0 findings

#### P0-1: The documented installation does not exist

`README.md` starts with `pip install smallbatch`, but the PyPI JSON endpoint
for `smallbatch` returned HTTP 404 on 2026-07-09. The repository contains
ignored local v0.1.0 distributions, the branch still reports version 0.1.0,
and there are no release tags.

Impact: the first command for every potential user fails. This alone prevents
adoption and makes every quickstart unverifiable.

Direction: either publish a tested release with CI and trusted publishing, or
document an exact Git commit install while clearly calling it unreleased.

#### P0-2: Compile can train against labels from a different function spec

[`doctor.py`](src/smallbatch/doctor.py#L103) only warns when dataset
`spec_hash` differs from the current spec. [`api.py`](src/smallbatch/api.py#L106)
loads the split files without checking that hash at all. A user can change the
rubric, labels, reference files, or teacher settings and then train against old
labels while the artifact records the new spec hash.

Impact: the resulting function and report can claim to represent a definition
that never produced its training or gate labels. This invalidates provenance,
staleness, and quality evidence.

Direction: compilation must fail closed on semantic-spec, prompt-version, and
dataset snapshot mismatches. Explicit migration or relabel flags can override
this with a recorded reason.

#### P0-3: The gate measures imitation, not correctness

Every real item is labeled once by the teacher, then the student is evaluated
against a held-out subset of those outputs. No independent gold-label concept
exists. A teacher that consistently misunderstands the rubric can produce a
passing student. A teacher that is stochastic can make a correct student look
worse.

Impact: `PASS` means "matched this teacher draw," not "safe to replace the
application's decision." README language such as "quality check" and "only a
right or wrong answer" overstates what is known.

Direction: report teacher agreement and gold accuracy separately. Deployment
gates should use user-defined gold outcomes when available. Without gold data,
the verdict must be named an imitation gate and prominently qualified.

#### P0-4: Model selection consumes the supposed final gate

`smallbatch sweep` evaluates every model/technique cell on the same gate and
presents a comparison table. Choosing the best cell therefore tunes on the
gate. The README pilot compares four models on a gate of only 22 items and then
highlights the best two.

Impact: the final estimate is optimistically biased and no untouched evidence
remains after model selection.

Direction: use train data for fitting, dev or cross-validation for candidate
selection, and one locked test set exactly once for the chosen build. A failed
final test requires a new versioned benchmark, not continued selection on the
same rows.

#### P0-5: Expensive labeling is not resumable

[`build_dataset`](src/smallbatch/labeling.py#L312) writes the dataset only after
all real labeling, split assignment, variant generation, and variant labeling
complete. A process exit, invalid response, terminal loss, or provider failure
near the end loses all successful in-memory work and its cost.

Impact: this is unacceptable for a workflow explicitly intended to spend
teacher calls once.

Direction: journal request batches and parsed rows as they complete; make each
operation idempotent; store prompt/model/request metadata and token usage; and
resume by content hash.

#### P0-6: Shared artifacts can disclose user data

[`report.py`](src/smallbatch/report.py#L119) places gate-input excerpts and
teacher rationales in both report formats. [`hub.py`](src/smallbatch/hub.py#L176)
uploads the full artifact directory, including those reports. `export` copies
`report.md` into the supposedly shareable runtime bundle.

Impact: `push --public`, a private repo with broad membership, or simply
sending an export bundle can disclose customer tickets, identifiers, or other
proprietary text. This directly conflicts with the privacy value proposition.

Direction: make reports private diagnostic artifacts by default; generate a
separate redacted share manifest; require an explicit `--include-examples`
choice; scan for known sensitive fields; and show the exact upload file list
before transfer.

#### P0-7: Artifact staleness is incorrect for normal `spec_files` and CLI overrides

Compile copies `spec.yaml` but not relative `spec_files` into the version
directory. [`artifacts.staleness`](src/smallbatch/artifacts.py#L97) reloads the
archived YAML relative to that new directory, so a normal relative reference is
reported missing immediately. This was reproduced in the audit.

Separately, `compile --base` and `--precision` mutate the in-memory spec, hash
that mutation, and record it in the manifest, but [`api.py`](src/smallbatch/api.py#L181)
copies the original unmodified YAML into the artifact. Such a build is also
self-inconsistent.

Impact: freshness warnings cannot be trusted, exported specs can misstate how
the model was built, and an artifact is not self-contained.

Direction: archive a fully resolved semantic spec plus a separate resolved
build record; copy or content-address every referenced file; and test the
archive from a different directory with the source tree removed.

#### P0-8: The starter can approve a plan with no gate

The generated starter has three placeholder items. In the smoke test,
`doctor --no-probe` reported a planned split of `3 train / 0 dev / 0 gate`,
classified the zero-sized gate only as a warning, and ended with `doctor: ok`.
Compile later rejects an empty gate, after labeling may already have incurred
cost.

Impact: preflight explicitly fails to prevent a known downstream hard error.

Direction: zero train, dev, or required gate counts are failures. The starter
should distinguish illustrative placeholder rows from the minimum data needed
for a meaningful attempt.

#### P0-9: The default model license conflicts with the target market

The default base, `LiquidAI/LFM2.5-350M-Base`, uses the LFM license, whose
commercial terms change at $10 million in annual revenue. The documentation
does disclose this, but a default intended for company function libraries
should not surprise larger adopters with a revenue threshold.

Direction: benchmark and select a permissive default. Apache-2.0 candidates of
similar scale currently include
[`Qwen/Qwen3-0.6B-Base`](https://huggingface.co/Qwen/Qwen3-0.6B-Base) and
[`ibm-granite/granite-4.0-350m-base`](https://huggingface.co/ibm-granite/granite-4.0-350m-base).
Keep license metadata and an explicit policy check in every build regardless
of the default.

### P1 findings

#### P1-1: The implementation assumes a generative LM before testing cheaper models

Every compile trains a causal LM adapter. For bounded classification, a linear
classifier over TF-IDF or frozen embeddings, SetFit, or an encoder with a
classification head can be much smaller, faster, easier to export, and lower
energy.

Direction: define a candidate interface around `fit`, `predict`, `package`, and
`measure`. Make causal LoRA one backend selected by evidence.

#### P1-2: The "shared base" RAM claim is not implemented

[`load_fn`](src/smallbatch/runtime.py#L38) loads a fresh base model for each
function. `serve` requires a merged per-function GGUF. Adapter-only export
exists, but no catalog runtime loads one base with multiple adapters or routes
calls among them. PEFT itself supports loading and selecting multiple adapters,
but smallbatch does not expose it.

Impact: ten functions can share Hugging Face cache storage, but normal Python
loading can consume ten base models of RAM. Merged exports also duplicate the
base on disk. The README claim is materially inaccurate.

Direction: either implement a shared runtime grouped by exact base revision or
limit the claim to adapter artifact size and cached base downloads.

#### P1-3: Input schemas are descriptive, not typed contracts

`input_schema` maps names to unparsed strings. Runtime and server code check
only for missing keys; they accept wrong types, unexpected keys, unbounded
payloads, and values that serialize ambiguously. The docs call the hints
informational in one place while prominent feature text says input validation.

Direction: adopt a deliberately small JSON Schema subset, validate identically
at label and runtime time, and record how unknown fields are handled.

#### P1-4: Output-spec validation permits ambiguous or pathological contracts

`FieldSpec` accepts reversed ranges, duplicate labels, newline-containing
labels, and case-insensitive collisions. A reversed range was reproduced and
produced no legal values. YAML also parses unquoted `yes`/`no` as booleans,
yielding a low-level validation error rather than an actionable hint. Gate
thresholds and most training counts/rates lack bounds.

Direction: validate ranges before enumerating them, constrain label syntax,
reject duplicates after normalization, validate thresholds and positive
counts, and provide YAML-specific guidance.

#### P1-5: Prompt boundaries and length behavior are unsafe and inconsistent

Raw values are interpolated as `field: value` lines. A value can contain
newlines, fake field names, or `output:` and interfere with the student prompt.
Teacher content is embedded in a single user message without a system-level
untrusted-data boundary or a provider JSON schema. Training uses
`max_seq_len`; evaluation hard-codes 2,048 tokens; truncation is silent.

Direction: serialize a canonical structured payload, mark all item content as
untrusted, use provider-native structured output when available, calculate
token budgets before requests, and report/reject truncation rather than hiding
it.

#### P1-6: Reproducibility metadata is insufficient

The manifest records model names but not model/tokenizer commit revisions,
package versions, CUDA/driver information, hardware, exact dataset content
hashes, prompt implementation hash, or source revision. Provider aliases such
as `sonnet` can change underneath the same string.

Direction: create a build lock containing immutable revisions and environment
facts. Rebuilding should either reproduce the same candidate or explain every
changed input.

#### P1-7: Builds and dataset writes are not transactional

A normal version directory is created before training. Failure can leave a
large incomplete directory. Sweep reruns delete the previous populated cell
before the replacement succeeds. Dataset files and manifests are overwritten
in place, with no lock or atomic rename.

Direction: build in a temporary state directory, write a state record, fsync
where appropriate, atomically promote a completed build, retain failed logs,
and lock per function/dataset.

#### P1-8: The dependency and compatibility policy is too optimistic

The base install pulls Torch, Transformers, PEFT, TRL, Datasets, and Accelerate.
The audit environment was 5.5 GB; Torch alone occupied about 1.5 GB. Minimum
versions have no tested upper constraints even though these libraries change
rapidly. The local environment already runs versions far above several listed
minimums.

Direction: split `smallbatch` core/runtime/train/export extras, keep a tested
constraints file or lock for releases, and test a small supported matrix.
GPU-specific Torch installation should be an explicit pre-install step rather
than an incidental dependency resolution outcome.

#### P1-9: No end-to-end behavior runs in CI

There is no `.github/workflows` directory. The 95 CPU tests pass in 0.42
seconds, but training, a real HTTP teacher exchange, model loading, adapter
loading, GGUF conversion, llama.cpp serving, and Hub packaging are mocked or
unexercised. Coverage tooling is not in the dev extra.

Direction: add lint/type/build/install tests, a mock HTTP integration, a tiny
local model train-load-call test, and scheduled GPU/export tests. Label feature
claims according to the highest level actually exercised.

#### P1-10: Runtime versioning is incomplete for a function library

`load_fn` always chooses the latest passing version and cannot select an
explicit version or stable alias. `serve` and `export` do accept a version.
Lexicographic date/revision sorting also misorders `-r10` relative to `-r9`.

Direction: use sortable build IDs, explicit semantic aliases, `load_fn(...,
version=...)`, promotion, rollback, and garbage collection.

#### P1-11: Structured runtime validation can accept partial outputs

For a multi-field contract, `parse_output` returns a dictionary when any field
parses and sets missing fields to `None`. `handle_call` rejects only a wholly
`None` result, so an unconstrained or faulty backend can return HTTP 200 with a
partially invalid object.

Direction: expose one contract validator and require all fields to validate at
labeling, evaluation, Python runtime, and HTTP runtime boundaries.

#### P1-12: Error handling is inconsistent

Some commands translate expected failures into concise messages, while common
file errors still print tracebacks. A missing items file reproduced a full
traceback. Network errors do not preserve response bodies/status in structured
run logs.

Direction: define typed domain errors, concise default CLI messages, an
optional `--debug`, and structured JSON errors for automation.

### P2 findings

- `--max-variants` is documented as a cap but is passed as `n_new`, which
  overrides the calculated need and can request that many variants even when
  fewer are needed to reach the target.
- Variant allocation independently rounds each band and can underproduce or
  overproduce the requested total.
- Function, sweep, arm, field, and label names need consistent filesystem and
  grammar safety. `FunctionSpec.name` currently flows into artifact paths
  without validation.
- `status` assumes manifest shapes and has no machine-readable output.
- Export can leave cached outputs from a previous quantization choice and
  chooses the first sorted GGUF when serving, not an explicit preferred build.
- The teacher-consistency script handles scalar numeric scores only, can divide
  by zero, and describes pairwise self-agreement as a "ceiling," which is not a
  generally valid statistical interpretation.
- The bundled ticket example calls its 71 hand-written synthetic inputs
  "real" once ingested, while the methodology repeatedly says gate and dev are
  real data. The provenance vocabulary should distinguish user-provided,
  imported-gold, teacher-generated, and augmented sources.
- The 1.3 MB banner is excluded from built distributions while README metadata
  refers to it with a relative path, so package-index rendering is likely to
  lose the primary visual asset even after publishing.

## Evaluation of the current empirical claims

The current "Does it work?" section should be removed, not softened. The pilot
is useful engineering exploration, but it is not yet product evidence.

### What the pilot does establish

- The training loop ran on four model architectures without a task-specific
  code branch in the public pipeline.
- Fine-tuning produced a large increase in agreement over the same small base
  models when they were prompted zero-shot.
- Plain LoRA was at least as useful as the two tested technique arms on this
  task.
- Consumer-GPU run times can be short for some small models.

Those are legitimate internal observations. They justify further work.

### What it does not establish

- It does not establish task correctness because all targets come from the
  teacher rather than independent labels.
- It does not establish an acceptable function because every highlighted
  student missed the configured 85% point-estimate gate.
- It does not establish performance above 85%. An 18/22 result (81.8%) has an
  approximate Wilson 95% interval of 61.5% to 92.7%.
- It does not establish generalization because there is one task, one tiny
  gate, no released dataset, and no repeated training seeds.
- It does not establish model-agnosticism. Four successful experimental loads
  are compatibility observations, not support for arbitrary open-weight
  architectures.
- It does not establish cost savings because teacher-label cost, training
  electricity, engineering time, local inference cost, and call-volume
  break-even were not measured.
- It does not establish energy savings because neither local nor remote energy
  was measured. Smaller parameter count is not itself an energy result.
- It does not establish that causal-LM fine-tuning is the right method because
  no majority, rules, TF-IDF, embedding, SetFit, or encoder baseline appears.
- It does not establish a teacher "ceiling." Pairwise self-agreement between
  two stochastic draws is not generally the maximum agreement a model can
  achieve against one draw or against an underlying gold outcome.
- It does not preserve an untouched final test after comparing four models and
  several technique arms on the same 22 rows.

The phrase "Compilation added +68-77 points" is especially likely to mislead.
It compares a trained specialist with an untrained base language model, not
with the hosted teacher, a cheap prompted model, or a conventional classifier.
It shows that supervised training changed behavior, not that smallbatch is an
economically or operationally superior replacement.

### Honest replacement text for now

Until a reproducible benchmark exists, the README should say something like:

> **Project status: experimental.** The CPU-only unit suite covers the spec,
> dataset, metric, report, and orchestration helpers. The complete pipeline has
> been exercised on one exploratory task, but smallbatch has not yet been
> validated on a public benchmark or in production. A passing teacher-agreement
> gate is not the same as ground-truth accuracy.

This level of candor will attract the right early contributors and prevent an
interesting prototype from looking like an untrustworthy mature product.

## Required benchmark program

The benchmark should test the product decision, not just the current
implementation. Its purpose is to answer:

> For which tasks and data volumes does smallbatch find a local function that
> preserves acceptable real-world quality and reaches a useful cost, latency,
> memory, or energy break-even?

### Task selection

Use at least three public, redistributable tasks plus one private task used only
as a non-reproducible case study:

- a 3-10 class operational intent or ticket-routing task;
- a binary or three-way policy classifier with asymmetric error costs;
- an ordinal rating task; and
- the maintainer's real repeated workflow, anonymized in aggregate reporting.

Public possibilities include Banking77 subsets, CLINC intent subsets, public
review sentiment/rating datasets, or another dataset whose license and target
semantics are unambiguous. Avoid a benchmark made only of synthetic tickets.
The exact dataset matters less than releasing every split and build input.

### Data protocol

1. Freeze a representative input pool before generating labels.
2. Keep existing gold labels hidden from every teacher prompt.
3. Split into train, model-selection dev, and locked final test before any
   augmentation.
4. Let the teacher label only the train inputs for the primary distillation
   condition.
5. Use human/gold labels for dev and final quality selection. Also report
   teacher-to-gold agreement.
6. Run at multiple train sizes, for example 25, 50, 100, 250, and 500.
7. Repeat stochastic training with at least three seeds.
8. Run teacher relabeling on a sample to quantify label instability, without
   calling it a ceiling.
9. Never expose final-test results until all candidate and hyperparameter
   choices for that benchmark version are frozen.

For private tasks without pre-existing gold labels, have a domain owner review
a statistically meaningful locked sample and preserve the review protocol.

### Candidate ladder

Every task should compare, in order:

1. constant/majority prediction;
2. a short deterministic rule baseline where plausible;
3. TF-IDF plus logistic regression or linear SVM;
4. frozen sentence embeddings plus a linear head;
5. SetFit or a small encoder classifier;
6. the current small causal-LM LoRA path;
7. the teacher called directly;
8. a cheaper hosted or local prompted model; and
9. optionally ProgramAsWeights for task shapes it supports.

If a linear model wins, smallbatch should celebrate and package it. That is the
most credible expression of "smallest acceptable function."

### Quality metrics

For enum classifiers, report:

- exact accuracy;
- macro and weighted F1;
- per-class precision and recall;
- balanced accuracy for skewed datasets;
- confusion matrix;
- an explicit cost-weighted severe-error metric where domain errors differ;
- paired bootstrap confidence intervals; and
- calibration/Brier score if confidence is exposed.

For ordinal scores, report:

- exact accuracy and within-one agreement;
- mean absolute error;
- quadratic weighted kappa or another justified ordinal agreement metric;
- rank correlation where appropriate; and
- severe-error rate based on task-specific costs.

Always show three relationships separately:

```text
teacher vs gold
candidate vs gold
candidate vs teacher
```

The deploy gate should support a minimum final-test size and a confidence rule,
such as requiring the lower bound of the selected metric to exceed the user's
minimum. A point estimate can still be shown, but it should not silently pass a
noise-dominated test.

### Systems and economics metrics

Record for every candidate:

- teacher requests, input/output tokens, failures, retries, wall time, and
  estimated monetary cost;
- training wall time, peak RAM/VRAM, and measured local energy;
- artifact and shared-runtime sizes;
- cold-load time and steady-state memory;
- single-item p50/p95 latency on named CPU and GPU hardware;
- batch throughput at several batch sizes;
- measured local inference energy per 1,000 items; and
- any remote-energy estimate clearly labeled as an estimate.

[CodeCarbon](https://github.com/mlco2/codecarbon) is a reasonable optional
starting point for CPU/GPU/RAM energy and emissions measurement. Its own
methodology distinguishes measured counters from estimated power. Preserve
that distinction. Carbon intensity and energy are different metrics; report
kWh first and location-dependent CO2e separately.

Calculate monetary break-even explicitly:

```text
build_cost = teacher_label_cost + training_compute_cost
per_call_savings = hosted_api_cost - local_variable_cost
break_even_calls = build_cost / per_call_savings
```

Do the analogous calculation for measured energy only when both sides are
available under comparable assumptions. "No API fee" is accurate; "no
per-call cost" is not, because local electricity, hardware, and operations
still exist.

### Ablations that matter

- teacher labels versus human labels for training;
- real inputs only versus synthetic augmentation;
- current band balancing versus no balancing;
- user review versus no review;
- base versus instruct checkpoint;
- candidate backend;
- train-set size;
- one versus multiple teacher label draws; and
- spec/rubric quality improvements.

DoRA and rationale distillation should not consume more benchmark budget until
a core candidate reliably works.

### Benchmark exit criteria

A credible first public result should meet all of these:

1. released specs, data splits, scripts, raw per-run results, and immutable
   model revisions;
2. at least three task families;
3. at least three training seeds per learned candidate;
4. one locked final test per benchmark version;
5. conventional and encoder baselines;
6. quality against gold labels, not only teacher agreement;
7. latency, memory, size, cost, and energy results on named hardware;
8. a documented case where the compiler correctly chooses a non-LoRA model or
   declines to compile; and
9. reproduction by a clean machine or independent person.

Only after this should the README contain a "Does it work?" section.

## Recommended target architecture

The current module boundaries are workable, but the data model should be
recentered around semantic identity, candidate selection, and portable
artifacts.

### 1. Model decision calibration explicitly

Add first-class records for:

- `DecisionDraft`: mutable labels, rubric clauses, open questions, and seed
  cases;
- `JudgmentEvent`: one teacher or human decision tied to an exact draft hash;
- `DecisionTest`: an adjudicated anchor, boundary, or regression case with an
  explanation of the controlling rule; and
- `DecisionVersion`: an immutable promoted draft, its accepted test suite,
  known ambiguities, abstention/escalation policy, and change log.

The calibration loop should be:

```text
seed cases -> draft -> probe -> adjudicate -> revise -> regression test -> freeze
```

Probe operations can ask the teacher to apply the current draft repeatedly,
generate close boundary cases, or critique conflicting clauses. They must not
silently rewrite the rubric. Every suggested semantic change is a diff the
domain owner accepts or rejects. Every judgment remains attached to the draft
that produced it.

This state model is the necessary bridge between the user's broader use case
and the existing label/review machinery.

### 2. Separate the semantic spec from the build recipe

The current YAML mixes what the function means with teacher provider,
augmentation size, gate policy, base model, precision, LoRA parameters, and
training hyperparameters. That makes a shareable function definition noisy and
causes build changes to look like semantic drift.

Use two logical documents, even if the CLI can render them into one file:

```yaml
# function.yaml: stable and shareable
name: ticket-priority
description: Assign support triage priority.
input:
  type: object
  required: [subject, body, customer_tier]
  properties:
    subject: {type: string, maxLength: 300}
    body: {type: string, maxLength: 8000}
    customer_tier: {type: string}
output:
  type: enum
  labels:
    urgent: Customer is blocked now or damage is ongoing.
    normal: A real problem exists but impact is contained.
    low: Nothing is currently broken.
rubric:
  tie_breakers:
    - Ongoing money, data, or security impact is urgent.
quality:
  metric: macro_f1
  minimum: 0.90
  minimum_test_items: 100
```

```yaml
# build.yaml: replaceable implementation policy
teacher:
  endpoint: ollama
  model: qwen3:8b
candidates: [tfidf, setfit, causal_lora]
runtime_budget:
  max_artifact_mb: 500
  max_p95_ms: 100
  device: cpu
```

The semantic hash should cover the function contract and referenced policy
content. The build ID should cover semantic hash, dataset snapshot, candidate
recipe, model revisions, code version, and environment lock.

### 3. Make the dataset an append-only ledger

Store raw inputs, label events, reviews, split assignments, and exclusions as
versioned records rather than repeatedly rewriting one canonical row. Materialize
train/dev/test views from that ledger. This supports:

- resumable teacher work;
- multiple teachers or repeated labels;
- human corrections without losing history;
- explicit migration after a rubric change;
- reliable dataset hashes; and
- future drift/capture without inventing a second data system.

JSONL remains sufficient initially. A database is not required.

### 4. Define a candidate backend protocol

```python
class CandidateBackend(Protocol):
    def fit(self, spec, train, dev, work_dir) -> BuildCandidate: ...
    def predict(self, artifact, items) -> list: ...
    def benchmark(self, artifact, items) -> RuntimeMetrics: ...
    def package(self, artifact, out_dir) -> PackageManifest: ...
```

This is a justified abstraction because it changes the product from a wrapper
around one training method into a method selector. Keep implementations few and
opinionated. Do not create a public plugin ecosystem before two backends work.

### 5. Make evaluation independent of a backend

One evaluation layer should render distinct decision-definition,
teacher-application, and implementation reports. The implementation evaluator
should validate input/output contracts and score predictions from rules,
sklearn, SetFit, causal LM, or a hosted model identically. It should own
adjudicated/teacher metric separation, confidence intervals, cost weights,
final-test locking, and report rendering.

### 6. Build a content-addressed artifact

A completed artifact should include:

```text
artifact/
  function.yaml          # resolved semantic contract
  decision-version.json  # immutable version/test provenance, no private cases
  build.lock.json        # immutable inputs and dependency/model revisions
  manifest.json          # backend, license, quality basis, runtime requirements
  model/                 # backend-specific files
  runtime/               # generated invocation metadata, not a training env
  report.public.json     # redacted summary safe to share
  test-vectors.json      # explicitly non-sensitive smoke vectors
```

Private evaluation cases and rationales should remain in the local build work
directory, not in the distributable artifact.

### 7. Split builder and runtime packages

A likely package shape is:

```text
smallbatch                 # spec, data, evaluation, CLI core
smallbatch[train]          # Torch/Transformers/PEFT/TRL backends
smallbatch[setfit]         # encoder backend
smallbatch[export]         # conversion tooling
smallbatch-runtime         # minimal loader/validator, if a separate package helps
```

Do not finalize package names until the first two artifact backends exist, but
make "recipient does not install the builder" a design invariant.

### 8. Add a real function catalog only after artifacts are stable

The catalog should be a local manifest index, not a platform:

- function name and semantic version;
- promoted artifact alias;
- semantic and build hashes;
- backend and exact shared base revision;
- license/redistribution policy;
- quality metric and whether it is gold- or teacher-based;
- runtime size/latency requirements; and
- staleness/drift state.

For compatible causal adapters, one process can load a base once and select
adapters per function. For sklearn/ONNX functions, each artifact can remain
independent. The catalog should abstract both without pretending their runtime
shapes are identical.

## Documentation evaluation

The current documentation is readable, concrete, and better than the evidence
base. That is the central documentation problem: persuasive polish makes
unvalidated behavior sound settled.

### Claims to remove or qualify

| Current claim/theme | Problem | Safer replacement |
|---|---|---|
| "uses the big model once" | Labeling uses many batched calls, variant generation, retries, and relabeling; recompilation may repeat them. | "uses a teacher during dataset creation, then needs no teacher at runtime." |
| "no per-call cost" | Local electricity, hardware, maintenance, and cold-start costs remain. | "no hosted-model API fee at runtime." |
| "milliseconds per call" | No public measurement, hardware, input size, batch size, or percentile is given. | Publish a named benchmark or omit it. |
| "a fraction of the energy" | No energy was measured and a generative LM may lose to a classical classifier. | "designed to reduce runtime compute; measure your build's break-even." |
| "your data stays local" | Hosted teachers receive every item used for labeling. Push/export reports may expose excerpts. | "runtime can be fully local; labeling locality depends on the teacher you choose." |
| "any open-weights model" | Architecture fallbacks are narrow and only four models were explored. | "tested models: ...; other compatible causal LMs may work." |
| "runs anywhere" | Training requires CUDA; Python runtime is heavyweight; GGUF depends on model support and external tools. | Name tested operating systems, devices, runtimes, and resource numbers. |
| "ten functions don't cost ten models of RAM" | Current `load_fn` loads one base per function and merged GGUF duplicates it. | "LoRA adapter files share a cached base model on disk." |
| "right or wrong answer" | Constrained decoding prevents invalid syntax but does not establish semantic correctness. | "always returns a value inside the contract; quality is measured separately." |
| "quality gate" | The current gate is teacher imitation, not ground-truth quality. | "teacher-agreement gate" until gold evaluation ships. |
| "extraction" | Outputs cannot contain free spans or arbitrary extracted text. | "classification and bounded structured decisions." |
| "model you own" | Base licenses and teacher terms can restrict use or distribution. | "a local artifact you control, subject to its recorded model and data licenses." |
| "model-agnosticism held" | Four exploratory loads do not establish general model support. | List the exact tested matrix and versions. |

### README information architecture

The README should be a decision and activation document, not a complete
reference. Recommended order:

1. **Name and one-sentence outcome.** No banner-only positioning.
2. **Experimental status.** State what has and has not been validated.
3. **When it is useful / when it is not.** Let unsuitable users self-select.
4. **Choose a path.** "Develop a decision" for builders or "call a compiled
   function" for artifact consumers.
5. **Develop one decision.** Start with a goal and seed cases; show calibration,
   adjudication, decision tests, and semantic freeze before training.
6. **Compile the frozen version.** State exact prerequisites, representative
   data minimum, expected teacher calls/time, and candidate selection.
7. **Read the reports.** Explain decision-definition quality, teacher
   application, implementation quality, runtime budget, and honest failure.
8. **Try a prebuilt artifact.** A CPU-only, no-teacher, no-GPU path that proves
   the callable-function experience in under five minutes.
9. **How it works.** One small diagram with calibration and compilation loops.
10. **Evidence.** Only reproducible calibration studies and compilation
   benchmarks, including rejected suggestions, losing baselines, and
   uncertainty.
11. **Why not alternatives?** A compact comparison with SetFit, PAW, OpenPipe,
   DSPy, and plain API calls.
12. **Responsible use and licenses.** A short accurate summary linking to the
    detailed living document.
13. **Development and contribution.** Link rather than embedding every detail.

The current long "What you get" section should become task-oriented docs. It
front-loads reports, early stopping, GGUF, serving, sweeps, and adapters before
a user knows whether they have enough data or a suitable task.

### Proposed README opening

```markdown
# smallbatch

Turn an evolving judgment into a tested local decision function.

Start with a goal and representative cases. smallbatch helps a domain owner
refine labels, rubric clauses, tie-breakers, and boundary tests with teacher and
human feedback. Once a decision version is frozen, it compares small local
implementations and packages the smallest candidate that meets the quality and
runtime requirements. A completed function runs without a hosted-model API
call.

> Experimental: the public benchmark and artifact format are still being
> stabilized, and the rubric-development workflow has not yet been externally
> validated. Teacher agreement is not the same as a coherent decision or
> ground-truth accuracy.

Use smallbatch when a fuzzy judgment can be expressed as a bounded output, you
have representative cases, and an accountable domain owner can resolve
ambiguity. Compile only after a semantic version is stable enough for one
deployment interval. Keep refining or keep the hosted implementation when it
is not.
```

Do not mention LoRA in the first screen. It is an implementation choice, not
the value proposition.

### Two audience tracks, not one overloaded quickstart

#### Builder track: develop, freeze, and compile a decision

This is the primary data-scientist/domain-owner path. It should show:

- starting from a goal and a small set of representative cases;
- drafting label definitions and tie-breakers;
- using disagreement and generated boundary probes to find ambiguity;
- reviewing and promoting cases into permanent decision tests;
- comparing and freezing a semantic version; and
- only then running candidate compilation and final evaluation.

It should state teacher cost, minimum reviewed evidence, hardware requirements,
and the meaning of all three failure outcomes: refine, relabel, and decline.

#### Consumer track: call a finished function

This is the adoption path for application developers. It should install a
small runtime and one released example artifact, then call it from CLI and
Python with no teacher, model training, CUDA, Ollama, or llama.cpp checkout.
It proves the destination before asking the user to build.

The current ticket example is useful as a smoke fixture but not as evaluation
evidence because all 71 inputs are synthetic. Add one released benchmark
example with real public gold labels and a frozen expected report.

### Documentation set

Recommended structure:

```text
README.md                         # decision + activation
docs/getting-started-runtime.md   # call a prebuilt artifact
docs/getting-started-decision.md  # first calibration + semantic freeze
docs/getting-started-build.md     # compile a frozen decision
docs/concepts/function-spec.md
docs/concepts/decision-lifecycle.md
docs/concepts/decision-tests.md
docs/concepts/data-and-labels.md
docs/concepts/evaluation.md
docs/concepts/artifacts.md
docs/guides/reviewing-data.md
docs/guides/refining-rubrics.md
docs/guides/local-gpu.md
docs/guides/cloud-build.md
docs/guides/sharing-safely.md
docs/reference/cli.md              # generated/tested
docs/reference/python-api.md
docs/reference/spec-schema.md      # generated from Pydantic/JSON Schema
docs/reference/artifact-schema.md
docs/troubleshooting.md
docs/responsible-use.md
CONTRIBUTING.md
SECURITY.md
CHANGELOG.md
ROADMAP.md
```

`docs/how-it-works.md` currently mixes concept, schema reference, algorithm,
runtime, sharing, research lineage, and advanced guides. Splitting it will
make answers easier to find and reduce duplicated claims.

### Roadmap documentation

The current roadmap is mostly a shipped-feature checklist. Move shipped work
to `CHANGELOG.md`. A roadmap should contain user outcomes, evidence required,
and explicit non-goals, for example:

```text
Outcome: one fuzzy judgment can be calibrated into a frozen decision version,
then independently validated, compiled, and shared.
Exit criteria: external calibration study, executable decision tests, public
compilation benchmark, adjudicated gate, resumable labels, redacted bundle.
```

This keeps development from drifting back toward feature count.

### Responsible-use documentation

The existence of this document is a strength, but it should be generated or
reviewed as a time-stamped compliance matrix rather than woven into broad
marketing claims.

Current authoritative points include:

- Anthropic's March 16, 2026 help article says specialized non-competing tools
  such as sentiment and content categorization can be trained from outputs,
  while competitive/general model development is prohibited. The page also
  uses broader warning language, so the exact customer agreement and use case
  still matter: [Anthropic guidance](https://support.claude.com/en/articles/12326764-can-i-use-my-outputs-to-train-an-ai-model).
- OpenAI's services agreement defines an exception for classifiers that are
  not distributed or made commercially available to third parties:
  [OpenAI Services Agreement](https://openai.com/policies/services-agreement/).
- The default LFM license has a revenue threshold:
  [LFM license](https://huggingface.co/LiquidAI/LFM2.5-350M-Base/blob/main/LICENSE).

The artifact manifest should therefore record separate statuses for:

```text
base_model_use
teacher_output_use
internal_deployment
third_party_distribution
commercial_distribution
reviewed_at
source_urls
```

Unknown should remain unknown, not silently become allowed. `push --public`
should not be available when recorded teacher terms restrict distribution.

## Open-source adoption readiness

### Immediate release infrastructure

Before asking for users or contributors, add:

- GitHub Actions for supported Python versions, tests, lint, type checks,
  package build, wheel install, and documentation links;
- one release tag and a reproducible PyPI publish flow using trusted
  publishing;
- `CONTRIBUTING.md` with environment and test tiers;
- `SECURITY.md` covering model/artifact and data-disclosure reports;
- a changelog and compatibility policy;
- issue templates for bug, failed build, model compatibility, and benchmark
  contribution;
- dependency update automation with actual compatibility tests; and
- a clean-room smoke script that starts from a fresh virtual environment.

### Support policy

Do not imply all models and GPUs are supported. Publish a small tested table:

| Layer | Tested combinations |
|---|---|
| Python | Exact minor versions exercised in CI |
| Training | Named NVIDIA architectures, Torch/CUDA pairs, and precision |
| Candidate models | Exact repository and immutable revision |
| Runtime | CPU/GPU hardware and operating systems |
| Export | llama.cpp commit and GGUF quantizations |
| Teachers | Exact endpoint implementations and models |

Everything else can be "community-reported" or "best effort." Narrow support
is more robust than nominal universality.

### Contribution strategy

The first useful community contributions are not new backends. Ask for:

- reproducible public task specs and gold datasets;
- independent benchmark reproductions;
- tested hardware/model compatibility reports;
- failure fixtures for teacher parsing and prompt boundaries;
- privacy-safe report improvements; and
- one conventional classifier backend.

Label "add another training technique" issues as out of scope until the
benchmark says a technique gap matters.

### Project name

`smallbatch` is concise and fits the data-volume idea, but it is generic and
does not explain classifiers or local runtime. Searchability will depend on a
consistent descriptor. Keep the name for now, reserve the package while it is
available, and always pair it with "local classifier compiler." Rebranding
before proof would add work without reducing the central risk.

## Prioritized roadmap

The roadmap is sequenced by uncertainty removed, not by engineering novelty.

### Phase 0: truthful, safe alpha

Goal: a new user can install the project and cannot accidentally produce or
share a materially misrepresented artifact.

Required work:

1. publish/install the actual branch version and add CI;
2. replace the README empirical section with experimental-status language;
3. fail compilation on semantic spec/dataset mismatch;
4. fix artifact self-containment and override serialization;
5. make zero-sized required splits preflight failures;
6. make teacher labeling resumable and cost-accounted;
7. remove evaluation inputs/rationales from distributable bundles by default;
8. switch to or explicitly require a permissive default base;
9. harden function/output names, ranges, thresholds, and typed inputs; and
10. mark `export`, `serve`, `push`, and arbitrary model support experimental.

Exit criteria:

- clean virtual-environment install succeeds from the documented command;
- all CPU CI tiers pass on every supported Python version;
- an interrupted label run resumes without repeating successful paid calls;
- changing a rubric or referenced file cannot silently reuse old labels;
- a shared artifact contains no source input text by default; and
- a clean machine can run the documented smoke example.

### Phase 1: prove one classifier end to end

Goal: establish that the product can make a correct build decision.

Required work:

1. add gold datasets and separate teacher/gold metrics;
2. implement majority and TF-IDF/logistic-regression candidates;
3. add a SetFit, embedding, or small encoder candidate;
4. make model selection use dev only and final test exactly once;
5. record cost, latency, memory, size, and local energy;
6. implement the benchmark protocol above;
7. release raw reproducible results, including failures; and
8. change the report from "did LoRA pass?" to "which candidate, if any, meets
   the requirement?"

Exit criteria:

- three public task families reproduce on a clean machine;
- selected candidates are evaluated against gold labels with uncertainty;
- at least one task selects a non-generative candidate or declines to compile;
- break-even is reported rather than asserted; and
- an independent user reproduces at least one result.

### Phase 2: make artifacts genuinely shareable

Goal: a developer can consume a compiled function without becoming a model
training operator.

Required work:

1. freeze a versioned, backend-neutral artifact schema;
2. split training and runtime dependencies;
3. add artifact install/import, explicit version load, promotion, and rollback;
4. package conventional candidates with a small CPU runtime, likely ONNX or a
   similarly stable format where appropriate;
5. validate the GGUF path end to end on a pinned llama.cpp release;
6. add public redacted reports and test vectors;
7. implement a shared-base multi-adapter runtime for compatible functions; and
8. benchmark cold start, concurrency, batching, and multiple-function memory.

Exit criteria:

- a recipient calls an artifact with no training dependencies;
- two artifact backends obey the same input/output contract;
- ten compatible adapter functions demonstrably share one loaded base;
- rollback is one command/API argument; and
- packages disclose all model/data license constraints before installation.

### Phase 3: personal reliance and feedback loop

Goal: keep a small library of functions correct as real usage accumulates.

Required work:

1. opt-in capture with redaction and retention controls;
2. user corrections as append-only label events;
3. scheduled gold spot checks;
4. drift and out-of-distribution signals tied to observed failure rates;
5. one-command rebuild from newly reviewed data; and
6. catalog-level usage, cost saved, and quality history.

Exit criteria:

- at least one maintainer function handles a real recurring workload for 90
  days;
- incidents and corrections feed a reproducible new build;
- the previous artifact remains available for rollback; and
- measured savings exceed build and maintenance cost for that workload.

### Explicitly defer

- GUI development;
- general hosted inference;
- multi-GPU/distributed training;
- open-ended generation;
- a general synthetic-data framework;
- technique arms without benchmark motivation;
- a custom cloud scheduler;
- a public function marketplace; and
- free-position extraction.

## Success metrics

### North-star metric

**Validated local functions in recurring use.** Count a function only when it:

- has a locked gold evaluation;
- meets an explicit quality threshold;
- has recorded cost/runtime evidence;
- has been called in a real workflow during the last 30 days; and
- has an owner and rollback path.

Stars, downloads, and adapters trained are useful acquisition indicators, not
proof of value.

### Activation funnel

- documented install success rate;
- time to first prebuilt function call;
- time to first completed build report;
- percentage of labeling jobs resumed after interruption without duplicate
  charge;
- percentage of builds that produce a clear accept/decline decision;
- percentage of accepted artifacts called again after 7 and 30 days; and
- percentage of first-time users who complete the flow without maintainer
  intervention.

### Quality and economics

- gold metric and lower confidence bound;
- teacher-to-gold gap and candidate-to-gold gap;
- severe-error rate by class;
- cost and energy break-even calls;
- p95 latency, throughput, cold start, memory, and artifact size;
- actual hosted API calls avoided; and
- correction/rebuild rate after deployment.

### Reliability

- expensive-stage resume success;
- duplicate teacher-call rate;
- reproducible-build rate under the declared environment;
- artifact load success on supported platforms;
- raw-data disclosure incidents, target zero;
- incomplete/corrupt artifact rate, target zero; and
- CI coverage across declared support combinations.

## Recommended first issue set

In strict order:

1. Publish a truthful installable alpha and CI matrix.
2. Introduce semantic spec hash, dataset snapshot hash, and fail-closed compile.
3. Make labeling journaled, resumable, and token/cost-accounted.
4. Separate private diagnostic report from redacted distributable report.
5. Add gold labels and rename the existing result "teacher agreement."
6. Add majority and TF-IDF/logistic-regression candidates.
7. Lock dev selection and one-use final test semantics.
8. Build one reproducible public benchmark fixture.
9. Switch the default to a permissive, benchmarked model.
10. Freeze a minimal artifact schema and runtime smoke test.
11. Add an encoder/SetFit candidate.
12. Measure latency, memory, cost, and energy; add break-even to the report.
13. Simplify the README and demote experimental commands.
14. Add explicit version load, promotion, and rollback.
15. Only then implement the shared-base function library runtime.

## Final assessment

The project should continue, but its next milestone should be evidence, not
surface area. The existing implementation shows enough care to make that work
worth doing. Its most valuable traits are the constrained task boundary,
real-input orientation, local artifact goal, and willingness to record honest
failure.

The strongest version of smallbatch is not a cheaper imitation of every LLM
call and not a more configurable fine-tuning tool. It is a disciplined compiler
that sometimes returns a tiny classifier, sometimes returns an adapter, and
sometimes tells the user to keep the API call. That last outcome is essential:
the tool becomes trustworthy when its recommendation is more important than
its preferred technique.

## Audit verification and limits

### Local verification performed

- `pytest -q`: **95 passed in 0.42 seconds**.
- CLI help inspected for all 11 commands.
- Fresh `init classifier` output generated under `/tmp` and loaded by
  `doctor --no-probe`.
- The fresh three-item starter produced `3 train / 0 dev / 0 gate` and doctor
  still exited successfully with warnings.
- `uv pip check` against the existing virtual environment: all 83 packages
  compatible.
- Existing environment/package inspection: Python 3.12, smallbatch 0.1.0,
  Torch 2.6.0+cu124, Transformers 5.13.0, PEFT 0.19.1, TRL 1.7.1, and related
  libraries.
- Existing `.venv` size: 5.5 GB; installed Torch package directory: about 1.5
  GB. This is an observed development environment, not a clean minimal-install
  benchmark.
- Relative-`spec_files` artifact staleness failure reproduced in a temporary
  directory.
- Reversed ranges, duplicate labels, and newline labels confirmed accepted by
  the current schema.
- Unquoted YAML `yes`/`no` label behavior confirmed to surface as boolean type
  validation errors.
- Missing items file confirmed to produce a traceback rather than a concise
  domain error.
- PyPI `smallbatch` JSON endpoint checked directly and returned HTTP 404.
- Git worktree was clean before the evaluation document was added.

### Not verified

- No live hosted or local teacher labeling was performed.
- No GPU training run was performed.
- No model weights were downloaded.
- No adapter was loaded for real inference.
- No GGUF export, Ollama call, llama.cpp server, SkyPilot job, or Hub upload was
  run.
- Coverage percentage was not measured because `coverage` is not installed in
  the development extra.
- Ignored `.env`, private/local experiment files, and ignored datasets were not
  inspected.
- The private pilot's raw rows, run manifests, and hardware telemetry were not
  available in the tracked repository, so its numbers were evaluated only as
  documented.

These omissions are themselves relevant: the repository currently offers no
clean, public, non-sensitive fixture through which an evaluator can verify the
central label-train-evaluate-export claim end to end.
