# Responsible Use

Smallbatch reproduces decisions; it does not validate them.

Before compiling a function, the user remains responsible for deciding that:

- The prompt expresses an appropriate and lawful decision.
- Imported decisions or teacher behavior are acceptable for the intended use.
- Representative inputs cover the production distribution and important
  failure cases.
- The constrained output is sufficient for downstream handling.
- Human review, appeal, monitoring, and fallback paths are appropriate.

Calibration lets a user inspect repeated teacher decisions before paying for a
full labeling run. It is not a correctness, fairness, safety, or policy audit.
Smallbatch intentionally provides no gold-label subsystem or acceptance gate.

Reports measure agreement and error against supplied decisions. A highly
faithful candidate can reproduce a bad teacher, biased historical decisions,
or an ambiguous prompt. Candidate selection on the same evaluation data makes
the chosen result optimistic; v0.2 discloses this and provides no confirmation
set.

Do not use a generated function as the sole decision-maker for high-impact
medical, legal, employment, credit, housing, insurance, policing, or similar
decisions without the domain controls those uses require.

## Teacher Provider Terms

Training candidates on a teacher's outputs makes the teacher provider's terms
part of your compliance surface. Major hosted providers expressly permit
building specialized, non-competing tools (content categorization, scoring,
sentiment) on their outputs — the bounded-decision functions Smallbatch
compiles fit that category — while prohibiting the use of outputs to train
general-purpose or competing generative models, and some restrict
distributing models trained on their outputs at all.

**Text output fields change this calculus.** A function with a `type: text`
field is trained on the teacher's *generated prose*, not just its selections
from a fixed list. A narrow, length-bounded transformation (a query rewrite,
a normalized message, a one-line rationale beside a decision) is still a
specialized tool, not a general assistant — Smallbatch enforces one required
text field, a hard character limit, and no open-ended generation — but it
sits meaningfully closer to the conduct hosted-provider terms restrict, and
whether a given text function crosses a given provider's line is a judgment
Smallbatch cannot make for you.

Concretely:

- **Self-hosted open-weights teachers** (Ollama, vLLM, or any
  `openai-compatible` endpoint you operate) sidestep the question entirely
  for labeling. This is the recommended path for text functions.
- **Hosted teachers** (`claude-cli`, `codex-cli`, or a hosted
  `openai-compatible` endpoint) require you to read the provider's current
  output-use terms against your specific text field before labeling.
  `smallbatch doctor` warns — it never blocks — when a text-bearing spec is
  configured with a teacher that looks hosted.
- Check the base model's license too: the adapter you distribute remains
  subject to it regardless of the teacher.

## Data And Model Privacy

Teacher calls send prompt and input content to the configured provider. Review
provider authorization, retention, and output-use terms before labeling.

Aggregate reports omit inputs and rationales; local detail reports retain them.
Standalone functions include the prompt and trained model state. TF-IDF
vocabularies can contain source tokens, embedding models may memorize, and
LoRA adapters can retain training information. Treat trained state as derived
data, not as redacted data.

Review generated source, dependency versions, model licenses, base-model
requirements, and every file in the package before sharing it.

## Operating Claims

Smallbatch records CPU latency, memory, and footprint. These can inform cost
and resource decisions but do not constitute direct energy or emissions
measurements. Do not convert them into energy claims without a separate,
documented measurement protocol.
