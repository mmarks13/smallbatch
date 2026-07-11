# Responsible use

smallbatch trains small models on labels produced by another model (the
"teacher"). Whether that is allowed depends on **your agreement with the
teacher's provider**, and it is your responsibility to check before labeling.
This page summarizes the landscape as of mid-2026; it is not legal advice, and
provider terms change — read the current documents yourself.

## The short version

- smallbatch functions are **narrow, constrained classifiers/scorers**: the
  output type is an integer in a fixed range or one label from a fixed list.
  The library cannot produce a chatbot or an open-ended text generator, and
  the trained adapter's only contract is score-in-range or label-from-list.
- Major providers generally distinguish between using outputs to build
  **competing, general-purpose models** (prohibited) and building
  **non-competing specialized tools** (often expressly allowed).
- Self-hosted and open-license teachers (e.g. a local model served by Ollama
  or vLLM) avoid the question entirely for the labeling step, subject to the
  model's own license.

## Anthropic

Anthropic's [Usage Policy](https://www.anthropic.com/legal/aup) prohibits
using inputs and outputs to train an AI model ("model scraping" or "model
distillation") **without prior authorization**. Anthropic's Help Center
article [“Can I use my Outputs to train an AI
model?”](https://support.claude.com/en/articles/12326764-can-i-use-my-outputs-to-train-an-ai-model)
clarifies what that means in practice:

- **Allowed** (non-competing specialized tools): sentiment analysis tools,
  content categorization systems, summarization, information extraction,
  semantic search, anomaly detection.
- **Prohibited**: general-purpose chatbots, models designed for open-ended
  text generation, training competitive models, reverse-engineering training
  methods.

A smallbatch function — a fixed-rubric scorer or categorizer — is shaped like
the allowed examples, but the judgment about *your* use case is yours to make
against the current policy, and asking Anthropic for authorization is the
unambiguous path.

Note on the `claude-cli` teacher backend: it drives Claude Code's supported
headless mode (`claude -p`). Using it does not change any of the above — the
training-use question depends on what you build, not how you call the model.

## OpenAI (and OpenAI-compatible hosted providers)

OpenAI's [Services Agreement](https://openai.com/policies/services-agreement/)
prohibits using output to develop competing AI models, with a defined
exception for models "primarily intended to categorize, classify, or organize
data" that are **not distributed or commercially made available to third
parties**, and for fine-tuning within OpenAI's own services. If you plan to
distribute or sell an adapter labeled with OpenAI outputs, read that clause
carefully first.

Other hosted providers reachable through the `openai-compatible` backend
(Gemini, etc.) have their own terms — check them.

Note on the `codex-cli` teacher backend: it drives the Codex CLI's supported
non-interactive mode (`codex exec`) using the account authenticated by
`codex login`. The implementation removes an inherited `OPENAI_API_KEY` so a
parent shell cannot silently change which account is used. As with
`claude-cli`, the training-use question depends on what you build, not how the
teacher is invoked.

## Practical guidance

1. **Prefer a teacher you unambiguously may use**: a self-hosted open-weights
   model, a provider that permits your use case in writing, or explicit
   authorization from the provider.
2. **Keep functions narrow.** That's also where small students actually match
   their teachers — see [how-it-works.md](how-it-works.md).
3. **Mind distribution.** Training an internal tool and publishing/selling an
   adapter are different acts under most terms.
4. Every labeled row records its teacher model and backend in provenance
   (`data/<fn>/meta.json`, artifact manifests), so you can always answer
   "what produced this training data?"
