# Teacher-output rights review — v0.2.0 benchmark

**Status: DRAFT — publication of teacher labels or trained artifacts is
blocked until the maintainer completes and dates this review.** If any item
below stays ambiguous, the v0.2.0 benchmark publishes aggregate metrics and
reproduction scripts only (no Claude-derived labels or weights).

## Facts to record (maintainer)

- Account/product used by `claude-cli`: ☐ *(e.g. Claude Pro/Max consumer,
  Team, Enterprise, API)*
- Applicable agreement(s) + URLs at review time: ☐
- Review date: ☐
- Reviewer: Michael Marks

## Guidance relied on

- Anthropic, "Can I use my Outputs to train an AI model?"
  (<https://support.claude.com/en/articles/12326764-can-i-use-my-outputs-to-train-an-ai-model>):
  specialized, **non-competing** tools — content categorization and sentiment
  are named — may be trained from Outputs; competitive/general model
  development is prohibited; broader written-permission language exists, so
  the controlling customer agreement matters.
- The benchmark's student models are bounded classifiers (an intent-routing
  subset task), squarely in the named permitted category.

## Per-item conclusions (complete before the tag)

| Published item | Conclusion | Basis |
|---|---|---|
| Aggregate metrics + scripts + frozen splits | ☐ | |
| Raw teacher labels (Sonnet 5 outputs) | ☐ | |
| Fitted TF-IDF artifact (trained on teacher labels) | ☐ | |
| LoRA adapter weights (trained on teacher labels) | ☐ | |

## Provenance recorded with the benchmark

- Teacher: `claude-cli`, model id `claude-sonnet-5`, CLI version ☐,
  invocation settings (spec `teacher:` block), prompt version, access date ☐.
- Reproducing the labels requires the reproducer's own Claude access and may
  not produce byte-identical outputs (sampling, model updates).
