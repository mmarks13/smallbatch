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
