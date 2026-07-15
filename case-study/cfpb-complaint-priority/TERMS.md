# Case Study Rights And Publication Review

**Status: reviewed and signed by the maintainer (final line).**

The teacher is the open-weights `openai/gpt-oss-120b` (Apache 2.0),
self-hosted behind an OpenAI-compatible endpoint on a rented single GPU. No
hosted-model provider holds rights over the teacher's outputs, so the
provider-terms questions that applied to hosted teachers (output-derived
training, artifact publication) do not arise. The case-study protocol still
publishes aggregate evidence only.

## Review Sources

Reviewed 2026-07-13:

CFPB:

- <https://www.consumerfinance.gov/data-research/consumer-complaints/>
- <https://www.consumerfinance.gov/complaint/data-use/>
- <https://cfpb.github.io/ccdb5-api/documentation/>
- <https://github.com/cfpb/ccdb5-api/blob/main/LICENSE>
- <https://files.consumerfinance.gov/f/documents/cfpb_narrative-scrubbing-standard_2023-05.pdf>

Teacher (self-hosted open weights):

- <https://huggingface.co/openai/gpt-oss-120b> (model card; license: Apache 2.0)
- <https://www.apache.org/licenses/LICENSE-2.0>

Infrastructure:

- <https://vast.ai/terms> (GPU rental terms; the host provides compute only
  and asserts no rights over workloads or outputs)

## Public Source Review

- CFPB source and API URLs reviewed: [x] 2026-07-13 (automated preflight;
  official API endpoint confirmed current in the OpenAPI document)
- API/data license recorded at review time: [x] `CC0` as reported by the API
  and repository license file
- Narrative consent and scrubbing documentation reviewed: [x] narratives are
  opt-in and scrubbed per the CFPB scrubbing standard; scrubbing is not
  infallible and the freeze was additionally pattern-scanned for PII
- Frozen complaint IDs rechecked for continued public availability: [x]
  2026-07-13 (all 600 fetched from the live official API at freeze time)
- Review date and reviewer: [x] Michael Marks, 2026-07-14 (see final line)

## Teacher Review

- Model and hosting: `openai/gpt-oss-120b`, Apache 2.0 open weights,
  self-hosted with vLLM on a rented single H100 (vast.ai marketplace).
- Applicable terms: Apache 2.0 (weights); no separate output-use policy
  restricts training on the model's outputs. Infrastructure rental terms
  grant the host no rights over the workload.
- Data path note: labeling prompts (public CC0 complaint text only) transit
  the rented marketplace host through an SSH tunnel. No private data is sent.
  Decisions and journals are written only on the maintainer's machine.
- Model ID, serving stack, and access date: recorded in `protocol.json` at
  run time.
- Permission to publish raw teacher decisions: allowed by license; withheld
  by case-study protocol (aggregates only)
- Permission to publish TF-IDF state trained on decisions: allowed
- Permission to publish SetFit state trained on decisions: allowed
- Permission to publish LoRA adapter trained on decisions: allowed

The case-study protocol still publishes only frozen inputs, protocol, and
aggregate evidence regardless of the permissions above.

## Required Disclosures

- The priority scale is constructed and is not CFPB policy.
- Narratives are not verified and are not representative of all consumers.
- Decision agreement is not correctness, fairness, or legal validation.
- CPU latency, memory, and footprint are not energy measurements.

Reviewed by Michael Marks, 2026-07-14
