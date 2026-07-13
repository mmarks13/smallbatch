# Case Study Rights And Publication Review

**Status: incomplete. Teacher decisions and trained artifacts must not be
published until this review is completed and dated.**

## Review Sources

Automated preflight located these current official sources on 2026-07-13. This
list is not maintainer approval and does not complete any checkbox below.

CFPB:

- <https://www.consumerfinance.gov/data-research/consumer-complaints/>
- <https://www.consumerfinance.gov/complaint/data-use/>
- <https://cfpb.github.io/ccdb5-api/documentation/>
- <https://github.com/cfpb/ccdb5-api/blob/main/LICENSE>
- <https://files.consumerfinance.gov/f/documents/cfpb_narrative-scrubbing-standard_2023-05.pdf>

OpenAI teacher access:

- <https://openai.com/policies/terms-of-use/>
- <https://openai.com/policies/services-agreement/>
- <https://openai.com/policies/service-terms/>
- <https://openai.com/policies/sharing-publication-policy/>
- <https://openai.com/policies/usage-policies/>

The reviewer must identify which agreement governs the exact `codex-cli`
account used for labeling. In particular, review programmatic output access,
using output to train another model or classifier, publication of derived
artifacts, and publication of aggregate evidence. Do not infer permission from
output ownership language alone.

## Public Source Review

- CFPB source and API URLs reviewed: [ ]
- API/data license recorded at review time: [ ]
- Narrative consent and scrubbing documentation reviewed: [ ]
- Frozen complaint IDs rechecked for continued public availability: [ ]
- Review date and reviewer: [ ]

## Teacher Review

- Account/product used by `codex-cli`: [ ]
- Applicable agreement and policy URLs: [ ]
- Model ID, CLI version, reasoning effort, and access date: [ ]
- Permission to publish raw teacher decisions: allowed / prohibited / unclear
- Permission to publish TF-IDF state trained on decisions: allowed / prohibited / unclear
- Permission to publish SetFit state trained on decisions: allowed / prohibited / unclear
- Permission to publish LoRA adapter trained on decisions: allowed / prohibited / unclear

Any `unclear` result means aggregate metrics and reproduction scripts only.

## Required Disclosures

- The priority scale is constructed and is not CFPB policy.
- Narratives are not verified and are not representative of all consumers.
- Decision agreement is not correctness, fairness, or legal validation.
- CPU latency, memory, and footprint are not energy measurements.
