# Cloud Training

Smallbatch does not provide a cloud runner. Run the same CLI inside a machine
you control, copy in the spec and decision dataset, and copy out the immutable
build and selected standalone package.

Only LoRA normally justifies a rented GPU. TF-IDF and SetFit should be tried
first because Smallbatch's purpose is to find the cheapest acceptable local
implementation, not to assume an adapted language model.

Recommended sequence:

1. Run `label` locally or import existing decisions.
2. Transfer `spec.yaml` and `data/<function>/` through an approved channel.
3. Run `doctor`, then `compile` on the training machine.
4. Review `report.md`; explicitly run `select` for one candidate or select none.
5. Transfer the generated wheel or source package, not the decision dataset.

Keep model caches on persistent storage when retrying. Build state and LoRA
checkpoints resume only when the decision, dataset, and candidate identities
match.

The CPU profile describes the machine that ran `compile` or `select`. If the
deployment CPU differs materially, run the standalone function's own benchmark
there before relying on the recorded latency and memory numbers.

## Practical notes for rented GPUs

- `cloud/skypilot.yaml` shows the shape of a launch; any orchestration
  (SkyPilot, plain SSH, a provider console) works because the artifact
  directory is the entire interface between machines.
- Marketplace GPUs (vast.ai and similar) churn minute to minute and their
  base images are minimal. Expect to pin your own Python environment rather
  than trusting the image's, and prefer `--retry-until-up`-style patience.
- The HF Hub sometimes rate-limits marketplace provider IPs. With pre-cached
  weights, set `HF_HUB_OFFLINE=1` in setup to avoid mid-run surprises.
- Always tear the instance down after copying artifacts out; a forgotten
  instance costs more than the compile.

## Self-hosted open-weights teachers

Labeling normally runs locally because it needs your teacher credentials. A
self-hosted open-weights teacher is the exception worth knowing: serve the
model with any OpenAI-compatible server (for example vLLM) on a rented GPU,
tunnel the port to your machine, and point the spec's `openai-compatible`
teacher at `http://127.0.0.1:<port>/v1`. Labeling, calibration, journals, and
decisions all stay local; only prompts transit the tunnel. Give reasoning
models generous context on the server — a truncated response returns no text
content and the teacher retries it as a failed attempt.
