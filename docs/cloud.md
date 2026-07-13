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
