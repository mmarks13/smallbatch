# Roadmap

Roadmap items must directly improve defining or obtaining a constrained
decision, building and comparing CPU candidates, understanding their tradeoffs,
explicitly selecting one or none, or packaging and running the selected local
function. They are not promises and are not part of the v0.2 public contract.

## Runtime Portability

- **GGUF/llama.cpp optimization after selection.** Convert a selected LoRA,
  then rerun the complete behavior and CPU profile before it can replace the
  Python PEFT package.
- **ONNX packages.** Evaluate TF-IDF and SetFit conversion as derived packages,
  with explicit package-vs-candidate drift evidence.
- **Smaller runtime dependencies.** Split training dependencies from runtime
  libraries only after standalone package compatibility is stable.
- **Deployment adapters.** Optional MLflow, BentoML, container, and Hub exports
  around the canonical standalone function package.

## Stronger Evidence

- Independent confirmation data after candidate selection.
- Bootstrap uncertainty for non-proportion metrics.
- Repeated cross-machine package evaluation and numerical-drift reporting.
- Direct energy measurement only with a defensible hardware protocol.
- Scheduled production-distribution and decision-drift checks.

## Candidate Development

- Evidence-driven SetFit default/model recommendations across more task types.
- Validated augmentation policies and diagnostics.
- Controlled candidate sweeps that do not imply automatic winner selection.
- Shared-base multi-adapter storage without misrepresenting runtime footprint.

## Deliberately Deferred Product Surface

- HTTP serving, Hub publishing, generated containers, and cloud orchestration.
- Open-ended generation or unconstrained outputs.
- Gold-label management, policy validation, or general annotation workflows.
