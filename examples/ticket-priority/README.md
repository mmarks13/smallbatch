# Ticket Priority Example

This example compiles a prompt-driven `urgent | normal | low` ticket decision.
It defaults to a local OpenAI-compatible Ollama teacher; edit the teacher block
for another provider you are authorized to use.

```bash
ollama serve
ollama pull qwen3:8b

smallbatch doctor examples/ticket-priority/spec.yaml \
  --items examples/ticket-priority/items.json

smallbatch label examples/ticket-priority/spec.yaml \
  --items examples/ticket-priority/items.json

smallbatch compile examples/ticket-priority/spec.yaml
smallbatch status
```

`label` first shows ten inputs decided twice. Approve, review another ten, or
decline and revise the prompt/teacher. Calibration is behavior inspection, not
correctness validation.

Review the generated `report.md`. It compares TF-IDF, SetFit, and LoRA where
available but does not choose one.

```bash
smallbatch select ticket-priority tfidf
smallbatch run ticket-priority \
  --json '{"subject":"API down","body":"All calls return 503","product_area":"api","customer_tier":"enterprise"}'
```

Selection produces an installable standalone wheel and prints its path and
import statement. The wheel does not depend on Smallbatch.
