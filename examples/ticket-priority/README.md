# Example: support-ticket priority

A self-contained smallbatch function: classify inbound support tickets as
`urgent` / `normal` / `low`. Everything you need is in this directory —
`spec.yaml` (the function definition) and `items.json` (71 synthetic tickets).

## 0. Pick a teacher

The spec defaults to a **local Ollama server** so the whole example runs with
no API key:

```bash
ollama pull qwen3:8b        # or any capable instruct model you have
ollama serve                # if not already running
```

To use a hosted provider instead, edit the `teacher:` block in `spec.yaml`
(any `/chat/completions` endpoint works; set `base_url`, `model`, and
`api_key_env`). Make sure your use of a provider as a labeling teacher is
allowed by its terms — see [docs/responsible-use.md](../../docs/responsible-use.md).

## 1. Label

The teacher scores the 71 real tickets against the rubric, generates
band-targeted variants up to `teacher.examples` (150), and writes a
train/holdout split:

```bash
smallbatch label examples/ticket-priority/spec.yaml \
    --items examples/ticket-priority/items.json
```

Output lands in `data/ticket-priority/` (train.jsonl, holdout.jsonl, meta.json).
Expect a JSON summary with a `label_histogram`; if >50% of labels fall in one
bin you'll get a warning to sharpen the rubric.

## 2. Compile

Fine-tunes the base model (default `LiquidAI/LFM2.5-350M-Base`, ~minutes on
any CUDA GPU) and gates the result against the teacher's held-out labels:

```bash
smallbatch compile examples/ticket-priority/spec.yaml
```

Exit code 0 = gate PASS, 2 = trained fine but didn't clear the bar (an honest
FAIL), 1 = real error. The artifact (adapter + manifest) lands in
`artifacts/ticket-priority/<date>/`.

## 3. Run it

```bash
smallbatch run ticket-priority --json \
  '{"subject": "Site down", "body": "Nobody can log in since 9am and we take orders through the portal.", "product_area": "auth", "customer_tier": "pro"}'
# -> urgent
```

Or from Python:

```python
from smallbatch import load_fn
priority = load_fn("ticket-priority")
priority({"subject": "...", "body": "...", "product_area": "...", "customer_tier": "..."})
```

## 4. (Optional) Export it

Once the gate passes, turn the function into a single CPU-runnable file — a
quantized GGUF plus an Ollama Modelfile and a llama.cpp grammar that makes
invalid outputs impossible (needs a llama.cpp checkout and `pip install
gguf`; see [docs](../../docs/how-it-works.md#exporting-to-a-zero-pytorch-runtime)):

```bash
smallbatch export ticket-priority --llama-cpp ~/llama.cpp
ollama create ticket-priority -f artifacts/ticket-priority/*/export/Modelfile
```

The same flow works programmatically end to end:

```python
import json, smallbatch

items = json.load(open("examples/ticket-priority/items.json"))
smallbatch.label("examples/ticket-priority/spec.yaml", items)
result = smallbatch.compile("examples/ticket-priority/spec.yaml")
print(result.passed, result.metrics["adapter"])
```
