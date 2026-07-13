# Local Training Hardware

TF-IDF trains on CPU. SetFit can train on CPU or GPU. LoRA training is normally
the only GPU-dependent stage; every completed candidate must subsequently run
the full evaluation split through its CPU runtime.

`precision: auto` uses BF16 on Ampere-or-newer NVIDIA GPUs and FP32 on older
cards such as Pascal. `qlora` requires the optional `bitsandbytes` dependency
and a compatible CUDA environment.

```yaml
candidates:
  granite-350m:
    type: lora
    model: ibm-granite/granite-4.0-350m
    precision: auto
```

Before a long run:

```bash
smallbatch doctor path/to/spec.yaml --items path/to/items.json
nvidia-smi
```

Candidate errors are isolated. If LoRA cannot train, completed TF-IDF and
SetFit candidates remain available for comparison and selection.

The selected LoRA package uses a Python PEFT CPU runtime in v0.2. GGUF and
llama.cpp conversion are deferred because conversion and quantization can
change behavior and therefore require a separate full evaluation.
