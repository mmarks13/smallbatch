# Running compiles on your own GPU

Labeling is CPU-and-network only; **compiling** (train + eval) needs a CUDA
GPU. This page covers picking precision, fitting models in VRAM, and the extra
care old cards need. No GPU at all? See [cloud.md](cloud.md).

## Precision: what `auto` picks and why

| Your GPU | `auto` resolves to | Notes |
|---|---|---|
| Ampere or newer (RTX 30xx+, A10/A100, L4, ...) | `bf16` | The happy path. |
| Pre-Ampere (Pascal/Turing, GTX 10xx/16xx, RTX 20xx) | `fp32` | No (or poor) bf16 support; fp16 needs fragile loss scaling and is crippled on Pascal. fp32 is fine for sub-2B students. |
| Any card, base model 3–8B | set `precision: qlora` | 4-bit NF4 frozen base + fp16 LoRA; fits ~8B on 11–12GB. Requires `pip install smallbatch[qlora]`. |

Rough VRAM budget: the default 350M base (or anything up to ~1B) in fp32 +
LoRA optimizer state trains comfortably in ~11GB; 1–2B in bf16 needs a few
GB; 3–9B needs `qlora` on consumer cards. A few-hundred-example compile of a
sub-2B student is minutes of GPU time.

## Old GPUs (Pascal / sm_61 and friends)

Aging cards work — the library auto-selects fp32 and forces qlora's
`bnb_4bit_compute_dtype` to fp32 on pre-Ampere — but the *environment* needs
pinning:

- **torch:** recent CUDA wheels drop Pascal kernels entirely (torch built for
  cu13x reports `cuda=False` on these cards). Install a `>=2.4,<2.7` build
  with CUDA 12.4-era wheels:

  ```bash
  pip install "torch==2.6.*" --index-url https://download.pytorch.org/whl/cu124
  ```

- **bitsandbytes (qlora only):** needs `>=0.46.1` — transformers' quantizer
  rejects older builds. qlora works on sm_61 but is slow (no tensor cores).
- **No flash-attention, no Unsloth** on pre-Ampere — neither is needed;
  vanilla PEFT is the path.
- **Power-limit aging cards under sustained training load.** Long fine-tunes
  are steadier load than gaming; older cards near their power limit can
  brown-out off the PCIe bus (kernel log: `Xid 79 — GPU has fallen off the
  bus`, unrecoverable without a power cycle). Capping helps, e.g.:

  ```bash
  sudo nvidia-smi -pl 180        # cap at 180W; pick ~70% of the card's default
  ```

Blackwell-era cards (RTX 50xx) sit at the other end: they need `torch>=2.7`
with cu128+ wheels. This is why `pyproject.toml` doesn't pin an upper bound —
match the torch build to your card.

## Fitting bigger models: knobs that matter

- `precision: qlora` — the main lever for 3–8B bases.
- `batch_size` / `eval_batch_size` — first thing to lower on OOM.
- `loss_type: nll` — TRL's default `chunked_nll` loss casts the LM head to
  fp32 (~3.8GB for a 150k-vocab model) and OOMs 12GB cards; `nll` materializes
  plain logits instead, which is smaller for huge-vocab models.
- Models with huge vocabularies also skip PEFT's fp32 embedding upcast
  automatically under qlora (another multi-GB saving; handled in
  `training.py`).

## Sanity-checking your setup

```bash
python -c "import torch; print(torch.__version__, torch.cuda.is_available(), torch.cuda.get_device_capability())"
pytest -q          # CPU-only unit suite; no GPU or network needed
```

Then run the [ticket-priority example](../examples/ticket-priority/README.md)
end to end — label (needs a teacher endpoint), compile (needs the GPU), run.
