"""Precision auto-selection. Pascal (sm_61, e.g. GTX 1080 Ti) has no bf16 and
crippled fp16 compute, so small models train in fp32 there; Ampere+ gets bf16."""

from __future__ import annotations


def pick_precision(requested: str) -> str:
    if requested != "auto":
        return requested
    import torch

    if not torch.cuda.is_available():
        return "fp32"
    major, _ = torch.cuda.get_device_capability()
    return "bf16" if major >= 8 else "fp32"
