"""Opt-in real-GPU tier.

Nothing here is collected unless SMALLBATCH_GPU_TESTS=1, so the default
CPU-only `pytest -q` is unaffected. When the tier IS requested, a missing or
broken CUDA device fails loudly instead of skipping: this tier backs required
checks, and a silently green run on a machine with no GPU would be a lie.

Run it:

    SMALLBATCH_GPU_TESTS=1 pytest tests_gpu -m gpu_smoke -q    # PR gate
    SMALLBATCH_GPU_TESTS=1 pytest tests_gpu -q                 # release gate
"""

from __future__ import annotations

import os

import pytest

if os.environ.get("SMALLBATCH_GPU_TESTS") != "1":
    collect_ignore_glob = ["*"]


@pytest.fixture(scope="session", autouse=True)
def require_cuda():
    import torch

    if not torch.cuda.is_available():
        pytest.fail(
            "SMALLBATCH_GPU_TESTS=1 but torch.cuda.is_available() is False — "
            "wrong torch build for this GPU (Blackwell needs a cu128+ wheel) "
            "or a driver problem. A skipped GPU tier must never look green."
        )
    yield
