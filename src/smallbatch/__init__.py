"""Distill prompt-driven LLM decisions into tested local CPU functions."""

from .api import CompileResult, LabelResult, SelectionResult, label, select
from .api import compile as compile  # noqa: A004 - deliberate product verb
from .runtime import load_fn
from .spec import load_spec

__version__ = "0.2.0"
__all__ = [
    "label",
    "compile",
    "select",
    "load_fn",
    "load_spec",
    "LabelResult",
    "CompileResult",
    "SelectionResult",
    "__version__",
]
