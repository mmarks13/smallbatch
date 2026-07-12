"""smallbatch — compile a frontier model's ability on one narrow task into a
small local model (spec -> teacher-labeled data -> LoRA adapter -> local fn)."""

from .api import CompileResult, LabelResult, label
from .api import compile as compile  # noqa: A004 - smallbatch.compile IS the product verb
from .export import ExportResult, export
from .hub import push
from .runtime import load_fn
from .spec import load_spec

__version__ = "0.2.0"
__all__ = [
    "label",
    "compile",
    "export",
    "push",
    "load_fn",
    "load_spec",
    "LabelResult",
    "CompileResult",
    "ExportResult",
    "__version__",
]
