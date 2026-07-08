"""Function specs: the YAML definition of a fuzzy function."""

from __future__ import annotations

import hashlib
import json
from pathlib import Path
from typing import Literal, Optional

import yaml
from pydantic import BaseModel, Field, ValidationError, model_validator


class OutputSpec(BaseModel):
    type: Literal["int", "enum"]
    range: Optional[tuple[int, int]] = None
    labels: Optional[list[str]] = None

    @model_validator(mode="after")
    def _check(self) -> "OutputSpec":
        if self.type == "int" and self.range is None:
            raise ValueError("output.type=int requires output.range")
        if self.type == "enum" and not self.labels:
            raise ValueError("output.type=enum requires output.labels")
        return self


class TeacherSpec(BaseModel):
    # no defaults: picking a labeling provider (and being authorized to use it)
    # is the user's explicit choice — see docs/responsible-use.md
    backend: Literal["claude-cli", "openai-compatible"]
    model: str
    examples: int = 600
    # gate/dev split sizes: a float < 1 is a fraction of the real rows, an
    # int is an absolute row count
    holdout: float | int = 0.15
    dev: float | int = 0.1
    batch_size: int = 40
    # openai-compatible only:
    base_url: Optional[str] = None
    api_key_env: str = "OPENAI_API_KEY"

    @model_validator(mode="after")
    def _check_splits(self) -> "TeacherSpec":
        for name in ("holdout", "dev"):
            v = getattr(self, name)
            if isinstance(v, float) and not 0 <= v < 1:
                raise ValueError(f"teacher.{name} fraction must be in [0, 1)")
            if isinstance(v, int) and v < 0:
                raise ValueError(f"teacher.{name} count must be >= 0")
        return self


class GateSpec(BaseModel):
    # for int outputs: fraction of the gate split within +/-1 of the teacher
    # label; for enum outputs: exact-match fraction. `agreement` is the
    # preferred name; `agreement_pm1` is kept as the legacy alias.
    agreement: Optional[float] = None
    agreement_pm1: float = 0.85
    must_beat_zeroshot: bool = True

    @property
    def threshold(self) -> float:
        return self.agreement if self.agreement is not None else self.agreement_pm1


class TrainSpec(BaseModel):
    # note: an adapter inherits its base model's license. LFM2.5 ships under
    # the LFM Open License (commercial use conditioned above $10M revenue) —
    # swap the base if that matters for you.
    base: str = "LiquidAI/LFM2.5-350M-Base"
    precision: Literal["auto", "fp32", "bf16", "qlora"] = "auto"
    lora_r: int = 16
    lora_alpha: Optional[int] = None  # defaults to 2*r
    lora_dropout: float = 0.05
    use_dora: bool = False
    rationale_distillation: bool = False
    # training runs to max_epochs unless dev agreement stops improving for
    # `patience` epochs (patience: null disables early stopping). `epochs` is
    # the legacy alias for max_epochs.
    max_epochs: int = 12
    epochs: Optional[int] = None
    patience: Optional[int] = 2
    min_delta: float = 0.0
    learning_rate: float = 2e-4
    batch_size: int = 8
    eval_batch_size: int = 16
    max_seq_len: int = 1024
    seed: int = 17
    # TRL's default "chunked_nll" loss casts the lm_head weight to fp32
    # (~3.8GB for Qwen's 151k vocab) and OOMs 12GB cards; "nll" materializes
    # plain logits instead, which is smaller for huge-vocab models
    loss_type: Optional[Literal["nll", "chunked_nll"]] = None

    # reject unknown keys so a typo'd sweep override (e.g. bathc_size) fails
    # loudly at spec-load time instead of being silently ignored
    model_config = {"extra": "forbid"}

    @model_validator(mode="after")
    def _epochs_alias(self) -> "TrainSpec":
        if self.epochs is not None and "max_epochs" not in self.model_fields_set:
            self.max_epochs = self.epochs
        return self

    @property
    def alpha(self) -> int:
        return self.lora_alpha if self.lora_alpha is not None else 2 * self.lora_r


class FunctionSpec(BaseModel):
    name: str
    description: str
    input_schema: dict[str, str]  # field name -> type hint (informational)
    output: OutputSpec
    rubric: str
    spec_files: list[str] = Field(default_factory=list)
    teacher: TeacherSpec
    gate: GateSpec = Field(default_factory=GateSpec)
    train: TrainSpec = Field(default_factory=TrainSpec)

    # set by load_spec so spec_files resolve relative to the YAML's directory
    _base_dir: Path = Path(".")
    # set by load_spec; compile() copies the original YAML into the artifact
    _source_path: Optional[Path] = None

    model_config = {"extra": "forbid"}

    def resolved_spec_files(self) -> list[Path]:
        out = []
        for f in self.spec_files:
            p = Path(f).expanduser()
            if not p.is_absolute():
                p = self._base_dir / p
            out.append(p)
        return out

    def spec_files_text(self) -> str:
        """Contents of all spec_files, for embedding in teacher prompts."""
        parts = []
        for p in self.resolved_spec_files():
            parts.append(f"--- {p.name} ---\n{p.read_text()}")
        return "\n\n".join(parts)

    def spec_hash(self) -> str:
        """Hash of the spec plus the contents of every spec_file.

        A compiled artifact records this; a mismatch later means the adapter
        is stale (the spec or a referenced file changed).
        """
        h = hashlib.sha256()
        h.update(json.dumps(self.model_dump(mode="json"), sort_keys=True).encode())
        for p in self.resolved_spec_files():
            h.update(p.read_bytes())
        return h.hexdigest()


_TEACHER_HELP = """\
every spec needs an explicit `teacher` block naming the labeling provider
you are authorized to use (see docs/responsible-use.md). Examples:

  teacher:                      # any /chat/completions endpoint
    backend: openai-compatible
    model: qwen3:8b             # e.g. a local Ollama model — no API key
    base_url: http://localhost:11434/v1

  teacher:                      # logged-in Claude Code CLI
    backend: claude-cli
    model: sonnet
"""


def load_spec(path: str | Path) -> FunctionSpec:
    path = Path(path)
    try:
        spec = FunctionSpec(**yaml.safe_load(path.read_text()))
    except ValidationError as e:
        if any(err["loc"][:1] == ("teacher",) for err in e.errors()):
            raise ValueError(f"{path}: {e}\n\n{_TEACHER_HELP}") from e
        raise
    spec._base_dir = path.parent.resolve()
    spec._source_path = path.resolve()
    return spec
