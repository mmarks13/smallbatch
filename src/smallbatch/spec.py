"""Function specs: the YAML definition of a fuzzy function."""

from __future__ import annotations

import hashlib
import json
import re
from pathlib import Path
from typing import Literal, Optional

import yaml
from pydantic import BaseModel, Field, ValidationError, model_validator

# function/sweep/arm names and run tags become filesystem path components
# (data/<name>, artifacts/<name>/<sweep>/<tag>), so they must be conservative
# slugs: ASCII letter/digit start, then letters/digits/._-, no "..", <= 80.
_SLUG_RE = re.compile(r"[A-Za-z0-9][A-Za-z0-9._-]{0,79}\Z")


def validate_slug(value: str, what: str) -> str:
    """Reject names that could escape or mangle the artifact/data layout:
    path separators, '..', control characters, non-ASCII, leading dots or
    dashes, and empty/overlong names."""
    if not isinstance(value, str) or not _SLUG_RE.fullmatch(value) or ".." in value:
        raise ValueError(
            f"{what} {value!r} must be a filesystem-safe slug "
            "(ASCII letters/digits/._-, starting with a letter or digit, "
            "no '..', at most 80 chars)"
        )
    return value


class FieldSpec(BaseModel):
    """One constrained output field: an int range or an enum label set
    (a controlled reason code is just an enum). Type is inferred from which
    constraint is present — no free-text fields."""

    range: Optional[tuple[int, int]] = None
    labels: Optional[list[str]] = None

    model_config = {"extra": "forbid"}

    @model_validator(mode="before")
    @classmethod
    def _hint_yaml_booleans(cls, data):
        # unquoted yes/no/true/false/on/off in YAML parse as booleans, which
        # would otherwise surface as an opaque string-type pydantic error
        if isinstance(data, dict) and any(
            isinstance(v, bool) for v in (data.get("labels") or [])
        ):
            raise ValueError(
                "labels contain YAML booleans — unquoted yes/no/true/false/on/off "
                'parse as booleans; quote them: labels: ["yes", "no"]'
            )
        return data

    @model_validator(mode="after")
    def _check(self) -> "FieldSpec":
        if (self.range is None) == (self.labels is None):
            raise ValueError("an output field needs exactly one of `range` or `labels`")
        if self.range is not None:
            lo, hi = self.range
            if lo > hi:
                raise ValueError(f"range [{lo}, {hi}] is reversed — no legal values")
        if self.labels is not None:
            if not self.labels:
                raise ValueError("`labels` must be non-empty")
            seen: dict[str, str] = {}
            for lb in self.labels:
                if "\n" in lb or "\r" in lb:
                    raise ValueError(f"label {lb!r} contains a newline")
                if not lb.strip():
                    raise ValueError("labels must be non-blank")
                key = lb.strip().casefold()
                if key in seen:
                    raise ValueError(
                        f"labels {seen[key]!r} and {lb!r} collide "
                        "(duplicate after trimming/case-folding — parsing is "
                        "case-insensitive)"
                    )
                seen[key] = lb
        return self

    @property
    def type(self) -> str:
        return "int" if self.range is not None else "enum"

    def values(self) -> list:
        if self.range is not None:
            lo, hi = self.range
            return list(range(lo, hi + 1))
        return list(self.labels)


# key names that mean "legacy scalar form", and are therefore unusable as
# output field names
_RESERVED_OUTPUT_KEYS = {"type", "range", "labels"}

# the implicit field name a legacy scalar output contract maps to
SCALAR_FIELD = "score"


class OutputSpec(BaseModel):
    """The output contract: an ordered map of field name -> FieldSpec.

    Two YAML spellings normalize here:

        output:                       output:
          priority:                     type: int      # legacy scalar form ==
            labels: [high, low]         range: [0, 10] # single field "score"
          confidence:
            range: [1, 5]
    """

    fields: dict[str, FieldSpec]

    model_config = {"extra": "forbid"}

    @model_validator(mode="before")
    @classmethod
    def _normalize(cls, data):
        if not isinstance(data, dict) or "fields" in data:
            return data
        if "type" in data:  # legacy scalar form
            t = data.get("type")
            if t == "int" and data.get("range") is None:
                raise ValueError("output.type=int requires output.range")
            if t == "enum" and not data.get("labels"):
                raise ValueError("output.type=enum requires output.labels")
            if t not in ("int", "enum"):
                raise ValueError(f"output.type must be int or enum, got {t!r}")
            field = {k: v for k, v in data.items() if k in ("range", "labels") and v}
            return {"fields": {SCALAR_FIELD: field}}
        bad = _RESERVED_OUTPUT_KEYS & set(data)
        if bad:
            raise ValueError(
                f"output field name(s) {sorted(bad)} are reserved; "
                "rename the field or use the scalar form with `type:`"
            )
        if not data:
            raise ValueError("output needs at least one field")
        for name in data:
            if not re.fullmatch(r"[A-Za-z_][A-Za-z0-9_-]*", str(name)):
                raise ValueError(f"output field name {name!r} must be a simple identifier")
        return {"fields": data}

    @property
    def is_scalar(self) -> bool:
        """Single-field contract emitting/storing a bare value (legacy form)."""
        return list(self.fields) == [SCALAR_FIELD]

    @property
    def scalar(self) -> FieldSpec:
        (field,) = self.fields.values()
        return field

    # -- legacy accessors: much of the single-output path (and old tests/specs)
    # speaks spec.output.type/range/labels; keep them meaningful there.
    @property
    def type(self) -> str:
        return self.scalar.type if self.is_scalar else "object"

    @property
    def range(self) -> Optional[tuple[int, int]]:
        return self.scalar.range if self.is_scalar else None

    @property
    def labels(self) -> Optional[list[str]]:
        return self.scalar.labels if self.is_scalar else None


class TeacherSpec(BaseModel):
    # no defaults: picking a labeling provider (and being authorized to use it)
    # is the user's explicit choice — see docs/responsible-use.md
    backend: Literal["claude-cli", "codex-cli", "openai-compatible"]
    model: str
    examples: int = 600
    # gate/dev split sizes: a float < 1 is a fraction of the real rows, an
    # int is an absolute row count
    holdout: float | int = 0.15
    dev: float | int = 0.1
    batch_size: int = 40
    # self-consistency probe: after labeling, re-send this many real rows
    # (stratified, input fields shuffled) and report how often the teacher
    # agrees with itself — the ceiling on any student's agreement. 0 = off.
    consistency: int = 0
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
        if self.consistency < 0:
            raise ValueError("teacher.consistency must be >= 0")
        if self.batch_size < 1:
            raise ValueError("teacher.batch_size must be >= 1")
        if self.examples < 0:
            raise ValueError("teacher.examples must be >= 0")
        return self


class GateSpec(BaseModel):
    # for int outputs: fraction of the gate split within +/-1 of the teacher
    # label; for enum outputs: exact-match fraction. `agreement` is the
    # preferred name; `agreement_pm1` is kept as the legacy alias.
    agreement: Optional[float] = None
    agreement_pm1: float = 0.85
    must_beat_zeroshot: bool = True
    # the model must also beat the BEST constant predictor on the gate labels
    # (±1 rule for int fields, majority class for enums) — on concentrated
    # labels a model that regressed to the prior can otherwise PASS
    must_beat_constant: bool = True
    # structured outputs: per-field threshold overrides, e.g. {reason: 0.7}
    fields: dict[str, float] = Field(default_factory=dict)
    # int outputs: |pred - reference| >= severe_delta counts as a severe miss
    # in reports and decision tables
    severe_delta: int = 3

    @model_validator(mode="after")
    def _check_bounds(self) -> "GateSpec":
        for label, v in [
            ("gate.agreement", self.agreement),
            ("gate.agreement_pm1", self.agreement_pm1),
            *((f"gate.fields.{k}", t) for k, t in self.fields.items()),
        ]:
            if v is not None and not 0 <= v <= 1:
                raise ValueError(f"{label} must be a fraction in [0, 1], got {v}")
        if self.severe_delta < 1:
            raise ValueError("gate.severe_delta must be >= 1")
        return self

    @property
    def threshold(self) -> float:
        return self.agreement if self.agreement is not None else self.agreement_pm1

    def field_threshold(self, name: str) -> float:
        return self.fields.get(name, self.threshold)


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

    @model_validator(mode="after")
    def _check_bounds(self) -> "TrainSpec":
        positive = {
            "lora_r": self.lora_r,
            "max_epochs": self.max_epochs,
            "learning_rate": self.learning_rate,
            "batch_size": self.batch_size,
            "eval_batch_size": self.eval_batch_size,
            "max_seq_len": self.max_seq_len,
        }
        if self.epochs is not None:
            positive["epochs"] = self.epochs
        if self.patience is not None:
            positive["patience"] = self.patience
        if self.lora_alpha is not None:
            positive["lora_alpha"] = self.lora_alpha
        for name, v in positive.items():
            if v <= 0:
                raise ValueError(f"train.{name} must be positive, got {v}")
        if not 0 <= self.lora_dropout < 1:
            raise ValueError(f"train.lora_dropout must be in [0, 1), got {self.lora_dropout}")
        if self.min_delta < 0:
            raise ValueError(f"train.min_delta must be >= 0, got {self.min_delta}")
        return self

    @property
    def alpha(self) -> int:
        return self.lora_alpha if self.lora_alpha is not None else 2 * self.lora_r


class ParaphraseSpec(BaseModel):
    cap: int = 50  # max paraphrase variants generated per label run
    model_config = {"extra": "forbid"}


class FieldDropoutSpec(BaseModel):
    # ablation probes: copies of train reals with one input field blanked,
    # RELABELED by the teacher (if the field mattered, the label honestly
    # moves; if not, the pair teaches invariance — either way it breaks
    # "field present -> memorized label" shortcuts)
    fields: list[str]
    cap: int = 30  # per field
    model_config = {"extra": "forbid"}


class CounterfactualSpec(BaseModel):
    # minimal label-moving edits of train reals, targeted at thin label
    # bands and independently relabeled — the highest-information examples
    # per teacher call (they trace the decision boundary)
    cap: int = 30
    model_config = {"extra": "forbid"}


class AugmentSpec(BaseModel):
    """Data-augmentation plan. When this block is present it fully replaces
    the legacy behavior (variants toward teacher.examples): only the kinds
    listed here run."""

    paraphrase: Optional[ParaphraseSpec] = None
    field_dropout: Optional[FieldDropoutSpec] = None
    counterfactual: Optional[CounterfactualSpec] = None
    model_config = {"extra": "forbid"}


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
    augment: Optional[AugmentSpec] = None

    # set by load_spec so spec_files resolve relative to the YAML's directory
    _base_dir: Path = Path(".")
    # set by load_spec; compile() copies the original YAML into the artifact
    _source_path: Optional[Path] = None

    model_config = {"extra": "forbid"}

    @model_validator(mode="after")
    def _check_name(self) -> "FunctionSpec":
        validate_slug(self.name, "function name")
        return self

    @model_validator(mode="after")
    def _check_augment_fields(self) -> "FunctionSpec":
        if self.augment and self.augment.field_dropout:
            unknown = [
                f for f in self.augment.field_dropout.fields
                if f not in self.input_schema
            ]
            if unknown:
                raise ValueError(
                    f"augment.field_dropout names unknown input fields: {unknown}"
                )
        return self

    @model_validator(mode="after")
    def _check_rationale_name(self) -> "FunctionSpec":
        # multi-field completions prefix the teacher rationale as `rationale:`
        if (
            self.train.rationale_distillation
            and not self.output.is_scalar
            and "rationale" in self.output.fields
        ):
            raise ValueError(
                "an output field named 'rationale' clashes with "
                "train.rationale_distillation — rename the field"
            )
        return self

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
        """Build identity: hash of the full resolved spec plus the contents of
        every spec_file. Covers training/gate/build settings too, so it changes
        on any spec edit — use `labeling_hash()` to ask the narrower question
        "are existing labels still valid for this spec?".
        """
        h = hashlib.sha256()
        h.update(json.dumps(self.model_dump(mode="json"), sort_keys=True).encode())
        for p in self.resolved_spec_files():
            h.update(p.read_bytes())
        return h.hexdigest()

    def labeling_hash(self) -> str:
        """Labeling identity: hash of everything that can change generated
        dataset rows or their meaning — the input/output contract, description,
        rubric, referenced-file contents (not paths), teacher identity, prompt
        version, and split/augment recipe.

        Deliberately EXCLUDES train hyperparameters, base model, precision, and
        gate thresholds: changing those must not invalidate a labeled dataset.
        Compile fails closed when a dataset's recorded labeling_hash differs
        from the current spec's.
        """
        from . import prompts  # lazy: prompts imports this module

        payload = {
            "description": self.description,
            "input_schema": self.input_schema,
            "output": self.output.model_dump(mode="json"),
            "rubric": self.rubric,
            "teacher": {
                "backend": self.teacher.backend,
                "model": self.teacher.model,
                "base_url": self.teacher.base_url,
            },
            "split": {"holdout": self.teacher.holdout, "dev": self.teacher.dev},
            "augment": self.augment.model_dump(mode="json") if self.augment else None,
            "prompt_version": prompts.PROMPT_VERSION,
        }
        h = hashlib.sha256()
        h.update(json.dumps(payload, sort_keys=True).encode())
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
