"""Prompt-first function specifications for smallbatch."""

from __future__ import annotations

import hashlib
import json
import re
from pathlib import Path
from typing import Annotated, Any, Literal

import yaml
from pydantic import (
    BaseModel,
    Field,
    PrivateAttr,
    ValidationError,
    field_validator,
    model_validator,
)

_ID_RE = re.compile(r"[a-z][a-z0-9]*(?:-[a-z0-9]+)*\Z")
_FIELD_RE = re.compile(r"[A-Za-z_][A-Za-z0-9_]*\Z")


def validate_id(value: str, what: str) -> str:
    if not isinstance(value, str) or len(value) > 80 or not _ID_RE.fullmatch(value):
        raise ValueError(
            f"{what} {value!r} must be lowercase kebab-case, start with a letter, "
            "and contain at most 80 characters"
        )
    return value


InputType = Literal["string", "integer", "number", "boolean"]


class FieldSpec(BaseModel):
    """One constrained output field: a bounded integer or an enum."""

    range: tuple[int, int] | None = None
    labels: list[str] | None = None
    model_config = {"extra": "forbid"}

    @model_validator(mode="before")
    @classmethod
    def _hint_yaml_booleans(cls, data):
        if isinstance(data, dict) and any(
            isinstance(value, bool) for value in (data.get("labels") or [])
        ):
            raise ValueError(
                "labels contain YAML booleans; quote yes/no/true/false/on/off labels"
            )
        return data

    @model_validator(mode="after")
    def _check(self) -> FieldSpec:
        if (self.range is None) == (self.labels is None):
            raise ValueError("an output field needs exactly one of `range` or `labels`")
        if self.range is not None and self.range[0] > self.range[1]:
            raise ValueError(f"range {list(self.range)} is reversed")
        if self.range is not None and not (0 <= self.range[0] and self.range[1] <= 9):
            raise ValueError(
                f"integer range {list(self.range)} must lie within 0-9: a level must be "
                "one token so the decision is one choice the student can be trained and "
                "scored on as an ordered scale. Rescale the decision (a 0-10 scale "
                "becomes 0-9), or use `labels` if the values are unordered"
            )
        if self.labels is not None:
            if not self.labels:
                raise ValueError("`labels` must be non-empty")
            seen: dict[str, str] = {}
            for label in self.labels:
                if not label.strip() or "\n" in label or "\r" in label:
                    raise ValueError(f"invalid output label {label!r}")
                key = label.strip().casefold()
                if key in seen:
                    raise ValueError(f"output labels {seen[key]!r} and {label!r} collide")
                seen[key] = label
        return self

    @property
    def type(self) -> str:
        return "int" if self.range is not None else "enum"

    def values(self) -> list[Any]:
        if self.range is not None:
            return list(range(self.range[0], self.range[1] + 1))
        return list(self.labels or [])


SCALAR_FIELD = "score"
_RESERVED_OUTPUT_KEYS = {"type", "range", "labels"}


class OutputSpec(BaseModel):
    fields: dict[str, FieldSpec]
    model_config = {"extra": "forbid"}

    @model_validator(mode="before")
    @classmethod
    def _normalize(cls, data):
        if not isinstance(data, dict) or "fields" in data:
            return data
        if "type" in data:
            kind = data.get("type")
            if kind not in ("int", "enum"):
                raise ValueError("output.type must be `int` or `enum`")
            allowed = {key: data[key] for key in ("range", "labels") if key in data}
            return {"fields": {SCALAR_FIELD: allowed}}
        if not data:
            raise ValueError("output needs at least one field")
        if _RESERVED_OUTPUT_KEYS & set(data):
            raise ValueError("structured output uses a reserved field name")
        return {"fields": data}

    @field_validator("fields")
    @classmethod
    def _field_names(cls, fields: dict[str, FieldSpec]) -> dict[str, FieldSpec]:
        for name in fields:
            if not _FIELD_RE.fullmatch(name):
                raise ValueError(f"output field {name!r} must be a Python-style identifier")
        return fields

    @property
    def is_scalar(self) -> bool:
        return list(self.fields) == [SCALAR_FIELD]

    @property
    def scalar(self) -> FieldSpec:
        if not self.is_scalar:
            raise ValueError("structured output has no scalar field")
        return self.fields[SCALAR_FIELD]

    @property
    def type(self) -> str:
        return self.scalar.type if self.is_scalar else "object"

    @property
    def range(self) -> tuple[int, int] | None:
        return self.scalar.range if self.is_scalar else None

    @property
    def labels(self) -> list[str] | None:
        return self.scalar.labels if self.is_scalar else None


class TeacherSpec(BaseModel):
    backend: Literal["claude-cli", "codex-cli", "openai-compatible"]
    model: str
    batch_size: int = 40
    reasoning_effort: Literal["minimal", "low", "medium", "high"] | None = None
    base_url: str | None = None
    api_key_env: str = "OPENAI_API_KEY"
    model_config = {"extra": "forbid"}

    @field_validator("batch_size")
    @classmethod
    def _positive_batch(cls, value: int) -> int:
        if value < 1:
            raise ValueError("teacher.batch_size must be positive")
        return value


class TfidfCandidateSpec(BaseModel):
    type: Literal["tfidf"]
    model_config = {"extra": "forbid"}


class SetFitCandidateSpec(BaseModel):
    type: Literal["setfit"]
    model: str
    embedding_samples_per_class: int = Field(default=8, ge=1)
    training_args: dict[str, Any] = Field(default_factory=dict)
    model_config = {"extra": "forbid"}

    @field_validator("training_args")
    @classmethod
    def _safe_args(cls, args: dict[str, Any]) -> dict[str, Any]:
        reserved = {
            "output_dir",
            "logging_dir",
            "report_to",
            "run_name",
            "loss",
            "distance_metric",
            "save_strategy",
            "save_steps",
            "save_total_limit",
            "load_best_model_at_end",
        }
        bad = reserved & set(args)
        if bad:
            raise ValueError(
                f"SetFit arguments {sorted(bad)} are managed by smallbatch or require code"
            )
        try:
            json.dumps(args)
        except TypeError as exc:
            raise ValueError("SetFit training_args must be JSON/YAML values") from exc
        return args


class LoraCandidateSpec(BaseModel):
    type: Literal["lora"]
    model: str = "ibm-granite/granite-4.0-350m"
    precision: Literal["auto", "fp32", "bf16", "qlora"] = "auto"
    lora_r: int = 16
    lora_alpha: int | None = None
    lora_dropout: float = 0.05
    use_dora: bool = False
    rationale_distillation: bool = False
    # ordinal: score the legal completions and optimize class NLL + RPS so an
    # integer scale trains as ordered levels rather than unrelated symbols.
    # auto uses it for scalar integer decisions without rationale distillation.
    objective: Literal["auto", "token", "ordinal"] = "auto"
    # which point of the level distribution to report: the mode (argmax, the
    # level constrained generation would emit), the median (the first level
    # whose cumulative probability reaches one half, which trades exact hits
    # for smaller misses), or within_one (the level whose ±1 window holds the
    # most mass). auto lets the development split decide.
    decode: Literal["auto", "argmax", "median", "within_one"] = "auto"
    # recompute activations in the backward pass instead of holding them: buys
    # a large amount of GPU memory for roughly a third more compute, which is
    # what lets a bigger student train on a small card
    gradient_checkpointing: bool = False
    max_epochs: int = 12
    patience: int | None = 2
    min_delta: float = 0.0
    learning_rate: float = 2e-4
    batch_size: int = 8
    eval_batch_size: int = 16
    max_seq_len: int = 1024
    seed: int = 17
    loss_type: Literal["nll", "chunked_nll"] | None = None
    model_config = {"extra": "forbid"}

    @model_validator(mode="after")
    def _bounds(self) -> LoraCandidateSpec:
        positive = {
            "lora_r": self.lora_r,
            "max_epochs": self.max_epochs,
            "learning_rate": self.learning_rate,
            "batch_size": self.batch_size,
            "eval_batch_size": self.eval_batch_size,
            "max_seq_len": self.max_seq_len,
        }
        if self.patience is not None:
            positive["patience"] = self.patience
        if self.lora_alpha is not None:
            positive["lora_alpha"] = self.lora_alpha
        for name, value in positive.items():
            if value <= 0:
                raise ValueError(f"{name} must be positive")
        if not 0 <= self.lora_dropout < 1 or self.min_delta < 0:
            raise ValueError("invalid LoRA dropout or min_delta")
        return self

    @property
    def alpha(self) -> int:
        return self.lora_alpha if self.lora_alpha is not None else 2 * self.lora_r


CandidateSpec = Annotated[
    TfidfCandidateSpec | SetFitCandidateSpec | LoraCandidateSpec,
    Field(discriminator="type"),
]


class ParaphraseSpec(BaseModel):
    cap: int = 50
    model_config = {"extra": "forbid"}


class FieldDropoutSpec(BaseModel):
    fields: list[str]
    cap: int = 30
    model_config = {"extra": "forbid"}


class CounterfactualSpec(BaseModel):
    cap: int = 30
    model_config = {"extra": "forbid"}


class AugmentationSpec(BaseModel):
    paraphrase: ParaphraseSpec | None = None
    field_dropout: FieldDropoutSpec | None = None
    counterfactual: CounterfactualSpec | None = None
    model_config = {"extra": "forbid"}


class FunctionSpec(BaseModel):
    name: str
    description: str = ""
    input_schema: dict[str, InputType]
    output: OutputSpec
    prompt: str
    teacher: TeacherSpec | None = None
    candidates: dict[str, CandidateSpec]
    augmentation: AugmentationSpec | None = None

    _base_dir: Path = PrivateAttr(default=Path("."))
    _source_path: Path | None = PrivateAttr(default=None)
    model_config = {"extra": "forbid"}

    @model_validator(mode="after")
    def _validate_contract(self) -> FunctionSpec:
        validate_id(self.name, "function name")
        if not self.prompt.strip():
            raise ValueError("prompt must not be blank")
        if not self.input_schema:
            raise ValueError("input_schema must not be empty")
        for name in self.input_schema:
            if not _FIELD_RE.fullmatch(name):
                raise ValueError(f"input field {name!r} must be a Python-style identifier")
        if not self.candidates:
            raise ValueError("configure at least one candidate")
        for name in self.candidates:
            validate_id(name, "candidate id")
        if self.augmentation and self.augmentation.field_dropout:
            unknown = set(self.augmentation.field_dropout.fields) - set(self.input_schema)
            if unknown:
                raise ValueError(f"field_dropout names unknown fields: {sorted(unknown)}")
        return self

    def decision_hash(self) -> str:
        from . import prompts

        payload = {
            "input_schema": self.input_schema,
            "output": self.output.model_dump(mode="json"),
            "prompt": self.prompt,
            "teacher": self.teacher.model_dump(mode="json") if self.teacher else None,
            "augmentation": (
                self.augmentation.model_dump(mode="json") if self.augmentation else None
            ),
            "prompt_version": prompts.PROMPT_VERSION,
            "split": {"train": 0.7, "dev": 0.1, "eval": 0.2, "seed": 17},
        }
        return _hash(payload)

    def build_hash(self) -> str:
        return _hash(self.model_dump(mode="json"))


def _hash(value: Any) -> str:
    return hashlib.sha256(
        json.dumps(value, sort_keys=True, separators=(",", ":")).encode()
    ).hexdigest()


def validate_input(spec: FunctionSpec, item: Any) -> dict[str, Any]:
    """Validate and normalize one public input object."""
    if not isinstance(item, dict):
        raise ValueError("input must be a JSON object")
    missing = [name for name in spec.input_schema if name not in item]
    extra = [name for name in item if name not in spec.input_schema]
    if missing or extra:
        parts = []
        if missing:
            parts.append(f"missing fields {missing}")
        if extra:
            parts.append(f"unexpected fields {extra}")
        raise ValueError("input " + "; ".join(parts))
    for name, kind in spec.input_schema.items():
        value = item[name]
        valid = {
            "string": isinstance(value, str),
            "integer": isinstance(value, int) and not isinstance(value, bool),
            "number": isinstance(value, (int, float)) and not isinstance(value, bool),
            "boolean": isinstance(value, bool),
        }[kind]
        if not valid:
            raise ValueError(f"input field {name!r} must be {kind}, got {type(value).__name__}")
    return {name: item[name] for name in spec.input_schema}


def validate_output(spec: FunctionSpec, value: Any) -> Any:
    """Normalize one constrained output or raise a useful contract error."""
    if spec.output.is_scalar:
        return _validate_field(spec.output.scalar, value, SCALAR_FIELD)
    if not isinstance(value, dict):
        raise ValueError("structured output must be an object")
    missing = [name for name in spec.output.fields if name not in value]
    extra = [name for name in value if name not in spec.output.fields]
    if missing or extra:
        raise ValueError(f"structured output mismatch: missing={missing}, extra={extra}")
    return {
        name: _validate_field(field, value[name], name)
        for name, field in spec.output.fields.items()
    }


def _validate_field(field: FieldSpec, value: Any, name: str) -> Any:
    if field.type == "int":
        if not isinstance(value, int) or isinstance(value, bool):
            raise ValueError(f"output field {name!r} must be an integer")
    elif not isinstance(value, str):
        raise ValueError(f"output field {name!r} must be a string label")
    if value not in field.values():
        raise ValueError(f"output field {name!r} value {value!r} is outside the contract")
    return value


_MIGRATION_HELP = """\
v0.2 uses a prompt-first spec. Rename `rubric` to `prompt`, move each model
under `candidates: <id>:`, and remove `gate`, top-level `train`, and `spec_files`.
Run `smallbatch init` to scaffold a spec in the new format.
"""


def load_spec(path: str | Path) -> FunctionSpec:
    path = Path(path)
    raw = yaml.safe_load(path.read_text())
    if isinstance(raw, dict) and ({"rubric", "gate", "train", "spec_files"} & set(raw)):
        raise ValueError(f"{path}: legacy pre-v0.2 spec\n\n{_MIGRATION_HELP}")
    try:
        spec = FunctionSpec(**raw)
    except ValidationError as exc:
        raise ValueError(f"{path}: {exc}") from exc
    spec._base_dir = path.parent.resolve()
    spec._source_path = path.resolve()
    return spec
