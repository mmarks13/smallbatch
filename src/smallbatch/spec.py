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

# Length-bounded text fields count Unicode code points in the decoded value —
# not bytes, tokens, JSON escapes, or the surrounding quotes.
TEXT_MAX_CHARS_DEFAULT = 300
TEXT_MAX_CHARS_LIMIT = 2000


class FieldSpec(BaseModel):
    """One declared output field: a bounded integer, an enum, or one
    length-bounded text value (declared as `type: text`)."""

    range: tuple[int, int] | None = None
    labels: list[str] | None = None
    max_chars: int | None = None
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

    @model_validator(mode="before")
    @classmethod
    def _text_shorthand(cls, data):
        """`{type: text, max_chars: N}` declares a text field; the stored
        discriminator is `max_chars`, which text fields always resolve."""
        if not isinstance(data, dict) or "type" not in data:
            return data
        kind = data.get("type")
        if kind != "text":
            raise ValueError(
                f"output field type {kind!r} is not declared with `type`: integer "
                "scales use `range`, enums use `labels`; only `type: text` exists"
            )
        translated = {key: value for key, value in data.items() if key != "type"}
        translated.setdefault("max_chars", TEXT_MAX_CHARS_DEFAULT)
        return translated

    @model_validator(mode="after")
    def _check(self) -> FieldSpec:
        declared = [
            name
            for name, value in (
                ("range", self.range),
                ("labels", self.labels),
                ("max_chars", self.max_chars),
            )
            if value is not None
        ]
        if len(declared) != 1:
            raise ValueError(
                "an output field needs exactly one of `range`, `labels`, or "
                "`type: text` (got " + (", ".join(declared) or "none") + ")"
            )
        if self.max_chars is not None and not (
            1 <= self.max_chars <= TEXT_MAX_CHARS_LIMIT
        ):
            raise ValueError(
                f"max_chars {self.max_chars} must lie within 1-{TEXT_MAX_CHARS_LIMIT} "
                "Unicode code points"
            )
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
        if self.range is not None:
            return "int"
        return "enum" if self.labels is not None else "text"

    def values(self) -> list[Any]:
        if self.range is not None:
            return list(range(self.range[0], self.range[1] + 1))
        if self.labels is not None:
            return list(self.labels)
        raise ValueError("a text field has no enumerable values")


SCALAR_FIELD = "score"
_RESERVED_OUTPUT_KEYS = {"type", "range", "labels", "max_chars"}


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
            if kind not in ("int", "enum", "text"):
                raise ValueError("output.type must be `int`, `enum`, or `text`")
            allowed = {
                key: data[key]
                for key in ("range", "labels", "max_chars")
                if key in data
            }
            if kind == "text":
                allowed["type"] = "text"
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
            if name.startswith("__"):
                raise ValueError(
                    f"output field {name!r} is reserved (dunder names collide "
                    "with internal loss bookkeeping)"
                )
        return fields

    @model_validator(mode="after")
    def _at_most_one_text(self) -> OutputSpec:
        text_fields = [name for name, field in self.fields.items() if field.type == "text"]
        if len(text_fields) > 1:
            raise ValueError(
                f"output declares {len(text_fields)} text fields ({text_fields}); "
                "a function may contain at most one"
            )
        return self

    @property
    def has_text(self) -> bool:
        return any(field.type == "text" for field in self.fields.values())

    @property
    def text_field(self) -> str | None:
        """Name of the single text field, or None."""
        for name, field in self.fields.items():
            if field.type == "text":
                return name
        return None

    @property
    def bounded_fields(self) -> dict[str, FieldSpec]:
        return {name: f for name, f in self.fields.items() if f.type != "text"}

    @property
    def is_text_only(self) -> bool:
        return self.has_text and not self.bounded_fields

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
    # decision-noise measurement. 2 labels every item twice (the second pass
    # with shuffled batch composition), tie-breaks flips with one targeted
    # third draw, and records the teacher's self-agreement stability; items with
    # three distinct categorical answers are set aside for optional user
    # resolution. 1 is a single draw with stability unknown.
    passes: Literal[1, 2] = 1
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


# Options deleted in the v0.3 schema break, each with the correction. They
# fail loudly instead of being ignored so an old spec cannot silently train
# under different semantics.
_REMOVED_CANDIDATE_OPTIONS = {
    "decode": (
        "`decode` was removed in v0.3: ordinal fields always decode argmax, "
        "the level constrained generation would emit. Delete the key."
    ),
    "objective": (
        "`objective` was removed in v0.3: each output field's declared type "
        "sets its training objective (int: class NLL + ranked probability "
        "score; enum: legal-value NLL; text: mean next-token NLL). Delete "
        "the key."
    ),
    "rationale_distillation": (
        "`rationale_distillation` was removed in v0.3: declare an ordinary "
        "text output field instead, e.g. `rationale: {type: text}` before "
        "the decision field. Delete the key."
    ),
    "loss_type": (
        "`loss_type` was removed in v0.3: the per-field objective replaces "
        "TRL token-loss variants. Delete the key."
    ),
}


def _reject_removed_options(cls, data):
    if isinstance(data, dict):
        for key, message in _REMOVED_CANDIDATE_OPTIONS.items():
            if key in data:
                raise ValueError(message)
    return data


class TfidfCandidateSpec(BaseModel):
    type: Literal["tfidf"]
    # ordered scales only: the softmax head's capacity. auto lets the
    # development split choose between a linear layer and one hidden layer;
    # pin it only to remove that search deliberately.
    head: Literal["auto", "linear", "mlp"] = "auto"
    model_config = {"extra": "forbid"}

    _removed = model_validator(mode="before")(classmethod(_reject_removed_options))


class SetFitCandidateSpec(BaseModel):
    type: Literal["setfit"]
    model: str
    # None (the default) fine-tunes the embedding on every training row.
    # SetFit's own 8-per-class few-shot recipe is the wrong regime for a tool
    # whose labeling pipeline produces thousands of decisions; set a number
    # only to deliberately restrict the contrastive phase.
    embedding_samples_per_class: int | None = Field(default=None, ge=1)
    training_args: dict[str, Any] = Field(default_factory=dict)
    # ordered scales only: the softmax head's capacity. auto lets the
    # development split choose between a linear layer and one hidden layer;
    # pin it only to remove that search deliberately.
    head: Literal["auto", "linear", "mlp"] = "auto"
    model_config = {"extra": "forbid"}

    _removed = model_validator(mode="before")(classmethod(_reject_removed_options))

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
    model_config = {"extra": "forbid"}

    _removed = model_validator(mode="before")(classmethod(_reject_removed_options))

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


# Default relative loss weights, per field type. A text field beside bounded
# fields defaults low so its many tokens support the decision instead of
# dominating it; alone it is the whole objective.
DEFAULT_BOUNDED_WEIGHT = 1.0
DEFAULT_TEXT_BESIDE_BOUNDED_WEIGHT = 0.25


class TrainingSpec(BaseModel):
    """Function-level training configuration shared by all LoRA candidates.

    `loss_weights` are the user's relative task priorities across output
    fields. Partial overrides are allowed: unspecified fields keep their
    defaults (bounded 1.0; text 0.25 beside bounded fields, 1.0 alone).
    Weights never vary by candidate and never include the internal JSON
    format objective.
    """

    loss_weights: dict[str, float] = Field(default_factory=dict)
    model_config = {"extra": "forbid"}

    @field_validator("loss_weights", mode="before")
    @classmethod
    def _finite_positive(cls, weights):
        import math

        if not isinstance(weights, dict):
            raise ValueError("loss_weights must map output field names to numbers")
        for name, value in weights.items():
            if (
                isinstance(value, bool)
                or not isinstance(value, (int, float))
                or not math.isfinite(value)
                or value <= 0
            ):
                raise ValueError(
                    f"loss_weights[{name!r}] must be a finite number greater than "
                    f"zero, got {value!r}"
                )
        return weights


def effective_loss_weights(spec: FunctionSpec) -> dict[str, float]:
    """Resolved per-field loss weights: defaults filled, overrides applied.

    These are recorded in build evidence as training intent; they are not a
    ranking formula across candidates.
    """
    overrides = spec.training.loss_weights if spec.training else {}
    weights: dict[str, float] = {}
    for name, field in spec.output.fields.items():
        if name in overrides:
            weights[name] = float(overrides[name])
        elif field.type != "text" or spec.output.is_text_only:
            weights[name] = DEFAULT_BOUNDED_WEIGHT
        else:
            weights[name] = DEFAULT_TEXT_BESIDE_BOUNDED_WEIGHT
    return weights


class FunctionSpec(BaseModel):
    name: str
    description: str = ""
    input_schema: dict[str, InputType]
    output: OutputSpec
    prompt: str
    teacher: TeacherSpec | None = None
    candidates: dict[str, CandidateSpec]
    augmentation: AugmentationSpec | None = None
    training: TrainingSpec | None = None

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
        if self.output.has_text:
            generative = {
                name: config
                for name, config in self.candidates.items()
                if config.type != "lora"
            }
            if generative:
                raise ValueError(
                    f"candidates {sorted(generative)} cannot produce the declared "
                    f"text field {self.output.text_field!r}: classifiers select "
                    "from fixed values, but this output contract requires "
                    "generation. Remove the text field or keep only `type: lora` "
                    "candidates"
                )
            if self.augmentation is not None:
                raise ValueError(
                    "augmentation is not supported for functions with a text "
                    "field: synthetic variants would need generated text targets "
                    "smallbatch cannot validate. Remove the `augmentation` block"
                )
            if self.output.is_text_only and self.teacher and self.teacher.passes == 2:
                raise ValueError(
                    "teacher.passes: 2 measures self-agreement on bounded "
                    "decisions; a text-only function has none (two harmless "
                    "rewordings are not a disagreement). Use passes: 1"
                )
        if self.training and self.training.loss_weights:
            unknown = set(self.training.loss_weights) - set(self.output.fields)
            if unknown:
                raise ValueError(
                    f"loss_weights name unknown output fields {sorted(unknown)}; "
                    f"declared fields are {list(self.output.fields)}"
                )
            if len(self.output.fields) == 1:
                raise ValueError(
                    "loss_weights have no effect on a single-field output "
                    "(one weight always normalizes to itself); delete the block"
                )
        return self

    def decision_hash(self) -> str:
        from . import prompts

        payload = {
            "input_schema": self.input_schema,
            "output": self.output.model_dump(mode="json"),
            "prompt": self.prompt,
            # `passes` is a measurement protocol, not part of the decision's
            # identity: the same approved teacher answers the same prompt, we
            # just draw more than once. Excluding it keeps existing datasets
            # and journals valid when a user turns measurement on or off.
            "teacher": (
                self.teacher.model_dump(mode="json", exclude={"passes"})
                if self.teacher
                else None
            ),
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


# C0 controls other than newline and tab, plus NUL and DEL, are unsafe in a
# text value. Carriage returns are rejected rather than normalized: silently
# rewriting \r\n would be repair, which smallbatch never does to outputs.
_UNSAFE_TEXT_CHARS = re.compile(r"[\x00-\x08\x0b-\x1f\x7f]")


def _validate_text(field: FieldSpec, value: Any, name: str) -> str:
    if not isinstance(value, str):
        raise ValueError(f"output field {name!r} must be a string")
    stripped = value.strip()
    if not stripped:
        raise ValueError(
            f"output field {name!r} must contain at least one non-whitespace "
            "character; empty text is not a valid result"
        )
    bad = _UNSAFE_TEXT_CHARS.search(value)
    if bad:
        raise ValueError(
            f"output field {name!r} contains unsafe control character "
            f"{bad.group()!r}; only newline and tab are allowed"
        )
    if len(value) > field.max_chars:
        raise ValueError(
            f"output field {name!r} is {len(value)} characters; the contract "
            f"allows at most {field.max_chars}. Smallbatch never truncates: "
            "shorten the value or raise max_chars"
        )
    return value


def _validate_field(field: FieldSpec, value: Any, name: str) -> Any:
    if field.type == "text":
        return _validate_text(field, value, name)
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
