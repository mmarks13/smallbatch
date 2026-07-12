"""Conventional classifier candidates (torch-free).

The compiler is a method selector, not a LoRA wrapper: every compile also
fits a TF-IDF + logistic-regression candidate on the same teacher labels and
scores it on the same gate. When the linear model wins, that's the artifact —
~KBs, CPU-only, no base model.

Persistence is skops (audited loading, no pickle code execution). Loading
refuses any type outside the expected sklearn pipeline before construction.
"""

from __future__ import annotations

from pathlib import Path
from typing import Any

from . import prompts
from .labeling import Row, row_output
from .spec import FunctionSpec

MODEL_FILE = "model.skops"
TFIDF_DIR = "tfidf"


def _texts(spec: FunctionSpec, rows: list[Row]) -> list[str]:
    """One canonical text per row — the same input serialization the prompt
    path uses, so both candidates read identical evidence."""
    return [prompts.render_input(r["input"], spec.input_schema) for r in rows]


def _check_class_coverage(spec: FunctionSpec, train_rows: list[Row]) -> None:
    """A data problem must read as one: logistic regression needs at least two
    observed classes per field, and silent single-class fits would be
    meaningless anyway."""
    for name, field in spec.output.fields.items():
        observed = set()
        for r in train_rows:
            out = row_output(spec, r)
            observed.add(out[name] if isinstance(out, dict) else out)
        if len(observed) < 2:
            raise ValueError(
                f"train split has a single observed class for '{name}' "
                f"({observed or '{}'}) — the tfidf candidate needs at least two; "
                "label more varied data"
            )
        missing = set(field.values()) - observed
        if missing:
            print(
                f"note: contract classes never observed in train for '{name}': "
                f"{sorted(map(str, missing))} — the tfidf candidate cannot "
                "predict them"
            )


def train_tfidf(spec: FunctionSpec, train_rows: list[Row], out_dir: Path) -> dict[str, Any]:
    """Fit one TfidfVectorizer+LogisticRegression pipeline per output field
    (int ranges are classification over the discrete values, matching the
    gate's agreement rules) and persist with skops. Returns format metadata
    for the candidate record."""
    import sklearn
    import skops
    import skops.io as sio
    from sklearn.feature_extraction.text import TfidfVectorizer
    from sklearn.linear_model import LogisticRegression
    from sklearn.pipeline import Pipeline

    _check_class_coverage(spec, train_rows)
    texts = _texts(spec, train_rows)
    pipelines: dict[str, Any] = {}
    for name in spec.output.fields:
        outs = [row_output(spec, r) for r in train_rows]
        labels = [out[name] if isinstance(out, dict) else out for out in outs]
        pipe = Pipeline(
            [
                ("tfidf", TfidfVectorizer(sublinear_tf=True, ngram_range=(1, 2))),
                ("clf", LogisticRegression(max_iter=1000)),
            ]
        )
        pipe.fit(texts, labels)
        pipelines[name] = pipe

    out_dir.mkdir(parents=True, exist_ok=True)
    sio.dump(pipelines, out_dir / MODEL_FILE)
    return {
        "format": "skops",
        "sklearn_version": sklearn.__version__,
        "skops_version": skops.__version__,
        "fields": list(spec.output.fields),
    }


def _load_pipelines(model_dir: Path) -> dict[str, Any]:
    """skops loading under an explicit trust policy: the standard sklearn
    pipeline needs no extra trusted types; anything outside that set is
    refused BEFORE construction (never blanket-trust get_untrusted_types)."""
    import skops.io as sio

    path = model_dir / MODEL_FILE
    untrusted = sio.get_untrusted_types(file=path)
    if untrusted:
        raise ValueError(
            f"refusing to load {path}: unexpected types {sorted(untrusted)} — "
            "a smallbatch tfidf artifact contains only standard sklearn "
            "pipeline components"
        )
    return sio.load(path, trusted=[])


def predict_tfidf(
    model_dir: Path, spec: FunctionSpec, items: list[dict]
) -> list[Any]:
    """Outputs aligned with `items`, in the contract's shape (bare value for
    scalar specs, {field: value} for structured). Never imports torch."""
    pipelines = _load_pipelines(model_dir)
    texts = [prompts.render_input(it, spec.input_schema) for it in items]
    if not texts:
        return []
    per_field = {name: list(pipe.predict(texts)) for name, pipe in pipelines.items()}
    if spec.output.is_scalar:
        (only,) = per_field.values()
        return [_native(v) for v in only]
    return [
        {name: _native(per_field[name][i]) for name in spec.output.fields}
        for i in range(len(texts))
    ]


def _native(v: Any) -> Any:
    """numpy scalars -> JSON-serializable Python natives."""
    return v.item() if hasattr(v, "item") else v
