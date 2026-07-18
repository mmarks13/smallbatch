"""TF-IDF candidate training and safe persistence."""

from __future__ import annotations

import json
from pathlib import Path
from typing import Any

from . import prompts
from .labeling import Row, row_output
from .spec import FunctionSpec

MODEL_FILE = "model.skops"
TFIDF_DIR = "tfidf"
DEV_DISTRIBUTIONS_FILE = "dev_distributions.local.json"


def dev_distribution_rows(
    head: dict, features, rows: list[Row], references: list, levels: list | None = None
) -> dict:
    """Per-row development class distributions, for local diagnostics only."""
    from . import ordinal

    distribution = ordinal.class_distribution(head, features)
    return {
        "levels": head["values"] if levels is None else levels,
        "decoder": head.get("decoder", "argmax"),
        "rows": [
            {
                "id": row.get("id"),
                "reference": reference,
                "distribution": [round(float(p), 4) for p in probabilities],
            }
            for row, reference, probabilities in zip(rows, references, distribution)
        ],
    }


def write_dev_distributions(out_dir: Path, per_field: dict) -> None:
    """Row-level distributions carry decision evidence, so they live in a
    `.local` file that packaging refuses to ship."""
    if per_field:
        (out_dir / DEV_DISTRIBUTIONS_FILE).write_text(
            json.dumps(per_field, indent=2, ensure_ascii=False)
        )


def _texts(spec: FunctionSpec, rows: list[Row]) -> list[str]:
    """One canonical text per row — the same input serialization the prompt
    path uses, so both candidates read identical evidence."""
    return [prompts.render_input(r["input"], spec.input_schema) for r in rows]


def _check_class_coverage(spec: FunctionSpec, train_rows: list[Row]) -> None:
    """Reject single-class candidate training and disclose missing classes."""
    for name, field in spec.output.fields.items():
        observed = set()
        for r in train_rows:
            out = row_output(spec, r)
            observed.add(out[name] if isinstance(out, dict) else out)
        if len(observed) < 2:
            raise ValueError(
                f"train split has a single observed class for '{name}' "
                f"({observed or '{}'}) — candidate training needs at least two; "
                "label more varied data"
            )
        missing = set(field.values()) - observed
        if missing:
            print(
                f"note: contract classes never observed in train for '{name}': "
                f"{sorted(map(str, missing))} — trained candidates cannot "
                "predict them"
            )


def train_tfidf(
    spec: FunctionSpec,
    train_rows: list[Row],
    out_dir: Path,
    dev_rows: list[Row] | None = None,
    decode: str = "auto",
) -> dict[str, Any]:
    """Fit a TF-IDF vectorizer plus one head per output field and persist with
    skops. Integer scales get an ordered head (`P(y > level)` per boundary) so
    the levels train as a scale, with its decode rule selected on the
    development rows; enum labels get a multinomial classifier. Returns format
    metadata for the candidate record."""
    import sklearn
    import skops
    import skops.io as sio
    from sklearn.feature_extraction.text import TfidfVectorizer
    from sklearn.linear_model import LogisticRegression

    from . import ordinal

    _check_class_coverage(spec, train_rows)
    texts = _texts(spec, train_rows)
    dev_texts = _texts(spec, dev_rows) if dev_rows else []
    models: dict[str, Any] = {}
    objectives: dict[str, str] = {}
    decoders: dict[str, str] = {}
    dev_decode_comparison: dict[str, Any] = {}
    head_diagnostics: dict[str, Any] = {}
    head_tuning: dict[str, Any] = {}
    dev_distributions: dict[str, Any] = {}
    for name, field in spec.output.fields.items():
        outs = [row_output(spec, r) for r in train_rows]
        labels = [out[name] if isinstance(out, dict) else out for out in outs]
        vectorizer = TfidfVectorizer(sublinear_tf=True, ngram_range=(1, 2))
        features = vectorizer.fit_transform(texts)
        if ordinal.applies(spec, name):
            dev_outs = [row_output(spec, r) for r in dev_rows or []]
            dev_references = [out[name] if isinstance(out, dict) else out for out in dev_outs]
            dev_features = vectorizer.transform(dev_texts) if dev_texts else None
            head, tuning = ordinal.fit_head(
                field.values(), features, labels, dev_features, dev_references
            )
            if tuning is not None:
                head_tuning[name] = tuning
            comparison = ordinal.select_decoder(head, dev_features, dev_references, decode)
            decoders[name] = head["decoder"]
            if comparison is not None:
                dev_decode_comparison[name] = comparison
            if dev_features is not None and dev_references:
                head_diagnostics[name] = ordinal.head_diagnostics(
                    head, dev_features, dev_references
                )
                dev_distributions[name] = dev_distribution_rows(
                    head, dev_features, dev_rows, dev_references
                )
            objectives[name] = "ordinal"
        else:
            head = LogisticRegression(max_iter=1000).fit(features, labels)
            objectives[name] = "multinomial"
        models[name] = {"vectorizer": vectorizer, "head": head}

    out_dir.mkdir(parents=True, exist_ok=True)
    sio.dump(models, out_dir / MODEL_FILE)
    write_dev_distributions(out_dir, dev_distributions)
    return {
        "format": "skops",
        "sklearn_version": sklearn.__version__,
        "skops_version": skops.__version__,
        "fields": list(spec.output.fields),
        "objective": objectives,
        "decode": decoders,
        "dev_decode_comparison": dev_decode_comparison,
        "head_diagnostics": head_diagnostics,
        "head_tuning": head_tuning,
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
    return predict_tfidf_pipelines(_load_pipelines(model_dir), spec, items)


def predict_tfidf_pipelines(
    pipelines: dict[str, Any], spec: FunctionSpec, items: list[dict]
) -> list[Any]:
    """Predict with already-loaded models for honest batch-one timing."""
    from . import ordinal

    texts = [prompts.render_input(it, spec.input_schema) for it in items]
    if not texts:
        return []
    per_field = {}
    for name, model in pipelines.items():
        features = model["vectorizer"].transform(texts)
        head = model["head"]
        if isinstance(head, dict) and head.get("kind") == ordinal.KIND:
            per_field[name] = ordinal.predict(head, features)
        else:
            per_field[name] = list(head.predict(features))
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
