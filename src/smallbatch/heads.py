"""The shared softmax ordinal head for constrained integer scales.

Every candidate family expresses an ordinal decision the same way: a softmax
distribution over the scale's levels, trained with cross-entropy plus the
ranked probability score (RPS) so distant misses cost more than adjacent ones,
and read out by argmax. LoRA gets the distribution
from the language model's own logits at the deciding token; SetFit and TF-IDF
get it from this head — a linear layer, or one hidden layer when the
development split says the extra capacity earns its keep.

`ordinal_loss` is the single definition of that training objective; the LoRA
trainer imports it rather than restating it.

Torch is a training-time dependency only. A persisted head is plain numpy
arrays in plain containers — skops-loadable under a strict trust policy, and
generated packages run it with a few lines of numpy, so the TF-IDF wheel
stays free of torch.
"""

from __future__ import annotations

from typing import Any

KIND = "ordinal-softmax"

# The default first: ties resolve to it, and without development rows it is
# the head. The linear entry keeps its own weight decay; the hidden entries
# share the decay that won on every encoder in the head bake-off, and search
# width and dropout, the two knobs whose best value moved between datasets.
HEAD_GRID = (
    {"hidden": 0, "dropout": 0.0, "weight_decay": 1e-4},
    {"hidden": 64, "dropout": 0.3, "weight_decay": 1e-3},
    {"hidden": 64, "dropout": 0.5, "weight_decay": 1e-3},
    {"hidden": 256, "dropout": 0.3, "weight_decay": 1e-3},
    {"hidden": 256, "dropout": 0.5, "weight_decay": 1e-3},
)
# Single-seed head training swung eval agreement by several points in the
# bake-off, so the seed is part of the development selection, not a constant.
SEEDS = (0, 1, 2)

MAX_EPOCHS = 400
NO_DEV_EPOCHS = 200
PATIENCE = 50
LEARNING_RATE = 1e-2


def applies(spec, field_name: str) -> bool:
    """Ordered levels exist only for integer ranges, not for enum labels."""
    return spec.output.fields[field_name].type == "int"


def ordinal_loss(class_logits, targets):
    """Class NLL plus the span-normalized ranked probability score.

    Token or class cross-entropy alone treats the levels as unrelated symbols:
    a near miss and a far one cost the same. The RPS term accumulates squared
    error across the ordered levels' cumulative distribution, so probability
    mass far from the decision is punished more than mass beside it. Shared
    verbatim between the LoRA trainer and `fit_head`.
    """
    import torch
    import torch.nn.functional as F

    classes = class_logits.shape[-1]
    nll = F.cross_entropy(class_logits, targets)
    probabilities = F.softmax(class_logits, dim=-1)
    cumulative = probabilities.cumsum(dim=-1)
    steps = (
        torch.arange(classes, device=class_logits.device).unsqueeze(0)
        >= targets.unsqueeze(1)
    ).to(cumulative.dtype)
    rps = ((cumulative - steps) ** 2).sum(dim=-1).mean() / max(classes - 1, 1)
    return nll + rps


def _is_sparse(features) -> bool:
    return hasattr(features, "tocsr")


def _as_torch_features(features):
    """Dense arrays become dense tensors; scipy matrices become sparse CSR.

    TF-IDF features over an open bigram vocabulary would not survive
    densification, and torch differentiates sparse @ dense natively, so the
    first layer multiplies in sparse form and everything after is dense.
    """
    import numpy as np
    import torch

    if _is_sparse(features):
        csr = features.tocsr()
        return torch.sparse_csr_tensor(
            torch.from_numpy(csr.indptr.astype("int64")),
            torch.from_numpy(csr.indices.astype("int64")),
            torch.from_numpy(csr.data.astype("float32")),
            size=csr.shape,
        )
    return torch.from_numpy(np.asarray(features, dtype="float32"))


def _fit_scaler(features):
    """Standardization statistics for dense features; none for sparse.

    The bake-off trained on standardized embeddings, so dense inputs keep
    that. Sparse TF-IDF rows are already L2-normalized by the vectorizer and
    centering would densify them, so they pass through untouched.
    """
    import numpy as np

    if _is_sparse(features):
        return None
    dense = np.asarray(features, dtype="float64")
    scale = dense.std(axis=0)
    scale[scale == 0.0] = 1.0
    return {"mean": dense.mean(axis=0), "scale": scale}


def _apply_scaler(scaler, features):
    import numpy as np

    if scaler is None:
        return features
    return (np.asarray(features, dtype="float64") - scaler["mean"]) / scaler["scale"]


def _train_one(features, labels, dev_features, dev_references, values, config, seed):
    """One gradient-trained head; returns (state, dev within-one or None).

    Full-batch Adam with the shared ordinal loss. With development rows every
    epoch is scored by the same loss on the development split — a smooth
    signal, where within-one agreement saturates on short scales and would
    halt training at the first plateau — and the best state wins, the same
    snapshot-best protection the embedding fine-tune has. Without development
    rows the fit runs a fixed schedule and stands. The returned score is the
    best state's dev within-one agreement, the metric the grid selects on.

    `labels` arrive in the scale's value space — a range need not start at
    zero — and train as indices of `values`.
    """
    import numpy as np
    import torch
    import torch.nn.functional as F

    torch.manual_seed(seed)
    classes = len(values)
    columns = features.shape[1]
    hidden = config["hidden"]
    index_of = {value: index for index, value in enumerate(values)}
    x = _as_torch_features(features)
    y = torch.tensor(np.asarray([index_of[label] for label in labels], dtype="int64"))

    first = torch.nn.Linear(columns, hidden or classes)
    second = torch.nn.Linear(hidden, classes) if hidden else None
    parameters = list(first.parameters()) + (
        list(second.parameters()) if second else []
    )
    optimizer = torch.optim.Adam(
        parameters, lr=LEARNING_RATE, weight_decay=config["weight_decay"]
    )

    def logits(batch, train: bool):
        if batch.layout == torch.sparse_csr:
            out = torch.sparse.mm(batch, first.weight.t()) + first.bias
        else:
            out = first(batch)
        if second is None:
            return out
        out = F.relu(out)
        if train and config["dropout"]:
            out = F.dropout(out, p=config["dropout"], training=True)
        return second(out)

    def state() -> dict[str, Any]:
        layers = [
            {
                "weight": first.weight.detach().numpy().copy(),
                "bias": first.bias.detach().numpy().copy(),
            }
        ]
        if second is not None:
            layers.append(
                {
                    "weight": second.weight.detach().numpy().copy(),
                    "bias": second.bias.detach().numpy().copy(),
                }
            )
        return {"layers": layers}

    select = dev_features is not None and bool(dev_references)
    dev_x = _as_torch_features(dev_features) if select else None
    dev_y = (
        torch.tensor(
            np.asarray(
                [index_of[reference] for reference in dev_references], dtype="int64"
            )
        )
        if select
        else None
    )

    best: tuple[float, dict[str, Any]] | None = None
    waited = 0
    epochs = MAX_EPOCHS if select else NO_DEV_EPOCHS
    for _ in range(epochs):
        optimizer.zero_grad()
        loss = ordinal_loss(logits(x, train=True), y)
        loss.backward()
        optimizer.step()
        if not select:
            continue
        with torch.no_grad():
            dev_loss = float(ordinal_loss(logits(dev_x, train=False), dev_y))
        if best is None or dev_loss < best[0]:
            best = (dev_loss, state())
            waited = 0
        else:
            waited += 1
            if waited >= PATIENCE:
                break
    if best is None:
        return state(), None

    # the grid selects on within-one agreement, measured once on the best state
    from . import decode

    interim = {"kind": KIND, "values": list(values), "scaler": None, **best[1]}
    predictions = decode.decode_levels(class_distribution(interim, dev_features), values)
    within_one = sum(
        abs(prediction - reference) <= 1
        for prediction, reference in zip(predictions, dev_references)
    ) / len(dev_references)
    return best[1], within_one


def fit_head(
    values: list[Any],
    features,
    labels: list[Any],
    dev_features=None,
    dev_references: list | None = None,
    setting: str = "auto",
) -> tuple[dict, dict | None]:
    """Fit the softmax head, choosing capacity on the development rows.

    The grid spans a linear head and one hidden layer — the bake-off found
    each wins on some encoders — with the seed searched alongside, scored by
    dev within-one agreement under the argmax read (the decode rule is
    selected afterwards, on the winning head). `setting` pins the family:
    `linear` or `mlp` restricts the grid, `auto` searches all of it. Without
    development rows the default entry stands. Returns (head, tuning record
    or None).
    """
    scaler = _fit_scaler(features)
    scaled = _apply_scaler(scaler, features)
    grid = tuple(
        config
        for config in HEAD_GRID
        if setting == "auto"
        or (setting == "linear" and config["hidden"] == 0)
        or (setting == "mlp" and config["hidden"] > 0)
    )
    if not grid:
        raise ValueError(f"unknown head setting: {setting!r}")

    def assemble(state: dict[str, Any], config: dict[str, Any]) -> dict[str, Any]:
        return {
            "kind": KIND,
            "values": list(values),
            "scaler": scaler,
            "layers": state["layers"],
            "hidden": config["hidden"],
        }

    if dev_features is None or not dev_references:
        state, _ = _train_one(scaled, labels, None, None, values, grid[0], SEEDS[0])
        return assemble(state, grid[0]), None

    scaled_dev = _apply_scaler(scaler, dev_features)
    trials: list[dict[str, Any]] = []
    best: tuple[float, dict[str, Any], dict[str, Any]] | None = None
    for config in grid:
        for seed in SEEDS:
            state, value = _train_one(
                scaled, labels, scaled_dev, dev_references, values, config, seed
            )
            trials.append(
                {**config, "seed": seed, "dev_within_one": round(value, 4)}
            )
            if best is None or value > best[0]:
                best = (value, state, {**config, "seed": seed})
    return assemble(best[1], best[2]), {
        "selected": best[2],
        "metric": "within_one",
        "trials": trials,
    }


def probe_within_one(
    values: list[Any], features, labels: list[Any], dev_features, dev_references: list
) -> float:
    """Score a feature space by the default head's dev within-one agreement.

    The embedding fine-tune snapshots the epoch whose space decodes best. That
    needs a fast probe whose score is comparable across epochs — the default
    linear entry with one seed — not the full capacity search, which the
    winning space gets exactly once afterwards.
    """
    scaler = _fit_scaler(features)
    _, value = _train_one(
        _apply_scaler(scaler, features),
        labels,
        _apply_scaler(scaler, dev_features),
        dev_references,
        values,
        HEAD_GRID[0],
        SEEDS[0],
    )
    return value


def class_distribution(head: dict, features):
    """Per-row probability of every level — plain numpy, no torch.

    The exact arithmetic generated packages inline: scale, one or two affine
    layers, softmax. Sparse feature matrices multiply without densifying.
    """
    import numpy as np

    x = _apply_scaler(head["scaler"], features)
    layers = head["layers"]
    out = x @ layers[0]["weight"].T + layers[0]["bias"]
    out = np.asarray(out, dtype="float64")
    if len(layers) > 1:
        out = np.maximum(out, 0.0)
        out = out @ layers[1]["weight"].T + layers[1]["bias"]
    out -= out.max(axis=1, keepdims=True)
    exp = np.exp(out)
    return exp / exp.sum(axis=1, keepdims=True)


def predict(head: dict, features) -> list[Any]:
    """Argmax class values via the head's distribution, one per feature row."""
    from . import decode

    return decode.decode_levels(class_distribution(head, features), head["values"])


def head_diagnostics(
    head: dict, features, references: list, levels: list | None = None
) -> dict:
    """Aggregate development diagnostics for the softmax head.

    Everything here is a summary — no inputs, no per-row values — so it can
    sit in public evidence. Each level's mean predicted probability next to
    the observed rate shows calibration in the large; a rare level whose mass
    never arrives is visible here long before it shows up as tail bias in
    evaluation. Mean confidence is the average probability of the decoded
    level.

    `references` live in the head's own value space; `levels` relabels the
    reported rows for heads that train on indices of the real scale.
    """
    distribution = class_distribution(head, features)
    rows = distribution.shape[0]
    values = head["values"]
    labels = values if levels is None else levels

    per_level = [
        {
            "level": labels[index],
            "mean_probability": round(float(distribution[:, index].mean()), 4),
            "observed_rate": round(
                sum(reference == value for reference in references) / len(references),
                4,
            ),
        }
        for index, value in enumerate(values)
    ]
    return {
        "rows": rows,
        "hidden": head.get("hidden", 0),
        "mean_confidence": round(float(distribution.max(axis=1).mean()), 4),
        "levels": per_level,
    }
