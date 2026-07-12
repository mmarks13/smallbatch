import json
import random

from smallbatch.labeling import (
    _closer_to_band,
    _dedupe_labeled_rows,
    build_dataset,
    consistency_probe,
    counterfactual_rows,
    generate_field_dropout,
    plan_variant_bands,
    split_holdout,
)
from smallbatch.spec import FunctionSpec

SPEC = FunctionSpec(
    name="toy",
    description="Score.",
    input_schema={"title": "str"},
    output={"type": "int", "range": [0, 4]},
    rubric="-",
    teacher={
        "backend": "claude-cli",
        "model": "sonnet",
        "examples": 40,
        "holdout": 0.2,
        "batch_size": 50,
    },
)


def rows(scores, origin="real"):
    return [
        {"input": {"title": f"t{i}"}, "score": s, "reason": "", "origin": origin}
        for i, s in enumerate(scores)
    ]


def test_plan_targets_sparse_bands():
    real = rows([2] * 20 + [3] * 10)
    plan = plan_variant_bands(SPEC, real, target_total=40)
    assert sum(plan.values()) in range(9, 12)  # ~10 needed, rounding tolerated
    assert 2 not in plan  # already over uniform target
    assert plan.get(0) and plan.get(4)  # empty bands get coverage


def test_holdout_real_only_and_stratified():
    data = rows([0] * 10 + [4] * 10) + rows([2] * 30, origin="variant")
    train, holdout = split_holdout(data, frac=0.2)
    assert all(r["origin"] == "real" for r in holdout)
    assert len(holdout) == 4 and {r["score"] for r in holdout} == {0, 4}
    assert len(train) + len(holdout) == 50


class FakeTeacher:
    """Labels everything score=1; generates items named v<i>."""

    def __init__(self):
        self.calls = 0

    def complete(self, prompt: str) -> str:
        self.calls += 1
        if "data labeler" in prompt:
            items = json.loads(prompt[prompt.index("[") :][: self._arr_len(prompt)])
            return json.dumps([{"id": it["id"], "score": 1, "reason": "r"} for it in items])
        return json.dumps([{"title": f"v{i}"} for i in range(5)])

    @staticmethod
    def _arr_len(prompt):
        text = prompt[prompt.index("[") :]
        depth = 0
        for i, c in enumerate(text):
            depth += c == "["
            depth -= c == "]"
            if depth == 0:
                return i + 1
        raise ValueError


def test_build_dataset_end_to_end(tmp_path):
    teacher = FakeTeacher()
    items = [{"title": f"real{i}"} for i in range(10)]
    meta = build_dataset(teacher, SPEC, items, tmp_path)
    assert meta["real"] == 10
    assert meta["variants"] > 0
    for split in ("train", "dev", "gate"):
        assert (tmp_path / f"{split}.jsonl").exists()
    labeled = (tmp_path / "labeled.jsonl").read_text().splitlines()
    assert len(labeled) == meta["real"] + meta["variants"]
    first = json.loads(labeled[0])
    assert first["teacher_model"] and first["origin"] == "real"
    assert first["id"] and first["split"] in ("train", "dev", "gate")
    # variants are train-only and carry the ids of the reals that seeded them
    rows = [json.loads(line) for line in labeled]
    real_ids = {r["id"] for r in rows if r["origin"] == "real"}
    train_real_ids = {r["id"] for r in rows if r["origin"] == "real" and r["split"] == "train"}
    for v in (r for r in rows if r["origin"] == "variant"):
        assert v["split"] == "train"
        assert v["source_ids"] and set(v["source_ids"]) <= train_real_ids, (
            "variant seeded from a non-train real"
        )
    assert real_ids  # sanity


# --- augment ------------------------------------------------------------


DROPOUT_SPEC = FunctionSpec(
    name="toy",
    description="Score.",
    input_schema={"title": "str", "signals": "str"},
    output={"type": "int", "range": [0, 4]},
    rubric="-",
    teacher={"backend": "claude-cli", "model": "sonnet", "batch_size": 50},
    augment={"field_dropout": {"fields": ["signals"], "cap": 3}},
)


def test_generate_field_dropout_blanks_and_links():
    reals = [
        {"id": f"r{i}", "input": {"title": f"t{i}", "signals": f"upvotes: {i}"},
         "score": 2, "origin": "real", "split": "train"}
        for i in range(5)
    ]
    reals.append({"id": "r9", "input": {"title": "t9", "signals": ""},
                  "score": 2, "origin": "real", "split": "train"})
    items, sources = generate_field_dropout(
        DROPOUT_SPEC, reals, ["signals"], cap=3, rng=random.Random(0)
    )
    assert len(items) == 3  # cap respected; the empty-signals row was never eligible
    assert all(it["signals"] == "" and it["title"] for it in items)
    assert all(len(src) == 1 and src[0].startswith("r") for src in sources)


class CFTeacher:
    """Counterfactual fake. Generation returns titles tagged with the target
    band; labeling scores tagged titles at their band (moved) or, in
    `stubborn` mode, always at 2 (miss -> retry)."""

    def __init__(self, stubborn=False):
        self.stubborn = stubborn
        self.cf_calls = 0
        self.retry_seen = False

    def complete(self, prompt: str) -> str:
        items = json.loads(prompt[prompt.index("[") :][: FakeTeacher._arr_len(prompt)])
        if "COUNTERFACTUAL" in prompt:
            self.cf_calls += 1
            self.retry_seen |= "did not move the label closer" in prompt
            import re
            band = re.search(r"score around (\d+)", prompt).group(1)
            return json.dumps(
                [{"id": it["id"], "title": f"cf{band} {it['title']}"} for it in items]
            )
        assert "data labeler" in prompt
        out = []
        for it in items:
            if it["title"].startswith("cf") and not self.stubborn:
                score = int(it["title"][2])
            else:
                score = 2
            out.append({"id": it["id"], "score": score, "reason": ""})
        return json.dumps(out)


CF_SPEC = FunctionSpec(
    name="toy",
    description="Score.",
    input_schema={"title": "str"},
    output={"type": "int", "range": [0, 4]},
    rubric="-",
    teacher={"backend": "claude-cli", "model": "sonnet", "batch_size": 50},
)


def cf_reals():
    return [
        {"id": f"r{i}", "input": {"title": f"t{i}"}, "score": 2, "reason": "",
         "origin": "real", "split": "train"}
        for i in range(10)
    ]


def test_counterfactuals_move_labels_and_record_intent():
    teacher = CFTeacher()
    rows_out, hit_rate = counterfactual_rows(
        teacher, CF_SPEC, cf_reals(), cap=8, seed=3, hist_rows=cf_reals()
    )
    assert rows_out and hit_rate == 1.0
    assert not teacher.retry_seen  # every edit moved: no retry pass
    for r in rows_out:
        assert r["origin"] == "counterfactual"
        assert r["intended_band"] is not None
        assert r["source_ids"] and r["source_ids"][0].startswith("r")
        assert r["score"] != 2  # moved off the source band


def test_counterfactual_misses_retry_once_and_are_kept():
    teacher = CFTeacher(stubborn=True)
    rows_out, hit_rate = counterfactual_rows(
        teacher, CF_SPEC, cf_reals(), cap=6, seed=3, hist_rows=cf_reals()
    )
    assert teacher.retry_seen  # a second, feedback-carrying pass ran
    assert hit_rate == 0.0  # stubborn teacher never moves the label
    assert all(r["score"] == 2 for r in rows_out)  # kept as invariance rows


def test_counterfactual_progress_requires_moving_toward_target():
    source = {"score": 4}
    assert _closer_to_band(CF_SPEC, {"score": 2}, source, 0)
    assert not _closer_to_band(CF_SPEC, {"score": 4}, source, 0)
    assert not _closer_to_band(CF_SPEC, {"score": 5}, source, 0)


def test_exact_input_duplicates_are_collapsed_with_observation():
    original = {
        "id": "same",
        "input": {"title": "x"},
        "score": 2,
        "reason": "first",
        "origin": "real",
        "split": "train",
    }
    duplicate = {
        **original,
        "score": 3,
        "reason": "second",
        "origin": "counterfactual",
        "source_ids": ["same"],
        "intended_band": 4,
    }
    unique, n = _dedupe_labeled_rows([original, duplicate])
    assert n == 1 and unique == [original]
    assert original["duplicate_observations"] == [{
        "origin": "counterfactual",
        "score": 3,
        "reason": "second",
        "source_ids": ["same"],
        "intended_band": 4,
    }]


def test_build_dataset_augment_block_replaces_legacy(tmp_path):
    class DropoutFake(FakeTeacher):
        def complete(self, prompt):
            assert "synthetic inputs" not in prompt, "paraphrase ran but block omits it"
            return super().complete(prompt)

    meta = build_dataset(
        DropoutFake(), DROPOUT_SPEC,
        [{"title": f"real{i}", "signals": f"upvotes: {i}"} for i in range(10)],
        tmp_path,
    )
    assert meta["variants"] == 0 and meta["dropout"] == 3
    rows_all = [json.loads(line) for line in (tmp_path / "labeled.jsonl").read_text().splitlines()]
    dropped = [r for r in rows_all if r["origin"] == "dropout"]
    assert len(dropped) == 3
    for r in dropped:
        assert r["split"] == "train" and r["source_ids"]
        assert r["input"]["signals"] == ""


class SplitBrainTeacher:
    """Probe fake: relabels t0-t4 as 1 (stable) and t5+ as 4 (unstable)."""

    def complete(self, prompt: str) -> str:
        items = json.loads(prompt[prompt.index("[") :][: FakeTeacher._arr_len(prompt)])
        return json.dumps(
            [
                {"id": it["id"], "score": 1 if int(it["title"][1:]) < 5 else 4,
                 "reason": ""}
                for it in items
            ]
        )


def test_consistency_probe_measures_self_agreement():
    data = rows([1] * 10)
    for r in data:
        r["split"] = "train"
    data[9]["split"] = "gate"
    probe = consistency_probe(SplitBrainTeacher(), SPEC, data, n=10)
    assert probe["n"] == 10
    assert probe["self_agreement"] == 0.5  # t5-t9 flip 1 -> 4 (beyond ±1)
    assert probe["unstable"] == 5 and probe["unstable_in_gate"] == 1
    assert all("probe_output" in r for r in data)  # n covered every row
    assert data[9]["probe_output"] == 4


def test_consistency_probe_off_and_empty():
    assert consistency_probe(SplitBrainTeacher(), SPEC, rows([1] * 4), n=0) is None
    assert consistency_probe(SplitBrainTeacher(), SPEC, [], n=5) is None


def test_build_dataset_runs_probe_into_meta(tmp_path):
    spec = SPEC.model_copy(
        update={"teacher": SPEC.teacher.model_copy(update={"consistency": 5})}
    )
    meta = build_dataset(FakeTeacher(), spec, [{"title": f"real{i}"} for i in range(10)],
                         tmp_path, max_variants=0)
    assert meta["teacher_self_agreement"] == 1.0  # FakeTeacher always says 1
    assert meta["probe_n"] == 5
    probed = [
        json.loads(line)
        for line in (tmp_path / "labeled.jsonl").read_text().splitlines()
        if "probe_output" in line
    ]
    assert len(probed) == 5


def test_append_keeps_gate_sticky_and_dedupes(tmp_path):
    teacher = FakeTeacher()
    items = [{"title": f"real{i}"} for i in range(10)]
    build_dataset(teacher, SPEC, items, tmp_path)
    gate_before = {
        json.loads(line)["id"] for line in (tmp_path / "gate.jsonl").read_text().splitlines()
    }

    more = items[:5] + [{"title": f"new{i}"} for i in range(20)]  # 5 dupes + 20 new
    meta = build_dataset(teacher, SPEC, more, tmp_path, append=True, max_variants=0)
    assert meta["real"] == 30  # dupes skipped, not relabeled
    gate_after = {
        json.loads(line)["id"] for line in (tmp_path / "gate.jsonl").read_text().splitlines()
    }
    assert gate_before <= gate_after  # sticky: nothing ever leaves the gate
    assert len(gate_after) == 6  # 0.2 of 30 reals


def test_legacy_dataset_migrates_holdout_to_gate(tmp_path):
    # simulate a pre-v0.2 layout: labeled/train/holdout, no ids or splits
    real = rows([0, 1, 2, 3, 4] * 4)
    holdout = real[:4]
    for path, rs in (("labeled.jsonl", real), ("holdout.jsonl", holdout)):
        (tmp_path / path).write_text("\n".join(json.dumps(r) for r in rs))

    meta = build_dataset(
        FakeTeacher(), SPEC, [{"title": "brand-new"}], tmp_path,
        append=True, max_variants=0,
    )
    gate = [json.loads(line) for line in (tmp_path / "gate.jsonl").read_text().splitlines()]
    old_titles = {r["input"]["title"] for r in holdout}
    assert old_titles <= {r["input"]["title"] for r in gate}
    assert meta["real"] == 21


def test_dataset_hash_order_invariant_and_content_sensitive():
    from smallbatch.labeling import dataset_hash

    rows = [
        {"id": "a", "input": {"t": "x"}, "score": 2, "split": "train", "origin": "real"},
        {"id": "b", "input": {"t": "y"}, "score": 3, "split": "gate", "origin": "real"},
    ]
    h = dataset_hash(rows)
    assert dataset_hash(list(reversed(rows))) == h  # row order is presentation
    assert dataset_hash([dict(rows[0], score=4), rows[1]]) != h  # label edit
    assert dataset_hash([dict(rows[0], split="gate"), rows[1]]) != h  # rerouting
    assert dataset_hash([dict(rows[0], gold=1), rows[1]]) != h  # gold annotation


def test_plan_variant_bands_cap_is_a_ceiling():
    # a cap can only reduce the balance-derived need, never force generation
    full = rows([0, 1, 2, 3, 4] * 10)  # 50 >= target_total
    assert plan_variant_bands(SPEC, full, target_total=40, cap=25) == {}
    sparse = rows([2] * 20 + [3] * 10)
    capped = plan_variant_bands(SPEC, sparse, target_total=40, cap=4)
    assert 0 < sum(capped.values()) <= 5  # ~10 needed, capped (± band rounding)


def test_max_variants_is_a_global_budget(tmp_path):
    from smallbatch.spec import AugmentSpec

    aug_spec = SPEC.model_copy(update={"augment": AugmentSpec(paraphrase={"cap": 50})})
    meta = build_dataset(
        FakeTeacher(), aug_spec, [{"title": f"real{i}"} for i in range(10)],
        tmp_path, max_variants=2,
    )
    assert meta["variants"] <= 2  # stage cap 50 bounded by the global budget


def test_max_variants_budget_covers_dropout_stage(tmp_path):
    from smallbatch.spec import AugmentSpec

    aug_spec = SPEC.model_copy(
        update={"augment": AugmentSpec(field_dropout={"fields": ["title"], "cap": 30})}
    )
    meta = build_dataset(
        FakeTeacher(), aug_spec, [{"title": f"real{i}"} for i in range(10)],
        tmp_path, max_variants=1,
    )
    assert meta.get("dropout", 0) <= 1
    assert meta["variants"] == 0  # no paraphrase stage configured


def test_max_variants_zero_makes_no_generation_calls(tmp_path):
    from smallbatch.spec import AugmentSpec

    aug_spec = SPEC.model_copy(update={"augment": AugmentSpec(
        paraphrase={"cap": 50},
        field_dropout={"fields": ["title"], "cap": 30},
    )})
    teacher = FakeTeacher()
    meta = build_dataset(
        teacher, aug_spec, [{"title": f"real{i}"} for i in range(10)],
        tmp_path, max_variants=0,
    )
    assert meta["variants"] == 0 and meta.get("dropout", 0) == 0
    assert teacher.calls == 1  # exactly the one real-labeling batch
