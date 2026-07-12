"""Label-review loop: filters, decisions, edits, and split rewriting."""

import argparse
import json

from smallbatch.review import apply_edit, matches, run_review, save
from smallbatch.spec import FunctionSpec

SPEC = FunctionSpec(
    name="toy",
    description="-",
    input_schema={"title": "str"},
    output={"type": "int", "range": [0, 4]},
    rubric="-",
    teacher={"backend": "claude-cli", "model": "sonnet"},
)


def make_args(**over):
    base = dict(split=None, origin=None, label=None, field=None, status="unreviewed")
    base.update(over)
    return argparse.Namespace(**base)


def rows():
    return [
        {"id": f"r{i}", "input": {"title": f"t{i}"}, "score": i % 5, "reason": "why",
         "origin": "real", "split": "train", "teacher_model": "sonnet",
         "labeled_at": "2026-07-08"}
        for i in range(6)
    ]


def write_dataset(tmp_path, rs):
    (tmp_path / "labeled.jsonl").write_text("\n".join(json.dumps(r) for r in rs))
    (tmp_path / "meta.json").write_text("{}")


def test_matches_unstable_filter():
    rs = rows()
    rs[0]["probe_output"] = rs[0]["score"]  # stable: probe agrees
    rs[1]["probe_output"] = (rs[1]["score"] + 3) % 5  # unstable
    args = make_args(status="all", unstable=True)
    picked = [r for r in rs if matches(SPEC, r, args)]
    assert picked == [rs[1]]  # unprobed and stable rows excluded


def test_matches_filters():
    row = rows()[2]  # score 2
    assert matches(SPEC, row, make_args())
    assert matches(SPEC, row, make_args(label="2"))
    assert not matches(SPEC, row, make_args(label="3"))
    assert not matches(SPEC, row, make_args(origin="variant"))
    row["review"] = {"status": "accepted"}
    assert not matches(SPEC, row, make_args())  # default: unreviewed only
    assert matches(SPEC, row, make_args(status="accepted"))
    assert matches(SPEC, row, make_args(status="all"))


def test_apply_edit_validates_contract():
    row = rows()[0]
    assert apply_edit(SPEC, row, lambda _: "9") is not None  # out of range
    assert apply_edit(SPEC, row, lambda _: "3") is None
    assert row["score"] == 3
    assert row["review"]["status"] == "edited" and row["review"]["original"] == 0


def test_run_review_accept_reject_and_save(tmp_path):
    rs = rows()
    write_dataset(tmp_path, rs)
    answers = iter(["a", "r", "q"])  # accept #1, reject #2, quit (saves)
    out = []
    run_review(SPEC, tmp_path, make_args(), input_fn=lambda _: next(answers),
               print_fn=out.append)
    saved = [json.loads(line) for line in (tmp_path / "labeled.jsonl").read_text().splitlines()]
    assert saved[0]["review"]["status"] == "accepted"
    assert saved[1]["review"]["status"] == "rejected"
    train = [json.loads(line) for line in (tmp_path / "train.jsonl").read_text().splitlines()]
    assert len(train) == 5  # rejected row excluded from splits
    assert {r["id"] for r in train} == {"r0", "r2", "r3", "r4", "r5"}
    meta = json.loads((tmp_path / "meta.json").read_text())
    assert meta["review"]["accepted"] == 1 and meta["review"]["rejected"] == 1


def test_save_writes_all_split_files(tmp_path):
    rs = rows()
    rs[0]["split"] = "gate"
    rs[1]["split"] = "dev"
    rs[1]["review"] = {"status": "rejected"}
    counts = save(SPEC, rs, tmp_path)
    assert counts == {"rejected": 1, "unreviewed": 5}
    dev = (tmp_path / "dev.jsonl").read_text().strip()
    assert dev == ""  # the only dev row was rejected
    gate = [json.loads(line) for line in (tmp_path / "gate.jsonl").read_text().splitlines()]
    assert len(gate) == 1
