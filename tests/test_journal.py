"""Stage-event journal: crash recovery at every paid stage.

A CrashingTeacher dies after N completions; the rerun must finish without
repeating the calls that already succeeded.
"""

import json

import pytest

from smallbatch.journal import EVENTS_FILE, JournalLocked, LabelJournal
from smallbatch.labeling import build_dataset
from smallbatch.spec import AugmentSpec, FunctionSpec

SPEC = FunctionSpec(
    name="toy",
    description="Score.",
    input_schema={"title": "str"},
    output={"type": "int", "range": [0, 4]},
    rubric="-",
    teacher={
        "backend": "claude-cli",
        "model": "sonnet",
        "examples": 0,  # no legacy variant fill — tests count calls exactly
        "holdout": 0.2,
        "batch_size": 5,  # small batches so a crash bisects real labeling
        "consistency": 0,
    },
)


class Boom(RuntimeError):
    pass


class CrashingTeacher:
    """Answers like FakeTeacher but raises Boom after `survive` completions."""

    def __init__(self, survive=10**9):
        self.survive = survive
        self.calls = 0
        self.label_calls = 0

    def complete(self, prompt: str) -> str:
        self.calls += 1
        if self.calls > self.survive:
            raise Boom()
        if "data labeler" in prompt:
            self.label_calls += 1
            n = prompt.count('"title"')
            return json.dumps([{"id": i, "score": 1, "reason": "r"} for i in range(n)])
        return json.dumps([{"title": f"v{self.calls}-{i}"} for i in range(3)])


def items(n):
    return [{"title": f"t{i}"} for i in range(n)]


def test_crash_mid_real_labeling_resumes_without_repeat(tmp_path):
    # 10 items, batch 5 -> 2 label calls; crash after the first
    with pytest.raises(Boom):
        build_dataset(CrashingTeacher(survive=1), SPEC, items(10), tmp_path)
    journal_events = (tmp_path / "journal" / EVENTS_FILE).read_text().splitlines()
    assert len([line for line in journal_events if '"row"' in line]) == 5

    resumed = CrashingTeacher()
    meta = build_dataset(resumed, SPEC, items(10), tmp_path)
    assert meta["real"] == 10
    assert resumed.label_calls == 1  # only the second batch was re-sent
    # fold succeeded -> journal archived, lock released
    assert not (tmp_path / "journal" / EVENTS_FILE).exists()
    assert not (tmp_path / "journal" / "lock").exists()
    assert list((tmp_path / "journal").glob("archived-*.jsonl"))


def test_crash_after_variant_generation_reuses_generated_inputs(tmp_path):
    aug_spec = SPEC.model_copy(update={
        "augment": AugmentSpec(paraphrase={"cap": 3}),
        "teacher": SPEC.teacher.model_copy(update={"examples": 20}),
    })
    # calls: 2 real-label batches, 4 band-generation calls (1 item each),
    # then crash on the variant-labeling call
    with pytest.raises(Boom):
        build_dataset(CrashingTeacher(survive=6), aug_spec, items(10), tmp_path)

    resumed = CrashingTeacher()
    meta = build_dataset(resumed, aug_spec, items(10), tmp_path)
    assert meta["variants"] > 0
    # generation was NOT repeated: the resumed teacher only labeled
    generation_calls = resumed.calls - resumed.label_calls
    assert generation_calls == 0
    rows = [json.loads(line) for line in (tmp_path / "labeled.jsonl").read_text().splitlines()]
    variant = next(r for r in rows if r["origin"] == "variant")
    assert variant["source_ids"]  # provenance survived the crash


def test_crash_during_probe_replays_probe_results(tmp_path):
    probe_spec = SPEC.model_copy(
        update={"teacher": SPEC.teacher.model_copy(update={"consistency": 8})}
    )
    # 2 real batches + probe batches (8 rows, batch 5 -> 2 calls); crash mid-probe
    with pytest.raises(Boom):
        build_dataset(CrashingTeacher(survive=3), probe_spec, items(10), tmp_path)
    resumed = CrashingTeacher()
    meta = build_dataset(resumed, probe_spec, items(10), tmp_path)
    assert meta["probe_n"] == 8
    assert resumed.calls < 4  # replays: no real labels, at most one probe batch


def test_torn_final_journal_line_is_tolerated(tmp_path):
    with pytest.raises(Boom):
        build_dataset(CrashingTeacher(survive=1), SPEC, items(10), tmp_path)
    events = tmp_path / "journal" / EVENTS_FILE
    events.write_text(events.read_text() + '{"event": "row", "fing')  # torn tail
    resumed = CrashingTeacher()
    meta = build_dataset(resumed, SPEC, items(10), tmp_path)
    assert meta["real"] == 10


def test_stale_fingerprint_events_are_ignored(tmp_path):
    with pytest.raises(Boom):
        build_dataset(CrashingTeacher(survive=1), SPEC, items(10), tmp_path)
    changed = SPEC.model_copy(deep=True)
    changed.rubric = "completely different"
    resumed = CrashingTeacher()
    meta = build_dataset(resumed, changed, items(10), tmp_path)
    assert meta["real"] == 10
    assert resumed.label_calls == 2  # nothing replayed: labels were re-earned


def test_concurrent_labeling_locked_out(tmp_path):
    j = LabelJournal.open(tmp_path, "f" * 64)
    try:
        with pytest.raises(JournalLocked, match="remove the file"):
            build_dataset(CrashingTeacher(), SPEC, items(10), tmp_path)
    finally:
        j.close()
    # lock released -> labeling proceeds
    meta = build_dataset(CrashingTeacher(), SPEC, items(10), tmp_path)
    assert meta["real"] == 10
