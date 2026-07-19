import json

import pytest

from smallbatch.journal import JournalLocked, LabelJournal


def test_journal_replays_matching_decision_identity_and_ignores_torn_line(tmp_path):
    journal = LabelJournal.open(tmp_path, "decision-a")
    row = {"id": "row", "input": {"x": "y"}, "output": "a", "origin": "real"}
    journal.record_row(row)
    journal.close()
    events = tmp_path / "journal" / "events.jsonl"
    with events.open("a") as handle:
        handle.write('{"torn":')
    replay = LabelJournal.open(tmp_path, "decision-a")
    assert replay.rows[("row", "real")] == row
    replay.close()
    stale = LabelJournal.open(tmp_path, "decision-b")
    assert stale.rows == {}
    stale.close()


def test_journal_keeps_one_row_per_labeling_pass(tmp_path):
    """Multi-pass labeling draws the same row id once per pass; each draw is
    separately paid work and must replay independently."""
    journal = LabelJournal.open(tmp_path, "decision")
    first = {"id": "row", "input": {"x": "y"}, "output": "a", "origin": "real#1"}
    second = {"id": "row", "input": {"x": "y"}, "output": "b", "origin": "real#2"}
    journal.record_row(first)
    journal.record_row(second)
    journal.close()
    replay = LabelJournal.open(tmp_path, "decision")
    assert replay.rows[("row", "real#1")] == first
    assert replay.rows[("row", "real#2")] == second
    replay.close()


def test_journal_lock_is_single_writer(tmp_path):
    first = LabelJournal.open(tmp_path, "decision")
    with pytest.raises(JournalLocked):
        LabelJournal.open(tmp_path, "decision")
    first.close()


def test_archive_rotates_completed_events(tmp_path):
    journal = LabelJournal.open(tmp_path, "decision")
    journal.record_stage_item("augmentation", {"x": "y"})
    journal.record_stage_done("augmentation")
    assert journal.cached_stage("augmentation")[0]["item"] == {"x": "y"}
    journal.archive()
    assert not (tmp_path / "journal" / "events.jsonl").exists()
    archived = list((tmp_path / "journal").glob("archived-*.jsonl"))
    assert len(archived) == 1
    assert json.loads(archived[0].read_text().splitlines()[0])["event"] == "stage_item"
