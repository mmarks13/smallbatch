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
    assert replay.rows["row"] == row
    replay.close()
    stale = LabelJournal.open(tmp_path, "decision-b")
    assert stale.rows == {}
    stale.close()


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
