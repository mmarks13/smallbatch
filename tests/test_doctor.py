import json

from conftest import imported_records, make_spec
from smallbatch.doctor import inspect_data, inspect_items, inspect_spec
from smallbatch.labeling import build_dataset


def test_items_distinguish_imported_and_teacher_modes():
    spec = make_spec()
    assert inspect_items(spec, imported_records(30))[0] == ("ok", "30 valid imported decisions")
    unlabeled = [{"input": record["input"]} for record in imported_records(30)]
    findings = inspect_items(spec, unlabeled)
    assert any(level == "fail" and "teacher" in message for level, message in findings)


def test_data_identity_and_files(tmp_path):
    spec = make_spec()
    build_dataset(spec, imported_records(30), tmp_path)
    assert inspect_data(spec, tmp_path) == [
        ("ok", "decision dataset matches: {'train': 21, 'dev': 3, 'eval': 6}")
    ]
    meta = json.loads((tmp_path / "meta.json").read_text())
    meta["decision_hash"] = "changed"
    (tmp_path / "meta.json").write_text(json.dumps(meta))
    assert inspect_data(spec, tmp_path)[0][0] == "fail"


def test_spec_lists_candidate_requirements():
    findings = inspect_spec(make_spec())
    assert ("ok", "tfidf: CPU TF-IDF candidate") in findings
