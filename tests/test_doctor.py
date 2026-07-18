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


class WordTokenizer:
    model_max_length = 512

    def __call__(self, text, add_special_tokens=True):
        return {"input_ids": list(range(len(text.split()) + 2))}


def setfit_spec():
    return make_spec(
        candidates={"bge-small": {"type": "setfit", "model": "BAAI/bge-small-en-v1.5"}}
    )


def test_truncation_check_warns_when_items_exceed_the_encoder_window(monkeypatch):
    from smallbatch import doctor

    monkeypatch.setattr(doctor, "_encoder_tokenizer", lambda model: WordTokenizer())
    monkeypatch.setattr(doctor, "_encoder_sequence_limit", lambda model: 16)
    records = imported_records(4)
    records[0]["input"]["body"] = "word " * 40

    findings = doctor.inspect_setfit_truncation(setfit_spec(), records)
    assert len(findings) == 1
    level, message = findings[0]
    assert level == "warn"
    assert "1/4 items exceed its 16-token window" in message
    assert "silently truncated" in message


def test_truncation_check_passes_items_that_fit(monkeypatch):
    from smallbatch import doctor

    monkeypatch.setattr(doctor, "_encoder_tokenizer", lambda model: WordTokenizer())
    monkeypatch.setattr(doctor, "_encoder_sequence_limit", lambda model: None)

    findings = doctor.inspect_setfit_truncation(setfit_spec(), imported_records(4))
    # limit falls back to the tokenizer's own claim (512)
    assert findings == [
        (
            "ok",
            findings[0][1],
        )
    ]
    assert "512-token window" in findings[0][1]


def test_truncation_check_stays_offline_when_the_encoder_is_not_cached(monkeypatch):
    from smallbatch import doctor

    monkeypatch.setattr(doctor, "_encoder_tokenizer", lambda model: None)
    findings = doctor.inspect_setfit_truncation(setfit_spec(), imported_records(4))
    assert findings == [
        ("warn", "BAAI/bge-small-en-v1.5: not cached locally; cannot check input truncation")
    ]


def test_truncation_check_skips_specs_without_setfit_candidates():
    from smallbatch import doctor

    assert doctor.inspect_setfit_truncation(make_spec(), imported_records(4)) == []
