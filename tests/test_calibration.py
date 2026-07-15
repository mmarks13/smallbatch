import json
import re

import pytest

from conftest import imported_records, make_spec
from smallbatch.calibration import CalibrationDeclined, calibrate_teacher


class Teacher:
    def __init__(self, drift=False):
        self.calls = 0
        self.drift = drift
        self.prompts = []

    def complete(self, prompt):
        self.calls += 1
        self.prompts.append(prompt)
        count = len(re.findall(r'"id":\s*\d+', prompt.split("Reply with", 1)[0]))
        return json.dumps(
            [
                {
                    "id": index,
                    "output": "urgent" if (index + (self.calls if self.drift else 0)) % 2 else "normal",
                    "reason": "test",
                }
                for index in range(count)
            ]
        )


def inputs(count=20):
    return [record["input"] for record in imported_records(count)]


def test_calibration_approves_and_reuses_identity(tmp_path):
    spec = make_spec(teacher={"backend": "codex-cli", "model": "test"})
    teacher = Teacher()
    result = calibrate_teacher(
        teacher, spec, inputs(), tmp_path, interactive=True, input_fn=lambda _: "a", print_fn=lambda _: None
    )
    assert result.status == "approved" and len(result.row_ids) == 10
    first_items = json.loads(teacher.prompts[0].split("Items:\n", 1)[1].split("\n\nReply", 1)[0])
    second_items = json.loads(teacher.prompts[1].split("Items:\n", 1)[1].split("\n\nReply", 1)[0])
    assert [item["title"] for item in first_items] != [item["title"] for item in second_items]
    calls = teacher.calls
    replay = calibrate_teacher(teacher, spec, inputs(), tmp_path, interactive=False)
    assert replay.status == "approved" and teacher.calls == calls


def test_calibration_more_then_decline_preserves_record(tmp_path):
    spec = make_spec(teacher={"backend": "codex-cli", "model": "test"})
    answers = iter(["m", "d"])
    with pytest.raises(CalibrationDeclined):
        calibrate_teacher(
            Teacher(True),
            spec,
            inputs(25),
            tmp_path,
            interactive=True,
            input_fn=lambda _: next(answers),
            print_fn=lambda _: None,
        )
    value = json.loads((tmp_path / "calibration.json").read_text())
    assert value["status"] == "declined" and value["batches_reviewed"] == 2


def test_noninteractive_requires_explicit_bypass(tmp_path):
    spec = make_spec(teacher={"backend": "codex-cli", "model": "test"})
    with pytest.raises(ValueError, match="--skip-calibration"):
        calibrate_teacher(Teacher(), spec, inputs(), tmp_path, interactive=False)
    result = calibrate_teacher(Teacher(), spec, inputs(), tmp_path, skip=True, interactive=False)
    assert result.status == "bypassed"
