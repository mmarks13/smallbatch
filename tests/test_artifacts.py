import textwrap

from smallbatch import artifacts


def make_artifact(tmp_path, name, passed=True, spec_files_content="likes: cats"):
    ref = tmp_path / "prefs.yaml"
    ref.write_text(spec_files_content)
    spec_yaml = textwrap.dedent(
        f"""
        name: {name}
        description: Score.
        input_schema: {{title: str}}
        output: {{type: int, range: [0, 10]}}
        rubric: "-"
        teacher: {{backend: claude-cli, model: sonnet}}
        spec_files: ["{ref}"]
        """
    )
    from smallbatch.spec import load_spec

    v = artifacts.new_version_dir(tmp_path / "artifacts", name)
    (v / "spec.yaml").write_text(spec_yaml)
    spec = load_spec(v / "spec.yaml")
    artifacts.write_manifest(
        v,
        {
            "function": name,
            "spec_hash": spec.spec_hash(),
            "gate": {"passed": passed, "reasons": []},
        },
    )
    return v, ref


def test_latest_prefers_passing(tmp_path):
    v1, _ = make_artifact(tmp_path, "toy", passed=True)
    v2, _ = make_artifact(tmp_path, "toy", passed=False)
    assert v2.name.endswith("-r2")
    assert artifacts.latest(tmp_path / "artifacts", "toy") == v1
    assert artifacts.latest(tmp_path / "artifacts", "toy", passing_only=False) == v2


def test_staleness_detects_spec_file_change(tmp_path):
    v, ref = make_artifact(tmp_path, "toy")
    assert artifacts.staleness(v) is None
    ref.write_text("likes: dogs")
    assert "changed" in artifacts.staleness(v)


def test_sweep_run_dir_replaces_populated_dir(tmp_path):
    root = tmp_path / "artifacts"
    d1 = artifacts.sweep_run_dir(root, "toy", "sw", "m-plain")
    (d1 / "adapter").mkdir()
    (d1 / "adapter" / "weights.bin").write_text("old")
    d2 = artifacts.sweep_run_dir(root, "toy", "sw", "m-plain")
    assert d1 == d2
    assert not (d2 / "adapter").exists()  # stateless: prior contents wiped


def test_sweep_runs_found_but_ignored_by_versions(tmp_path):
    root = tmp_path / "artifacts"
    # a normal dated version at depth 1
    v = artifacts.new_version_dir(root, "toy")
    artifacts.write_manifest(v, {"function": "toy", "gate": {"passed": True}})
    # sweep runs nested at depth 2
    for tag in ("m-plain", "m-rationale"):
        d = artifacts.sweep_run_dir(root, "toy", "sw", tag)
        artifacts.write_manifest(d, {"function": "toy", "gate": {"passed": True}})
    assert artifacts.versions(root, "toy") == [v]  # depth-1 only
    runs = artifacts.sweep_runs(root, "toy")
    assert {p.name for p in runs} == {"m-plain", "m-rationale"}


def _v2_manifest(tfidf_passed, lora_passed, winner="tfidf", accepted=None):
    return {
        "manifest_schema_version": 2,
        "function": "toy",
        "candidates": {
            "tfidf": {
                "backend": "tfidf", "status": "completed",
                "gate": {"passed": tfidf_passed, "reasons": []},
            },
            "lora": {
                "backend": "lora", "status": "completed",
                "gate": {"passed": lora_passed, "reasons": []},
            },
        },
        "selection": {"winner": winner, "reason": "highest gate agreement"},
        "deployment": (
            {"accepted_despite_gate": True, "accepted_candidate": accepted}
            if accepted else None
        ),
        "gate": {"passed": tfidf_passed if winner == "tfidf" else lora_passed},
    }


def test_candidate_usability_is_candidate_scoped():
    # both fail, tfidf accepted: only tfidf becomes usable
    m = _v2_manifest(False, False, winner="tfidf", accepted="tfidf")
    assert artifacts.candidate_is_usable(m, "tfidf")
    assert not artifacts.candidate_is_usable(m, "lora")
    assert artifacts.candidate_is_usable(m, "lora", allow_failed=True)
    assert artifacts.artifact_is_usable(m)


def test_passing_winner_does_not_unlock_failed_secondary():
    m = _v2_manifest(True, False, winner="tfidf")
    assert artifacts.candidate_is_usable(m, "tfidf")
    assert not artifacts.candidate_is_usable(m, "lora")


def test_unaccepted_all_fail_artifact_unusable():
    m = _v2_manifest(False, False, winner="tfidf")
    assert not artifacts.artifact_is_usable(m)
    assert not artifacts.candidate_is_usable(m, "tfidf")


def test_errored_candidate_never_usable():
    m = _v2_manifest(False, False, winner="tfidf", accepted="tfidf")
    m["candidates"]["tfidf"]["status"] = "error"
    assert not artifacts.candidate_is_usable(m, "tfidf")
    assert not artifacts.candidate_is_usable(m, "tfidf", allow_failed=True)


def test_legacy_v1_manifest_synthesized():
    m = {"function": "toy", "gate": {"passed": True},
         "base_model": "b", "inference_precision": "fp32",
         "metrics": {"adapter": {"agreement": 0.9}}}
    assert artifacts.winner(m) == "lora"
    rec = artifacts.candidate_record(m)
    assert rec["backend"] == "lora" and rec["gate"]["passed"]
    assert artifacts.artifact_is_usable(m)
    assert artifacts.candidate_record(m, "tfidf") is None


def test_resolve_version_candidate_scoped(tmp_path):
    import pytest

    root = tmp_path / "artifacts"
    v = artifacts.new_version_dir(root, "toy")
    artifacts.write_manifest(v, _v2_manifest(False, False, accepted="tfidf"))
    assert artifacts.resolve_version(root, "toy", None, allow_failed=False) == v
    with pytest.raises(ValueError, match="lora"):
        artifacts.resolve_version(root, "toy", None, False, candidate="lora")
    assert artifacts.resolve_version(root, "toy", None, True, candidate="lora") == v
    with pytest.raises(ValueError, match="no 'setfit' candidate"):
        artifacts.resolve_version(root, "toy", None, True, candidate="setfit")


def test_version_sort_handles_double_digit_revisions(tmp_path):
    root = tmp_path / "artifacts"
    base = root / "toy"
    for name in ["2026-07-11"] + [f"2026-07-11-r{i}" for i in range(2, 12)]:
        d = base / name
        d.mkdir(parents=True)
        artifacts.write_manifest(d, {"function": "toy", "gate": {"passed": True}})
    vs = [p.name for p in artifacts.versions(root, "toy")]
    assert vs[-1] == "2026-07-11-r11"
    assert vs.index("2026-07-11-r9") < vs.index("2026-07-11-r10")
    assert artifacts.latest(root, "toy").name == "2026-07-11-r11"
    # dates still order before revisions of a later date
    d = base / "2026-07-12"
    d.mkdir()
    artifacts.write_manifest(d, {"function": "toy", "gate": {"passed": True}})
    assert artifacts.latest(root, "toy").name == "2026-07-12"
