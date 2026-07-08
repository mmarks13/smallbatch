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
