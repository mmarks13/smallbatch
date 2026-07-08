"""Sweep orchestration tests — torch-free, compile stubbed via an injected
`run` callable so the whole grid runs on CPU with no HF/GPU access."""

import subprocess
from pathlib import Path

import pytest
import yaml
from pydantic import ValidationError

from smallbatch import artifacts
from smallbatch.sweep import (
    SweepSpec,
    expand_grid,
    load_sweep,
    make_tag,
    merged_train,
    render_table,
    run_sweep,
)


def write_base_spec(tmp_path, spec_files=None):
    ref = tmp_path / "prefs.yaml"
    ref.write_text("likes: ai")
    spec = {
        "name": "triage",
        "description": "Score.",
        "input_schema": {"title": "str"},
        "output": {"type": "int", "range": [0, 10]},
        "rubric": "-",
        "teacher": {"backend": "claude-cli", "model": "sonnet"},
        "spec_files": [str(ref)] if spec_files is None else spec_files,
        "train": {"base": "base/model", "epochs": 4},
    }
    p = tmp_path / "triage.yaml"
    p.write_text(yaml.safe_dump(spec))
    return p, ref


def write_sweep(tmp_path, **over):
    doc = {
        "name": "sw",
        "spec": "triage.yaml",
        "models": {"org/Big-4B": {"batch_size": 2}, "org/Small-1B": {}},
        "arms": {"plain": {}, "rationale": {"rationale_distillation": True}},
    }
    doc.update(over)
    p = tmp_path / "sweep.yaml"
    p.write_text(yaml.safe_dump(doc))
    return p


class FakeRun:
    """Stands in for subprocess.run: answers preflight probes and writes fake
    manifests for compile calls, keyed by --tag."""

    def __init__(self, artifacts_root, *, agreement=0.9, rc_by_tag=None,
                 bad_models=(), timeout_tags=()):
        self.root = Path(artifacts_root)
        self.agreement = agreement
        self.rc_by_tag = rc_by_tag or {}
        self.bad_models = set(bad_models)
        self.timeout_tags = set(timeout_tags)
        self.calls = []

    def __call__(self, argv, env=None, capture_output=False, text=False, timeout=None):
        self.calls.append({"argv": argv, "env": env, "timeout": timeout})
        if "compile" not in argv:  # preflight probe: [py, -c, SRC, *models]
            bad = [m for m in argv[3:] if m in self.bad_models]
            out = "".join(f"PREFLIGHT_FAIL:: {m}: OSError: gone\n" for m in bad)
            return subprocess.CompletedProcess(argv, 1 if bad else 0, out, "")

        opt = lambda n: argv[argv.index(n) + 1]
        tag = opt("--tag")
        if tag in self.timeout_tags:
            raise subprocess.TimeoutExpired(argv, timeout)
        rc = self.rc_by_tag.get(tag, 0)
        if rc not in (0, 2):
            return subprocess.CompletedProcess(argv, rc, "", "boom traceback")
        spec = yaml.safe_load(Path(argv[argv.index("compile") + 1]).read_text())
        d = artifacts.sweep_run_dir(self.root, spec["name"], opt("--sweep-name"), tag)
        artifacts.write_manifest(d, {
            "function": spec["name"],
            "version": f"{opt('--sweep-name')}/{tag}",
            "base_model": spec["train"]["base"],
            "metrics": {"adapter": {"agreement": self.agreement}, "zeroshot": None},
            "gate": {"passed": rc == 0, "reasons": [] if rc == 0 else ["below gate"]},
        })
        return subprocess.CompletedProcess(argv, rc, "PASS\n", "")


# --- pure grid math ---------------------------------------------------------

def test_make_tag():
    assert make_tag("Qwen/Qwen3.5-4B", "plain") == "qwen3.5-4b-plain"
    assert make_tag("bare-id", "rationale") == "bare-id-rationale"


def test_merged_train_precedence():
    base = {"epochs": 4, "batch_size": 8}
    out = merged_train(base, "org/M", {"batch_size": 2}, {"batch_size": 1, "use_dora": True})
    assert out == {"epochs": 4, "batch_size": 1, "base": "org/M", "use_dora": True}


def test_sweepspec_rejects_extra_key(tmp_path):
    with pytest.raises(ValidationError):
        SweepSpec(name="x", spec="s.yaml", models={"m": {}}, arms={"a": {}}, bogus=1)


def test_sweepspec_requires_nonempty(tmp_path):
    with pytest.raises(ValidationError):
        SweepSpec(name="x", spec="s.yaml", models={}, arms={"a": {}})
    with pytest.raises(ValidationError):
        SweepSpec(name="x", spec="s.yaml", models={"m": {}}, arms={})


def test_expand_grid_size_and_abs_spec_files(tmp_path):
    base, ref = write_base_spec(tmp_path)
    sweep = load_sweep(write_sweep(tmp_path))
    plans = expand_grid(sweep, base)
    assert len(plans) == 4  # 2 models x 2 arms
    tags = {p.tag for p in plans}
    assert tags == {"big-4b-plain", "big-4b-rationale", "small-1b-plain", "small-1b-rationale"}
    for p in plans:
        assert Path(p.spec_dict["spec_files"][0]).is_absolute()
    big_plain = next(p for p in plans if p.tag == "big-4b-plain")
    assert big_plain.spec_dict["train"]["base"] == "org/Big-4B"
    assert big_plain.spec_dict["train"]["batch_size"] == 2
    rat = next(p for p in plans if p.tag == "small-1b-rationale")
    assert rat.spec_dict["train"]["rationale_distillation"] is True


def test_expand_grid_rejects_typo_override(tmp_path):
    base, _ = write_base_spec(tmp_path)
    sweep = load_sweep(write_sweep(tmp_path, models={"org/M": {"bathc_size": 2}}))
    with pytest.raises(ValidationError):
        expand_grid(sweep, base)


def test_render_table():
    results = [
        {"model": "org/Big-4B", "arm": "plain", "status": "pass", "agreement": 0.9},
        {"model": "org/Big-4B", "arm": "rationale", "status": "fail", "agreement": 0.7},
    ]
    t = render_table(results, ["org/Big-4B"], ["plain", "rationale"])
    assert "0.90 PASS" in t and "0.70 FAIL" in t and "| Big-4B |" in t


# --- orchestration ----------------------------------------------------------

def test_happy_path(tmp_path, capsys):
    base, _ = write_base_spec(tmp_path)
    sweep = load_sweep(write_sweep(tmp_path))
    root = tmp_path / "artifacts"
    fake = FakeRun(root, agreement=0.88)
    rc = run_sweep(sweep, data_dir=None, artifacts_root=root,
                   compile_prefix=["py"], run=fake)
    assert rc == 0
    out = capsys.readouterr().out
    assert out.count("MANIFEST::") == 4
    res = (root / "triage" / "sw" / "results.json")
    assert res.exists()
    assert (root / "triage" / "sw" / "summary.md").exists()
    # flags reached the stubbed compile
    compile_calls = [c for c in fake.calls if "compile" in c["argv"]]
    assert len(compile_calls) == 4
    for c in compile_calls:
        assert "--sweep-name" in c["argv"] and "--tag" in c["argv"] and "--arm" in c["argv"]
    data = yaml.safe_load(res.read_text())
    assert all(r["status"] == "pass" for r in data["runs"])


def test_error_recorded_and_continues(tmp_path, capsys):
    base, _ = write_base_spec(tmp_path)
    sweep = load_sweep(write_sweep(tmp_path))
    root = tmp_path / "artifacts"
    fake = FakeRun(root, rc_by_tag={"big-4b-plain": 3})
    rc = run_sweep(sweep, data_dir=None, artifacts_root=root,
                   compile_prefix=["py"], run=fake)
    assert rc == 1  # an error occurred
    data = yaml.safe_load((root / "triage" / "sw" / "results.json").read_text())
    by = {r["tag"]: r for r in data["runs"]}
    assert by["big-4b-plain"]["status"] == "error"
    assert "boom" in by["big-4b-plain"]["error"]
    assert by["small-1b-plain"]["status"] == "pass"  # continued after the error
    assert len(data["runs"]) == 4


def test_timeout_recorded_and_continues(tmp_path):
    base, _ = write_base_spec(tmp_path)
    sweep = load_sweep(write_sweep(tmp_path, timeout_minutes=0.01))
    root = tmp_path / "artifacts"
    fake = FakeRun(root, timeout_tags={"big-4b-plain"})
    rc = run_sweep(sweep, data_dir=None, artifacts_root=root,
                   compile_prefix=["py"], run=fake)
    assert rc == 1
    data = yaml.safe_load((root / "triage" / "sw" / "results.json").read_text())
    by = {r["tag"]: r for r in data["runs"]}
    assert by["big-4b-plain"]["status"] == "timeout"
    assert by["small-1b-rationale"]["status"] == "pass"


def test_env_block_applied(tmp_path):
    base, _ = write_base_spec(tmp_path)
    sweep = load_sweep(write_sweep(tmp_path, env={"HF_HUB_OFFLINE": "1"}))
    root = tmp_path / "artifacts"
    fake = FakeRun(root)
    run_sweep(sweep, data_dir=None, artifacts_root=root, compile_prefix=["py"], run=fake)
    for c in fake.calls:
        assert c["env"]["HF_HUB_OFFLINE"] == "1"


def test_preflight_aborts_before_any_run(tmp_path, capsys):
    base, _ = write_base_spec(tmp_path)
    sweep = load_sweep(write_sweep(tmp_path, models={"org/Good": {}, "org/Bad": {}}))
    root = tmp_path / "artifacts"
    fake = FakeRun(root, bad_models={"org/Bad"})
    rc = run_sweep(sweep, data_dir=None, artifacts_root=root,
                   compile_prefix=["py"], run=fake)
    assert rc == 1
    # only the preflight probe ran; no compile calls
    assert not any("compile" in c["argv"] for c in fake.calls)
    assert "org/Bad" in capsys.readouterr().err
