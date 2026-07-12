"""Model x arm sweeps: `smallbatch sweep <sweep.yaml>`.

A sweep is a grid of (base model) x (arm) compiles of one function spec. Each
cell is run as an isolated `smallbatch compile` subprocess so VRAM is fully
released between runs and one run's crash can't poison the rest. This module is
deliberately torch-free (stdlib + yaml + pydantic + artifacts) so the grid math
and orchestration are unit-testable on CPU.

Merge precedence for a run's train config is: base spec < model override <
arm override. Runs are fully stateless — rerunning a sweep replaces its
artifact dirs; there is no resume (subset == a smaller yaml).
"""

from __future__ import annotations

import copy
import json
import os
import subprocess
import sys
import tempfile
from dataclasses import dataclass, field
from pathlib import Path

import yaml
from pydantic import BaseModel, model_validator

from . import artifacts
from .spec import FunctionSpec, validate_slug


class SweepSpec(BaseModel):
    name: str
    spec: str  # path to the base function spec yaml (relative to this file)
    models: dict[str, dict] = {}
    arms: dict[str, dict] = {}
    timeout_minutes: float = 120
    env: dict[str, str] = {}

    # set by load_sweep so `spec` resolves relative to the sweep yaml's dir
    _base_dir: Path = Path(".")

    model_config = {"extra": "forbid"}

    @model_validator(mode="after")
    def _nonempty(self) -> "SweepSpec":
        if not self.models:
            raise ValueError("sweep needs at least one model")
        if not self.arms:
            raise ValueError("sweep needs at least one arm")
        validate_slug(self.name, "sweep name")
        for arm in self.arms:
            validate_slug(arm, "arm name")
        return self

    def resolved_spec_path(self) -> Path:
        p = Path(self.spec).expanduser()
        if not p.is_absolute():
            p = (self._base_dir / p).resolve()
        return p


def load_sweep(path: str | Path) -> SweepSpec:
    path = Path(path)
    sweep = SweepSpec(**yaml.safe_load(path.read_text()))
    sweep._base_dir = path.parent.resolve()
    return sweep


def make_tag(model_id: str, arm: str) -> str:
    """Stable per-run tag, e.g. Qwen/Qwen3.5-4B + plain -> qwen3.5-4b-plain.

    Matches the tags the ad-hoc sweep scripts produced, so old artifacts and
    manifests keep the same names. Validated: the tag becomes an artifact
    path component, and a hostile/typo'd model id must fail here.
    """
    return validate_slug(f"{model_id.split('/')[-1]}-{arm}".lower(), "run tag")


def merged_train(
    base_train: dict, model_id: str, model_ovr: dict, arm_ovr: dict
) -> dict:
    """Compose a run's train block: base spec < model override < arm override."""
    return {**base_train, "base": model_id, **model_ovr, **arm_ovr}


@dataclass
class RunPlan:
    tag: str
    model: str
    arm: str
    spec_dict: dict = field(repr=False)


def expand_grid(sweep: SweepSpec, base_spec_path: str | Path) -> list[RunPlan]:
    """Cartesian product of models x arms as validated spec dicts.

    Loads the base spec yaml, rewrites its relative spec_files to absolute
    (temp specs live in a scratch dir where relative paths would break), and
    validates every merged spec through FunctionSpec so an override typo fails
    here — before a single GPU-hour is spent — rather than mid-run.
    """
    base_spec_path = Path(base_spec_path)
    base_yaml = yaml.safe_load(base_spec_path.read_text())
    base_dir = base_spec_path.parent
    base_train = dict(base_yaml.get("train") or {})

    abs_spec_files = []
    for f in base_yaml.get("spec_files") or []:
        p = Path(f).expanduser()
        if not p.is_absolute():
            p = (base_dir / p).resolve()
        abs_spec_files.append(str(p))

    plans: list[RunPlan] = []
    for model_id, model_ovr in sweep.models.items():
        for arm, arm_ovr in sweep.arms.items():
            spec_dict = copy.deepcopy(base_yaml)
            spec_dict["train"] = merged_train(
                base_train, model_id, model_ovr or {}, arm_ovr or {}
            )
            if abs_spec_files:
                spec_dict["spec_files"] = abs_spec_files
            FunctionSpec(**spec_dict)  # raises on unknown/typo'd override keys
            plans.append(RunPlan(make_tag(model_id, arm), model_id, arm, spec_dict))
    return plans


_PROBE = """
import sys
from huggingface_hub import hf_hub_download
bad = []
for m in sys.argv[1:]:
    try:
        hf_hub_download(m, "config.json")
    except Exception as e:
        bad.append(f"{m}: {type(e).__name__}: {e}")
for line in bad:
    print("PREFLIGHT_FAIL:: " + line)
sys.exit(1 if bad else 0)
"""


def preflight(models, env: dict, *, run=subprocess.run) -> list[str]:
    """Cheap existence/access check on every model id before the sweep starts.

    One subprocess downloads (or cache-hits) each model's config.json — instant
    when cached (incl. HF_HUB_OFFLINE=1), seconds otherwise. Catches typo'd,
    gated, or missing ids and anonymous rate-limiting before hour one. Returns
    the list of failure lines (empty == all good).
    """
    proc = run(
        [sys.executable, "-c", _PROBE, *models],
        env=env,
        capture_output=True,
        text=True,
    )
    if proc.returncode == 0:
        return []
    return [
        line[len("PREFLIGHT_FAIL:: ") :]
        for line in (proc.stdout or "").splitlines()
        if line.startswith("PREFLIGHT_FAIL:: ")
    ] or [f"preflight probe failed (exit {proc.returncode}): {proc.stderr[-500:]}"]


def _cell(r: dict) -> str:
    status = r["status"]
    if status in ("pass", "fail"):
        agr = r.get("agreement")
        if agr is None:
            return status.upper()
        ci = r.get("agreement_ci")
        half = f"±{(ci[1] - ci[0]) / 2:.2f}" if ci else ""
        return f"{agr:.2f}{half} {status.upper()}"
    return status.upper()


def render_table(results: list[dict], models: list[str], arms: list[str]) -> str:
    """Markdown table, rows = models, cols = arms."""
    by = {(r["model"], r["arm"]): r for r in results}
    header = "| model | " + " | ".join(arms) + " |"
    sep = "|" + "---|" * (len(arms) + 1)
    lines = [header, sep]
    for m in models:
        cells = [_cell(by[(m, a)]) if (m, a) in by else "-" for a in arms]
        lines.append(f"| {m.split('/')[-1]} | " + " | ".join(cells) + " |")
    return "\n".join(lines)


def run_sweep(
    sweep: SweepSpec,
    *,
    data_dir: str | None,
    artifacts_root: str | Path,
    compile_prefix: list[str] | None = None,
    run=subprocess.run,
    do_preflight: bool = True,
) -> int:
    """Execute every cell of the grid; return 0 iff all runs were honest.

    Each run is an isolated `compile` subprocess (record-failure-and-continue):
    exit 0/2 are honest gate PASS/FAIL, anything else is recorded as error and
    the sweep continues. Streams `MANIFEST::<json>` per successful run, then
    writes results.json + summary.md and prints a table. Exit 1 if any run
    errored or timed out.
    """
    artifacts_root = Path(artifacts_root)
    base_spec_path = sweep.resolved_spec_path()
    plans = expand_grid(sweep, base_spec_path)
    function = plans[0].spec_dict["name"]
    env = {**os.environ, **{k: str(v) for k, v in sweep.env.items()}}

    if do_preflight:
        bad = preflight(list(sweep.models), env, run=run)
        if bad:
            print("preflight failed — aborting before any run:", file=sys.stderr)
            for b in bad:
                print(f"  {b}", file=sys.stderr)
            return 1

    prefix = compile_prefix or [sys.executable, "-m", "smallbatch.cli"]
    sweep_dir = artifacts_root / function / sweep.name
    sweep_dir.mkdir(parents=True, exist_ok=True)
    timeout = sweep.timeout_minutes * 60

    results: list[dict] = []
    with tempfile.TemporaryDirectory() as tmp:
        for plan in plans:
            spec_path = Path(tmp) / f"{plan.tag}.yaml"
            spec_path.write_text(yaml.safe_dump(plan.spec_dict, sort_keys=False))
            argv = [
                *prefix, "compile", str(spec_path),
                *(["--data", data_dir] if data_dir else []),
                "--artifacts", str(artifacts_root),
                "--sweep-name", sweep.name,
                "--tag", plan.tag,
                "--arm", plan.arm,
            ]
            rec = {"tag": plan.tag, "model": plan.model, "arm": plan.arm}
            try:
                proc = run(argv, env=env, capture_output=True, text=True, timeout=timeout)
            except subprocess.TimeoutExpired:
                rec["status"] = "timeout"
                rec["error"] = f"exceeded {sweep.timeout_minutes} min"
                results.append(rec)
                print(f"TIMEOUT:: {plan.tag}", flush=True)
                continue

            if proc.returncode in (0, 2):
                # the compile subprocess created + populated this dir; only read
                # it (sweep_run_dir is destructive, so don't call it here)
                run_dir = artifacts_root / function / sweep.name / plan.tag
                manifest = artifacts.read_manifest(run_dir)
                rec["status"] = "pass" if proc.returncode == 0 else "fail"
                # sweep cells compare adapters; the winner's metrics are used
                # when the lora candidate errored (rare inside a sweep). Full
                # metric dicts stay in results.json via the manifest line.
                adapter = (
                    manifest["metrics"].get("adapter")
                    or (artifacts.candidate_record(manifest) or {}).get("metrics")
                    or {}
                )
                rec["agreement"] = adapter.get("agreement")
                rec["agreement_ci"] = adapter.get("agreement_ci")
                rec["metrics"] = {k: v for k, v in adapter.items() if k != "preds"}
                rec["best_epoch"] = manifest.get("best_epoch")
                rec["epochs_run"] = manifest.get("epochs_run")
                rec["zeroshot"] = (manifest["metrics"].get("zeroshot") or {}).get(
                    "agreement"
                )
                rec["gate_reasons"] = manifest["gate"].get("reasons", [])
                print(
                    "MANIFEST::" + json.dumps(manifest, separators=(",", ":")),
                    flush=True,
                )
            else:
                rec["status"] = "error"
                rec["error"] = (proc.stderr or "")[-2500:]
                print(f"ERROR:: {plan.tag} (exit {proc.returncode})", flush=True)
            results.append(rec)

    (sweep_dir / "results.json").write_text(
        json.dumps({"sweep": sweep.name, "function": function, "runs": results}, indent=2)
    )
    table = render_table(results, list(sweep.models), list(sweep.arms))
    summary = f"# Sweep `{sweep.name}` ({function})\n\n{table}\n"
    (sweep_dir / "summary.md").write_text(summary)
    print(table, flush=True)

    return 0 if all(r["status"] in ("pass", "fail") for r in results) else 1
