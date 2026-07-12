"""smallbatch CLI: label, compile, run, status."""

from __future__ import annotations

import argparse
import json
import os
import sys
from pathlib import Path

from . import artifacts
from .spec import load_spec


def _load_items(path: Path) -> list[dict]:
    text = path.read_text()
    if path.suffix == ".jsonl":
        return [json.loads(l) for l in text.splitlines() if l.strip()]
    data = json.loads(text)
    if isinstance(data, dict):  # tolerate {"items": [...]}-shaped files
        for v in data.values():
            if isinstance(v, list):
                return v
        raise ValueError(f"no item list found in {path}")
    return data


def cmd_label(args) -> int:
    from .api import label

    spec = load_spec(args.spec)
    items = _load_items(Path(args.items))
    result = label(
        spec, items, out_dir=args.out,
        append=args.append, max_variants=args.max_variants,
    )
    print(json.dumps(result.meta, indent=2))
    if result.compressed:
        hist = result.meta["label_histogram"]
        top_share = max(hist.values()) / sum(hist.values())
        print(
            f"note: labels are compressed ({top_share:.0%} in one bin) — "
            "consider sharpening the rubric anchors and relabeling"
        )
    return 0


def cmd_compile(args) -> int:
    from .api import compile as compile_fn

    spec = load_spec(args.spec)
    try:
        result = compile_fn(
            spec,
            data_dir=args.data,
            artifacts_root=args.artifacts,
            base=args.base,
            precision=args.precision,
            sweep_name=getattr(args, "sweep_name", None),
            tag=getattr(args, "tag", None),
            arm=getattr(args, "arm", None),
            allow_stale_labels=getattr(args, "allow_stale_labels", False),
        )
    except ValueError as e:
        print(f"error: {e}", file=sys.stderr)
        return 1

    # report first: the metrics are the result; the verdict is one line
    h = (result.report or {}).get("headline", {})
    ci = h.get("agreement_ci")
    ci_txt = f" (95% CI {ci[0]:.0%}-{ci[1]:.0%})" if ci else ""
    print(f"agreement {h.get('agreement', 0):.1%}{ci_txt} on {h.get('n', 0)} gate items")
    tr = (result.report or {}).get("training", {})
    if tr.get("best_epoch") is not None:
        print(
            f"best epoch {tr['best_epoch']}/{tr.get('epochs_run')} "
            f"({tr.get('stopped_reason')})"
        )
    if result.report_path:
        print(f"report: {result.report_path}")
    print(json.dumps({"metrics": result.metrics, "gate": result.gate}, indent=2))
    for w in (result.report or {}).get("warnings") or []:
        print(f"warning: {w}", file=sys.stderr)
    print(f"{'PASS' if result.passed else 'FAIL'}: {result.version_dir}")
    if not result.passed:
        _offer_acceptance(args, spec, result)
    return 0 if result.passed else 2


def _offer_acceptance(args, spec, result) -> None:
    """All-candidates-failed flow (CLI only): show the full decision summary
    and let the user explicitly deploy the best candidate. Acceptance never
    rewrites the gate; the exit code stays 2 either way. Never interactive
    inside sweep subprocesses, the Python API, or non-TTY runs."""
    from . import decision

    manifest = result.manifest
    if not decision.all_completed_failed(manifest):
        return
    if getattr(args, "sweep_name", None) is not None:
        return  # sweeps stay non-interactive; cells are research results
    winner = manifest["selection"]["winner"]
    use_flag = getattr(args, "use_best_anyway", False)
    interactive = sys.stdin.isatty() and sys.stdout.isatty()
    if not use_flag and not interactive:
        print(
            f"hint: `smallbatch run {manifest['function']} --allow-failed` uses "
            "the best candidate once; `compile --use-best-anyway` accepts it as "
            "the default"
        )
        return

    root = Path(args.artifacts)
    current = artifacts.latest(root, manifest["function"])
    displaced = (
        current.name if current is not None and current != result.version_dir else None
    )
    print()
    print(decision.build_decision_text(spec, manifest, result.report, displaced))
    if result.report_path:
        print(f"\nDetails: {result.report_path}")
    if use_flag:
        accept, via = True, "flag_use_best_anyway"
        print(f"--use-best-anyway: accepting '{winner}' for deployment")
    else:
        reply = input(f"\nUse {winner} as the default despite the failed gate? [y/N] ")
        accept, via = reply.strip().lower() in ("y", "yes"), "interactive_compile"
    if accept:
        artifacts.accept_candidate(result.version_dir, winner, via)
        print(
            f"{winner} accepted for use.\n"
            "The quality result remains FAIL; run/load will show a warning.\n"
            "Accepted for runtime; gate remains FAIL, so compile exits 2."
        )
    elif displaced:
        print(f"declined — {displaced} remains the deployed default")


def cmd_run(args) -> int:
    from .runtime import load_fn

    try:
        fn = load_fn(
            args.name,
            artifacts_root=args.artifacts,
            allow_failed=args.allow_failed,
            version=args.version,
            candidate=args.candidate,
        )
    except (FileNotFoundError, ValueError) as e:
        print(f"error: {e}", file=sys.stderr)
        return 1
    if args.json:
        print(fn(json.loads(args.json)))
    else:
        items = _load_items(Path(args.input_file))
        for item, out in zip(items, fn.batch(items)):
            print(json.dumps({"output": out, "input": item}, ensure_ascii=False))
    return 0


def cmd_export(args) -> int:
    from .export import export

    try:
        export(
            args.name,
            artifacts_root=args.artifacts,
            version=args.version,
            quant=args.quant,
            adapter_only=args.adapter_only,
            allow_failed=args.allow_failed,
            llama_cpp=args.llama_cpp,
            keep_merged=args.keep_merged,
        )
    except (FileNotFoundError, ValueError, RuntimeError) as e:
        print(f"error: {e}", file=sys.stderr)
        return 1
    return 0


def cmd_push(args) -> int:
    from .hub import push

    try:
        push(
            args.name,
            repo_id=args.repo,
            artifacts_root=args.artifacts,
            version=args.version,
            private=not args.public,
            allow_failed=args.allow_failed,
        )
    except (FileNotFoundError, ValueError, RuntimeError) as e:
        print(f"error: {e}", file=sys.stderr)
        return 1
    return 0


def cmd_sweep(args) -> int:
    import shlex

    from .sweep import load_sweep, run_sweep

    sweep = load_sweep(args.sweep_yaml)
    override = os.environ.get("SMALLBATCH_COMPILE")
    compile_prefix = shlex.split(override) if override else None
    return run_sweep(
        sweep,
        data_dir=args.data,
        artifacts_root=args.artifacts,
        compile_prefix=compile_prefix,
    )


def cmd_init(args) -> int:
    from .init_cmd import init

    try:
        out = init(args.template, args.name, directory=args.dir)
    except (ValueError, FileExistsError) as e:
        print(f"error: {e}", file=sys.stderr)
        return 1
    print(f"created {out}/spec.yaml and {out}/items.json")
    print(f"next: fill in the TODOs, then `smallbatch doctor {out}/spec.yaml --items {out}/items.json`")
    return 0


def cmd_serve(args) -> int:
    from .serve import serve

    try:
        return serve(
            args.name,
            artifacts_root=args.artifacts,
            version=args.version,
            port=args.port,
            llama_server=args.llama_server,
            allow_failed=args.allow_failed,
            candidate=args.candidate,
        )
    except (FileNotFoundError, ValueError) as e:
        print(f"error: {e}", file=sys.stderr)
        return 1


def cmd_doctor(args) -> int:
    from .doctor import run_doctor

    try:
        spec = load_spec(args.spec)
    except ValueError as e:
        print(f"FAIL  spec did not validate:\n{e}", file=sys.stderr)
        return 1
    items = _load_items(Path(args.items)) if args.items else None
    data = Path(args.data) if args.data else Path(f"data/{spec.name}")
    return run_doctor(
        spec,
        items=items,
        data_dir=data if data.is_dir() else None,
        probe=not args.no_probe,
    )


def cmd_review(args) -> int:
    from .review import run_review

    spec = load_spec(args.spec)
    data = Path(args.data or f"data/{spec.name}")
    if not (data / "labeled.jsonl").exists():
        print(f"error: no labeled dataset under {data}", file=sys.stderr)
        return 1
    return run_review(spec, data, args)


def cmd_status(args) -> int:
    root = Path(args.artifacts)
    if not root.is_dir():
        print(f"no artifacts under {root}")
        return 0
    for fn_dir in sorted(root.iterdir()):
        for v in artifacts.versions(root, fn_dir.name):
            m = artifacts.read_manifest(v)
            flags = []
            deployment = m.get("deployment") or {}
            if m["gate"]["passed"]:
                flags.append("PASS")
            elif deployment.get("accepted_despite_gate"):
                flags.append(
                    f"IN USE - GATE FAIL (accepted: {deployment.get('accepted_candidate')})"
                )
            else:
                flags.append("FAIL")
            broken = artifacts.artifact_integrity(v)
            drift = artifacts.source_drift(v)
            if broken:
                flags.append(f"INTEGRITY ({broken})")
            elif drift == artifacts.SOURCE_UNAVAILABLE:
                pass  # immutable snapshot; missing source is not a problem
            elif drift:
                flags.append(f"SOURCE DRIFT ({drift})")
            winner_rec = artifacts.candidate_record(m) or {}
            agr = (winner_rec.get("metrics") or {}).get("agreement")
            agr_txt = f"{agr:.2%}" if agr is not None else "-"
            backend = winner_rec.get("backend", "lora")
            print(
                f"{fn_dir.name}/{v.name}  [{' '.join(flags)}]  "
                f"agreement={agr_txt}  winner={backend}  base={m.get('base_model')}"
            )
        for v in artifacts.sweep_runs(root, fn_dir.name):
            m = artifacts.read_manifest(v)
            gate = "PASS" if m["gate"]["passed"] else "FAIL"
            agr = ((m.get("metrics") or {}).get("adapter") or {}).get("agreement")
            agr_txt = f"{agr:.2%}" if agr is not None else "-"
            print(f"{fn_dir.name}/{m['version']}  [{gate}]  agreement={agr_txt}  base={m.get('base_model')}")
    return 0


def main(argv: list[str] | None = None) -> int:
    p = argparse.ArgumentParser(prog="smallbatch")
    sub = p.add_subparsers(dest="cmd", required=True)

    lp = sub.add_parser("label", help="generate a teacher-labeled dataset for a spec")
    lp.add_argument("spec")
    lp.add_argument("--items", required=True, help="JSON/JSONL file of real input items")
    lp.add_argument("--out", help="output dir (default data/<name>)")
    lp.add_argument(
        "--append", action="store_true",
        help="keep existing rows + split assignments (sticky gate); only label unseen items",
    )
    lp.add_argument(
        "--max-variants", type=int,
        help="global budget: max total new synthetic rows this run across "
        "paraphrase/field-dropout/counterfactual (0 = no synthetic work)",
    )
    lp.set_defaults(fn=cmd_label)

    cp = sub.add_parser("compile", help="train + evaluate + gate an adapter")
    cp.add_argument("spec")
    cp.add_argument("--data", help="labeled data dir (default data/<name>)")
    cp.add_argument("--base", help="override train.base model id")
    cp.add_argument("--precision", choices=["auto", "fp32", "bf16", "qlora"])
    cp.add_argument(
        "--allow-stale-labels",
        action="store_true",
        help="train even though the dataset was labeled under a different "
        "rubric/contract/teacher (recorded in the artifact manifest)",
    )
    cp.add_argument(
        "--use-best-anyway",
        action="store_true",
        help="if every candidate fails the gate, accept the best one as the "
        "deployed default without prompting (gate stays FAIL, exit stays 2)",
    )
    cp.add_argument("--artifacts", default=str(artifacts.DEFAULT_ROOT))
    # sweep-internal: route the artifact into artifacts/<fn>/<sweep>/<tag> and
    # stamp the manifest, instead of a dated version dir (see cmd_sweep)
    cp.add_argument("--sweep-name", help=argparse.SUPPRESS)
    cp.add_argument("--tag", help=argparse.SUPPRESS)
    cp.add_argument("--arm", help=argparse.SUPPRESS)
    cp.set_defaults(fn=cmd_compile)

    rp = sub.add_parser("run", help="call a compiled function")
    rp.add_argument("name")
    rp.add_argument("--json", help="single input item as JSON")
    rp.add_argument("--input-file", help="JSON/JSONL file of items")
    rp.add_argument("--allow-failed", action="store_true")
    rp.add_argument("--version", help="artifact version dir name (default: latest usable)")
    rp.add_argument(
        "--candidate", help="run a specific retained candidate (e.g. lora, tfidf)"
    )
    rp.add_argument("--artifacts", default=str(artifacts.DEFAULT_ROOT))
    rp.set_defaults(fn=cmd_run)

    ep = sub.add_parser(
        "export", help="export a compiled function to GGUF (+ grammar + Modelfile)"
    )
    ep.add_argument("name")
    ep.add_argument("--version", help="artifact version dir name (default: latest passing)")
    ep.add_argument("--quant", default="q4_k_m", choices=["f16", "q8_0", "q4_k_m"])
    ep.add_argument(
        "--adapter-only", action="store_true",
        help="convert just the LoRA for llama-server --lora over a shared base",
    )
    ep.add_argument("--allow-failed", action="store_true")
    ep.add_argument("--llama-cpp", help="llama.cpp checkout dir (or set LLAMA_CPP_DIR)")
    ep.add_argument("--keep-merged", action="store_true", help=argparse.SUPPRESS)
    ep.add_argument("--artifacts", default=str(artifacts.DEFAULT_ROOT))
    ep.set_defaults(fn=cmd_export)

    pp = sub.add_parser(
        "push", help="upload an artifact to the Hugging Face Hub (private by default)"
    )
    pp.add_argument("name")
    pp.add_argument("--repo", required=True, help="Hub repo id, e.g. you/fn-name")
    pp.add_argument("--version", help="artifact version dir name (default: latest passing)")
    pp.add_argument("--public", action="store_true", help="create the repo public")
    pp.add_argument("--allow-failed", action="store_true")
    pp.add_argument("--artifacts", default=str(artifacts.DEFAULT_ROOT))
    pp.set_defaults(fn=cmd_push)

    wp = sub.add_parser("sweep", help="run a grid of model x arm compiles")
    wp.add_argument("sweep_yaml")
    wp.add_argument("--data", help="labeled data dir (default data/<name>)")
    wp.add_argument("--artifacts", default=str(artifacts.DEFAULT_ROOT))
    wp.set_defaults(fn=cmd_sweep)

    ip = sub.add_parser("init", help="create a starter spec.yaml + items.json from a template")
    ip.add_argument("template", choices=["classifier", "scorer", "structured"])
    ip.add_argument("name", help="function name (also the output directory)")
    ip.add_argument("--dir", help="output directory (default: ./<name>)")
    ip.set_defaults(fn=cmd_init)

    vs = sub.add_parser(
        "serve", help="serve a compiled function over HTTP (llama-server + validation)"
    )
    vs.add_argument("name")
    vs.add_argument("--port", type=int, default=8080)
    vs.add_argument("--version", help="artifact version dir name (default: latest passing)")
    vs.add_argument("--llama-server", help="path to the llama-server binary")
    vs.add_argument(
        "--candidate", help="serve a specific retained candidate (e.g. lora, tfidf)"
    )
    vs.add_argument("--allow-failed", action="store_true")
    vs.add_argument("--artifacts", default=str(artifacts.DEFAULT_ROOT))
    vs.set_defaults(fn=cmd_serve)

    dp = sub.add_parser(
        "doctor", help="preflight a spec: contract, teacher, data, hardware, disk, export"
    )
    dp.add_argument("spec")
    dp.add_argument("--items", help="JSON/JSONL items file to check against the spec")
    dp.add_argument("--data", help="labeled data dir (default data/<name>)")
    dp.add_argument(
        "--no-probe", action="store_true",
        help="skip the single live teacher call (reachability/parse check)",
    )
    dp.set_defaults(fn=cmd_doctor)

    vp = sub.add_parser(
        "review", help="step through teacher labels: accept/reject/edit before training"
    )
    vp.add_argument("spec")
    vp.add_argument("--data", help="labeled data dir (default data/<name>)")
    vp.add_argument("--split", choices=["train", "dev", "gate"])
    vp.add_argument(
        "--origin", choices=["real", "variant", "dropout", "counterfactual"]
    )
    vp.add_argument("--label", help="filter by (primary) label/score value")
    vp.add_argument("--field", help="structured outputs: filter by field, e.g. reason or reason=outage")
    vp.add_argument(
        "--unstable", action="store_true",
        help="only rows where the teacher's self-consistency probe disagreed",
    )
    vp.add_argument(
        "--status", default="unreviewed",
        choices=["unreviewed", "accepted", "rejected", "edited", "all"],
    )
    vp.set_defaults(fn=cmd_review)

    sp = sub.add_parser("status", help="list compiled functions and staleness")
    sp.add_argument("--artifacts", default=str(artifacts.DEFAULT_ROOT))
    sp.set_defaults(fn=cmd_status)

    args = p.parse_args(argv)
    if args.cmd == "run" and not (args.json or args.input_file):
        p.error("run requires --json or --input-file")
    return args.fn(args)


if __name__ == "__main__":
    raise SystemExit(main())
