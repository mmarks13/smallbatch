"""Smallbatch command-line interface."""

from __future__ import annotations

import argparse
import json
import sys
from pathlib import Path

from . import artifacts
from .spec import load_spec


def _load_items(path: Path) -> list[dict]:
    text = path.read_text()
    if path.suffix == ".jsonl":
        return [json.loads(line) for line in text.splitlines() if line.strip()]
    value = json.loads(text)
    if isinstance(value, dict) and isinstance(value.get("items"), list):
        return value["items"]
    if not isinstance(value, list):
        raise ValueError(f"{path} must contain a JSON list or JSONL records")
    return value


def cmd_init(args) -> int:
    from .init_cmd import init

    try:
        directory = init(args.template, args.name, directory=args.dir)
    except (ValueError, FileExistsError) as exc:
        print(f"error: {exc}", file=sys.stderr)
        return 1
    print(f"created {directory}/spec.yaml and {directory}/items.json")
    return 0


def cmd_doctor(args) -> int:
    from .doctor import run_doctor

    try:
        spec = load_spec(args.spec)
        items = _load_items(Path(args.items)) if args.items else None
        return run_doctor(
            spec,
            items=items,
            data_dir=Path(args.data or f"data/{spec.name}"),
            probe=not args.no_probe,
        )
    except (ValueError, FileNotFoundError) as exc:
        print(f"error: {exc}", file=sys.stderr)
        return 1


def cmd_label(args) -> int:
    from .api import label
    from .calibration import CalibrationDeclined

    try:
        result = label(
            args.spec,
            _load_items(Path(args.items)),
            out_dir=args.out,
            append=args.append,
            max_variants=args.max_variants,
            skip_calibration=args.skip_calibration,
        )
    except CalibrationDeclined as exc:
        print(str(exc))
        return 0
    except (ValueError, FileNotFoundError) as exc:
        print(f"error: {exc}", file=sys.stderr)
        return 1
    print(json.dumps(result.meta, indent=2))
    return 0


def cmd_compile(args) -> int:
    from .api import compile as compile_fn

    try:
        result = compile_fn(
            args.spec,
            data_dir=args.data,
            artifacts_root=args.artifacts,
            cpu_threads=args.cpu_threads,
        )
    except (ValueError, FileNotFoundError) as exc:
        print(f"error: {exc}", file=sys.stderr)
        return 1
    for name, record in result.candidates.items():
        if record.get("status") == "completed":
            print(f"{name}: completed")
        else:
            print(f"{name}: ERROR - {record.get('error')}")
    print(f"report: {result.report_path}")
    print("No candidate was selected. Use `smallbatch select` after reviewing the report.")
    return 0


def cmd_select(args) -> int:
    root = Path(args.artifacts)
    if args.clear:
        event = artifacts.clear_active(root, args.name)
        print(json.dumps(event, indent=2))
        return 0
    if not args.candidate:
        print("error: select requires a candidate or --clear", file=sys.stderr)
        return 1
    from .api import select

    try:
        result = select(
            args.name,
            args.candidate,
            version=args.version,
            artifacts_root=root,
            accept_package_drift=args.accept_package_drift,
        )
    except (ValueError, FileNotFoundError, RuntimeError) as exc:
        print(f"error: {exc}", file=sys.stderr)
        return 1
    print(f"selected {result.candidate} from {result.build_id}")
    print(f"source: {result.package_dir / 'source'}")
    print(f"wheel: {result.wheel}")
    print(
        f"import: from smallbatch_functions.{args.name.replace('-', '_')} "
        "import run"
    )
    return 0


def cmd_run(args) -> int:
    from .runtime import load_fn

    try:
        function = load_fn(
            args.name,
            artifacts_root=args.artifacts,
            version=args.version,
            candidate=args.candidate,
        )
        if args.json:
            print(json.dumps(function(json.loads(args.json)), ensure_ascii=False))
        else:
            records = _load_items(Path(args.input_file))
            inputs = [record["input"] if "input" in record else record for record in records]
            for item, output in zip(inputs, function.batch(inputs)):
                print(json.dumps({"input": item, "output": output}, ensure_ascii=False))
    except (ValueError, FileNotFoundError) as exc:
        print(f"error: {exc}", file=sys.stderr)
        return 1
    return 0


def cmd_status(args) -> int:
    root = Path(args.artifacts)
    if not root.is_dir():
        print(f"no artifacts under {root}")
        return 0
    for function_dir in sorted(path for path in root.iterdir() if path.is_dir()):
        active = artifacts.read_active(root, function_dir.name)
        print(f"{function_dir.name}: active={active['candidate'] if active else 'none'}")
        for build in artifacts.versions(root, function_dir.name):
            try:
                manifest = artifacts.read_manifest(build)
                broken = artifacts.artifact_integrity(build)
                drift = artifacts.source_drift(build)
                flags = []
                if broken:
                    flags.append(f"INTEGRITY: {broken}")
                if drift and drift != artifacts.SOURCE_UNAVAILABLE:
                    flags.append(f"SOURCE DRIFT: {drift}")
                summary = ", ".join(
                    f"{name}={record.get('status')}"
                    for name, record in manifest["candidates"].items()
                )
                print(f"  {build.name}: {summary}" + (f" [{'; '.join(flags)}]" if flags else ""))
            except (ValueError, FileNotFoundError) as exc:
                print(f"  {build.name}: ERROR {exc}")
    return 0


def main(argv: list[str] | None = None) -> int:
    parser = argparse.ArgumentParser(prog="smallbatch")
    sub = parser.add_subparsers(dest="cmd", required=True)

    init_parser = sub.add_parser("init", help="create a prompt-first function project")
    init_parser.add_argument("template", choices=["classifier", "scorer", "structured"])
    init_parser.add_argument("name")
    init_parser.add_argument("--dir")
    init_parser.set_defaults(fn=cmd_init)

    doctor = sub.add_parser("doctor", help="validate a spec, data, teacher, and hardware")
    doctor.add_argument("spec")
    doctor.add_argument("--items")
    doctor.add_argument("--data")
    doctor.add_argument("--no-probe", action="store_true")
    doctor.set_defaults(fn=cmd_doctor)

    label_parser = sub.add_parser("label", help="import or generate decisions")
    label_parser.add_argument("spec")
    label_parser.add_argument("--items", required=True)
    label_parser.add_argument("--out")
    label_parser.add_argument("--append", action="store_true")
    label_parser.add_argument("--max-variants", type=int)
    label_parser.add_argument("--skip-calibration", action="store_true")
    label_parser.set_defaults(fn=cmd_label)

    compile_parser = sub.add_parser("compile", help="build and compare CPU candidates")
    compile_parser.add_argument("spec")
    compile_parser.add_argument("--data")
    compile_parser.add_argument("--artifacts", default=str(artifacts.DEFAULT_ROOT))
    compile_parser.add_argument("--cpu-threads", type=int)
    compile_parser.set_defaults(fn=cmd_compile)

    select_parser = sub.add_parser("select", help="package and activate one candidate")
    select_parser.add_argument("name")
    select_parser.add_argument("candidate", nargs="?")
    select_parser.add_argument("--version")
    select_parser.add_argument("--clear", action="store_true")
    select_parser.add_argument("--accept-package-drift", action="store_true")
    select_parser.add_argument("--artifacts", default=str(artifacts.DEFAULT_ROOT))
    select_parser.set_defaults(fn=cmd_select)

    run_parser = sub.add_parser("run", help="call an active or explicit candidate")
    run_parser.add_argument("name")
    run_parser.add_argument("--json")
    run_parser.add_argument("--input-file")
    run_parser.add_argument("--version")
    run_parser.add_argument("--candidate")
    run_parser.add_argument("--artifacts", default=str(artifacts.DEFAULT_ROOT))
    run_parser.set_defaults(fn=cmd_run)

    status = sub.add_parser("status", help="show builds, candidates, and active selections")
    status.add_argument("--artifacts", default=str(artifacts.DEFAULT_ROOT))
    status.set_defaults(fn=cmd_status)

    args = parser.parse_args(argv)
    if args.cmd == "run" and not (args.json or args.input_file):
        parser.error("run requires --json or --input-file")
    if args.cmd == "run" and bool(args.version) != bool(args.candidate):
        parser.error("explicit run requires both --version and --candidate")
    if args.cmd == "select" and args.clear and args.candidate:
        parser.error("select accepts either a candidate or --clear")
    return args.fn(args)


if __name__ == "__main__":
    raise SystemExit(main())
