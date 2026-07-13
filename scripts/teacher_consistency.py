"""Measure the teacher's self-agreement: relabel a function's real items with
the same teacher/prompt and compare to the stored labels. The result is the
ceiling any student can be expected to hit — the gate bar should sit below it.

Usage: python scripts/teacher_consistency.py <spec.yaml> [--data data/<name>]
"""

import argparse
import datetime
import json
import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parents[1] / "src"))

from smallbatch.evaluate import pearson_r
from smallbatch.labeling import label_items, read_jsonl
from smallbatch.spec import load_spec
from smallbatch.teacher import make_teacher


def main() -> int:
    ap = argparse.ArgumentParser()
    ap.add_argument("spec")
    ap.add_argument("--data", default=None)
    args = ap.parse_args()

    spec = load_spec(args.spec)
    data_dir = Path(args.data or f"data/{spec.name}")
    original = [r for r in read_jsonl(data_dir / "labeled.jsonl") if r["origin"] == "real"]

    teacher = make_teacher(spec.teacher)
    relabeled = label_items(teacher, spec, [r["input"] for r in original], origin="relabel")

    # align by serialized input (label_items drops items that fail twice)
    def key(row):
        return json.dumps(row["input"], sort_keys=True)

    second = {key(r): r["score"] for r in relabeled}
    pairs = [(r["score"], second[key(r)]) for r in original if key(r) in second]

    n = len(pairs)
    exact = sum(1 for a, b in pairs if a == b)
    within1 = sum(1 for a, b in pairs if abs(a - b) <= 1)
    r = pearson_r([a for a, _ in pairs], [b for _, b in pairs])
    diffs: dict[int, int] = {}
    for a, b in pairs:
        diffs[b - a] = diffs.get(b - a, 0) + 1

    out = {
        "n": n,
        "exact": round(exact / n, 4),
        "within1": round(within1 / n, 4),
        "pearson_r": round(r, 4) if r is not None else None,
        "diff_histogram": {str(k): diffs[k] for k in sorted(diffs)},
        "teacher_model": spec.teacher.model,
        "measured_at": datetime.date.today().isoformat(),
    }
    (data_dir / "teacher_consistency.json").write_text(json.dumps(out, indent=2))
    print(json.dumps(out, indent=2))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
