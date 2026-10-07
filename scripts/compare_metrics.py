"""Fail (exit 1) if a metric dropped more than allowed between two metrics.json files."""

from __future__ import annotations

import argparse
import json
import sys
from pathlib import Path

KEYS = ["roc_auc", "pr_auc", "ks", "brier", "expected_cost", "recall_at_1pct_fpr"]


def main() -> int:
    ap = argparse.ArgumentParser()
    ap.add_argument("--base", type=Path, required=True)
    ap.add_argument("--head", type=Path, required=True)
    ap.add_argument("--metric", default="roc_auc")
    ap.add_argument("--max-drop", type=float, default=0.01)
    ap.add_argument("--summary", type=Path)
    a = ap.parse_args()

    base = json.loads(a.base.read_text())
    head = json.loads(a.head.read_text())
    drop = base[a.metric] - head[a.metric]
    ok = drop <= a.max_drop
    lines = [
        f"### ML quality gate ({a.head.parent.parent.name}): {'PASS' if ok else 'FAIL'}",
        "",
        "| metric | main | PR | delta |",
        "|---|---|---|---|",
    ]
    for k in KEYS:
        if k in base and k in head:
            lines.append(f"| {k} | {base[k]:.4f} | {head[k]:.4f} | {head[k] - base[k]:+.4f} |")
    lines += ["", f"Rule: {a.metric} may drop at most {a.max_drop} (dropped {drop:+.4f})."]
    text = "\n".join(lines)
    print(text)
    if a.summary:
        with a.summary.open("a", encoding="utf-8") as fh:
            fh.write(text + "\n\n")
    return 0 if ok else 1


if __name__ == "__main__":
    sys.exit(main())
