#!/usr/bin/env python3
# Copyright 2025 the LlamaFactory team.
#
# Licensed under the Apache License, Version 2.0.

r"""Aggregate **eval cluster-A position** jsonl lines across trainer steps.

Input lines are produced when ``logits_analysis_eval_save_cluster_a_detail`` is True (see
``logits_analysis_callback._append_eval_cluster_a_positions_jsonl``). Each line tags one
supervised position classified as cluster A at one eval batch.

For each logical key ``(metric_key_prefix, sample_id, seq_pos)`` this script reports:

- ``n_lines``: total jsonl lines (eval batches × recurring A hits).
- ``n_distinct_trainer_steps``: how many distinct ``trainer_step`` values appear (training
  checkpoints at which eval was run and this position was A).

Example::

    python scripts/stat_eval_cluster_a_positions.py --input_dir ./analysis_logs/eval/cluster_a_positions

    python scripts/stat_eval_cluster_a_positions.py \\
        --glob "./runs/**/eval/cluster_a_positions/*.jsonl" --top 80 --output_csv summary.csv
"""

from __future__ import annotations

import argparse
import csv
import glob as glob_mod
import json
from pathlib import Path


def _iter_jsonl_files(input_dir: Path | None, glob_pat: str | None) -> list[Path]:
    if glob_pat:
        matches = glob_mod.glob(glob_pat, recursive=True)
        return sorted(Path(p).resolve() for p in matches if Path(p).suffix == ".jsonl")
    if input_dir is None:
        raise SystemExit("Provide --input_dir or --glob")
    if not input_dir.is_dir():
        raise SystemExit(f"Not a directory: {input_dir}")
    return sorted(input_dir.rglob("*.jsonl"))


def main() -> None:
    parser = argparse.ArgumentParser(
        description="Count recurrence of (metric_key_prefix, sample_id, seq_pos) for eval cluster-A positions."
    )
    parser.add_argument(
        "--input_dir",
        type=str,
        default=None,
        help="Directory to scan recursively for *.jsonl (cluster_a_positions output).",
    )
    parser.add_argument(
        "--glob",
        dest="glob_pat",
        type=str,
        default=None,
        help="Alternative: glob pattern for jsonl files (e.g. ./analysis_logs/eval/cluster_a_positions/*.jsonl).",
    )
    parser.add_argument("--top", type=int, default=100, help="Print only top N keys by n_lines (default: 100).")
    parser.add_argument(
        "--min_lines",
        type=int,
        default=1,
        help="Only include keys with at least this many lines (default: 1).",
    )
    parser.add_argument(
        "--output_csv",
        type=str,
        default=None,
        help="Optional path to write a CSV with all aggregated rows.",
    )
    args = parser.parse_args()

    input_dir = Path(args.input_dir).resolve() if args.input_dir else None
    files = _iter_jsonl_files(input_dir, args.glob_pat)
    if not files:
        raise SystemExit("No jsonl files found.")

    # key -> { "n_lines": int, "steps": set[int] }
    agg: dict[tuple[str, int, int], dict[str, Any]] = {}

    def _get_key(r: dict[str, Any]) -> tuple[str, int, int]:
        pfx = str(r.get("metric_key_prefix") or "")
        sid = int(r["sample_id"])
        sp = int(r["seq_pos"])
        return (pfx, sid, sp)

    n_read = 0
    n_skipped = 0
    for fp in files:
        with fp.open(encoding="utf-8") as f:
            for line in f:
                line = line.strip()
                if not line:
                    continue
                n_read += 1
                try:
                    r = json.loads(line)
                except json.JSONDecodeError:
                    n_skipped += 1
                    continue
                if r.get("analysis") != "eval_cluster_a_position":
                    n_skipped += 1
                    continue
                key = _get_key(r)
                step = int(r.get("trainer_step", -1))
                if key not in agg:
                    agg[key] = {"n_lines": 0, "steps": set()}
                agg[key]["n_lines"] += 1
                if step >= 0:
                    agg[key]["steps"].add(step)

    rows_out: list[dict[str, Any]] = []
    for (pfx, sid, sp), v in agg.items():
        nl = v["n_lines"]
        if nl < args.min_lines:
            continue
        nd = len(v["steps"])
        rows_out.append(
            {
                "metric_key_prefix": pfx,
                "sample_id": sid,
                "seq_pos": sp,
                "n_lines": nl,
                "n_distinct_trainer_steps": nd,
            }
        )

    rows_out.sort(key=lambda x: (-x["n_lines"], -x["n_distinct_trainer_steps"], x["sample_id"], x["seq_pos"]))

    print(f"files={len(files)} jsonl_lines_read={n_read} skipped_non_a={n_skipped} unique_keys={len(rows_out)}")
    top = rows_out[: max(0, args.top)]
    for i, row in enumerate(top, 1):
        print(
            f"{i:4d}  prefix={row['metric_key_prefix']!r}  sample_id={row['sample_id']}  "
            f"seq_pos={row['seq_pos']}  n_lines={row['n_lines']}  n_steps={row['n_distinct_trainer_steps']}"
        )

    if args.output_csv:
        outp = Path(args.output_csv).resolve()
        outp.parent.mkdir(parents=True, exist_ok=True)
        with outp.open("w", encoding="utf-8", newline="") as f:
            w = csv.DictWriter(
                f,
                fieldnames=[
                    "metric_key_prefix",
                    "sample_id",
                    "seq_pos",
                    "n_lines",
                    "n_distinct_trainer_steps",
                ],
            )
            w.writeheader()
            w.writerows(rows_out)
        print(outp)


if __name__ == "__main__":
    main()
