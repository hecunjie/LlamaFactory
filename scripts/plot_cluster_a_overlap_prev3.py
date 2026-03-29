#!/usr/bin/env python3
# Copyright 2025 the LlamaFactory team.
#
# Licensed under the Apache License, Version 2.0.

r"""Overlap rate: cluster-A positions in each step file vs union of the **previous three** step files.

For jsonl files named like ``*_trainer_step_<N>.jsonl`` (from
``logits_analysis_eval_save_cluster_a_detail``), files are sorted by ``N`` ascending.
For file at index ``i`` (0-based), let ``P = F[i-1] ∪ F[i-2] ∪ F[i-3]`` (existing indices only).

* overlap_rate(i) = |positions(F[i]) ∩ P| / |positions(F[i])|  (empty → NaN)

The script computes this for every file that has at least one preceding file, then plots
``trainer_step`` vs overlap rate (chronological order on x-axis).

Example::

    python scripts/plot_cluster_a_overlap_prev3.py \\
        --input_dir ./analysis_logs/eval/cluster_a_positions \\
        --output ./figures/cluster_a_overlap.png
"""

from __future__ import annotations

import argparse
import glob as glob_mod
import json
import re
from pathlib import Path
from typing import Any


def _load_position_keys(path: Path) -> set[tuple[str, int, int]]:
    keys: set[tuple[str, int, int]] = set()
    with path.open(encoding="utf-8") as f:
        for line in f:
            line = line.strip()
            if not line:
                continue
            try:
                r = json.loads(line)
            except json.JSONDecodeError:
                continue
            if r.get("analysis") != "eval_cluster_a_position":
                continue
            pfx = str(r.get("metric_key_prefix") or "")
            keys.add((pfx, int(r["sample_id"]), int(r["seq_pos"])))
    return keys


_STEP_RE = re.compile(r"_trainer_step_(\d+)\.jsonl$", re.IGNORECASE)


def _parse_trainer_step(path: Path) -> int | None:
    m = _STEP_RE.search(path.name)
    if m:
        return int(m.group(1))
    return None


def _group_files_by_prefix(paths: list[Path]) -> dict[str, list[Path]]:
    r"""Group ``*_trainer_step_N.jsonl`` by stem prefix before ``_trainer_step_``."""
    groups: dict[str, list[Path]] = {}
    for p in paths:
        name = p.name
        idx = name.find("_trainer_step_")
        if idx < 0:
            continue
        prefix = name[:idx]
        step = _parse_trainer_step(p)
        if step is None:
            continue
        groups.setdefault(prefix, []).append(p)
    for pref in groups:
        groups[pref].sort(key=lambda x: (_parse_trainer_step(x) or 0, str(x)))
    return groups


def _iter_jsonl_files(input_dir: Path | None, glob_pat: str | None) -> list[Path]:
    if glob_pat:
        matches = glob_mod.glob(glob_pat, recursive=True)
        return sorted(Path(p).resolve() for p in matches if Path(p).suffix == ".jsonl")
    if input_dir is None:
        raise SystemExit("Provide --input_dir or --glob")
    if not input_dir.is_dir():
        raise SystemExit(f"Not a directory: {input_dir}")
    return sorted(input_dir.rglob("*.jsonl"))


def compute_overlap_series(files_sorted: list[Path]) -> list[tuple[int, float, int, int]]:
    r"""Return list of (trainer_step, rate, n_curr, n_overlap). Skips files with no predecessors."""
    out: list[tuple[int, float, int, int]] = []
    n = len(files_sorted)
    # Cache loaded sets per index
    sets: list[set[tuple[str, int, int]] | None] = [None] * n
    for i in range(n):
        if sets[i] is None:
            sets[i] = _load_position_keys(files_sorted[i])
    for i in range(n):
        if i == 0:
            continue
        # union of up to three immediately preceding files
        start = max(0, i - 3)
        prev_union: set[tuple[str, int, int]] = set()
        for j in range(start, i):
            prev_union |= sets[j]  # type: ignore[operator]
        curr = sets[i]  # type: ignore[assignment]
        if not curr:
            step = _parse_trainer_step(files_sorted[i]) or -1
            out.append((step, float("nan"), 0, 0))
            continue
        inter = len(curr & prev_union)
        rate = inter / len(curr)
        step = _parse_trainer_step(files_sorted[i]) or -1
        out.append((step, rate, len(curr), inter))
    return out


def main() -> None:
    parser = argparse.ArgumentParser(
        description="Plot overlap rate of cluster-A positions vs union of previous 3 step files."
    )
    parser.add_argument("--input_dir", type=str, default=None)
    parser.add_argument("--glob", dest="glob_pat", type=str, default=None)
    parser.add_argument(
        "--prefix",
        type=str,
        default=None,
        help="Only use filename group whose stem starts with this (e.g. eval_logits_analysis_eval_mine).",
    )
    parser.add_argument("--output", type=str, required=True, help="Output PNG path.")
    parser.add_argument(
        "--newest_left",
        action="store_true",
        help="Plot with larger trainer_step on the left (reverse chronological on x-axis).",
    )
    parser.add_argument("--no_annotate", action="store_true", help="Do not draw n_overlap/n_curr labels on points.")
    parser.add_argument("--dpi", type=int, default=150)
    parser.add_argument(
        "--json_out",
        type=str,
        default=None,
        help="Optional path to save per-step stats as JSON.",
    )
    args = parser.parse_args()

    input_dir = Path(args.input_dir).resolve() if args.input_dir else None
    all_files = _iter_jsonl_files(input_dir, args.glob_pat)
    if not all_files:
        raise SystemExit("No jsonl files found.")

    groups = _group_files_by_prefix(all_files)
    if args.prefix:
        keys = [k for k in groups if k == args.prefix or k.startswith(args.prefix)]
        if len(keys) != 1:
            raise SystemExit(f"Expected exactly one prefix match for --prefix {args.prefix!r}, got {keys}")
        use_files = groups[keys[0]]
        group_name = keys[0]
    else:
        if len(groups) != 1:
            raise SystemExit(
                "Multiple filename prefixes found; pass --prefix to choose one: " + ", ".join(sorted(groups))
            )
        group_name = next(iter(groups))
        use_files = groups[group_name]

    if len(use_files) < 2:
        raise SystemExit("Need at least two trainer_step jsonl files to compute overlap.")

    series = compute_overlap_series(use_files)
    # First row corresponds to i=1 (second file); include step labels
    import matplotlib

    matplotlib.use("Agg")
    import matplotlib.pyplot as plt

    steps = [t[0] for t in series]
    rates = [t[1] for t in series]
    n_curr = [t[2] for t in series]
    n_ovl = [t[3] for t in series]

    fig, ax = plt.subplots(figsize=(9, 5))
    xs, ys = steps, rates
    ann_nc, ann_no = n_curr, n_ovl
    if args.newest_left:
        pairs = sorted(zip(steps, rates, n_curr, n_ovl), key=lambda t: t[0], reverse=True)
        xs = [p[0] for p in pairs]
        ys = [p[1] for p in pairs]
        ann_nc = [p[2] for p in pairs]
        ann_no = [p[3] for p in pairs]
    ax.plot(xs, ys, marker="o", linewidth=1.2, markersize=5, color="C0")
    ax.set_xlabel("trainer_step (current file)")
    ax.set_ylabel("overlap rate vs ∪(previous ≤3 files)")
    ax.set_ylim(-0.05, 1.05)
    ax.grid(True, alpha=0.3)
    ax.set_title(f"cluster-A overlap (prev 3 files)\n{group_name}  (n_files={len(use_files)})")
    if not args.no_annotate:
        for x, r, nc, no in zip(xs, ys, ann_nc, ann_no):
            if nc > 0 and r == r:
                ax.annotate(f"{no}/{nc}", (x, r), textcoords="offset points", xytext=(0, 6), ha="center", fontsize=7)
    fig.tight_layout()
    outp = Path(args.output).resolve()
    outp.parent.mkdir(parents=True, exist_ok=True)
    fig.savefig(str(outp), format="png", dpi=args.dpi, bbox_inches="tight")
    plt.close(fig)
    print(outp)

    if args.json_out:
        payload: dict[str, Any] = {
            "prefix": group_name,
            "files": [str(p) for p in use_files],
            "points": [
                {
                    "trainer_step": s,
                    "overlap_rate": None if (isinstance(r, float) and r != r) else r,
                    "n_positions_curr": nc,
                    "n_overlap": no,
                }
                for s, r, nc, no in zip(steps, rates, n_curr, n_ovl)
            ],
        }
        jp = Path(args.json_out).resolve()
        jp.parent.mkdir(parents=True, exist_ok=True)
        jp.write_text(json.dumps(payload, ensure_ascii=False, indent=2), encoding="utf-8")
        print(jp)


if __name__ == "__main__":
    main()
