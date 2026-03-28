#!/usr/bin/env python3
# Copyright 2025 the LlamaFactory team.
#
# Licensed under the Apache License, Version 2.0.

r"""Merge all logits-analysis jsonl under a directory and plot metrics vs ``step``.

Each file may contain only one JSON line. All lines are merged, sorted by ``step``.
**Each subplot is saved as its own PNG** (6 files by default).

Example::

    python scripts/plot_logits_analysis_jsonl.py --input_dir ./analysis_logs

    # Custom directory and file name prefix
    python scripts/plot_logits_analysis_jsonl.py --input_dir ./analysis_logs \\
        --output_dir ./figures --basename run1_curves

    # Legacy: --output path/to/name.png → saves as path/to/name_01_*.png ...
    python scripts/plot_logits_analysis_jsonl.py --input_dir ./analysis_logs \\
        --output ./analysis_plots/curves.png
"""

from __future__ import annotations

import argparse
import json
import warnings
from pathlib import Path
from typing import Any, Optional


def _get(d: dict[str, Any], *keys: str, default: Any = None) -> Any:
    cur: Any = d
    for k in keys:
        if not isinstance(cur, dict):
            return default
        cur = cur.get(k)
        if cur is None:
            return default
    return cur


def _load_all_records(input_dir: Path, pattern: str) -> list[dict[str, Any]]:
    r"""Read every non-empty line from every matching jsonl file."""
    files = sorted(input_dir.glob(pattern))
    if not files:
        return []
    rows: list[dict[str, Any]] = []
    for fp in files:
        with fp.open(encoding="utf-8") as f:
            for line in f:
                line = line.strip()
                if not line:
                    continue
                rows.append(json.loads(line))
    return rows


def _merge_sort_by_step(rows: list[dict[str, Any]]) -> list[dict[str, Any]]:
    r"""Sort by ``step`` ascending; duplicate ``step`` keeps the **last** record (with warning)."""
    by_step: dict[int, dict[str, Any]] = {}
    dup = 0
    for r in rows:
        s = int(r.get("step", -1))
        if s in by_step:
            dup += 1
        by_step[s] = r
    if dup:
        warnings.warn(
            f"Merged {dup} duplicate step value(s); kept last occurrence per step.",
            stacklevel=2,
        )
    return [by_step[k] for k in sorted(by_step.keys())]


def _series_masked(rows: list[dict[str, Any]], sub: str, stat: str) -> list[Optional[float]]:
    out: list[Optional[float]] = []
    for r in rows:
        v = _get(r, "masked_summary", sub, stat)
        out.append(float(v) if v is not None else None)
    return out


def _series_group(rows: list[dict[str, Any]], gid: str, sub: str, stat: str) -> list[Optional[float]]:
    out: list[Optional[float]] = []
    for r in rows:
        g = _get(r, "groups", gid)
        if not isinstance(g, dict):
            out.append(None)
            continue
        v = _get(g, sub, stat)
        out.append(float(v) if v is not None else None)
    return out


def _steps(rows: list[dict[str, Any]]) -> list[int]:
    return [int(r.get("step", 0)) for r in rows]


def _group_label(rows: list[dict[str, Any]], gid: str) -> str:
    for r in rows:
        g = _get(r, "groups", gid)
        if isinstance(g, dict) and g.get("name"):
            return f'{gid}: {g["name"]}'
    return gid


def _save_figure(path: Path, dpi: int) -> None:
    import matplotlib.pyplot as plt

    path.parent.mkdir(parents=True, exist_ok=True)
    plt.savefig(path, dpi=dpi, bbox_inches="tight")
    plt.close()


def plot_and_save_separate(
    rows: list[dict[str, Any]],
    *,
    output_dir: Path,
    basename: str,
    subtitle: str,
    dpi: int = 150,
) -> list[Path]:
    r"""Save 6 PNGs: ``{basename}_01_...png`` … ``{basename}_06_...png``."""
    import matplotlib.pyplot as plt

    if not rows:
        return []

    steps = _steps(rows)
    lr = [float(r.get("optimizer_lr") or 0.0) for r in rows]
    group_ids = ["A", "B", "C", "D"]
    colors = ["C0", "C1", "C2", "C3"]
    written: list[Path] = []

    def _name(suffix: str) -> Path:
        return output_dir / f"{basename}_{suffix}.png"

    # 01 optimizer lr
    fig, ax = plt.subplots(figsize=(8, 5))
    fig.suptitle(subtitle, fontsize=10, y=1.02)
    ax.plot(steps, lr, color="C0", linewidth=1.2, marker="o", markersize=3)
    ax.set_ylabel("optimizer_lr")
    ax.set_xlabel("step")
    ax.set_yscale("log")
    ax.set_title("optimizer learning rate")
    ax.grid(True, alpha=0.3)
    p = _name("01_optimizer_lr")
    plt.savefig(p, dpi=dpi, bbox_inches="tight")
    plt.close()
    written.append(p)

    # 02 masked summary
    fig, ax = plt.subplots(figsize=(8, 5))
    fig.suptitle(subtitle, fontsize=10, y=1.02)
    for name, sub, color in [
        ("|Δentropy| (masked)", "delta_entropy", "C0"),
        ("|Δmax_cos| (masked)", "delta_max_cosine", "C1"),
        ("|Δmax_logit| (masked)", "delta_max_logit", "C2"),
    ]:
        y = _series_masked(rows, sub, "mean_abs")
        ax.plot(steps, y, label=name, color=color, linewidth=1.0, marker="o", markersize=3)
    ax.set_ylabel("mean_abs")
    ax.set_xlabel("step")
    ax.legend(loc="best", fontsize=8)
    ax.grid(True, alpha=0.3)
    ax.set_title("masked_summary (mean_abs)")
    p = _name("02_masked_summary")
    plt.savefig(p, dpi=dpi, bbox_inches="tight")
    plt.close()
    written.append(p)

    # 03 groups entropy
    fig, ax = plt.subplots(figsize=(8, 5))
    fig.suptitle(subtitle, fontsize=10, y=1.02)
    for gid, c in zip(group_ids, colors):
        label = _group_label(rows, gid)
        y = _series_group(rows, gid, "delta_entropy", "mean_abs")
        ax.plot(steps, y, label=label, color=c, linewidth=1.0, marker="o", markersize=3)
    ax.set_ylabel("mean_abs")
    ax.set_xlabel("step")
    ax.legend(loc="best", fontsize=7)
    ax.grid(True, alpha=0.3)
    ax.set_title("groups: |Δentropy| mean_abs")
    p = _name("03_groups_entropy")
    plt.savefig(p, dpi=dpi, bbox_inches="tight")
    plt.close()
    written.append(p)

    # 04 groups max logit
    fig, ax = plt.subplots(figsize=(8, 5))
    fig.suptitle(subtitle, fontsize=10, y=1.02)
    for gid, c in zip(group_ids, colors):
        label = _group_label(rows, gid)
        y = _series_group(rows, gid, "delta_max_logit", "mean_abs")
        ax.plot(steps, y, label=label, color=c, linewidth=1.0, marker="o", markersize=3)
    ax.set_ylabel("mean_abs")
    ax.set_xlabel("step")
    ax.legend(loc="best", fontsize=7)
    ax.grid(True, alpha=0.3)
    ax.set_title("groups: |Δmax_logit| mean_abs")
    p = _name("04_groups_max_logit")
    plt.savefig(p, dpi=dpi, bbox_inches="tight")
    plt.close()
    written.append(p)

    # 05 groups max cosine
    fig, ax = plt.subplots(figsize=(8, 5))
    fig.suptitle(subtitle, fontsize=10, y=1.02)
    for gid, c in zip(group_ids, colors):
        label = _group_label(rows, gid)
        y = _series_group(rows, gid, "delta_max_cosine", "mean_abs")
        ax.plot(steps, y, label=label, color=c, linewidth=1.0, marker="o", markersize=3)
    ax.set_ylabel("mean_abs")
    ax.set_xlabel("step")
    ax.legend(loc="best", fontsize=7)
    ax.grid(True, alpha=0.3)
    ax.set_title("groups: |Δmax_cosine| mean_abs")
    p = _name("05_groups_max_cosine")
    plt.savefig(p, dpi=dpi, bbox_inches="tight")
    plt.close()
    written.append(p)

    # 06 param / logit diagnostics
    fig, ax = plt.subplots(figsize=(8, 5))
    fig.suptitle(subtitle, fontsize=10, y=1.02)
    pmax = [r.get("max_abs_trainable_param_delta") for r in rows]
    pmax_f = [float(x) if x is not None else float("nan") for x in pmax]
    mean_ml = [r.get("mean_abs_delta_max_logit_masked") for r in rows]
    mean_ml_f = [float(x) if x is not None else float("nan") for x in mean_ml]
    ax.plot(steps, pmax_f, label="max_abs_trainable_param_delta", color="C0", linewidth=1.0, marker="o", markersize=3)
    ax2 = ax.twinx()
    ax2.plot(
        steps,
        mean_ml_f,
        label="mean_abs_delta_max_logit_masked",
        color="C1",
        linewidth=1.0,
        alpha=0.85,
        marker="s",
        markersize=3,
    )
    ax.set_ylabel("param_delta", color="C0")
    ax2.set_ylabel("mean_abs_Δmax_logit", color="C1")
    ax.set_xlabel("step")
    ax.tick_params(axis="y", labelcolor="C0")
    ax2.tick_params(axis="y", labelcolor="C1")
    if all(x == x and x > 0 for x in pmax_f):
        ax.set_yscale("log")
    lines, labels = ax.get_legend_handles_labels()
    lines2, labels2 = ax2.get_legend_handles_labels()
    ax.legend(lines + lines2, labels + labels2, loc="best", fontsize=8)
    ax.grid(True, alpha=0.3)
    ax.set_title("param / logit delta diagnostics")
    p = _name("06_param_logit_diagnostics")
    plt.savefig(p, dpi=dpi, bbox_inches="tight")
    plt.close()
    written.append(p)

    return written


def main() -> None:
    parser = argparse.ArgumentParser(
        description="Merge all jsonl records under a directory and plot vs step; save each subplot as its own PNG."
    )
    parser.add_argument(
        "--input_dir",
        type=str,
        required=True,
        help="Directory containing *.jsonl (one or more lines per file).",
    )
    parser.add_argument(
        "--output_dir",
        type=str,
        default=None,
        help="Directory for PNG files (overrides --output if both set).",
    )
    parser.add_argument(
        "--basename",
        type=str,
        default="logits_analysis_by_step",
        help="Filename prefix for outputs: {basename}_01_....png (default: logits_analysis_by_step).",
    )
    parser.add_argument(
        "--output",
        type=str,
        default=None,
        help=(
            "Legacy: path ending in .png → PNGs go next to it with stem as basename "
            "(e.g. curves.png → curves_01_....png). Non-.png treated as output directory."
        ),
    )
    parser.add_argument(
        "--pattern",
        type=str,
        default="*.jsonl",
        help="Glob under input_dir (default: *.jsonl).",
    )
    parser.add_argument("--dpi", type=int, default=150, help="PNG resolution (default: 150).")
    args = parser.parse_args()

    input_dir = Path(args.input_dir).resolve()
    if not input_dir.is_dir():
        raise SystemExit(f"Not a directory: {input_dir}")

    raw = _load_all_records(input_dir, args.pattern)
    if not raw:
        raise SystemExit(f"No JSON lines found under {input_dir!s} with pattern {args.pattern!r}")

    rows = _merge_sort_by_step(raw)
    n_files = len(list(input_dir.glob(args.pattern)))

    if args.output_dir is not None:
        out_dir = Path(args.output_dir).resolve()
        basename = args.basename
    elif args.output is not None:
        op = Path(args.output).resolve()
        if op.suffix.lower() == ".png":
            out_dir, basename = op.parent, op.stem
        else:
            out_dir, basename = op, args.basename
    else:
        out_dir = (input_dir / "plots").resolve()
        basename = args.basename

    subtitle = (
        f"merged n={len(rows)} steps, files={n_files}, "
        f"step∈[{rows[0].get('step')}, {rows[-1].get('step')}]"
    )
    paths = plot_and_save_separate(
        rows,
        output_dir=out_dir,
        basename=basename,
        subtitle=subtitle,
        dpi=args.dpi,
    )
    for p in paths:
        print(p)
    print(f"done. {len(paths)} PNGs → {out_dir}")


if __name__ == "__main__":
    main()
