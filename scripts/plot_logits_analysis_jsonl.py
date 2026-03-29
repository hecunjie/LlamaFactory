#!/usr/bin/env python3
# Copyright 2025 the LlamaFactory team.
#
# Licensed under the Apache License, Version 2.0.

r"""Merge logits-analysis jsonl under a directory and plot metrics vs ``step``.

**Dataset grouping**: All ``*.jsonl`` under ``--input_dir`` (recursive) are grouped by filename stem
before the last ``_step_<digits>`` segment, e.g. ``eval_logits_analysis_eval_x_step_10`` → key
``eval_logits_analysis_eval_x``; ``analysis_step_100_100`` → ``analysis_step_train``. Each group is
processed and written with a distinct output basename.

**Multi-line files (eval)**: Multiple jsonl lines with the same ``step`` (batched eval) are merged
into one per-step record using count-weighted means for group / masked statistics.

**Plots** (train): 01–06 as before; **07** A–D token-count fractions; **10–11** cosine-mean **difference**
curves **A−D** and **A−B** vs step (train: ``delta_max_cosine.mean``; eval: ``max_cosine.mean``).
**Eval** snapshots use ``max_cosine.mean`` instead of deltas where applicable.

Also writes ``{basename}_contrast_stats.json`` (train delta records only) with paired contrast stats.
Use ``--no_contrast`` to skip; ``--contrast_full`` adds per-step arrays (large).

Example::

    python scripts/plot_logits_analysis_jsonl.py --input_dir ./analysis_logs

    python scripts/plot_logits_analysis_jsonl.py --input_dir ./analysis_logs/train \\
        --output_dir ./figures --basename run1_curves
"""

from __future__ import annotations

import argparse
import json
import math
import re
import warnings
from pathlib import Path
from typing import Any, Optional

import numpy as np


def _get(d: dict[str, Any], *keys: str, default: Any = None) -> Any:
    cur: Any = d
    for k in keys:
        if not isinstance(cur, dict):
            return default
        cur = cur.get(k)
        if cur is None:
            return default
    return cur


def _safe_dataset_basename_key(key: str) -> str:
    r"""Filesystem-safe suffix for output basename."""
    s = re.sub(r"[^\w\-.]+", "_", key.strip()).strip("_")
    return s[:200] if s else "dataset"


def dataset_group_key_from_path(path: Path) -> str:
    r"""Group jsonl files: stem without trailing ``_step_<digits>``; train ``analysis_step_N_N`` → one bucket."""
    stem = path.stem
    m_ev = re.match(r"^(eval_logits_analysis_.+)_step_(\d+)$", stem)
    if m_ev:
        return m_ev.group(1)
    m_tr = re.match(r"^analysis_step_(\d+)_(\d+)$", stem)
    if m_tr and m_tr.group(1) == m_tr.group(2):
        return "analysis_step_train"
    if stem.startswith("analysis_step_"):
        return "analysis_step_train"
    m_gen = re.match(r"^(.*)_step_(\d+)$", stem)
    if m_gen:
        return m_gen.group(1)
    return stem


def discover_jsonl_by_dataset(input_dir: Path, pattern: str) -> dict[str, list[Path]]:
    r"""All matching jsonl under ``input_dir`` (recursive), grouped by :func:`dataset_group_key_from_path`."""
    files = sorted(input_dir.rglob(pattern))
    groups: dict[str, list[Path]] = {}
    for fp in files:
        if not fp.is_file():
            continue
        key = dataset_group_key_from_path(fp)
        groups.setdefault(key, []).append(fp)
    for k in groups:
        groups[k].sort(key=lambda p: str(p))
    return groups


def _weighted_mean(values: list[float], weights: list[float]) -> float:
    sw = float(sum(weights))
    if sw <= 0 or not values:
        return float("nan")
    return float(sum(v * w for v, w in zip(values, weights)) / sw)


def _merge_group_cluster_dicts(parts: list[dict[str, Any]], metric_keys: tuple[str, ...]) -> dict[str, Any]:
    r"""Merge A/B/C/D group dicts across eval batches (count-weighted mean / mean_abs / std)."""
    count = sum(int(p.get("count") or 0) for p in parts)
    out: dict[str, Any] = {
        "count": count,
        "name": parts[0].get("name"),
        "description": parts[0].get("description"),
    }
    for mk in metric_keys:
        sub_list: list[dict[str, Any]] = []
        ws: list[float] = []
        for p in parts:
            d = p.get(mk)
            if not isinstance(d, dict):
                continue
            w = float(p.get("count") or 0)
            sub_list.append(d)
            ws.append(w)
        if not sub_list:
            continue
        sw = sum(ws)
        if sw <= 0:
            out[mk] = dict(sub_list[0])
            continue
        mean_v = _weighted_mean([float(d.get("mean") or 0) for d in sub_list], ws)
        mean_abs_v = _weighted_mean([float(d.get("mean_abs") or 0) for d in sub_list], ws)
        std_v = _weighted_mean([float(d.get("std") or 0) for d in sub_list], ws)
        out[mk] = {"mean": mean_v, "std": std_v, "mean_abs": mean_abs_v}
    return out


def _merge_masked_summary_parts(
    parts: list[dict[str, Any]], summary_keys: tuple[str, ...]
) -> dict[str, Any]:
    tc = sum(int(p.get("token_count") or 0) for p in parts)
    out: dict[str, Any] = {"token_count": tc}
    for sk in summary_keys:
        sub_list: list[dict[str, Any]] = []
        ws: list[float] = []
        for p in parts:
            d = p.get(sk)
            if not isinstance(d, dict):
                continue
            w = float(p.get("token_count") or 0)
            sub_list.append(d)
            ws.append(w)
        if not sub_list:
            continue
        sw = sum(ws)
        if sw <= 0:
            out[sk] = dict(sub_list[0])
            continue
        out[sk] = {
            "mean": _weighted_mean([float(d.get("mean") or 0) for d in sub_list], ws),
            "mean_abs": _weighted_mean([float(d.get("mean_abs") or 0) for d in sub_list], ws),
            "std": _weighted_mean([float(d.get("std") or 0) for d in sub_list], ws),
        }
    return out


def _merge_records_for_one_step(lines: list[dict[str, Any]]) -> dict[str, Any]:
    r"""Merge multiple jsonl lines (eval batches) sharing the same ``step``."""
    if len(lines) == 1:
        return lines[0]
    kind = lines[0].get("analysis")
    train_keys = ("delta_entropy", "delta_max_cosine", "delta_max_logit")
    eval_keys = ("entropy", "max_cosine", "max_logit")
    if kind == "eval_cluster_snapshot":
        mkeys = eval_keys
        sk_keys = eval_keys
    else:
        mkeys = train_keys
        sk_keys = train_keys
    base = dict(lines[0])
    base["step"] = int(lines[0].get("step", 0))
    groups_m: dict[str, Any] = {}
    for gid in ("A", "B", "C", "D"):
        gp = [_get(ln, "groups", gid) for ln in lines]
        gp = [g for g in gp if isinstance(g, dict)]
        if gp:
            groups_m[gid] = _merge_group_cluster_dicts(gp, mkeys)
    base["groups"] = groups_m
    ms_parts = [ln["masked_summary"] for ln in lines if isinstance(ln.get("masked_summary"), dict)]
    if ms_parts:
        base["masked_summary"] = _merge_masked_summary_parts(ms_parts, sk_keys)
    return base


def load_merge_dataset_group(files: list[Path]) -> list[dict[str, Any]]:
    r"""Load all lines from files, merge lines that share the same ``step``, sort by step."""
    raw: list[dict[str, Any]] = []
    for fp in files:
        with fp.open(encoding="utf-8") as f:
            for line in f:
                line = line.strip()
                if not line:
                    continue
                raw.append(json.loads(line))
    if not raw:
        return []
    by_step: dict[int, list[dict[str, Any]]] = {}
    for r in raw:
        s = int(r.get("step", -1))
        by_step.setdefault(s, []).append(r)
    merged: list[dict[str, Any]] = []
    for s in sorted(by_step.keys()):
        chunk = by_step[s]
        if len(chunk) > 1:
            merged.append(_merge_records_for_one_step(chunk))
        else:
            merged.append(chunk[0])
    return merged


def _rows_analysis_kind(rows: list[dict[str, Any]]) -> str:
    if not rows:
        return "train"
    if rows[0].get("analysis") == "eval_cluster_snapshot":
        return "eval"
    return "train"


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


def _series_group_count_fraction(rows: list[dict[str, Any]], gid: str) -> list[Optional[float]]:
    r"""Fraction of supervised tokens in cluster ``gid`` (count / sum_A-D counts)."""
    out: list[Optional[float]] = []
    idx = {"A": 0, "B": 1, "C": 2, "D": 3}[gid]
    for r in rows:
        counts = [float(_get(r, "groups", g, "count") or 0) for g in ("A", "B", "C", "D")]
        t = sum(counts)
        if t <= 0:
            out.append(None)
        else:
            out.append(counts[idx] / t)
    return out


def _series_group_cosine_mean(rows: list[dict[str, Any]], kind: str, gid: str) -> list[Optional[float]]:
    r"""``delta_max_cosine.mean`` (train) or ``max_cosine.mean`` (eval)."""
    sub = "max_cosine" if kind == "eval" else "delta_max_cosine"
    out: list[Optional[float]] = []
    for r in rows:
        v = _get(r, "groups", gid, sub, "mean")
        if v is None:
            out.append(None)
            continue
        fv = float(v)
        out.append(fv if fv == fv else None)
    return out


def _series_cosine_mean_pair_diff(
    rows: list[dict[str, Any]],
    kind: str,
    g_left: str,
    g_right: str,
) -> list[Optional[float]]:
    r"""Per step: ``cosine_mean(left) − cosine_mean(right)`` (same metric as :func:`_series_group_cosine_mean`)."""
    yl = _series_group_cosine_mean(rows, kind, g_left)
    yr = _series_group_cosine_mean(rows, kind, g_right)
    out: list[Optional[float]] = []
    for a, b in zip(yl, yr):
        if a is None or b is None:
            out.append(None)
            continue
        out.append(float(a) - float(b))
    return out


# (field under each group, stat key) — same metrics as in plots.
_CONTRAST_METRICS: tuple[tuple[str, str], ...] = (
    ("delta_entropy", "mean_abs"),
    ("delta_entropy", "mean"),
    ("delta_max_logit", "mean_abs"),
    ("delta_max_logit", "mean"),
    ("delta_max_cosine", "mean_abs"),
    ("delta_max_cosine", "mean"),
)


def _paired_diff_series(
    rows: list[dict[str, Any]],
    g_left: str,
    g_right: str,
    field: str,
    stat: str,
) -> tuple[list[float], list[int]]:
    r"""Per step: ``value_left - value_right`` when both finite."""
    diffs: list[float] = []
    steps_out: list[int] = []
    for r in rows:
        v1 = _get(r, "groups", g_left, field, stat)
        v2 = _get(r, "groups", g_right, field, stat)
        if v1 is None or v2 is None:
            continue
        f1, f2 = float(v1), float(v2)
        if math.isnan(f1) or math.isnan(f2):
            continue
        diffs.append(f1 - f2)
        steps_out.append(int(r.get("step", 0)))
    return diffs, steps_out


def _summarize_diffs(diffs: list[float]) -> dict[str, Any]:
    r"""Mean, population variance, sample variance / std; t-test vs 0 if ``n >= 2``."""
    arr = np.asarray(diffs, dtype=np.float64)
    n = int(arr.size)
    if n == 0:
        return {
            "n_paired": 0,
            "diff_mean": None,
            "diff_variance_population": None,
            "diff_variance_sample": None,
            "diff_std_sample": None,
            "paired_ttest_vs_zero": None,
        }
    mean = float(arr.mean())
    var_pop = float(arr.var(ddof=0))
    var_sam = float(arr.var(ddof=1)) if n > 1 else float("nan")
    std_sam = float(arr.std(ddof=1)) if n > 1 else float("nan")
    tinfo: Optional[dict[str, Any]] = None
    if n >= 2:
        from scipy import stats

        tstat, pval = stats.ttest_1samp(arr, popmean=0.0, alternative="two-sided")
        tinfo = {
            "statistic": float(tstat),
            "pvalue_two_sided": float(pval),
            "significant_at_0.05": bool(pval < 0.05),
            "significant_at_0.01": bool(pval < 0.01),
            "note": (
                "H0: mean(step-wise diff)=0; paired steps only. "
                "Low n ⇒ low power; interpret with care."
            ),
        }
    else:
        tinfo = {
            "statistic": None,
            "pvalue_two_sided": None,
            "note": "Need n>=2 steps for t-test.",
        }
    return {
        "n_paired": n,
        "diff_mean": mean,
        "diff_variance_population": var_pop,
        "diff_variance_sample": var_sam,
        "diff_std_sample": std_sam,
        "paired_ttest_vs_zero": tinfo,
    }


def compute_group_contrasts(
    rows: list[dict[str, Any]],
    *,
    include_per_step_series: bool = False,
) -> dict[str, Any]:
    r"""A vs D and A vs B for each metric; JSON-serializable."""
    out: dict[str, Any] = {
        "description": (
            "Per metric: at each step, diff = group_left - group_right. "
            "Then mean / variance of diff across steps; one-sample t-test H0: mean(diff)=0 "
            "(same as paired t-test on left vs right)."
        ),
        "pairs": {},
    }
    for pair_name, g1, g2 in (
        ("A_vs_D", "A", "D"),
        ("A_vs_B", "A", "B"),
    ):
        metrics_out: dict[str, Any] = {}
        for field, stat in _CONTRAST_METRICS:
            diffs, step_list = _paired_diff_series(rows, g1, g2, field, stat)
            key = f"{field}_{stat}"
            block: dict[str, Any] = {
                "group_left": g1,
                "group_right": g2,
                "definition": f"({g1}.{field}.{stat} - {g2}.{field}.{stat}) per step",
                **_summarize_diffs(diffs),
            }
            if include_per_step_series:
                block["steps"] = step_list
                block["diff_per_step"] = diffs
            metrics_out[key] = block
        out["pairs"][pair_name] = {
            "group_left": g1,
            "group_right": g2,
            "metrics": metrics_out,
        }
    return out


def plot_and_save_separate(
    rows: list[dict[str, Any]],
    *,
    output_dir: Path,
    basename: str,
    subtitle: str,
    dpi: int = 150,
) -> list[Path]:
    r"""Save PNGs: ``01``–``06`` (train-centric; eval skips 01/06), ``07`` (A–D fractions), ``10``–``11``
    (A−D / A−B cosine-mean differences vs step)."""
    import matplotlib

    matplotlib.use("Agg")
    import matplotlib.pyplot as plt

    if not rows:
        return []

    kind = _rows_analysis_kind(rows)
    steps = _steps(rows)
    group_ids = ["A", "B", "C", "D"]
    colors = ["C0", "C1", "C2", "C3"]
    written: list[Path] = []

    def _name(suffix: str) -> Path:
        return output_dir / f"{basename}_{suffix}.png"

    train = kind == "train"
    # 01 optimizer lr (train only)
    if train and any(r.get("optimizer_lr") is not None for r in rows):
        lr = [float(r.get("optimizer_lr") or 0.0) for r in rows]
        fig, ax = plt.subplots(figsize=(8, 5))
        fig.suptitle(subtitle, fontsize=10, y=1.02)
        ax.plot(steps, lr, color="C0", linewidth=1.2, marker="o", markersize=3)
        ax.set_ylabel("optimizer_lr")
        ax.set_xlabel("step")
        ax.set_yscale("log")
        ax.set_title("optimizer learning rate")
        ax.grid(True, alpha=0.3)
        p = _name("01_optimizer_lr")
        fig.savefig(str(p), format="png", dpi=dpi, bbox_inches="tight")
        plt.close(fig)
        written.append(p)

    # 02 masked summary
    fig, ax = plt.subplots(figsize=(8, 5))
    fig.suptitle(subtitle, fontsize=10, y=1.02)
    if train:
        series = [
            ("|Δentropy| (masked)", "delta_entropy", "C0"),
            ("|Δmax_cos| (masked)", "delta_max_cosine", "C1"),
            ("|Δmax_logit| (masked)", "delta_max_logit", "C2"),
        ]
        ttl = "masked_summary (mean_abs, train Δ)"
    else:
        series = [
            ("|entropy| (masked)", "entropy", "C0"),
            ("|max_cos| (masked)", "max_cosine", "C1"),
            ("|max_logit| (masked)", "max_logit", "C2"),
        ]
        ttl = "masked_summary (mean_abs, eval snapshot)"
    for name, sub, color in series:
        y = _series_masked(rows, sub, "mean_abs")
        ax.plot(steps, y, label=name, color=color, linewidth=1.0, marker="o", markersize=3)
    ax.set_ylabel("mean_abs")
    ax.set_xlabel("step")
    ax.legend(loc="best", fontsize=8)
    ax.grid(True, alpha=0.3)
    ax.set_title(ttl)
    p = _name("02_masked_summary")
    fig.savefig(str(p), format="png", dpi=dpi, bbox_inches="tight")
    plt.close(fig)
    written.append(p)

    # 03 groups entropy
    fig, ax = plt.subplots(figsize=(8, 5))
    fig.suptitle(subtitle, fontsize=10, y=1.02)
    ent_sub, ent_abs = ("delta_entropy", "mean_abs") if train else ("entropy", "mean_abs")
    for gid, c in zip(group_ids, colors):
        label = _group_label(rows, gid)
        y = _series_group(rows, gid, ent_sub, ent_abs)
        ax.plot(steps, y, label=label, color=c, linewidth=1.0, marker="o", markersize=3)
    ax.set_ylabel("mean_abs")
    ax.set_xlabel("step")
    ax.legend(loc="best", fontsize=7)
    ax.grid(True, alpha=0.3)
    ax.set_title("groups: entropy |Δ| (train) or |value| (eval)" if train else "groups: |entropy| mean_abs (eval)")
    p = _name("03_groups_entropy")
    fig.savefig(str(p), format="png", dpi=dpi, bbox_inches="tight")
    plt.close(fig)
    written.append(p)

    # 04 groups max logit
    fig, ax = plt.subplots(figsize=(8, 5))
    fig.suptitle(subtitle, fontsize=10, y=1.02)
    ml_sub = "delta_max_logit" if train else "max_logit"
    for gid, c in zip(group_ids, colors):
        label = _group_label(rows, gid)
        y = _series_group(rows, gid, ml_sub, "mean_abs")
        ax.plot(steps, y, label=label, color=c, linewidth=1.0, marker="o", markersize=3)
    ax.set_ylabel("mean_abs")
    ax.set_xlabel("step")
    ax.legend(loc="best", fontsize=7)
    ax.grid(True, alpha=0.3)
    ax.set_title("groups: |Δmax_logit| mean_abs" if train else "groups: |max_logit| mean_abs")
    p = _name("04_groups_max_logit")
    fig.savefig(str(p), format="png", dpi=dpi, bbox_inches="tight")
    plt.close(fig)
    written.append(p)

    # 05 groups max cosine
    fig, ax = plt.subplots(figsize=(8, 5))
    fig.suptitle(subtitle, fontsize=10, y=1.02)
    mc_sub = "delta_max_cosine" if train else "max_cosine"
    for gid, c in zip(group_ids, colors):
        label = _group_label(rows, gid)
        y = _series_group(rows, gid, mc_sub, "mean_abs")
        ax.plot(steps, y, label=label, color=c, linewidth=1.0, marker="o", markersize=3)
    ax.set_ylabel("mean_abs")
    ax.set_xlabel("step")
    ax.legend(loc="best", fontsize=7)
    ax.grid(True, alpha=0.3)
    ax.set_title("groups: |Δmax_cosine| mean_abs" if train else "groups: |max_cosine| mean_abs")
    p = _name("05_groups_max_cosine")
    fig.savefig(str(p), format="png", dpi=dpi, bbox_inches="tight")
    plt.close(fig)
    written.append(p)

    # 06 param / logit diagnostics (train only)
    if train:
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
        fig.savefig(str(p), format="png", dpi=dpi, bbox_inches="tight")
        plt.close(fig)
        written.append(p)

    # 07 A–D token count fractions
    fig, ax = plt.subplots(figsize=(8, 5))
    fig.suptitle(subtitle, fontsize=10, y=1.02)
    for gid, c in zip(group_ids, colors):
        label = _group_label(rows, gid)
        y = _series_group_count_fraction(rows, gid)
        ax.plot(steps, y, label=label, color=c, linewidth=1.2, marker="o", markersize=3)
    ax.set_ylabel("fraction of A–D tokens")
    ax.set_xlabel("step")
    ax.set_ylim(0.0, 1.0)
    ax.legend(loc="best", fontsize=7)
    ax.grid(True, alpha=0.3)
    ax.set_title("cluster token count share (A–D) vs step")
    p = _name("07_group_count_fractions")
    fig.savefig(str(p), format="png", dpi=dpi, bbox_inches="tight")
    plt.close(fig)
    written.append(p)

    ylab_cm = "delta_max_cosine.mean (train)" if train else "max_cosine.mean (eval)"
    # 10 cosine mean difference A − D vs step
    diff_ad = _series_cosine_mean_pair_diff(rows, kind, "A", "D")
    fig, ax = plt.subplots(figsize=(8, 5))
    fig.suptitle(subtitle, fontsize=10, y=1.02)
    ax.plot(steps, diff_ad, color="C2", linewidth=1.2, marker="o", markersize=3, label="A − D")
    ax.axhline(0.0, color="0.5", linewidth=0.8, linestyle="--", alpha=0.8)
    ax.set_ylabel(f"A − D ({ylab_cm})")
    ax.set_xlabel("step")
    ax.legend(loc="best", fontsize=8)
    ax.grid(True, alpha=0.3)
    ax.set_title("cosine mean difference (A − D) vs step")
    p = _name("10_cosine_mean_diff_A_minus_D")
    fig.savefig(str(p), format="png", dpi=dpi, bbox_inches="tight")
    plt.close(fig)
    written.append(p)

    # 11 cosine mean difference A − B vs step
    diff_ab = _series_cosine_mean_pair_diff(rows, kind, "A", "B")
    fig, ax = plt.subplots(figsize=(8, 5))
    fig.suptitle(subtitle, fontsize=10, y=1.02)
    ax.plot(steps, diff_ab, color="C2", linewidth=1.2, marker="o", markersize=3, label="A − B")
    ax.axhline(0.0, color="0.5", linewidth=0.8, linestyle="--", alpha=0.8)
    ax.set_ylabel(f"A − B ({ylab_cm})")
    ax.set_xlabel("step")
    ax.legend(loc="best", fontsize=8)
    ax.grid(True, alpha=0.3)
    ax.set_title("cosine mean difference (A − B) vs step")
    p = _name("11_cosine_mean_diff_A_minus_B")
    fig.savefig(str(p), format="png", dpi=dpi, bbox_inches="tight")
    plt.close(fig)
    written.append(p)

    return written


def _print_contrast_summary(contrast: dict[str, Any]) -> None:
    r"""Short human-readable summary (mean diff, var, p-value) for key metrics."""

    def _fmt(x: Any) -> str:
        if x is None:
            return "n/a"
        if isinstance(x, float) and (math.isnan(x) or math.isinf(x)):
            return "n/a"
        try:
            return f"{float(x):.6g}"
        except (TypeError, ValueError):
            return str(x)

    pairs = contrast.get("pairs") or {}
    headline = (
        "Contrast summary (A−D, A−B): diff = left_group − right_group per step; "
        "mean/var of diff; t-test H0: mean(diff)=0"
    )
    print(headline)
    for pname in ("A_vs_D", "A_vs_B"):
        block = pairs.get(pname)
        if not block:
            continue
        print(f"\n[{pname}] {block.get('group_left')} vs {block.get('group_right')}")
        metrics = block.get("metrics") or {}
        for mkey in (
            "delta_entropy_mean_abs",
            "delta_max_logit_mean_abs",
            "delta_max_cosine_mean_abs",
        ):
            m = metrics.get(mkey)
            if not m:
                continue
            tt = m.get("paired_ttest_vs_zero") or {}
            pval = tt.get("pvalue_two_sided")
            sig = tt.get("significant_at_0.05")
            print(
                f"  {mkey}: n={m.get('n_paired')} mean_diff={_fmt(m.get('diff_mean'))} "
                f"var_pop={_fmt(m.get('diff_variance_population'))} "
                f"p_two_sided={pval if pval is not None else 'n/a'} sig_0.05={sig}"
            )


def main() -> None:
    parser = argparse.ArgumentParser(
        description="Merge all jsonl records under a directory and plot vs step; save each subplot as its own PNG."
    )
    parser.add_argument(
        "--input_dir",
        type=str,
        required=True,
        help="Root directory; all matching jsonl files are found recursively and grouped by dataset (filename).",
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
        help="Filename prefix; each dataset group gets {basename}_{dataset_key}_01_....png.",
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
        help="Glob under input_dir, recursive (default: *.jsonl).",
    )
    parser.add_argument("--dpi", type=int, default=150, help="PNG resolution (default: 150).")
    parser.add_argument(
        "--no_contrast",
        action="store_true",
        help="Do not write {basename}_contrast_stats.json or print A vs D / A vs B stats.",
    )
    parser.add_argument(
        "--contrast_full",
        action="store_true",
        help="Include steps and diff_per_step arrays in contrast JSON (large).",
    )
    args = parser.parse_args()

    input_dir = Path(args.input_dir).resolve()
    if not input_dir.is_dir():
        raise SystemExit(f"Not a directory: {input_dir}")

    groups_map = discover_jsonl_by_dataset(input_dir, args.pattern)
    if not groups_map:
        raise SystemExit(f"No jsonl files under {input_dir!s} matching {args.pattern!r} (recursive)")

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

    out_dir.mkdir(parents=True, exist_ok=True)

    for group_key in sorted(groups_map.keys()):
        files = groups_map[group_key]
        rows = load_merge_dataset_group(files)
        if not rows:
            warnings.warn(f"Skipping empty dataset group {group_key!r}", stacklevel=2)
            continue
        kind = _rows_analysis_kind(rows)
        base = f"{basename}_{_safe_dataset_basename_key(group_key)}"
        subtitle = (
            f"{group_key} | kind={kind} | n_steps={len(rows)} | n_files={len(files)} | "
            f"step∈[{rows[0].get('step')}, {rows[-1].get('step')}]"
        )
        paths = plot_and_save_separate(
            rows,
            output_dir=out_dir,
            basename=base,
            subtitle=subtitle,
            dpi=args.dpi,
        )
        for p in paths:
            print(p)
        print(f"done [{group_key}]. {len(paths)} PNGs → {out_dir}")

        if not args.no_contrast:
            if kind == "train":
                contrast = compute_group_contrasts(
                    rows,
                    include_per_step_series=args.contrast_full,
                )
                contrast_path = out_dir / f"{base}_contrast_stats.json"
                with contrast_path.open("w", encoding="utf-8") as f:
                    json.dump(contrast, f, ensure_ascii=False, indent=2)
                print(contrast_path)
                _print_contrast_summary(contrast)
            else:
                print(
                    f"[{group_key}] skip contrast_stats.json (eval_cluster_snapshot has no Δ metrics; "
                    "contrast requires train clustered_token_deltas)."
                )


if __name__ == "__main__":
    main()
