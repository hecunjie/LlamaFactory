import argparse
import csv
import os
from collections import defaultdict

import matplotlib.pyplot as plt


def load_sweep_rows(path: str):
    rows = []
    with open(path, "r", encoding="utf-8") as f:
        reader = csv.DictReader(f)
        for row in reader:
            rows.append(row)
    return rows


def aggregate_by_alpha(rows):
    """Aggregate KL and overlap metrics by alpha across samples/positions."""
    stats = defaultdict(lambda: {"count": 0, "kl_blend_to_A": 0.0, "kl_A_to_blend": 0.0,
                                 "kl_blend_to_C": 0.0, "kl_C_to_blend": 0.0,
                                 "topk_overlap_with_A": 0.0, "topk_overlap_with_C": 0.0})
    for r in rows:
        alpha = float(r["alpha"])
        s = stats[alpha]
        s["count"] += 1
        s["kl_blend_to_A"] += float(r["kl_blend_to_A"])
        s["kl_A_to_blend"] += float(r["kl_A_to_blend"])
        s["kl_blend_to_C"] += float(r["kl_blend_to_C"])
        s["kl_C_to_blend"] += float(r["kl_C_to_blend"])
        s["topk_overlap_with_A"] += float(r["topk_overlap_with_A"])
        s["topk_overlap_with_C"] += float(r["topk_overlap_with_C"])

    alphas = sorted(stats.keys())
    agg = {
        "alpha": [],
        "kl_blend_to_A": [],
        "kl_A_to_blend": [],
        "kl_blend_to_C": [],
        "kl_C_to_blend": [],
        "topk_overlap_with_A": [],
        "topk_overlap_with_C": [],
    }
    for a in alphas:
        s = stats[a]
        c = s["count"]
        agg["alpha"].append(a)
        agg["kl_blend_to_A"].append(s["kl_blend_to_A"] / c)
        agg["kl_A_to_blend"].append(s["kl_A_to_blend"] / c)
        agg["kl_blend_to_C"].append(s["kl_blend_to_C"] / c)
        agg["kl_C_to_blend"].append(s["kl_C_to_blend"] / c)
        agg["topk_overlap_with_A"].append(s["topk_overlap_with_A"] / c)
        agg["topk_overlap_with_C"].append(s["topk_overlap_with_C"] / c)
    return agg


def main():
    parser = argparse.ArgumentParser(
        description="Plot KL and top-k overlap vs alpha from *_blend_alpha_sweep.csv files."
    )
    parser.add_argument(
        "--analysis_dir",
        type=str,
        required=True,
        help="Path to entropy_analysis directory (contains sample_*/ or sample_*_blend_alpha_sweep.csv).",
    )
    parser.add_argument(
        "--output",
        type=str,
        default="blend_alpha_sweep.png",
        help="Output PNG path for the plot.",
    )
    args = parser.parse_args()

    # Collect all sweep csvs (they may live either directly under analysis_dir
    # or inside per-sample subfolders like sample_<idx>/).
    sweep_files = []
    for root, _dirs, files in os.walk(args.analysis_dir):
        for name in files:
            if name.endswith("blend_alpha_sweep.csv"):
                sweep_files.append(os.path.join(root, name))

    if not sweep_files:
        raise SystemExit(f"No *_blend_alpha_sweep.csv found under {args.analysis_dir}")

    all_rows = []
    for path in sweep_files:
        all_rows.extend(load_sweep_rows(path))

    agg = aggregate_by_alpha(all_rows)

    fig, axes = plt.subplots(1, 2, figsize=(10, 4))

    # KL curves
    ax = axes[0]
    ax.plot(agg["alpha"], agg["kl_blend_to_A"], label="KL(blend || A)")
    ax.plot(agg["alpha"], agg["kl_A_to_blend"], label="KL(A || blend)")
    ax.plot(agg["alpha"], agg["kl_blend_to_C"], label="KL(blend || C)")
    ax.plot(agg["alpha"], agg["kl_C_to_blend"], label="KL(C || blend)")
    ax.set_xlabel("alpha in blend = alpha * A + (1-alpha) * B")
    ax.set_ylabel("KL divergence")
    ax.legend()
    ax.grid(True, alpha=0.3)

    # Top-k overlap curves
    ax = axes[1]
    ax.plot(agg["alpha"], agg["topk_overlap_with_A"], label="top-k overlap with A")
    ax.plot(agg["alpha"], agg["topk_overlap_with_C"], label="top-k overlap with C")
    ax.set_xlabel("alpha in blend = alpha * A + (1-alpha) * B")
    ax.set_ylabel("Average top-k overlap count")
    ax.legend()
    ax.grid(True, alpha=0.3)

    plt.tight_layout()
    plt.savefig(args.output, dpi=150)
    print(f"Saved plot to {args.output}")


if __name__ == "__main__":
    main()

