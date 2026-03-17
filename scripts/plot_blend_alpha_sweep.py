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
                                 "kl_blend_to_B": 0.0, "kl_B_to_blend": 0.0,
                                 "topk_overlap_with_A": 0.0, "topk_overlap_with_B": 0.0})
    for r in rows:
        alpha = float(r["alpha"])
        s = stats[alpha]
        s["count"] += 1
        s["kl_blend_to_A"] += float(r["kl_blend_to_A"])
        s["kl_A_to_blend"] += float(r["kl_A_to_blend"])
        s["kl_blend_to_B"] += float(r["kl_blend_to_B"])
        s["kl_B_to_blend"] += float(r["kl_B_to_blend"])
        s["topk_overlap_with_A"] += float(r["topk_overlap_with_A"])
        s["topk_overlap_with_B"] += float(r["topk_overlap_with_B"])

    alphas = sorted(stats.keys())
    agg = {
        "alpha": [],
        "kl_blend_to_A": [],
        "kl_A_to_blend": [],
        "kl_blend_to_B": [],
        "kl_B_to_blend": [],
        "topk_overlap_with_A": [],
        "topk_overlap_with_B": [],
    }
    for a in alphas:
        s = stats[a]
        c = s["count"]
        agg["alpha"].append(a)
        agg["kl_blend_to_A"].append(s["kl_blend_to_A"] / c)
        agg["kl_A_to_blend"].append(s["kl_A_to_blend"] / c)
        agg["kl_blend_to_B"].append(s["kl_blend_to_B"] / c)
        agg["kl_B_to_blend"].append(s["kl_B_to_blend"] / c)
        agg["topk_overlap_with_A"].append(s["topk_overlap_with_A"] / c)
        agg["topk_overlap_with_B"].append(s["topk_overlap_with_B"] / c)
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

    # Collect all sweep csvs
    sweep_files = []
    for name in os.listdir(args.analysis_dir):
        if name.endswith("_blend_alpha_sweep.csv"):
            sweep_files.append(os.path.join(args.analysis_dir, name))

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
    ax.plot(agg["alpha"], agg["kl_blend_to_B"], label="KL(blend || B)")
    ax.plot(agg["alpha"], agg["kl_B_to_blend"], label="KL(B || blend)")
    ax.set_xlabel("alpha in blend = alpha * A + (1-alpha) * B")
    ax.set_ylabel("KL divergence")
    ax.legend()
    ax.grid(True, alpha=0.3)

    # Top-k overlap curves
    ax = axes[1]
    ax.plot(agg["alpha"], agg["topk_overlap_with_A"], label="top-k overlap with A")
    ax.plot(agg["alpha"], agg["topk_overlap_with_B"], label="top-k overlap with B")
    ax.set_xlabel("alpha in blend = alpha * A + (1-alpha) * B")
    ax.set_ylabel("Average top-k overlap count")
    ax.legend()
    ax.grid(True, alpha=0.3)

    plt.tight_layout()
    plt.savefig(args.output, dpi=150)
    print(f"Saved plot to {args.output}")


if __name__ == "__main__":
    main()

