#!/usr/bin/env python3
from __future__ import annotations

import argparse
import os

import matplotlib.pyplot as plt
import numpy as np


def main() -> None:
    parser = argparse.ArgumentParser(description="UMAP scatter colored by cluster.")
    parser.add_argument("--input_dir", type=str, default="outputs/hidden_states_run")
    parser.add_argument("--output_name", type=str, default="umap_clusters.png")
    args = parser.parse_args()

    umap_path = os.path.join(args.input_dir, "umap_2d.npy")
    labels_path = os.path.join(args.input_dir, "cluster_labels.npy")
    hs_2d = np.load(umap_path)
    labels = np.load(labels_path)

    fig, ax = plt.subplots(figsize=(12, 8))
    sc = ax.scatter(hs_2d[:, 0], hs_2d[:, 1], c=labels, cmap="tab10", alpha=0.5, s=5)
    plt.colorbar(sc, ax=ax)
    ax.set_title("UMAP of high-entropy hidden states (by cluster)")
    out = os.path.join(args.input_dir, args.output_name)
    fig.tight_layout()
    fig.savefig(out, dpi=150)
    plt.close(fig)
    print(f"Saved {out}")


if __name__ == "__main__":
    main()
