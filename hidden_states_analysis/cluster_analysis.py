#!/usr/bin/env python3
from __future__ import annotations

import argparse
import json
import os

import numpy as np
from sklearn.cluster import KMeans
from sklearn.metrics import davies_bouldin_score, silhouette_score
from sklearn.preprocessing import normalize

try:
    import umap
except ImportError as e:
    raise SystemExit("Please install umap-learn: pip install umap-learn") from e


def main() -> None:
    parser = argparse.ArgumentParser(description="UMAP + KMeans on extracted hidden states.")
    parser.add_argument("--input_dir", type=str, default="outputs/hidden_states_run")
    parser.add_argument("--k_min", type=int, default=2)
    parser.add_argument("--k_max", type=int, default=8)
    parser.add_argument("--random_state", type=int, default=42)
    args = parser.parse_args()

    hs_path = os.path.join(args.input_dir, "hidden_states.npy")
    hs = np.load(hs_path)
    if hs.size == 0:
        raise SystemExit("No hidden states; run extract_hidden_states.py first.")

    hs_norm = normalize(hs)
    n = hs_norm.shape[0]
    if n < 2:
        raise SystemExit("Need at least 2 points for clustering.")
    # range(args.k_min, args.k_max) convention: try k in [k_min, k_max)
    k_hi_exclusive = min(args.k_max, n + 1)
    sil_sample = min(5000, n)
    reducer = umap.UMAP(n_components=2, metric="cosine", random_state=args.random_state)
    hs_2d = reducer.fit_transform(hs_norm)
    umap_path = os.path.join(args.input_dir, "umap_2d.npy")
    np.save(umap_path, hs_2d)

    results: dict = {}
    best_k = None
    best_sil = -1.0
    best_labels = None

    if args.k_min >= k_hi_exclusive:
        raise SystemExit(f"Invalid K range: n={n}, k_min={args.k_min}, k_max cap={k_hi_exclusive}")

    for k in range(args.k_min, k_hi_exclusive):
        km = KMeans(n_clusters=k, random_state=args.random_state, n_init=10)
        labels = km.fit_predict(hs_norm)
        sil = float(silhouette_score(hs_norm, labels, sample_size=sil_sample) if n > 1 else 0.0)
        db = float(davies_bouldin_score(hs_norm, labels)) if k > 1 else float("nan")
        results[k] = {"silhouette": sil, "davies_bouldin": db}
        print(f"K={k}: silhouette={sil:.4f}, davies_bouldin={db:.4f}")
        if sil > best_sil:
            best_sil = sil
            best_k = k
            best_labels = labels

    assert best_k is not None and best_labels is not None
    print(f"\nBest K: {best_k} (silhouette={best_sil:.4f})")

    labels_path = os.path.join(args.input_dir, "cluster_labels.npy")
    np.save(labels_path, best_labels)

    summary_path = os.path.join(args.input_dir, "cluster_summary.json")
    with open(summary_path, "w", encoding="utf-8") as f:
        json.dump(
            {"best_k": best_k, "best_silhouette": best_sil, "per_k": results, "umap_path": umap_path},
            f,
            ensure_ascii=False,
            indent=2,
        )
    print(f"Saved labels -> {labels_path}, UMAP -> {umap_path}, summary -> {summary_path}")


if __name__ == "__main__":
    main()
