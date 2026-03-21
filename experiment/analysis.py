"""Phase 4: aggregate intervention results and plot curves."""

from __future__ import annotations

import json
from pathlib import Path

import matplotlib.pyplot as plt


def analyze_and_plot(intervention_results_path, config, output_dir):
    """
    Read intervention jsonl and compute accuracy for residual/vocab/full by gamma.
    """
    output_dir = Path(output_dir)
    output_dir.mkdir(parents=True, exist_ok=True)
    rows = _load_jsonl(intervention_results_path)
    if not rows:
        raise RuntimeError("No intervention results found.")

    gamma_keys = [str(x) for x in config.GAMMA_VALUES]
    stats = {
        g: {"residual": 0, "vocab": 0, "full": 0, "n": 0}
        for g in gamma_keys
    }

    for row in rows:
        res = row.get("results", {})
        for g in gamma_keys:
            if g not in res:
                continue
            stats[g]["n"] += 1
            stats[g]["residual"] += int(_to_bool(res[g].get("residual", False)))
            stats[g]["vocab"] += int(_to_bool(res[g].get("vocab", False)))
            stats[g]["full"] += int(_to_bool(res[g].get("full", False)))

    xs = [float(g) for g in gamma_keys]
    ys_res, ys_voc, ys_full, ns = [], [], [], []
    for g in gamma_keys:
        n = max(stats[g]["n"], 1)
        ys_res.append(100.0 * stats[g]["residual"] / n)
        ys_voc.append(100.0 * stats[g]["vocab"] / n)
        ys_full.append(100.0 * stats[g]["full"] / n)
        ns.append(stats[g]["n"])

    plt.figure(figsize=(8, 5))
    plt.plot(xs, ys_res, marker="o", color="tab:blue", label="residual")
    plt.plot(xs, ys_voc, marker="o", color="tab:orange", label="vocab")
    plt.plot(xs, ys_full, marker="o", color="tab:green", label="full")
    plt.xlabel("gamma")
    plt.ylabel("Accuracy (%)")
    plt.title("Intervention Accuracy vs Gamma")
    plt.grid(True, alpha=0.3)
    plt.legend()
    fig_path = output_dir / "intervention_results.png"
    plt.tight_layout()
    plt.savefig(fig_path, dpi=150)
    plt.close()

    print("gamma | residual(%) | vocab(%) | full(%) | n_samples")
    for i, g in enumerate(gamma_keys):
        print(f"{g:>5} | {ys_res[i]:>11.2f} | {ys_voc[i]:>8.2f} | {ys_full[i]:>7.2f} | {ns[i]}")
    print(f"[analysis] plot saved to {fig_path}")

    return {
        "gamma": xs,
        "residual_pct": ys_res,
        "vocab_pct": ys_voc,
        "full_pct": ys_full,
        "n_samples": ns,
        "plot_path": str(fig_path),
    }


def _load_jsonl(path):
    rows = []
    with Path(path).open("r", encoding="utf-8") as f:
        for line in f:
            line = line.strip()
            if line:
                rows.append(json.loads(line))
    return rows


def _to_bool(v):
    if isinstance(v, dict):
        return bool(v.get("is_correct", False))
    return bool(v)
