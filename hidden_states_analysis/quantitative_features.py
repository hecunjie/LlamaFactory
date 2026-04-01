#!/usr/bin/env python3
from __future__ import annotations

import argparse
import json
import os
import re
from collections import defaultdict

import numpy as np


def is_numeric_token(token_str: str) -> bool:
    return bool(re.match(r"^\s*[\d\.\,\-\+]+\s*$", token_str))


def is_connector_token(token_str: str) -> bool:
    connectors = [
        "so",
        "therefore",
        "because",
        "thus",
        "hence",
        "since",
        "then",
        "if",
        "but",
        "and",
    ]
    return token_str.strip().lower() in connectors


def has_operator_in_context(context: str) -> bool:
    return bool(re.search(r"[\+\-\*\/\=\%]", context))


def main() -> None:
    parser = argparse.ArgumentParser(description="Per-cluster token/context statistics.")
    parser.add_argument("--input_dir", type=str, default="outputs/hidden_states_run")
    args = parser.parse_args()

    labels = np.load(os.path.join(args.input_dir, "cluster_labels.npy"))
    with open(os.path.join(args.input_dir, "hidden_states_meta.json"), encoding="utf-8") as f:
        meta = json.load(f)

    stats: dict = defaultdict(
        lambda: {"total": 0, "numeric": 0, "connector": 0, "has_operator": 0, "neg_logprob_mean": 0.0}
    )

    for lab, m in zip(labels.tolist(), meta):
        c = int(lab)
        stats[c]["total"] += 1
        stats[c]["neg_logprob_mean"] += float(m["neg_logprob"])
        ts = m["token_str"]
        if is_numeric_token(ts):
            stats[c]["numeric"] += 1
        if is_connector_token(ts):
            stats[c]["connector"] += 1
        if has_operator_in_context(m["context_window"]):
            stats[c]["has_operator"] += 1

    lines = [f"{'Cluster':<10} {'N':<8} {'Numeric%':<12} {'Connector%':<14} {'HasOp%':<10} {'AvgNegLogP':<12}"]
    for c in sorted(stats.keys()):
        s = stats[c]
        n = s["total"]
        if n == 0:
            continue
        lines.append(
            f"{c:<10} {n:<8} {100 * s['numeric'] / n:<12.1f} {100 * s['connector'] / n:<14.1f} "
            f"{100 * s['has_operator'] / n:<10.1f} {s['neg_logprob_mean'] / n:<12.3f}"
        )
    report = "\n".join(lines)
    print(report)
    out_path = os.path.join(args.input_dir, "cluster_stats.txt")
    with open(out_path, "w", encoding="utf-8") as f:
        f.write(report + "\n")
    print(f"Wrote {out_path}")


if __name__ == "__main__":
    main()
