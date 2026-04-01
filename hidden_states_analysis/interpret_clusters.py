#!/usr/bin/env python3
from __future__ import annotations

import argparse
import json
import os
import random
from collections import defaultdict

import numpy as np


def main() -> None:
    parser = argparse.ArgumentParser(description="Sample contexts per cluster for manual review.")
    parser.add_argument("--input_dir", type=str, default="outputs/hidden_states_run")
    parser.add_argument("--per_cluster", type=int, default=30)
    parser.add_argument("--seed", type=int, default=0)
    args = parser.parse_args()

    random.seed(args.seed)
    labels = np.load(os.path.join(args.input_dir, "cluster_labels.npy"))
    meta_path = os.path.join(args.input_dir, "hidden_states_meta.json")
    with open(meta_path, encoding="utf-8") as f:
        meta = json.load(f)

    by_c: dict[int, list[dict]] = defaultdict(list)
    for lab, m in zip(labels.tolist(), meta):
        by_c[int(lab)].append(m)

    lines: list[str] = []
    for cid in sorted(by_c.keys()):
        samples = by_c[cid]
        pick = random.sample(samples, min(args.per_cluster, len(samples)))
        block = [f"\n{'=' * 60}", f"CLUSTER {cid} (n={len(samples)})", "=" * 60]
        for s in pick:
            block.append(
                f"  neg_logprob={s['neg_logprob']:.3f} | token={s['token_str']!r} | "
                f"context: ...{s['context_window']}..."
            )
        print("\n".join(block))
        lines.extend(block)

    out_txt = os.path.join(args.input_dir, "cluster_interpret_samples.txt")
    with open(out_txt, "w", encoding="utf-8") as f:
        f.write("\n".join(lines))
    print(f"Wrote {out_txt}")


if __name__ == "__main__":
    main()
