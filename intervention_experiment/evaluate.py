#!/usr/bin/env python3
from __future__ import annotations

import argparse
import os
import sys

if __package__ is None or __package__ == "":
    sys.path.append(os.path.dirname(os.path.dirname(__file__)))
import json
from pathlib import Path


def read_jsonl(path: Path):
    rows = []
    with path.open("r", encoding="utf-8") as f:
        for line in f:
            line = line.strip()
            if line:
                rows.append(json.loads(line))
    return rows


def lse_bin(v: float) -> str:
    if v < 19.0:
        return "<19.0"
    if v < 19.76:
        return "[19.0,19.76)"
    if v < 20.5:
        return "[19.76,20.5)"
    return ">=20.5"


def summarize(rows: list[dict]):
    n = len(rows)
    acc = sum(1 for r in rows if r.get("is_correct", False)) / max(n, 1)
    avg_len = sum(int(r.get("response_length", 0)) for r in rows) / max(n, 1)

    total_steps = 0
    total_intervened = 0
    bin_stats = {}
    for r in rows:
        for step in r.get("step_records", []):
            total_steps += 1
            total_intervened += int(step.get("intervened", False))
            b = lse_bin(float(step.get("lse", 0.0)))
            x = bin_stats.setdefault(b, {"n": 0, "top1_selected": 0, "intervened_n": 0})
            x["n"] += 1
            x["top1_selected"] += int(step.get("picked_is_top1", False))
            x["intervened_n"] += int(step.get("intervened", False))

    trigger_rate = total_intervened / max(total_steps, 1)
    per_bin = {
        k: {
            "count": v["n"],
            "intervened_count": v["intervened_n"],
            "top1_selected_rate": v["top1_selected"] / max(v["n"], 1),
        }
        for k, v in sorted(bin_stats.items(), key=lambda x: x[0])
    }
    return {
        "n_samples": n,
        "accuracy": acc,
        "avg_response_length": avg_len,
        "trigger_rate": trigger_rate,
        "per_lse_bin": per_bin,
    }


def main():
    parser = argparse.ArgumentParser(description="Evaluate intervention experiment jsonl files.")
    parser.add_argument("--pred_path", type=str, required=True)
    parser.add_argument("--baseline_path", type=str, default=None)
    parser.add_argument("--save_path", type=str, default=None)
    args = parser.parse_args()

    pred_rows = read_jsonl(Path(args.pred_path))
    pred_summary = summarize(pred_rows)
    output = {"target": args.pred_path, "summary": pred_summary}

    if args.baseline_path:
        base_rows = read_jsonl(Path(args.baseline_path))
        base_summary = summarize(base_rows)
        output["baseline"] = {"path": args.baseline_path, "summary": base_summary}
        output["delta_vs_baseline"] = {
            "accuracy_pp": (pred_summary["accuracy"] - base_summary["accuracy"]) * 100.0,
            "avg_response_length": pred_summary["avg_response_length"] - base_summary["avg_response_length"],
            "top1_rate_delta_per_bin": {
                k: pred_summary["per_lse_bin"].get(k, {}).get("top1_selected_rate", 0.0)
                - base_summary["per_lse_bin"].get(k, {}).get("top1_selected_rate", 0.0)
                for k in set(pred_summary["per_lse_bin"]) | set(base_summary["per_lse_bin"])
            },
        }

    text = json.dumps(output, ensure_ascii=False, indent=2)
    print(text)
    if args.save_path:
        out = Path(args.save_path)
        out.parent.mkdir(parents=True, exist_ok=True)
        out.write_text(text, encoding="utf-8")


if __name__ == "__main__":
    main()
