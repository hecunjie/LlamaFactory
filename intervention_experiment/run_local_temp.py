#!/usr/bin/env python3
from __future__ import annotations

import argparse
import os
import sys

if __package__ is None or __package__ == "":
    sys.path.append(os.path.dirname(os.path.dirname(__file__)))

from intervention_experiment.common import ExperimentArgs, run_experiment


def parse_args():
    parser = argparse.ArgumentParser(description="Run LocalTemp intervention experiment on GSM8K.")
    parser.add_argument("--model_name_or_path", type=str, required=True)
    parser.add_argument("--output_path", type=str, required=True)
    parser.add_argument("--threshold", type=float, required=True)
    parser.add_argument("--local_temp", type=float, required=True)
    parser.add_argument("--dataset_name", type=str, default="openai/gsm8k")
    parser.add_argument("--dataset_config", type=str, default="main")
    parser.add_argument("--split", type=str, default="test")
    parser.add_argument("--dataset_path", type=str, default=None)
    parser.add_argument("--lf_dataset_name", type=str, default=None)
    parser.add_argument("--dataset_info_path", type=str, default="data/dataset_info.json")
    parser.add_argument("--lf_split", type=str, default=None)
    parser.add_argument("--max_samples", type=int, default=None)
    parser.add_argument("--seed", type=int, default=42)
    parser.add_argument("--max_new_tokens", type=int, default=512)
    parser.add_argument("--top_p", type=float, default=0.95)
    parser.add_argument("--base_temp", type=float, default=1.0)
    parser.add_argument("--device_map", type=str, choices=["single", "auto"], default="single")
    parser.add_argument("--dtype", type=str, choices=["bfloat16", "float16", "float32"], default="bfloat16")
    return parser.parse_args()


def main():
    args = parse_args()
    exp_args = ExperimentArgs(
        model_name_or_path=args.model_name_or_path,
        output_path=args.output_path,
        dataset_name=args.dataset_name,
        dataset_config=args.dataset_config,
        split=args.split,
        dataset_path=args.dataset_path,
        lf_dataset_name=args.lf_dataset_name,
        dataset_info_path=args.dataset_info_path,
        lf_split=args.lf_split,
        max_samples=args.max_samples,
        seed=args.seed,
        max_new_tokens=args.max_new_tokens,
        top_p=args.top_p,
        base_temp=args.base_temp,
        threshold=args.threshold,
        local_temp=args.local_temp,
        device_map=args.device_map,
        dtype=args.dtype,
    )
    run_experiment(exp_args, method="local_temp")


if __name__ == "__main__":
    main()
