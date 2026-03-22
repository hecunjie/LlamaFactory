#!/usr/bin/env python3
"""
Prepare Hendrycks MATH dataset for standard SFT in LlamaFactory.

Loads EleutherAI/hendrycks_math (per-subject configs), merges subjects, and
converts to alpaca format:
  - instruction: the problem statement (LaTeX)
  - output: official solution; if \\boxed{...} is present, appends a line
    \"#### <boxed content>\" so ComputeExactMatch (GSM8K-style delimiter) can
    compare answers. If no \\boxed is found, output is the raw solution only.

Usage:
    python scripts/prepare_math_sft.py [--output-dir data]
    python scripts/prepare_math_sft.py --subjects algebra prealgebra --output-dir data
"""

from __future__ import annotations

import argparse
import json
import os

# Subjects match HuggingFace config names for EleutherAI/hendrycks_math
DEFAULT_SUBJECTS = (
    "algebra",
    "counting_and_probability",
    "geometry",
    "intermediate_algebra",
    "number_theory",
    "prealgebra",
    "precalculus",
)


def extract_last_boxed(text: str) -> str | None:
    """Return inner content of the last \\boxed{...}, handling nested braces."""
    key = r"\boxed{"
    idx = text.rfind(key)
    if idx == -1:
        return None
    i = idx + len(key)
    depth = 1
    start = i
    while i < len(text) and depth:
        c = text[i]
        if c == "{":
            depth += 1
        elif c == "}":
            depth -= 1
        i += 1
    if depth != 0:
        return None
    return text[start : i - 1].strip()


def load_math_split(split: str, subjects: tuple[str, ...]) -> list[dict]:
    """Load and merge MATH rows from all requested subject configs."""
    from datasets import load_dataset

    rows: list[dict] = []
    for subj in subjects:
        ds = load_dataset("EleutherAI/hendrycks_math", subj, split=split)
        for ex in ds:
            rows.append({
                "problem": ex["problem"],
                "solution": ex["solution"],
                "subject": subj,
                "level": ex.get("level"),
                "type": ex.get("type"),
            })
    return rows


def convert_to_alpaca(examples: list[dict]) -> tuple[list[dict], int]:
    """
    Convert merged MATH rows to LlamaFactory alpaca format.

    Returns (records, num_missing_boxed) for logging.
    """
    results: list[dict] = []
    missing = 0
    for ex in examples:
        instruction = ex["problem"].strip()
        solution = ex["solution"].strip()
        boxed = extract_last_boxed(solution)
        if boxed is not None:
            output = f"{solution}\n#### {boxed}"
        else:
            output = solution
            missing += 1
        results.append({
            "instruction": instruction,
            "output": output,
        })
    return results, missing


def main():
    parser = argparse.ArgumentParser(description="Prepare MATH (hendrycks) for SFT")
    parser.add_argument(
        "--output-dir",
        type=str,
        default="data",
        help="Directory to save the converted JSONL files",
    )
    parser.add_argument(
        "--subjects",
        nargs="*",
        default=list(DEFAULT_SUBJECTS),
        help="Subject configs to include (default: all 7 MATH subjects)",
    )
    args = parser.parse_args()

    subjects = tuple(args.subjects)
    for s in subjects:
        if s not in DEFAULT_SUBJECTS:
            raise SystemExit(
                f"Unknown subject {s!r}. Expected one of: {', '.join(DEFAULT_SUBJECTS)}"
            )

    os.makedirs(args.output_dir, exist_ok=True)

    for split, filename in [("train", "math_sft_train.jsonl"), ("test", "math_sft_test.jsonl")]:
        print(f"Loading MATH {split} split ({len(subjects)} subject(s))...")
        raw = load_math_split(split, subjects)
        print(f"  {len(raw)} examples loaded")

        converted, n_missing = convert_to_alpaca(raw)
        if n_missing:
            print(
                f"  Warning: {n_missing} example(s) had no \\boxed{{...}}; "
                "output is solution only (no #### suffix)."
            )

        out_path = os.path.join(args.output_dir, filename)
        with open(out_path, "w", encoding="utf-8") as f:
            for item in converted:
                f.write(json.dumps(item, ensure_ascii=False) + "\n")
        print(f"  Saved to {out_path}")

        if converted:
            print(f"  Sample:\n    instruction: {converted[0]['instruction'][:80]}...")
            print(f"    output: {converted[0]['output'][:80]}...")
    print("Done!")


if __name__ == "__main__":
    main()
