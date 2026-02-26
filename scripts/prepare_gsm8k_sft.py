#!/usr/bin/env python3
"""
Prepare GSM8K dataset for standard SFT in LlamaFactory.

Converts HuggingFace GSM8K into alpaca format:
  - instruction: the math question
  - output: full CoT reasoning + final answer (e.g. "Step1... Step2... #### 42")

For eval, ComputeExactMatch extracts the last number from both prediction and
label, so keeping the "#### <number>" format in output works perfectly.

Usage:
    python scripts/prepare_gsm8k_sft.py [--output-dir data]
"""

import argparse
import json
import os
import re


def load_gsm8k(split: str) -> list[dict]:
    """Load GSM8K from HuggingFace datasets."""
    from datasets import load_dataset
    ds = load_dataset("openai/gsm8k", "main", split=split)
    return list(ds)


def convert_to_alpaca(examples: list[dict]) -> list[dict]:
    """
    Convert GSM8K format to LlamaFactory alpaca format.

    GSM8K format:
        {"question": "...", "answer": "reasoning steps\n#### 42"}

    Output (alpaca) format:
        {"instruction": "...", "output": "reasoning steps\nThe answer is 42."}
    """
    results = []
    for ex in examples:
        question = ex["question"].strip()
        answer_raw = ex["answer"].strip()

        # Extract the final numerical answer after ####
        match = re.search(r"####\s*(.+)$", answer_raw)
        if match:
            final_answer = match.group(1).strip()
            # Get the reasoning part (before ####)
            reasoning_part = answer_raw[:match.start()].strip()
            # Build a clean output: reasoning + clear final answer
            output = f"{reasoning_part}\n#### {final_answer}"
        else:
            # Fallback: use the raw answer as-is
            output = answer_raw

        results.append({
            "instruction": question,
            "output": output,
        })
    return results


def main():
    parser = argparse.ArgumentParser(description="Prepare GSM8K for SFT")
    parser.add_argument(
        "--output-dir", type=str, default="data",
        help="Directory to save the converted JSON files"
    )
    args = parser.parse_args()

    os.makedirs(args.output_dir, exist_ok=True)

    for split, filename in [("train", "gsm8k_sft_train.json"), ("test", "gsm8k_sft_test.json")]:
        print(f"Loading GSM8K {split} split...")
        raw = load_gsm8k(split)
        print(f"  {len(raw)} examples loaded")

        converted = convert_to_alpaca(raw)
        out_path = os.path.join(args.output_dir, filename)
        with open(out_path, "w", encoding="utf-8") as f:
            json.dump(converted, f, ensure_ascii=False, indent=2)
        print(f"  Saved to {out_path}")

        # Show a sample
        if converted:
            print(f"  Sample:\n    instruction: {converted[0]['instruction'][:80]}...")
            print(f"    output: {converted[0]['output'][:80]}...")
    print("Done!")


if __name__ == "__main__":
    main()
