#!/usr/bin/env python3
"""
Evaluate model predictions on GSM8K-style datasets.

Input JSONL format (one JSON object per line):
  {
    "prompt":    "<question text>",
    "label":     "<CoT reasoning> #### <answer>",
    "predicts":  ["<pred1>", "<pred2>", ...]
  }

Metrics computed:
  - pass@1        : accuracy of the first prediction
  - pass@k        : whether any of the k predictions is correct
  - majority@k    : accuracy when taking the majority-vote answer over k predictions
  - avg_correct   : average fraction of correct predictions per sample
  - mean_pred_len : average token length of predictions (whitespace-split)
"""

import argparse
import json
import math
import re
import sys
from collections import Counter
from pathlib import Path
from typing import Optional


# ---------------------------------------------------------------------------
# Answer extraction
# ---------------------------------------------------------------------------

def _normalize_number(text: str) -> Optional[str]:
    """
    Strip formatting (commas, dollar signs, percent signs, trailing dots)
    and return a canonical numeric string, or None if not parseable.
    """
    text = text.strip().replace(",", "").replace("$", "").replace("%", "").rstrip(".")
    # Accept integers and decimals (possibly negative)
    m = re.fullmatch(r"-?\d+(?:\.\d+)?", text)
    if m:
        # Normalise: remove leading zeros, keep decimals as-is
        try:
            val = float(text)
            if not math.isfinite(val):
                return None
            # Return int string when there is no fractional part
            if val == int(val):
                return str(int(val))
            return str(val)
        except (ValueError, OverflowError):
            pass
    # e.g. "540 meters" after #### or "The answer is ..."
    m2 = re.match(r"^(-?\d+(?:\.\d+)?)", text)
    if m2:
        try:
            val = float(m2.group(1))
            if not math.isfinite(val):
                return None
            if val == int(val):
                return str(int(val))
            return str(val)
        except (ValueError, OverflowError):
            pass
    return None


def extract_answer(text: str) -> Optional[str]:
    """
    Extract the final answer from a GSM8K-style string.

    Priority:
      1. Last occurrence of "#### <value>" (GSM8K)
      2. Last occurrence of "### <value>"
      3. Last occurrence of "The answer is <value>"
      4. Last standalone number in the text (fallback)
    """
    text = text.strip()

    # 1. #### pattern (GSM8K canonical)
    matches = list(re.finditer(r"####\s*(.+?)(?:\n|$)", text))
    if matches:
        raw = matches[-1].group(1).strip()
        norm = _normalize_number(raw)
        return norm if norm is not None else raw

    # 2. ### pattern (common in CoT outputs; do not match the first 3 # of ####)
    matches = list(re.finditer(r"(?<!#)###(?![#])\s*(.+?)(?:\n|$)", text))
    if matches:
        raw = matches[-1].group(1).strip()
        norm = _normalize_number(raw)
        return norm if norm is not None else raw

    # 3. "The answer is X" pattern
    matches = list(re.finditer(
        r"[Tt]he\s+answer\s+is\s+[:\-]?\s*([^\.\n,]+)", text
    ))
    if matches:
        raw = matches[-1].group(1).strip().rstrip(".")
        norm = _normalize_number(raw)
        return norm if norm is not None else raw

    # 4. Fallback: last number in the text
    numbers = re.findall(r"-?\d[\d,]*(?:\.\d+)?", text)
    if numbers:
        raw = numbers[-1].replace(",", "")
        return _normalize_number(raw) or raw

    return None


def answers_match(pred_ans: Optional[str], label_ans: Optional[str]) -> bool:
    """Return True when both answers are non-None and equal after normalisation."""
    if pred_ans is None or label_ans is None:
        return False
    # Try numeric comparison first to handle "42" vs "42.0"
    try:
        return float(pred_ans) == float(label_ans)
    except (ValueError, TypeError):
        return pred_ans.strip().lower() == label_ans.strip().lower()


def split_system_user_from_prompt(prompt: str) -> tuple[Optional[str], Optional[str], bool]:
    """
    Split chat-style prompt text into system/user sections.

    Expected prompt format (roughly):
      system
      <system content>
      user
      <user content>
      assistant
      ...
    """
    if not isinstance(prompt, str):
        return None, None, False

    text = prompt.strip()
    pattern = re.compile(
        r"^\s*system\s*\n(?P<system>.*?)\nuser\s*\n(?P<user>.*?)(?:\nassistant\b.*)?$",
        flags=re.IGNORECASE | re.DOTALL,
    )
    m = pattern.match(text)
    if not m:
        return None, None, False

    system_prompt = m.group("system").strip()
    user_prompt = m.group("user").strip()
    if not system_prompt or not user_prompt:
        return None, None, False
    return system_prompt, user_prompt, True


def normalize_record(record: dict) -> dict:
    """
    Normalize one JSONL record to a robust internal format.

    Added keys:
      - system_prompt
      - user_prompt
      - prompt_parse_ok
    """
    prompt = record.get("prompt", "")
    label = record.get("label", "")
    predicts = record.get("predicts", [])

    # Keep predicts as list[str] when possible.
    if isinstance(predicts, str):
        predicts = [predicts]
    elif not isinstance(predicts, list):
        predicts = []
    predicts = [p for p in predicts if isinstance(p, str)]

    system_prompt, user_prompt, parse_ok = split_system_user_from_prompt(prompt)

    normalized = dict(record)
    normalized["prompt"] = prompt if isinstance(prompt, str) else ""
    normalized["label"] = label if isinstance(label, str) else str(label)
    normalized["predicts"] = predicts
    normalized["system_prompt"] = system_prompt
    normalized["user_prompt"] = user_prompt
    normalized["prompt_parse_ok"] = parse_ok
    return normalized


# ---------------------------------------------------------------------------
# Per-sample metrics
# ---------------------------------------------------------------------------

def evaluate_sample(label: str, predicts: list[str], k: Optional[int] = None):
    """
    Compute metrics for a single sample.

    Args:
        label:    Ground-truth string (contains the answer).
        predicts: List of model predictions.
        k:        If given, only consider the first k predictions.

    Returns:
        dict with keys: pass@1, pass@k, majority@k, avg_correct,
                        label_answer, pred_answers
    """
    label_ans = extract_answer(label)
    preds = predicts[:k] if k is not None else predicts
    if not preds:
        return {
            "pass@1": False,
            "pass@k": False,
            "majority@k": False,
            "avg_correct": 0.0,
            "label_answer": label_ans,
            "pred_answers": [],
        }

    pred_answers = [extract_answer(p) for p in preds]
    correctness = [answers_match(pa, label_ans) for pa in pred_answers]

    # pass@1: first prediction correct
    pass1 = correctness[0]

    # pass@k: any prediction correct
    passk = any(correctness)

    # majority@k: most-voted answer is correct
    # Use string representation for voting; None counts as its own bucket
    vote_counter: Counter = Counter()
    for pa in pred_answers:
        vote_counter[pa if pa is not None else "__none__"] += 1
    majority_ans_key = vote_counter.most_common(1)[0][0]
    majority_ans = None if majority_ans_key == "__none__" else majority_ans_key
    majority_correct = answers_match(majority_ans, label_ans)

    avg_correct = sum(correctness) / len(correctness)

    return {
        "pass@1": pass1,
        "pass@k": passk,
        "majority@k": majority_correct,
        "avg_correct": avg_correct,
        "label_answer": label_ans,
        "pred_answers": pred_answers,
    }


# ---------------------------------------------------------------------------
# Dataset-level evaluation
# ---------------------------------------------------------------------------

def evaluate_dataset(records: list[dict], k: Optional[int] = None) -> dict:
    """
    Evaluate a list of records and aggregate metrics.

    Each record must have: "label" (str) and "predicts" (list[str]).
    "prompt" is optional and ignored here.

    Returns:
        Aggregated metric dict plus per-sample results.
    """
    per_sample = []
    total_pred_len = 0
    total_pred_count = 0
    total_prompt_parse_ok = 0

    for rec in records:
        label = rec.get("label", "")
        predicts = rec.get("predicts", [])
        result = evaluate_sample(label, predicts, k=k)
        result["prompt_parse_ok"] = bool(rec.get("prompt_parse_ok", False))
        result["system_prompt"] = rec.get("system_prompt")
        result["user_prompt"] = rec.get("user_prompt")
        per_sample.append(result)
        if result["prompt_parse_ok"]:
            total_prompt_parse_ok += 1

        # Accumulate prediction lengths (whitespace-tokenised)
        preds_used = predicts[:k] if k is not None else predicts
        for p in preds_used:
            total_pred_len += len(p.split())
            total_pred_count += 1

    n = len(per_sample)
    if n == 0:
        return {"error": "No records found"}

    agg = {
        "num_samples": n,
        "k": k,
        "pass@1":      sum(s["pass@1"]      for s in per_sample) / n,
        "pass@k":      sum(s["pass@k"]      for s in per_sample) / n,
        "majority@k":  sum(s["majority@k"]  for s in per_sample) / n,
        "avg_correct": sum(s["avg_correct"] for s in per_sample) / n,
        "prompt_parse_rate": total_prompt_parse_ok / n,
        "mean_pred_len": total_pred_len / total_pred_count if total_pred_count else 0.0,
        "per_sample": per_sample,
    }
    return agg


# ---------------------------------------------------------------------------
# I/O helpers
# ---------------------------------------------------------------------------

def load_jsonl(path: str) -> list[dict]:
    records = []
    with open(path, "r", encoding="utf-8") as f:
        for lineno, line in enumerate(f, 1):
            line = line.strip()
            if not line:
                continue
            try:
                raw = json.loads(line)
                if not isinstance(raw, dict):
                    print(f"[WARN] Line {lineno} is not a JSON object, skipping.", file=sys.stderr)
                    continue
                records.append(normalize_record(raw))
            except json.JSONDecodeError as e:
                print(f"[WARN] Line {lineno} is not valid JSON, skipping: {e}", file=sys.stderr)
    return records


def print_report(agg: dict) -> None:
    k_label = f"k={agg['k']}" if agg.get("k") else "k=all"
    print(f"\n{'='*50}")
    print(f"  GSM8K Evaluation Report  ({k_label}, n={agg['num_samples']})")
    print(f"{'='*50}")
    print(f"  pass@1        : {agg['pass@1']:.4f}  ({agg['pass@1']*100:.2f}%)")
    print(f"  pass@k        : {agg['pass@k']:.4f}  ({agg['pass@k']*100:.2f}%)")
    print(f"  majority@k    : {agg['majority@k']:.4f}  ({agg['majority@k']*100:.2f}%)")
    print(f"  avg_correct   : {agg['avg_correct']:.4f}")
    print(f"  prompt_parse  : {agg['prompt_parse_rate']:.4f}  ({agg['prompt_parse_rate']*100:.2f}%)")
    print(f"  mean_pred_len : {agg['mean_pred_len']:.1f} tokens")
    print(f"{'='*50}\n")


# ---------------------------------------------------------------------------
# CLI entry point
# ---------------------------------------------------------------------------

def main():
    parser = argparse.ArgumentParser(
        description="Evaluate GSM8K-style predictions from a JSONL file."
    )
    parser.add_argument("input", help="Path to the input JSONL file")
    parser.add_argument(
        "--k", type=int, default=None,
        help="Use only the first k predictions per sample (default: all)"
    )
    parser.add_argument(
        "--output", type=str, default=None,
        help="Optional path to save the full evaluation results as JSON"
    )
    parser.add_argument(
        "--no-per-sample", action="store_true",
        help="Exclude per-sample details from the saved output JSON"
    )
    args = parser.parse_args()

    print(f"Loading {args.input} ...")
    records = load_jsonl(args.input)
    print(f"  {len(records)} records loaded.")

    agg = evaluate_dataset(records, k=args.k)
    print_report(agg)

    if args.output:
        save_data = dict(agg)
        if args.no_per_sample:
            save_data.pop("per_sample", None)
        Path(args.output).write_text(
            json.dumps(save_data, ensure_ascii=False, indent=2), encoding="utf-8"
        )
        print(f"Results saved to {args.output}")


if __name__ == "__main__":
    main()
