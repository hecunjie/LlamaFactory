#!/usr/bin/env python3
"""
Stat <add_think> token ratio among correct/incorrect answers.

Input JSONL (one object per line), each record may contain:
  - "label": ground truth answer text
  - "predict": single prediction string (optional)
  - "predicts": list of prediction strings (optional)

Correctness is judged by `extract_answer` + `answers_match` from `cal_metric.py`.
Ratio is computed by token count:
  <add_think> token count / total token count
"""

import argparse
import json
from pathlib import Path
from typing import Iterable

from cal_metric import answers_match, extract_answer


ADD_THINK_TOKEN = "<add_think>"


def iter_predictions(record: dict, dedup_per_record: bool = False) -> Iterable[str]:
    preds: list[str] = []

    predict = record.get("predict")
    if isinstance(predict, str):
        preds.append(predict)

    predicts = record.get("predicts")
    if isinstance(predicts, list):
        preds.extend(p for p in predicts if isinstance(p, str))
    elif isinstance(predicts, str):
        preds.append(predicts)

    if not dedup_per_record:
        return preds

    seen = set()
    uniq = []
    for p in preds:
        if p in seen:
            continue
        seen.add(p)
        uniq.append(p)
    return uniq


def safe_ratio(num: int, den: int) -> float:
    return num / den if den else 0.0


def token_stats(text: str) -> tuple[int, int]:
    """
    Return (add_think_token_count, total_token_count).

    We force `<add_think>` to be tokenized as a standalone token even when it is
    attached to words like `abc<add_think>def`.
    """
    normalized = text.replace(ADD_THINK_TOKEN, f" {ADD_THINK_TOKEN} ")
    tokens = normalized.split()
    add_think_count = sum(1 for t in tokens if t == ADD_THINK_TOKEN)
    return add_think_count, len(tokens)


def load_jsonl(path: str) -> list[dict]:
    records = []
    with open(path, "r", encoding="utf-8") as f:
        for lineno, line in enumerate(f, 1):
            line = line.strip()
            if not line:
                continue
            try:
                obj = json.loads(line)
                if isinstance(obj, dict):
                    records.append(obj)
                else:
                    print(f"[WARN] Line {lineno} is not a JSON object, skipped.")
            except json.JSONDecodeError as e:
                print(f"[WARN] Line {lineno} invalid JSON, skipped: {e}")
    return records


def main() -> None:
    parser = argparse.ArgumentParser(
        description="Compute <add_think> ratio in correct/incorrect predictions."
    )
    parser.add_argument("input", help="Path to input JSONL file")
    parser.add_argument(
        "--dedup-per-record",
        action="store_true",
        help="Deduplicate identical prediction texts inside each record",
    )
    parser.add_argument(
        "--output",
        type=str,
        default=None,
        help="Optional path to save stats as JSON",
    )
    args = parser.parse_args()

    records = load_jsonl(args.input)
    print(f"Loaded records: {len(records)}")

    total_answers = 0
    total_add_think_tokens = 0
    total_tokens = 0

    correct_answers = 0
    correct_add_think_tokens = 0
    correct_tokens = 0

    wrong_answers = 0
    wrong_add_think_tokens = 0
    wrong_tokens = 0

    missing_label_answer = 0

    for rec in records:
        label = rec.get("label", "")
        label_ans = extract_answer(label if isinstance(label, str) else str(label))
        if label_ans is None:
            missing_label_answer += 1

        preds = iter_predictions(rec, dedup_per_record=args.dedup_per_record)
        for pred in preds:
            total_answers += 1
            add_think_cnt, token_cnt = token_stats(pred)
            total_add_think_tokens += add_think_cnt
            total_tokens += token_cnt

            pred_ans = extract_answer(pred)
            is_correct = answers_match(pred_ans, label_ans)

            if is_correct:
                correct_answers += 1
                correct_add_think_tokens += add_think_cnt
                correct_tokens += token_cnt
            else:
                wrong_answers += 1
                wrong_add_think_tokens += add_think_cnt
                wrong_tokens += token_cnt

    result = {
        "num_records": len(records),
        "num_answers_used": total_answers,
        "dedup_per_record": args.dedup_per_record,
        "missing_label_answer": missing_label_answer,
        "overall": {
            "add_think_token_ratio": safe_ratio(total_add_think_tokens, total_tokens),
            "add_think_tokens": total_add_think_tokens,
            "total_tokens": total_tokens,
        },
        "correct": {
            "add_think_token_ratio": safe_ratio(correct_add_think_tokens, correct_tokens),
            "add_think_tokens": correct_add_think_tokens,
            "total_tokens": correct_tokens,
            "answers": correct_answers,
        },
        "incorrect": {
            "add_think_token_ratio": safe_ratio(wrong_add_think_tokens, wrong_tokens),
            "add_think_tokens": wrong_add_think_tokens,
            "total_tokens": wrong_tokens,
            "answers": wrong_answers,
        },
    }

    print("\n=== <add_think> token statistics ===")
    print(f"answers used: {result['num_answers_used']}")
    print(
        f"correct   : {correct_add_think_tokens}/{correct_tokens} = "
        f"{result['correct']['add_think_token_ratio']:.4f} "
        f"({result['correct']['add_think_token_ratio'] * 100:.2f}%)"
    )
    print(
        f"incorrect : {wrong_add_think_tokens}/{wrong_tokens} = "
        f"{result['incorrect']['add_think_token_ratio']:.4f} "
        f"({result['incorrect']['add_think_token_ratio'] * 100:.2f}%)"
    )
    print(
        f"overall   : {total_add_think_tokens}/{total_tokens} = "
        f"{result['overall']['add_think_token_ratio']:.4f} "
        f"({result['overall']['add_think_token_ratio'] * 100:.2f}%)"
    )

    if args.output:
        Path(args.output).write_text(
            json.dumps(result, ensure_ascii=False, indent=2),
            encoding="utf-8",
        )
        print(f"\nSaved JSON stats to: {args.output}")


if __name__ == "__main__":
    main()
