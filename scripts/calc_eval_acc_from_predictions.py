#!/usr/bin/env python3
"""
Calculate exact-match accuracy from intermediate eval prediction JSONL files.

Expected JSONL fields per line:
  - predict_answer
  - label

Usage:
  python scripts/calc_eval_acc_from_predictions.py \
    --input_dir /path/to/output_dir/eval_predictions \
    --output_json /path/to/output_dir/eval_predictions/acc_summary.json
"""

import argparse
import json
import re
from pathlib import Path
from typing import Optional


def _normalize_extracted_answer(text: str) -> str:
    ans = text.strip()
    ans = ans.strip(" \t\r\n\"'`")
    ans = re.sub(r'[\"\'}\]\s]+$', "", ans).strip()
    paren_match = re.match(r"^\s*([^\(\)]+?)\s*\([^)]*\)\s*$", ans)
    if paren_match:
        ans = paren_match.group(1).strip()
    return ans


def _extract_boxed_content(text: str) -> Optional[str]:
    key = r"\boxed{"
    start = text.rfind(key)
    if start == -1:
        return None

    i = start + len(key)
    depth = 1
    buf = []
    while i < len(text):
        ch = text[i]
        if ch == "{":
            depth += 1
            buf.append(ch)
        elif ch == "}":
            depth -= 1
            if depth == 0:
                return "".join(buf).strip()
            buf.append(ch)
        else:
            buf.append(ch)
        i += 1
    return None


def extract_answer_from_output(text: str) -> str:
    text = (text or "").strip()

    boxed = _extract_boxed_content(text)
    if boxed:
        return _normalize_extracted_answer(boxed)

    if "####" in text:
        return _normalize_extracted_answer(text.split("####")[-1].strip())

    if "###" in text:
        return _normalize_extracted_answer(text.split("###")[-1].strip())

    final_answer_matches = re.findall(r"The final answer is\s*[:\-]?\s*(.+)", text, flags=re.IGNORECASE)
    if final_answer_matches:
        return _normalize_extracted_answer(final_answer_matches[-1].strip())

    answer_is_matches = re.findall(r"The answer is\s*(.+)", text, flags=re.IGNORECASE)
    if answer_is_matches:
        return _normalize_extracted_answer(answer_is_matches[-1].strip())

    return _normalize_extracted_answer(text)


def extract_last_number(text: str) -> Optional[float]:
    text = text.replace(",", "")
    numbers = re.findall(r"-?\d+\.?\d*", text)
    if numbers:
        try:
            return float(numbers[-1])
        except ValueError:
            return None
    return None


def answers_match(pred: str, label: str) -> bool:
    if pred == label:
        return True
    pred_num = extract_last_number(pred)
    label_num = extract_last_number(label)
    if pred_num is not None and label_num is not None:
        return abs(pred_num - label_num) < 1e-6
    return False


def evaluate_file(path: Path) -> dict:
    total = 0
    correct = 0
    with path.open("r", encoding="utf-8") as f:
        for line in f:
            line = line.strip()
            if not line:
                continue
            rec = json.loads(line)
            pred = extract_answer_from_output(str(rec.get("predict_answer", "")))
            label = extract_answer_from_output(str(rec.get("label", "")))
            total += 1
            if answers_match(pred, label):
                correct += 1

    acc = (correct / total) if total > 0 else 0.0
    m = re.search(r"_step_(\d+)\.jsonl$", path.name)
    step = int(m.group(1)) if m else -1
    return {"file": path.name, "step": step, "num_samples": total, "num_correct": correct, "acc": acc}


def main():
    parser = argparse.ArgumentParser(description="Compute step-wise eval acc from saved JSONL predictions.")
    parser.add_argument("--input_dir", type=str, required=True, help="Directory of eval prediction JSONL files.")
    parser.add_argument("--glob", type=str, default="*.jsonl", help="File glob pattern in input_dir.")
    parser.add_argument("--output_json", type=str, default=None, help="Optional output JSON path.")
    args = parser.parse_args()

    input_dir = Path(args.input_dir)
    files = sorted(input_dir.glob(args.glob))
    if not files:
        raise FileNotFoundError(f"No files matched {args.glob} under {input_dir}")

    rows = [evaluate_file(p) for p in files]
    rows.sort(key=lambda x: (x["step"], x["file"]))

    print("step\tacc\tnum_correct/num_samples\tfile")
    for r in rows:
        print(f"{r['step']}\t{r['acc']:.6f}\t{r['num_correct']}/{r['num_samples']}\t{r['file']}")

    if args.output_json:
        out_path = Path(args.output_json)
        out_path.parent.mkdir(parents=True, exist_ok=True)
        out_path.write_text(json.dumps(rows, ensure_ascii=False, indent=2), encoding="utf-8")
        print(f"\nSaved summary to: {out_path}")


if __name__ == "__main__":
    main()

