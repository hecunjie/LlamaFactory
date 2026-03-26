#!/usr/bin/env python3
"""
统计 token-level 数据中 `<add_think>` 出现在 top5 的频率，并按 case 分组。

输入支持：
1) JSONL（每行一个 dict）
2) JSON 文件（单个 dict 或 dict 列表）

每条记录至少应包含：
- top5_tokens: list[str]
- case: str (可缺省，缺省为 "__MISSING_CASE__")
- is_correct: bool (可缺省，缺省为 "__ALL__" 分组下统计)
"""

from __future__ import annotations

import argparse
import csv
import json
from pathlib import Path
from typing import Any


ADD_THINK_TOKEN = "<add_think>"
MISSING_CASE = "__MISSING_CASE__"
ALL_CORRECTNESS = "__ALL__"


def load_records(path: Path) -> list[dict[str, Any]]:
    if not path.exists():
        raise FileNotFoundError(f"输入文件不存在: {path}")

    if path.suffix.lower() == ".jsonl":
        records: list[dict[str, Any]] = []
        with path.open("r", encoding="utf-8") as f:
            for lineno, line in enumerate(f, 1):
                line = line.strip()
                if not line:
                    continue
                try:
                    obj = json.loads(line)
                except json.JSONDecodeError as e:
                    print(f"[WARN] 第 {lineno} 行 JSON 解析失败，跳过: {e}")
                    continue
                if isinstance(obj, dict):
                    records.append(obj)
                else:
                    print(f"[WARN] 第 {lineno} 行不是 JSON object，跳过。")
        return records

    raw = json.loads(path.read_text(encoding="utf-8"))
    if isinstance(raw, dict):
        return [raw]
    if isinstance(raw, list):
        out = []
        for idx, item in enumerate(raw):
            if isinstance(item, dict):
                out.append(item)
            else:
                print(f"[WARN] JSON 列表第 {idx} 项不是 object，跳过。")
        return out
    raise ValueError("JSON 文件内容必须是 object 或 object 列表。")


def normalize_top5_tokens(value: Any) -> list[str]:
    if isinstance(value, list):
        return [str(x) for x in value]
    return []


def has_add_think_in_top5(rec: dict[str, Any], target_token: str) -> bool:
    top5_tokens = normalize_top5_tokens(rec.get("top5_tokens"))
    return target_token in top5_tokens


def format_pct(numerator: int, denominator: int) -> float:
    if denominator == 0:
        return 0.0
    return numerator / denominator


def main() -> None:
    parser = argparse.ArgumentParser(
        description="统计 `<add_think>` 在 top5_tokens 中的出现频率，并按 case 分组。"
    )
    parser.add_argument("input", type=str, help="输入 JSONL/JSON 文件路径")
    parser.add_argument(
        "--token",
        type=str,
        default=ADD_THINK_TOKEN,
        help="目标 token，默认 <add_think>",
    )
    parser.add_argument(
        "--output-json",
        type=str,
        default=None,
        help="可选：保存汇总统计 JSON 的路径",
    )
    parser.add_argument(
        "--output-csv",
        type=str,
        default=None,
        help="可选：保存分组统计 CSV 的路径",
    )
    args = parser.parse_args()

    input_path = Path(args.input)
    records = load_records(input_path)
    if not records:
        print("输入数据为空，退出。")
        return

    # 分组键：
    # 1) 仅按 case
    # 2) 按 case + is_correct
    case_stats: dict[str, dict[str, int]] = {}
    case_correctness_stats: dict[str, dict[str, dict[str, int]]] = {}
    invalid_top5_count = 0

    for rec in records:
        case_name = str(rec.get("case", MISSING_CASE))
        is_correct_raw = rec.get("is_correct", ALL_CORRECTNESS)
        if isinstance(is_correct_raw, bool):
            correctness_name = "correct" if is_correct_raw else "incorrect"
        else:
            correctness_name = ALL_CORRECTNESS

        top5_tokens_raw = rec.get("top5_tokens")
        top5_tokens = normalize_top5_tokens(top5_tokens_raw)
        if not isinstance(top5_tokens_raw, list):
            invalid_top5_count += 1

        hit = args.token in top5_tokens

        if case_name not in case_stats:
            case_stats[case_name] = {"total": 0, "hit": 0}
        case_stats[case_name]["total"] += 1
        if hit:
            case_stats[case_name]["hit"] += 1

        if case_name not in case_correctness_stats:
            case_correctness_stats[case_name] = {}
        if correctness_name not in case_correctness_stats[case_name]:
            case_correctness_stats[case_name][correctness_name] = {"total": 0, "hit": 0}

        case_correctness_stats[case_name][correctness_name]["total"] += 1
        if hit:
            case_correctness_stats[case_name][correctness_name]["hit"] += 1

    overall_total = len(records)
    overall_hit = sum(1 for rec in records if has_add_think_in_top5(rec, args.token))

    case_rows = []
    for case_name, stat in sorted(case_stats.items(), key=lambda x: x[0]):
        total = stat["total"]
        hit = stat["hit"]
        case_rows.append(
            {
                "case": case_name,
                "total": total,
                "hit": hit,
                "frequency": format_pct(hit, total),
            }
        )

    case_correctness_rows = []
    for case_name in sorted(case_correctness_stats.keys()):
        for correctness_name in sorted(case_correctness_stats[case_name].keys()):
            stat = case_correctness_stats[case_name][correctness_name]
            total = stat["total"]
            hit = stat["hit"]
            case_correctness_rows.append(
                {
                    "case": case_name,
                    "correctness": correctness_name,
                    "total": total,
                    "hit": hit,
                    "frequency": format_pct(hit, total),
                }
            )

    summary = {
        "meta": {
            "input": str(input_path),
            "target_token": args.token,
            "num_records": overall_total,
            "invalid_top5_tokens_field_count": invalid_top5_count,
        },
        "overall": {
            "total": overall_total,
            "hit": overall_hit,
            "frequency": format_pct(overall_hit, overall_total),
        },
        "by_case": case_rows,
        "by_case_and_correctness": case_correctness_rows,
    }

    print("\n=== Overall ===")
    print(
        f"token={args.token} | hit={summary['overall']['hit']}/{summary['overall']['total']} "
        f"({summary['overall']['frequency'] * 100:.2f}%)"
    )
    if invalid_top5_count > 0:
        print(f"[WARN] {invalid_top5_count} 条记录的 top5_tokens 不是 list，按空列表处理。")

    print("\n=== By case ===")
    for row in case_rows:
        print(
            f"{row['case']}: {row['hit']}/{row['total']} "
            f"({row['frequency'] * 100:.2f}%)"
        )

    print("\n=== By case + correctness ===")
    for row in case_correctness_rows:
        print(
            f"{row['case']} | {row['correctness']}: {row['hit']}/{row['total']} "
            f"({row['frequency'] * 100:.2f}%)"
        )

    if args.output_json:
        out_json = Path(args.output_json)
        out_json.write_text(json.dumps(summary, ensure_ascii=False, indent=2), encoding="utf-8")
        print(f"\nSaved JSON: {out_json.resolve()}")

    if args.output_csv:
        out_csv = Path(args.output_csv)
        out_csv.parent.mkdir(parents=True, exist_ok=True)
        with out_csv.open("w", encoding="utf-8", newline="") as f:
            writer = csv.DictWriter(
                f,
                fieldnames=["case", "correctness", "total", "hit", "frequency"],
            )
            writer.writeheader()
            for row in case_correctness_rows:
                writer.writerow(row)
        print(f"Saved CSV: {out_csv.resolve()}")


if __name__ == "__main__":
    main()
