#!/usr/bin/env python3
"""
从 JSONL 中筛出 ``case`` 字段以指定前缀开头的行（默认 B_ 类，如 B_lowconf），写入新文件。

流式处理，适合大文件。
"""

from __future__ import annotations

import argparse
import json
from pathlib import Path


def default_output_path(in_path: Path, prefix: str) -> Path:
    if prefix == "B_":
        return in_path.with_name(f"{in_path.stem}_B_class.jsonl")
    safe = "".join(c if c.isalnum() or c in "-_" else "_" for c in prefix)[:48]
    return in_path.with_name(f"{in_path.stem}_case_{safe}.jsonl")


def main() -> None:
    parser = argparse.ArgumentParser(
        description="筛选 JSONL 中 case 以给定前缀开头的记录（默认前缀 B_）。"
    )
    parser.add_argument("input", type=str, help="输入 .jsonl 路径")
    parser.add_argument(
        "-o",
        "--output",
        type=str,
        default=None,
        help="输出路径；未指定时按前缀生成（B_ → <stem>_B_class.jsonl）",
    )
    parser.add_argument(
        "--prefix",
        type=str,
        default="B_",
        help='只保留 case 以此字符串开头的行（默认 "B_"）',
    )
    parser.add_argument(
        "--strict",
        action="store_true",
        help="任一行 JSON 解析失败或非 dict 时立即退出（默认跳过并警告）",
    )
    args = parser.parse_args()

    in_path = Path(args.input)
    if not in_path.is_file():
        raise SystemExit(f"输入文件不存在: {in_path}")

    prefix = args.prefix
    if args.output:
        out_path = Path(args.output)
    else:
        out_path = default_output_path(in_path, prefix)

    kept = 0
    total = 0
    bad = 0

    with in_path.open("r", encoding="utf-8") as fin, out_path.open("w", encoding="utf-8") as fout:
        for lineno, line in enumerate(fin, 1):
            line = line.strip()
            if not line:
                continue
            total += 1
            try:
                obj = json.loads(line)
            except json.JSONDecodeError as e:
                bad += 1
                if not args.strict:
                    print(f"[WARN] 第 {lineno} 行 JSON 解析失败，跳过: {e}")
                    continue
                raise SystemExit(f"第 {lineno} 行 JSON 解析失败: {e}") from e

            if not isinstance(obj, dict):
                bad += 1
                if not args.strict:
                    print(f"[WARN] 第 {lineno} 行不是 JSON object，跳过。")
                    continue
                raise SystemExit(f"第 {lineno} 行不是 JSON object。")

            case = obj.get("case")
            if isinstance(case, str) and case.startswith(prefix):
                fout.write(json.dumps(obj, ensure_ascii=False) + "\n")
                kept += 1

    print(f"读取行数（非空）: {total}")
    print(f"写入条数 ({prefix!r}): {kept}")
    if bad:
        print(f"跳过/异常: {bad}")
    print(f"输出: {out_path}")


if __name__ == "__main__":
    main()
