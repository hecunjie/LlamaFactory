#!/usr/bin/env python3
"""
分析仅包含低 LSE token 的 JSONL，进行 logit 分解并判断假说 B/C 的支持情况。

输入每行至少包含：
- log_sum_exp: float
- top5_log_probs: list[float] (长度 >= 1，通常为 5)
- top5_probs: list[float] (长度 >= 1，通常为 5)
- entropy: float
- is_correct: bool
"""

from __future__ import annotations

import argparse
import json
from pathlib import Path
from typing import Any

REQUIRED_KEYS = {"log_sum_exp", "top5_log_probs", "top5_probs", "entropy", "is_correct"}


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description="对低 LSE JSONL 做 LSE = top1_logit + tail_contribution 分解并绘图。"
    )
    parser.add_argument("input", type=str, help="输入 JSONL 路径（仅低 LSE token）")
    parser.add_argument(
        "--output-plot",
        type=str,
        default="lse_decomposition.png",
        help="输出图像路径（默认: lse_decomposition.png）",
    )
    parser.add_argument(
        "--output-csv",
        type=str,
        default=None,
        help="可选：保存分解后的逐条明细 CSV 路径",
    )
    parser.add_argument(
        "--strict",
        action="store_true",
        help="严格模式：任一行格式问题立即退出；默认跳过并警告。",
    )
    return parser.parse_args()


def _validate_record(rec: dict[str, Any], lineno: int) -> None:
    missing = REQUIRED_KEYS - set(rec.keys())
    if missing:
        raise ValueError(f"第 {lineno} 行缺少字段: {sorted(missing)}")

    if not isinstance(rec["top5_log_probs"], list) or len(rec["top5_log_probs"]) < 1:
        raise ValueError(f"第 {lineno} 行 top5_log_probs 不是非空列表")
    if not isinstance(rec["top5_probs"], list) or len(rec["top5_probs"]) < 1:
        raise ValueError(f"第 {lineno} 行 top5_probs 不是非空列表")


def load_dataframe(input_path: Path, strict: bool) -> pd.DataFrame:
    import pandas as pd

    rows: list[dict[str, Any]] = []
    bad = 0
    total_non_empty = 0

    with input_path.open("r", encoding="utf-8") as f:
        for lineno, raw_line in enumerate(f, 1):
            line = raw_line.strip()
            if not line:
                continue
            total_non_empty += 1

            try:
                rec = json.loads(line)
                if not isinstance(rec, dict):
                    raise ValueError(f"第 {lineno} 行不是 JSON object")
                _validate_record(rec, lineno)

                lse = float(rec["log_sum_exp"])
                top1_log_prob = float(rec["top5_log_probs"][0])  # 负数
                top1_logit = lse + top1_log_prob
                tail = -top1_log_prob

                rows.append(
                    {
                        "sample_idx": rec.get("sample_idx"),
                        "t": rec.get("t"),
                        "is_correct": bool(rec["is_correct"]),
                        "entropy": float(rec["entropy"]),
                        "log_sum_exp": lse,
                        "top1_prob": float(rec["top5_probs"][0]),
                        "top1_logit": top1_logit,
                        "tail_contribution": tail,
                    }
                )
            except Exception as e:  # noqa: BLE001
                bad += 1
                if strict:
                    raise SystemExit(f"解析失败: {e}") from e
                print(f"[WARN] 第 {lineno} 行跳过: {e}")

    df = pd.DataFrame(rows)
    print(f"读取非空行数: {total_non_empty}")
    print(f"有效记录数: {len(df)}")
    if bad:
        print(f"跳过异常行数: {bad}")
    return df


def analyze_and_plot(df: pd.DataFrame, output_plot: Path) -> None:
    import matplotlib.pyplot as plt
    import numpy as np
    from scipy import stats

    if df.empty:
        raise SystemExit("无有效记录，无法分析。")

    print("\n-- 描述统计 --")
    print(
        df[["log_sum_exp", "top1_logit", "tail_contribution", "entropy", "top1_prob"]]
        .describe()
        .round(4)
    )

    mean_lse = df["log_sum_exp"].mean()
    mean_top1 = df["top1_logit"].mean()
    mean_tail = df["tail_contribution"].mean()

    print("\n-- LSE 分解 --")
    print(f"mean LSE               = {mean_lse:.4f}")
    if abs(mean_lse) > 1e-12:
        print(f"mean top1_logit        = {mean_top1:.4f}  ({mean_top1 / mean_lse * 100:.1f}% of LSE)")
        print(f"mean tail_contribution = {mean_tail:.4f}  ({mean_tail / mean_lse * 100:.1f}% of LSE)")
    else:
        print(f"mean top1_logit        = {mean_top1:.4f}")
        print(f"mean tail_contribution = {mean_tail:.4f}")
    print(f"验证(应≈0): top1+tail-LSE = {(mean_top1 + mean_tail - mean_lse):.6f}")

    rho_top1, p_top1 = stats.spearmanr(df["log_sum_exp"], df["top1_logit"])
    rho_tail, p_tail = stats.spearmanr(df["log_sum_exp"], df["tail_contribution"])
    rho_entr, p_entr = stats.spearmanr(df["log_sum_exp"], df["entropy"])

    print("\n-- Spearman 相关（低 LSE 组内部）--")
    print(f"rho(LSE, top1_logit)        = {rho_top1:.4f}  p={p_top1:.2e}")
    print(f"rho(LSE, tail_contribution) = {rho_tail:.4f}  p={p_tail:.2e}")
    print(f"rho(LSE, entropy)           = {rho_entr:.4f}  p={p_entr:.2e}")

    print("\n解读参考：")
    print("  rho(LSE, top1_logit) 接近 +1 且 rho(LSE, tail) 接近 0 -> 更支持假说 B（top1_logit 偏低）")
    print("  rho(LSE, tail) 接近 -1 且 rho(LSE, top1_logit) 接近 0 -> 更支持假说 C（概率分散）")

    fig, axes = plt.subplots(1, 3, figsize=(15, 5))

    axes[0].scatter(df["log_sum_exp"], df["top1_logit"], alpha=0.15, s=8)
    axes[0].set_xlabel("log_sum_exp (LSE)")
    axes[0].set_ylabel("top1_logit")
    axes[0].set_title(f"LSE vs top1_logit\nSpearman rho={rho_top1:.3f}")
    lse_range = np.linspace(df["log_sum_exp"].min(), df["log_sum_exp"].max(), 100)
    axes[0].plot(lse_range, lse_range, "r--", linewidth=1, label="tail=0 baseline")
    axes[0].legend()

    axes[1].scatter(df["log_sum_exp"], df["tail_contribution"], alpha=0.15, s=8, color="orange")
    axes[1].set_xlabel("log_sum_exp (LSE)")
    axes[1].set_ylabel("tail_contribution = -log(p1)")
    axes[1].set_title(f"LSE vs tail_contribution\nSpearman rho={rho_tail:.3f}")

    axes[2].hist(
        df["top1_logit"],
        bins=80,
        alpha=0.6,
        label=f"top1_logit (mean={mean_top1:.2f})",
        density=True,
    )
    axes[2].hist(
        df["tail_contribution"],
        bins=80,
        alpha=0.6,
        label=f"tail (mean={mean_tail:.2f})",
        density=True,
    )
    axes[2].axvline(mean_top1, color="blue", linestyle="--")
    axes[2].axvline(mean_tail, color="orange", linestyle="--")
    axes[2].set_xlabel("value")
    axes[2].set_ylabel("density")
    axes[2].set_title("Distribution of LSE Components")
    axes[2].legend()

    plt.suptitle(f"LSE Decomposition on Low-LSE Tokens (n={len(df)})", fontsize=13)
    plt.tight_layout()
    output_plot.parent.mkdir(parents=True, exist_ok=True)
    plt.savefig(output_plot, dpi=150)
    plt.close(fig)
    print(f"\n图已保存: {output_plot}")


def main() -> None:
    args = parse_args()
    input_path = Path(args.input)
    if not input_path.is_file():
        raise SystemExit(f"输入文件不存在: {input_path}")

    df = load_dataframe(input_path=input_path, strict=args.strict)
    if args.output_csv:
        output_csv = Path(args.output_csv)
        output_csv.parent.mkdir(parents=True, exist_ok=True)
        df.to_csv(output_csv, index=False)
        print(f"明细 CSV 已保存: {output_csv}")

    analyze_and_plot(df=df, output_plot=Path(args.output_plot))


if __name__ == "__main__":
    main()
