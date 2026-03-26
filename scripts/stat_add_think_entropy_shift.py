#!/usr/bin/env python3
from __future__ import annotations

import argparse
import csv
import json
import math
import re
from pathlib import Path
from typing import Any, Optional

import matplotlib.pyplot as plt
import numpy as np
import torch
from tqdm import tqdm
from transformers import AutoModelForCausalLM, AutoTokenizer


def _normalize_number(text: str) -> Optional[str]:
    text = text.strip().replace(",", "").replace("$", "").replace("%", "").rstrip(".")
    m = re.fullmatch(r"-?\d+(?:\.\d+)?", text)
    if m:
        try:
            val = float(text)
            if not math.isfinite(val):
                return None
            if val == int(val):
                return str(int(val))
            return str(val)
        except (ValueError, OverflowError):
            pass

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
    text = str(text).strip()

    matches = list(re.finditer(r"####\s*(.+?)(?:\n|$)", text))
    if matches:
        raw = matches[-1].group(1).strip()
        norm = _normalize_number(raw)
        return norm if norm is not None else raw

    matches = list(re.finditer(r"(?<!#)###(?![#])\s*(.+?)(?:\n|$)", text))
    if matches:
        raw = matches[-1].group(1).strip()
        norm = _normalize_number(raw)
        return norm if norm is not None else raw

    matches = list(re.finditer(r"[Tt]he\s+answer\s+is\s+[:\-]?\s*([^\.\n,]+)", text))
    if matches:
        raw = matches[-1].group(1).strip().rstrip(".")
        norm = _normalize_number(raw)
        return norm if norm is not None else raw

    numbers = re.findall(r"-?\d[\d,]*(?:\.\d+)?", text)
    if numbers:
        raw = numbers[-1].replace(",", "")
        return _normalize_number(raw) or raw
    return None


def answers_match(pred_ans: Optional[str], label_ans: Optional[str]) -> bool:
    if pred_ans is None or label_ans is None:
        return False
    try:
        return float(pred_ans) == float(label_ans)
    except (ValueError, TypeError):
        return pred_ans.strip().lower() == label_ans.strip().lower()


def load_jsonl(path: Path) -> list[dict[str, Any]]:
    rows: list[dict[str, Any]] = []
    with path.open("r", encoding="utf-8") as f:
        for line in f:
            line = line.strip()
            if line:
                rows.append(json.loads(line))
    return rows


def summarize_deltas(deltas: list[float], eps: float) -> dict[str, Any]:
    if not deltas:
        return {
            "count": 0,
            "increase": 0,
            "decrease": 0,
            "unchanged": 0,
            "increase_ratio": 0.0,
            "decrease_ratio": 0.0,
            "unchanged_ratio": 0.0,
            "mean": float("nan"),
            "std": float("nan"),
            "min": float("nan"),
            "max": float("nan"),
            "p10": float("nan"),
            "p50": float("nan"),
            "p90": float("nan"),
        }

    arr = np.asarray(deltas, dtype=np.float64)
    inc = int((arr > eps).sum())
    dec = int((arr < -eps).sum())
    same = int(arr.size - inc - dec)
    return {
        "count": int(arr.size),
        "increase": inc,
        "decrease": dec,
        "unchanged": same,
        "increase_ratio": float(inc / arr.size),
        "decrease_ratio": float(dec / arr.size),
        "unchanged_ratio": float(same / arr.size),
        "mean": float(arr.mean()),
        "std": float(arr.std(ddof=0)),
        "min": float(arr.min()),
        "max": float(arr.max()),
        "p10": float(np.quantile(arr, 0.10)),
        "p50": float(np.quantile(arr, 0.50)),
        "p90": float(np.quantile(arr, 0.90)),
    }


def plot_delta_histogram(
    correct_deltas: list[float],
    wrong_deltas: list[float],
    output_path: Path,
    bins: int = 50,
) -> None:
    correct_arr = np.asarray(correct_deltas, dtype=np.float64)
    wrong_arr = np.asarray(wrong_deltas, dtype=np.float64)
    correct_arr = correct_arr[np.isfinite(correct_arr)]
    wrong_arr = wrong_arr[np.isfinite(wrong_arr)]

    fig, ax = plt.subplots(figsize=(10, 6))
    has_any = False

    if correct_arr.size > 0:
        ax.hist(
            correct_arr,
            bins=bins,
            alpha=0.55,
            color="tab:blue",
            label=f"correct (n={correct_arr.size})",
        )
        ax.axvline(float(correct_arr.mean()), color="tab:blue", linestyle="--", linewidth=1.5)
        has_any = True

    if wrong_arr.size > 0:
        ax.hist(
            wrong_arr,
            bins=bins,
            alpha=0.55,
            color="tab:red",
            label=f"wrong (n={wrong_arr.size})",
        )
        ax.axvline(float(wrong_arr.mean()), color="tab:red", linestyle="--", linewidth=1.5)
        has_any = True

    ax.axvline(0.0, color="black", linestyle="-", linewidth=1.2, alpha=0.8, label="delta=0")
    ax.set_xlabel("Entropy shift delta (after - before)")
    ax.set_ylabel("Count")
    ax.set_title("Entropy Shift Distribution Around <add_think>")
    ax.grid(True, alpha=0.3)

    if has_any:
        ax.legend()
    else:
        ax.text(0.5, 0.5, "No valid delta data.", ha="center", va="center", transform=ax.transAxes)

    fig.tight_layout()
    fig.savefig(output_path, dpi=160)
    plt.close(fig)


def main() -> None:
    parser = argparse.ArgumentParser(
        description="统计正确/错误样本在 <add_think> 触发前后的熵变化分布。"
    )
    parser.add_argument("--data", type=str, required=True, help="输入 JSONL，需包含 prompt/predict/label")
    parser.add_argument("--model", type=str, required=True, help="本地模型路径")
    parser.add_argument("--max_samples", type=int, default=2000, help="最多处理样本数")
    parser.add_argument("--batch_size", type=int, default=4, help="batch size")
    parser.add_argument(
        "--eps",
        type=float,
        default=1e-6,
        help="判定增大/减小/不变的阈值（|delta|<=eps 视为不变）",
    )
    parser.add_argument(
        "--output_dir",
        type=str,
        default="add_think_entropy_shift_outputs",
        help="输出目录：汇总 JSON、逐触发点 CSV、直方图都保存在该目录下",
    )
    parser.add_argument(
        "--hist_bins",
        type=int,
        default=50,
        help="直方图 bins 数量",
    )
    args = parser.parse_args()

    data_path = Path(args.data)
    out_dir = Path(args.output_dir)
    out_dir.mkdir(parents=True, exist_ok=True)
    out_json = out_dir / "add_think_entropy_shift_summary.json"
    out_csv = out_dir / "add_think_entropy_shift_rows.csv"
    out_plot = out_dir / "add_think_entropy_shift_hist.png"

    rows = load_jsonl(data_path)
    if not rows:
        print("输入数据为空，退出。")
        return
    rows = rows[: args.max_samples]

    tokenizer = AutoTokenizer.from_pretrained(args.model, trust_remote_code=True)
    if tokenizer.pad_token_id is None:
        tokenizer.pad_token = tokenizer.eos_token

    model = AutoModelForCausalLM.from_pretrained(
        args.model,
        torch_dtype=torch.float16,
        device_map="auto",
        trust_remote_code=True,
        low_cpu_mem_usage=True,
    ).eval()
    device = model.lm_head.weight.device

    add_think_id = tokenizer.convert_tokens_to_ids("<add_think>")
    if isinstance(add_think_id, list):
        add_think_id = add_think_id[0] if add_think_id else -1
    if add_think_id is None or int(add_think_id) < 0:
        raise ValueError("tokenizer 中未找到 <add_think>，无法统计。")
    add_think_id = int(add_think_id)

    per_trigger_rows: list[dict[str, Any]] = []
    correct_deltas: list[float] = []
    wrong_deltas: list[float] = []
    samples_with_trigger = 0

    for st in tqdm(range(0, len(rows), args.batch_size), desc="Analyzing"):
        batch = rows[st : st + args.batch_size]
        prompts = [str(r.get("prompt", "")) for r in batch]
        predicts = [str(r.get("predict", "")) for r in batch]
        labels = [str(r.get("label", "")) for r in batch]
        full_texts = [p + pr for p, pr in zip(prompts, predicts)]

        enc = tokenizer(
            full_texts,
            return_tensors="pt",
            padding=True,
            truncation=False,
            add_special_tokens=False,
        )
        input_ids = enc["input_ids"].to(device)
        attention_mask = enc["attention_mask"].to(device)

        prompt_lens = [
            len(tokenizer(p, add_special_tokens=False)["input_ids"])
            for p in prompts
        ]
        predict_ids_list = [
            tokenizer(pr, add_special_tokens=False)["input_ids"]
            for pr in predicts
        ]

        with torch.no_grad():
            outputs = model(input_ids=input_ids, attention_mask=attention_mask)
            logits = outputs.logits  # [B, L, V]

        for i in range(len(batch)):
            attn_len = int(attention_mask[i].sum().item())
            if attn_len <= 2:
                continue

            pred_ans = extract_answer(predicts[i])
            gold_ans = extract_answer(labels[i])
            is_correct = answers_match(pred_ans, gold_ans)
            correctness = "correct" if is_correct else "wrong"

            prompt_len = int(prompt_lens[i])
            predict_ids = predict_ids_list[i]
            if not predict_ids:
                continue

            # 在 response 区间中找 <add_think> 的位置（相对 response 起点）
            think_rel_positions = [idx for idx, tid in enumerate(predict_ids) if int(tid) == add_think_id]
            if not think_rel_positions:
                continue
            samples_with_trigger += 1

            # 熵序列：entropy_at_t 使用 logits[t] 的分布
            logits_i = logits[i, :attn_len, :].float()
            logits_i = torch.nan_to_num(logits_i, nan=0.0, posinf=1e4, neginf=-1e4)
            probs_i = torch.softmax(logits_i, dim=-1)
            entropy_i = -(probs_i * torch.log(probs_i.clamp_min(1e-12))).sum(dim=-1).cpu().numpy()

            # 触发点定义：
            # <add_think> 位于 full 序列位置 j（token 索引），
            # 则“前”熵取 entropy[j-1]（预测 <add_think> 本身），
            # “后”熵取 entropy[j]（预测 <add_think> 后一个 token）。
            for rel_pos in think_rel_positions:
                j = prompt_len + rel_pos
                if j - 1 < 0 or j >= attn_len - 1:
                    continue
                entropy_before = float(entropy_i[j - 1])
                entropy_after = float(entropy_i[j])
                delta = entropy_after - entropy_before

                if is_correct:
                    correct_deltas.append(delta)
                else:
                    wrong_deltas.append(delta)

                per_trigger_rows.append(
                    {
                        "sample_idx": st + i,
                        "is_correct": is_correct,
                        "group": correctness,
                        "think_rel_pos": int(rel_pos),
                        "think_full_pos": int(j),
                        "entropy_before": entropy_before,
                        "entropy_after": entropy_after,
                        "delta": float(delta),
                        "trend": "increase" if delta > args.eps else ("decrease" if delta < -args.eps else "unchanged"),
                    }
                )

    summary = {
        "meta": {
            "data": str(data_path),
            "model": str(args.model),
            "max_samples": int(args.max_samples),
            "batch_size": int(args.batch_size),
            "eps": float(args.eps),
            "num_samples_loaded": int(len(rows)),
            "num_samples_with_add_think": int(samples_with_trigger),
            "num_triggers_total": int(len(per_trigger_rows)),
        },
        "correct": summarize_deltas(correct_deltas, args.eps),
        "wrong": summarize_deltas(wrong_deltas, args.eps),
    }

    out_json.write_text(json.dumps(summary, ensure_ascii=False, indent=2), encoding="utf-8")

    fieldnames = [
        "sample_idx",
        "is_correct",
        "group",
        "think_rel_pos",
        "think_full_pos",
        "entropy_before",
        "entropy_after",
        "delta",
        "trend",
    ]
    with out_csv.open("w", encoding="utf-8", newline="") as f:
        writer = csv.DictWriter(f, fieldnames=fieldnames)
        writer.writeheader()
        for row in per_trigger_rows:
            writer.writerow(row)

    plot_delta_histogram(
        correct_deltas=correct_deltas,
        wrong_deltas=wrong_deltas,
        output_path=out_plot,
        bins=max(5, int(args.hist_bins)),
    )

    print(f"[done] summary json: {out_json.resolve()}")
    print(f"[done] per-trigger csv: {out_csv.resolve()}")
    print(f"[done] histogram: {out_plot.resolve()}")
    print(
        "[stats] "
        f"triggers={len(per_trigger_rows)}, "
        f"correct={summary['correct']['count']}, "
        f"wrong={summary['wrong']['count']}"
    )


if __name__ == "__main__":
    main()

