#!/usr/bin/env python3
"""
分析 LLM 推理轨迹中各 token 位置的熵分布，用于 latent thinking 实验中高熵阈值标定。

用法见 cursor_prompt/entropy_ana.md。
"""

from __future__ import annotations

import argparse
import json
import os
import pickle
import random
import re
import sys
import warnings
from pathlib import Path
from typing import Any, Optional

import matplotlib.pyplot as plt
import numpy as np
import torch
from tqdm import tqdm
from transformers import AutoModelForCausalLM, AutoTokenizer

# 与 run_experiment 一致：支持以脚本方式运行
if __package__ is None or __package__ == "":
    _ROOT = Path(__file__).resolve().parent.parent
    if str(_ROOT) not in sys.path:
        sys.path.insert(0, str(_ROOT))
    from experiment.utils import answers_equal, model_input_device, split_system_user_from_prompt
else:
    from .utils import answers_equal, model_input_device, split_system_user_from_prompt


def parse_boxed_answer(text: str) -> Optional[str]:
    """
    从文本中提取 \\boxed{X} 里的答案。
    处理嵌套花括号（如 \\boxed{\\frac{1}{2}}）。
    返回字符串，找不到返回 None。
    """
    if not isinstance(text, str) or not text:
        return None
    key = r"\boxed{"
    idx = 0
    while True:
        start = text.find(key, idx)
        if start < 0:
            return None
        i = start + len(key)
        depth = 1
        j = i
        while j < len(text) and depth > 0:
            c = text[j]
            if c == "{":
                depth += 1
            elif c == "}":
                depth -= 1
            j += 1
        if depth == 0:
            inner = text[i : j - 1].strip()
            return inner if inner else None
        idx = start + 1


def build_chat_input_ids(
    tokenizer: Any,
    prompt: str,
    response: str,
) -> tuple[torch.Tensor, int]:
    """
    用 apply_chat_template 构造与训练一致的 input_ids，并返回 prompt 部分长度（response 起始下标）。
    若无法解析 system/user，则退化为单条 user 消息（内容为原始 prompt）。
    """
    sys_s, user_s, ok = split_system_user_from_prompt(prompt)
    if ok and sys_s is not None and user_s is not None:
        messages = [
            {"role": "system", "content": sys_s},
            {"role": "user", "content": user_s},
        ]
    else:
        messages = [{"role": "user", "content": prompt}]

    # 含「助手开始」但尚无正文，长度即 response 第一个 token 的起始下标
    prompt_ids = tokenizer.apply_chat_template(
        messages,
        tokenize=True,
        add_generation_prompt=True,
        return_tensors="pt",
    )
    prompt_len = int(prompt_ids.shape[1])

    full_messages = messages + [{"role": "assistant", "content": response}]
    input_ids = tokenizer.apply_chat_template(
        full_messages,
        tokenize=True,
        add_generation_prompt=False,
        return_tensors="pt",
    )
    return input_ids, prompt_len


def compute_token_entropy(
    model: torch.nn.Module,
    tokenizer: Any,
    prompt: str,
    response: str,
    device: torch.device,
    gold_answer: Optional[str],
) -> tuple[list[str], torch.Tensor, bool]:
    """
    对单条 response 计算每个生成 token 位置的预测熵（仅 response 段）。

    返回：
      token_strs: response 每个 token 的字符串
      entropies: (response_len,) 每个位置的熵
      is_correct: 该轨迹答案是否与 gold（boxed）一致
    """
    if not response:
        return [], torch.tensor([], dtype=torch.float32), False

    pred_ans = parse_boxed_answer(response)
    is_correct = answers_equal(pred_ans, gold_answer) if gold_answer is not None else False

    input_ids, prompt_len = build_chat_input_ids(tokenizer, prompt, response)
    seq_len = int(input_ids.shape[1])
    if seq_len <= prompt_len:
        return [], torch.tensor([], dtype=torch.float32), is_correct

    input_ids = input_ids.to(device)

    with torch.no_grad():
        out = model(input_ids=input_ids)
        logits = out.logits  # (1, seq_len, vocab)

    # 位置 t 的 logits 预测 token t+1；response 第一个 token 在 prompt_len，由 logits[prompt_len-1] 预测
    resp_logits = logits[:, prompt_len - 1 : -1, :].squeeze(0)  # (response_len, vocab)
    if resp_logits.numel() == 0:
        return [], torch.tensor([], dtype=torch.float32), is_correct

    probs = torch.softmax(resp_logits, dim=-1)
    log_probs = torch.log(probs + 1e-10)
    entropies = -(probs * log_probs).sum(dim=-1).detach().cpu().float()

    resp_ids = input_ids[0, prompt_len:].tolist()
    token_strs = [tokenizer.convert_ids_to_tokens([tid])[0] for tid in resp_ids]

    # 对齐长度（极端情况下 tokenizer 边界可能导致 logits 与 ids 差 1）
    n = min(len(token_strs), int(entropies.shape[0]))
    token_strs = token_strs[:n]
    entropies = entropies[:n]

    return token_strs, entropies, is_correct


def _load_jsonl(path: Path) -> list[dict[str, Any]]:
    rows: list[dict[str, Any]] = []
    with path.open("r", encoding="utf-8") as f:
        for line in f:
            line = line.strip()
            if not line:
                continue
            rows.append(json.loads(line))
    return rows


def _collect_entropy_records(
    model: torch.nn.Module,
    tokenizer: Any,
    data: list[dict[str, Any]],
    device: torch.device,
    max_samples: int,
    max_predicts: int,
    use_cuda: bool,
) -> list[dict[str, Any]]:
    """对 JSONL 逐条、逐 predict 计算熵，返回可 pickle 的记录列表。"""
    records: list[dict[str, Any]] = []
    n = min(len(data), max_samples)

    outer = tqdm(range(n), desc="样本", leave=True)
    for si in outer:
        row = data[si]
        try:
            prompt = row.get("prompt", "")
            label = row.get("label", "")
            predicts = row.get("predicts") or []
            if not isinstance(predicts, list):
                predicts = []
            gold = parse_boxed_answer(label)

            for pi, resp in enumerate(predicts[:max_predicts]):
                if not isinstance(resp, str):
                    continue
                token_strs, entropies, is_correct = compute_token_entropy(
                    model, tokenizer, prompt, resp, device, gold
                )
                records.append(
                    {
                        "sample_idx": si,
                        "predict_idx": pi,
                        "token_strs": token_strs,
                        "entropies": entropies.numpy().tolist(),
                        "is_correct": bool(is_correct),
                    }
                )
        except Exception as e:  # noqa: BLE001
            warnings.warn(f"样本 {si} 处理失败，已跳过: {e}", UserWarning, stacklevel=1)
        finally:
            if use_cuda:
                torch.cuda.empty_cache()

    return records


def _flat_tokens_and_entropies(
    records: list[dict[str, Any]],
) -> tuple[np.ndarray, np.ndarray, np.ndarray]:
    """展开为逐 token 的熵、是否正确（轨迹级广播）、token 字符串索引。"""
    ents: list[float] = []
    corr: list[bool] = []
    toks: list[str] = []
    for r in records:
        ic = bool(r["is_correct"])
        for t, e in zip(r["token_strs"], r["entropies"]):
            toks.append(t)
            ents.append(float(e))
            corr.append(ic)
    return np.asarray(ents, dtype=np.float64), np.asarray(corr, dtype=bool), np.asarray(toks, dtype=object)


def _plot_entropy_distribution(
    entropies: np.ndarray,
    correct_mask: np.ndarray,
    percentiles: list[float],
    out_path: Path,
) -> dict[float, float]:
    """
    左：正确/错误轨迹 token 熵直方图叠加；右：全局经验 CDF，并在给定分位处标水平线与熵阈值。
    返回各分位对应的熵阈值。
    """
    fig, axes = plt.subplots(1, 2, figsize=(12, 4.5))

    e_cor = entropies[correct_mask]
    e_wro = entropies[~correct_mask]
    ax0 = axes[0]
    bins = 50
    ax0.hist(
        e_cor,
        bins=bins,
        alpha=0.45,
        color="blue",
        label="正确轨迹 token",
        density=True,
    )
    ax0.hist(
        e_wro,
        bins=bins,
        alpha=0.45,
        color="red",
        label="错误轨迹 token",
        density=True,
    )
    ax0.set_xlabel("熵")
    ax0.set_ylabel("密度")
    ax0.set_title("Token 熵分布（按轨迹正误着色）")
    ax0.legend()

    sorted_e = np.sort(entropies)
    n = len(sorted_e)
    y = (np.arange(1, n + 1, dtype=np.float64)) / max(n, 1)
    ax1 = axes[1]
    ax1.plot(sorted_e, y, color="black", linewidth=1.2, label="经验 CDF")
    ax1.set_xlabel("熵")
    ax1.set_ylabel("累积概率")
    ax1.set_title("熵的累积分布")
    ax1.grid(True, alpha=0.3)

    thresh: dict[float, float] = {}
    ann_lines: list[str] = []
    for p in percentiles:
        # 分位数：P(H <= t) = p
        t = float(np.quantile(entropies, p))
        thresh[p] = t
        ax1.axhline(p, color="gray", linestyle="--", linewidth=0.9, alpha=0.85)
        ax1.axvline(t, color="lightgray", linestyle=":", linewidth=0.7, alpha=0.9)
        ann_lines.append(f"p={p:.2f} → H={t:.3f}")
    ax1.text(
        0.02,
        0.98,
        "\n".join(ann_lines),
        transform=ax1.transAxes,
        va="top",
        ha="left",
        fontsize=8,
        bbox=dict(boxstyle="round", facecolor="wheat", alpha=0.35),
    )
    ax1.legend()
    fig.tight_layout()
    fig.savefig(out_path, dpi=150)
    plt.close(fig)
    return thresh


def _threshold_table(
    records: list[dict[str, Any]],
    entropies: np.ndarray,
    percentiles: list[float],
) -> None:
    """打印不同分位阈值下，每条轨迹平均「高熵」token 数及正误对比。"""
    print("\n阈值分析：")
    print("  百分位    熵阈值   每条轨迹平均选中token数   正确/错误轨迹高熵位置数比较")
    for p in percentiles:
        thr = float(np.quantile(entropies, p))
        per_traj_counts: list[int] = []
        cor_counts: list[int] = []
        wro_counts: list[int] = []
        for r in records:
            es = np.asarray(r["entropies"], dtype=np.float64)
            cnt = int((es >= thr).sum())
            per_traj_counts.append(cnt)
            if r["is_correct"]:
                cor_counts.append(cnt)
            else:
                wro_counts.append(cnt)
        mean_sel = float(np.mean(per_traj_counts)) if per_traj_counts else 0.0
        mean_cor = float(np.mean(cor_counts)) if cor_counts else 0.0
        mean_wro = float(np.mean(wro_counts)) if wro_counts else 0.0
        print(
            f"  p={p:.2f}  {thr:>6.3f}  {mean_sel:>20.1f}  "
            f"correct: {mean_cor:.1f}, wrong: {mean_wro:.1f}"
        )


def _print_top_high_entropy_tokens(
    tokens: np.ndarray,
    entropies: np.ndarray,
    correct_mask: np.ndarray,
    thr: float,
    top_k: int = 30,
) -> None:
    """在熵 >= 阈值的 token 中统计出现频次 Top-K。"""
    m = entropies >= thr
    sub_tok = tokens[m]
    sub_ent = entropies[m]
    sub_cor = correct_mask[m]
    if len(sub_tok) == 0:
        print("\n（无 token 达到该高熵阈值，跳过 Top-K 统计）")
        return

    uniq, inv = np.unique(sub_tok, return_inverse=True)
    counts = np.bincount(inv)
    order = np.argsort(-counts)[:top_k]

    print(f"\nTop-{top_k} 高熵 tokens（阈值 p=0.80，熵>={thr:.3f}）：")
    print(f"{'排名':>4} {'token字符串':<24} {'出现次数':>10} {'平均熵':>10} {'正确轨迹占比':>12}")
    for rank, ui in enumerate(order, start=1):
        t = uniq[ui]
        idx = np.where(sub_tok == t)[0]
        c = int(counts[ui])
        mean_e = float(sub_ent[idx].mean())
        frac_cor = float(sub_cor[idx].mean())
        disp = repr(t)[:22]
        print(f"{rank:>4} {disp:<24} {c:>10} {mean_e:>10.3f} {frac_cor:>12.2f}")


def _print_random_high_entropy_contexts(
    records: list[dict[str, Any]],
    thr: float,
    n_samples: int = 20,
    rng: Optional[random.Random] = None,
    window: int = 6,
) -> None:
    """随机抽取高熵位置，用同一条轨迹前后 token 拼接局部上下文（[[ ]] 标出高熵 token）。"""
    rng = rng or random.Random(0)
    candidates: list[tuple[bool, float, str, list[str], int]] = []
    for r in records:
        ic = bool(r["is_correct"])
        toks = r["token_strs"]
        for j, (ts, e) in enumerate(zip(toks, r["entropies"])):
            if float(e) >= thr:
                candidates.append((ic, float(e), ts, toks, j))
    if not candidates:
        print("\n（无高熵位置可供抽样）")
        return

    picks = rng.sample(candidates, k=min(n_samples, len(candidates)))
    print("\n=== 高熵 token 样例 ===")
    for i, (ic, e, ts, toks, j) in enumerate(picks, start=1):
        tag = "正确轨迹" if ic else "错误轨迹"
        lo = max(0, j - window)
        hi = min(len(toks), j + window + 1)
        parts: list[str] = []
        for k in range(lo, hi):
            piece = toks[k]
            if k == j:
                parts.append(f"[[{piece}]]")
            else:
                parts.append(piece)
        ctx = "... " + "".join(parts) + " ..."
        print(f"\n[样例 {i}] 熵={e:.3f}，来自：{tag}")
        print(f"  上下文：{ctx}")


def _plot_entropy_comparison(
    records: list[dict[str, Any]],
    out_path: Path,
) -> None:
    """
    左：正确 vs 错误轨迹的 token 熵（箱线图）；右：相对位置上的平均熵曲线。
    """
    cor_ent: list[float] = []
    wro_ent: list[float] = []
    # 相对位置分桶：20 个 bin
    n_bins = 20
    sum_cor = np.zeros(n_bins, dtype=np.float64)
    cnt_cor = np.zeros(n_bins, dtype=np.int64)
    sum_wro = np.zeros(n_bins, dtype=np.float64)
    cnt_wro = np.zeros(n_bins, dtype=np.int64)

    for r in records:
        es = np.asarray(r["entropies"], dtype=np.float64)
        L = len(es)
        if L == 0:
            continue
        ic = bool(r["is_correct"])
        if ic:
            cor_ent.extend(es.tolist())
        else:
            wro_ent.extend(es.tolist())

        if L == 1:
            rel = np.zeros(1, dtype=np.float64)
        else:
            rel = np.arange(L, dtype=np.float64) / float(L - 1)
        bin_idx = np.minimum((rel * n_bins).astype(np.int64), n_bins - 1)
        for j, b in enumerate(bin_idx):
            if ic:
                sum_cor[b] += es[j]
                cnt_cor[b] += 1
            else:
                sum_wro[b] += es[j]
                cnt_wro[b] += 1

    fig, axes = plt.subplots(1, 2, figsize=(12, 4.5))
    ax0 = axes[0]
    data = []
    labels = []
    if cor_ent:
        data.append(np.asarray(cor_ent, dtype=np.float64))
        labels.append("正确轨迹")
    if wro_ent:
        data.append(np.asarray(wro_ent, dtype=np.float64))
        labels.append("错误轨迹")
    if not data:
        ax0.text(0.5, 0.5, "无数据", ha="center", va="center", transform=ax0.transAxes)
    else:
        bp = ax0.boxplot(data, tick_labels=labels, patch_artist=True)
        colors = ["lightblue", "mistyrose"]
        for patch, c in zip(bp["boxes"], colors[: len(bp["boxes"])]):
            patch.set_facecolor(c)
    ax0.set_ylabel("Token 熵")
    ax0.set_title("正确 vs 错误轨迹（逐 token）")

    ax1 = axes[1]
    x = np.linspace(0, 1, n_bins)
    mean_cor = np.divide(sum_cor, np.maximum(cnt_cor, 1), out=np.zeros_like(sum_cor), where=cnt_cor > 0)
    mean_wro = np.divide(sum_wro, np.maximum(cnt_wro, 1), out=np.zeros_like(sum_wro), where=cnt_wro > 0)
    ax1.plot(x, mean_cor, label="正确轨迹", color="blue")
    ax1.plot(x, mean_wro, label="错误轨迹", color="red")
    ax1.set_xlabel("相对位置（0=首 token，1=末 token）")
    ax1.set_ylabel("平均熵")
    ax1.set_title("沿生成序列的平均熵")
    ax1.legend()
    ax1.grid(True, alpha=0.3)

    fig.tight_layout()
    fig.savefig(out_path, dpi=150)
    plt.close(fig)


def main() -> None:
    parser = argparse.ArgumentParser(description="推理轨迹 token 熵分析")
    parser.add_argument("--data_path", type=str, required=True, help="JSONL 数据路径")
    parser.add_argument("--model_name", type=str, required=True, help="HF 模型名或本地路径")
    parser.add_argument("--output_dir", type=str, required=True, help="输出目录")
    parser.add_argument("--max_samples", type=int, default=200, help="最多处理多少条样本")
    parser.add_argument("--max_predicts", type=int, default=5, help="每条样本最多处理几条 predicts")
    parser.add_argument("--device", type=str, default="cuda", choices=["cuda", "cpu"])
    parser.add_argument("--no_cache", action="store_true", help="忽略缓存，强制重算 forward")
    args = parser.parse_args()

    out_dir = Path(args.output_dir)
    out_dir.mkdir(parents=True, exist_ok=True)
    cache_path = out_dir / "entropy_cache.pkl"

    data_path = Path(args.data_path)
    data = _load_jsonl(data_path)

    records: list[dict[str, Any]]
    if cache_path.is_file() and not args.no_cache:
        print(f"[cache] 加载缓存: {cache_path}")
        with cache_path.open("rb") as f:
            records = pickle.load(f)
    else:
        print(f"[load] tokenizer / model: {args.model_name}")
        tokenizer = AutoTokenizer.from_pretrained(args.model_name, trust_remote_code=True)
        if tokenizer.pad_token_id is None:
            tokenizer.pad_token = tokenizer.eos_token

        use_cuda = args.device == "cuda" and torch.cuda.is_available()
        dtype = torch.bfloat16
        model = AutoModelForCausalLM.from_pretrained(
            args.model_name,
            torch_dtype=dtype,
            trust_remote_code=True,
            device_map="auto" if use_cuda else None,
            low_cpu_mem_usage=use_cuda,
        ).eval()

        if not use_cuda:
            model = model.to("cpu")

        dev = model_input_device(model)
        print(f"[load] forward 设备（输入）: {dev}")

        records = _collect_entropy_records(
            model,
            tokenizer,
            data,
            dev,
            max_samples=args.max_samples,
            max_predicts=args.max_predicts,
            use_cuda=use_cuda,
        )

        with cache_path.open("wb") as f:
            pickle.dump(records, f)
        print(f"[cache] 已写入: {cache_path}")

    if not records:
        print("无有效记录，退出。")
        return

    entropies, correct_mask, tokens = _flat_tokens_and_entropies(records)
    percentiles = [0.70, 0.80, 0.85, 0.90]

    p1 = out_dir / "entropy_distribution.png"
    thresh_map = _plot_entropy_distribution(entropies, correct_mask, percentiles, p1)
    print(f"\n[图] 已保存: {p1.resolve()}")

    p2 = out_dir / "entropy_comparison.png"
    _plot_entropy_comparison(records, p2)
    print(f"[图] 已保存: {p2.resolve()}")

    _threshold_table(records, entropies, percentiles)

    thr_80 = thresh_map.get(0.80, float(np.quantile(entropies, 0.80)))
    _print_top_high_entropy_tokens(tokens, entropies, correct_mask, thr_80, top_k=30)
    _print_random_high_entropy_contexts(records, thr_80, n_samples=20, rng=random.Random(42))


if __name__ == "__main__":
    main()
