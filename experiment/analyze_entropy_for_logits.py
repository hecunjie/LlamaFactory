#!/usr/bin/env python3
from __future__ import annotations

import argparse
import json
from pathlib import Path
from typing import Any

import matplotlib.pyplot as plt
import numpy as np
import torch
import torch.nn.functional as F
from tqdm import tqdm
from transformers import AutoModelForCausalLM, AutoTokenizer


CASE_NORMAL = "normal"
CASE_A = "A_multipeak"
CASE_B = "B_lowconf"
CASE_AMBIGUOUS = "ambiguous"


def extract_answer(text: str) -> str:
    if "###" in text:
        return text.split("###")[-1].strip().split()[0]
    return ""


def load_jsonl(path: Path) -> list[dict[str, Any]]:
    rows: list[dict[str, Any]] = []
    with path.open("r", encoding="utf-8") as f:
        for line in f:
            line = line.strip()
            if line:
                rows.append(json.loads(line))
    return rows


def classify_case(
    entropy: float,
    top1_prob: float,
    top5_mass: float,
    max_cosine_sim: float,
    entropy_threshold: float,
    sim_threshold: float,
) -> str:
    if entropy <= entropy_threshold:
        return CASE_NORMAL
    if top5_mass >= 0.6 and top1_prob < 0.4:
        return CASE_A
    if max_cosine_sim < sim_threshold:
        return CASE_B
    return CASE_AMBIGUOUS


def classify_high_case(
    top1_prob: float,
    top5_mass: float,
    max_cosine_sim: float,
    sim_threshold: float,
) -> str:
    if top5_mass >= 0.6 and top1_prob < 0.4:
        return CASE_A
    if max_cosine_sim < sim_threshold:
        return CASE_B
    return CASE_AMBIGUOUS


def _summarize(arr: np.ndarray, keys: list[float]) -> dict[float, float]:
    if arr.size == 0:
        return {k: float("nan") for k in keys}
    return {k: float(np.quantile(arr, k)) for k in keys}


def _plot_results(
    all_high_rows: list[dict[str, Any]],
    output_plot: Path,
) -> None:
    if not all_high_rows:
        fig, ax = plt.subplots(figsize=(8, 4))
        ax.text(0.5, 0.5, "No high-entropy positions.", ha="center", va="center", transform=ax.transAxes)
        ax.set_axis_off()
        fig.tight_layout()
        fig.savefig(output_plot, dpi=150)
        plt.close(fig)
        return

    ent = np.array([r["entropy"] for r in all_high_rows], dtype=np.float64)
    sim = np.array([r["max_cosine_sim"] for r in all_high_rows], dtype=np.float64)
    mass = np.array([r["top5_mass"] for r in all_high_rows], dtype=np.float64)
    cases = np.array([r["case"] for r in all_high_rows], dtype=object)
    finite_mask = np.isfinite(ent) & np.isfinite(sim) & np.isfinite(mass)
    ent = ent[finite_mask]
    sim = sim[finite_mask]
    mass = mass[finite_mask]
    cases = cases[finite_mask]
    if ent.size == 0:
        fig, ax = plt.subplots(figsize=(8, 4))
        ax.text(0.5, 0.5, "No finite high-entropy points.", ha="center", va="center", transform=ax.transAxes)
        ax.set_axis_off()
        fig.tight_layout()
        fig.savefig(output_plot, dpi=150)
        plt.close(fig)
        return

    case_to_color = {
        CASE_A: "tab:blue",
        CASE_B: "tab:red",
        CASE_AMBIGUOUS: "tab:green",
    }

    fig, axes = plt.subplots(2, 2, figsize=(13, 10))

    # 1) entropy vs max_sim
    ax = axes[0, 0]
    for c in [CASE_A, CASE_B, CASE_AMBIGUOUS]:
        m = cases == c
        if np.any(m):
            ax.scatter(ent[m], sim[m], s=10, alpha=0.55, label=c, color=case_to_color[c])
    ax.set_xlabel("entropy")
    ax.set_ylabel("max_cosine_sim")
    ax.set_title("Entropy vs Max Cosine Similarity")
    ax.grid(True, alpha=0.3)
    ax.legend()

    # 2) entropy histogram stacked by case
    ax = axes[0, 1]
    hist_data = [ent[cases == c] for c in [CASE_A, CASE_B, CASE_AMBIGUOUS] if np.any(cases == c)]
    labels = [c for c in [CASE_A, CASE_B, CASE_AMBIGUOUS] if np.any(cases == c)]
    colors = [case_to_color[c] for c in labels]
    if hist_data:
        ax.hist(hist_data, bins=50, stacked=True, label=labels, color=colors, alpha=0.8)
        ax.legend()
    ax.set_xlabel("entropy")
    ax.set_ylabel("count")
    ax.set_title("Entropy Distribution (Stacked by Case)")
    ax.grid(True, alpha=0.3)

    # 3) top5_mass vs max_sim
    ax = axes[1, 0]
    for c in [CASE_A, CASE_B, CASE_AMBIGUOUS]:
        m = cases == c
        if np.any(m):
            ax.scatter(mass[m], sim[m], s=10, alpha=0.55, label=c, color=case_to_color[c])
    ax.set_xlabel("top5_mass")
    ax.set_ylabel("max_cosine_sim")
    ax.set_title("Top-5 Mass vs Max Cosine Similarity")
    ax.grid(True, alpha=0.3)
    ax.legend()

    # 4) joint heatmap of entropy and max_sim
    ax = axes[1, 1]
    heat, xedges, yedges = np.histogram2d(ent, sim, bins=40)
    if np.allclose(xedges[0], xedges[-1]) or np.allclose(yedges[0], yedges[-1]):
        ax.scatter(ent, sim, s=10, alpha=0.5, color="tab:purple")
    else:
        im = ax.imshow(
            heat.T,
            origin="lower",
            aspect="auto",
            extent=[xedges[0], xedges[-1], yedges[0], yedges[-1]],
            cmap="magma",
        )
        fig.colorbar(im, ax=ax, fraction=0.046, pad=0.04)
    ax.set_xlabel("entropy")
    ax.set_ylabel("max_cosine_sim")
    ax.set_title("Joint Density Heatmap")

    fig.tight_layout()
    fig.savefig(output_plot, dpi=150)
    plt.close(fig)


def _collect_examples(all_high_rows: list[dict[str, Any]], per_case: int = 2) -> dict[str, list[dict[str, Any]]]:
    out = {CASE_A: [], CASE_B: [], CASE_AMBIGUOUS: []}
    for row in all_high_rows:
        c = row["case"]
        if c in out and len(out[c]) < per_case:
            out[c].append(row)
    return out


def _build_token_context(tokenizer: Any, token_ids: list[int], t: int, window: int = 6) -> str:
    lo = max(0, t - window)
    hi = min(len(token_ids), t + window + 1)
    parts: list[str] = []
    for i in range(lo, hi):
        tok = tokenizer.decode([int(token_ids[i])], skip_special_tokens=False)
        if i == t:
            parts.append(f"[[{tok}]]")
        else:
            parts.append(tok)
    return "".join(parts)


def _build_prompt_prefix(tokenizer: Any, prompt_text: str, token_strs: list[str], t: int) -> str:
    # 截止到当前 token（含当前 token）的完整前缀：prompt + response_prefix
    response_prefix = tokenizer.convert_tokens_to_string(token_strs[: t + 1])
    return prompt_text + response_prefix


def main() -> None:
    parser = argparse.ArgumentParser(description="Analyze logits entropy and hidden-state confidence.")
    parser.add_argument("--data", type=str, required=True, help="Input JSONL path")
    parser.add_argument("--model", type=str, required=True, help="Local model checkpoint path")
    parser.add_argument("--max_samples", type=int, default=200, help="Max samples to analyze")
    parser.add_argument("--batch_size", type=int, default=4, help="Batch size")
    parser.add_argument(
        "--high_entropy_topk",
        type=int,
        default=20,
        help="每个 batch 在错误样本路径上按熵全局选取的 top-k 数量",
    )
    parser.add_argument("--entropy_threshold", type=float, default=2.0, help="High entropy threshold")
    parser.add_argument("--sim_threshold", type=float, default=0.3, help="Low-confidence sim threshold")
    parser.add_argument("--only_wrong", action="store_true", default=True, help="Analyze only wrong samples")
    parser.add_argument("--all_samples", action="store_true", help="Analyze all samples (override only_wrong)")
    parser.add_argument("--output_plot", type=str, default="entropy_analysis.png", help="Output plot path")
    parser.add_argument("--output_jsonl", type=str, default="entropy_results.jsonl", help="Output result JSONL path")
    args = parser.parse_args()

    data_path = Path(args.data)
    output_plot = Path(args.output_plot)
    output_jsonl = Path(args.output_jsonl)
    output_plot.parent.mkdir(parents=True, exist_ok=True)
    output_jsonl.parent.mkdir(parents=True, exist_ok=True)

    print(f"[load] data: {data_path}")
    data = load_jsonl(data_path)
    if not data:
        print("输入数据为空，退出。")
        return

    rows_selected: list[dict[str, Any]] = []
    for row in data:
        pred = str(row.get("predict", ""))
        label = str(row.get("label", ""))
        is_correct = extract_answer(pred) == extract_answer(label)
        if args.all_samples:
            rows_selected.append(row)
        else:
            if args.only_wrong and (not is_correct):
                rows_selected.append(row)
            elif (not args.only_wrong):
                rows_selected.append(row)
        if len(rows_selected) >= args.max_samples:
            break

    if not rows_selected:
        print("没有满足筛选条件的样本，退出。")
        return

    print(f"[load] model/tokenizer: {args.model}")
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

    tied = bool(model.lm_head.weight is model.model.embed_tokens.weight)
    print(f"[check] tied embedding: {tied}")
    embed_matrix = model.model.embed_tokens.weight
    embed_norm = F.normalize(embed_matrix.detach(), dim=1)
    embed_norm_t = embed_norm.transpose(0, 1).contiguous()

    all_entropies: list[float] = []
    all_max_sims: list[float] = []
    case_counts_global = {CASE_NORMAL: 0, CASE_A: 0, CASE_B: 0, CASE_AMBIGUOUS: 0}
    case_counts_high = {CASE_A: 0, CASE_B: 0, CASE_AMBIGUOUS: 0}
    all_high_rows: list[dict[str, Any]] = []
    per_token_json_rows: list[dict[str, Any]] = []
    total_positions = 0
    high_entropy_count = 0

    for st in tqdm(range(0, len(rows_selected), args.batch_size), desc="Analyzing"):
        batch_rows = rows_selected[st : st + args.batch_size]
        full_texts = [str(r.get("prompt", "")) + str(r.get("predict", "")) for r in batch_rows]
        prompts = [str(r.get("prompt", "")) for r in batch_rows]
        predicts = [str(r.get("predict", "")) for r in batch_rows]
        labels = [str(r.get("label", "")) for r in batch_rows]

        enc = tokenizer(
            full_texts,
            return_tensors="pt",
            padding=True,
            truncation=False,
            add_special_tokens=False,
        )
        input_ids = enc["input_ids"].to(model.lm_head.weight.device)
        attention_mask = enc["attention_mask"].to(model.lm_head.weight.device)

        prompt_lens = []
        for p in prompts:
            p_ids = tokenizer(p, add_special_tokens=False)["input_ids"]
            prompt_lens.append(len(p_ids))

        with torch.no_grad():
            out = model(input_ids=input_ids, attention_mask=attention_mask, output_hidden_states=True)
            logits = out.logits  # [B, L, V]
            hidden = out.hidden_states[-1]  # [B, L, D]

        bsz, seq_len, vocab_size = logits.shape
        _ = vocab_size

        batch_sample_buffers: list[dict[str, Any]] = []
        batch_wrong_candidates: list[tuple[float, int, int]] = []  # (entropy, sample_i, token_t)

        for i in range(bsz):
            attn_len = int(attention_mask[i].sum().item())
            prompt_len = int(prompt_lens[i])
            pred_answer = extract_answer(predicts[i])
            gold_answer = extract_answer(labels[i])
            is_correct = pred_answer == gold_answer

            sample_case_counts = {CASE_NORMAL: 0, CASE_A: 0, CASE_B: 0, CASE_AMBIGUOUS: 0}
            sample_total_positions = 0

            if attn_len <= 1 or prompt_len >= attn_len:
                batch_sample_buffers.append(
                    {
                        "prompt_preview": prompts[i],
                        "is_correct": is_correct,
                        "total_positions": 0,
                        "case_counts": sample_case_counts,
                        "token_metrics": [],
                        "target_ids": None,
                        "top5_ids": None,
                        "top5_vals": None,
                        "top5_cosine_sim": None,
                        "sample_i": i,
                    }
                )
                continue

            pred_start = max(prompt_len - 1, 0)
            pred_end = attn_len - 1
            if pred_start >= pred_end:
                batch_sample_buffers.append(
                    {
                        "prompt_preview": prompts[i],
                        "is_correct": is_correct,
                        "total_positions": 0,
                        "case_counts": sample_case_counts,
                        "token_metrics": [],
                        "target_ids": None,
                        "top5_ids": None,
                        "top5_vals": None,
                        "top5_cosine_sim": None,
                        "sample_i": i,
                    }
                )
                continue

            logits_i = logits[i, pred_start:pred_end, :]
            hidden_i = hidden[i, pred_start:pred_end, :]
            target_ids = input_ids[i, prompt_len:attn_len]
            npos = int(min(logits_i.shape[0], target_ids.shape[0]))
            if npos <= 0:
                batch_sample_buffers.append(
                    {
                        "prompt_preview": prompts[i],
                        "is_correct": is_correct,
                        "total_positions": 0,
                        "case_counts": sample_case_counts,
                        "token_metrics": [],
                        "target_ids": None,
                        "top5_ids": None,
                        "top5_vals": None,
                        "top5_cosine_sim": None,
                        "sample_i": i,
                    }
                )
                continue

            logits_i = logits_i[:npos]
            hidden_i = hidden_i[:npos]
            target_ids = target_ids[:npos]

            # 数值稳定：在 fp32 上计算概率与熵，避免 fp16 下出现 NaN/Inf
            logits_i_fp32 = torch.nan_to_num(logits_i.float(), nan=0.0, posinf=1e4, neginf=-1e4)
            probs = torch.softmax(logits_i_fp32, dim=-1)
            log_probs = torch.log(probs.clamp_min(1e-12))
            entropy = -(probs * log_probs).sum(dim=-1)
            top1_prob = probs.max(dim=-1).values
            top5_vals, top5_ids = probs.topk(k=min(5, probs.shape[-1]), dim=-1)
            top5_mass = top5_vals.sum(dim=-1)

            hidden_i = torch.nan_to_num(hidden_i.float(), nan=0.0, posinf=1e4, neginf=-1e4)
            h_norm = F.normalize(hidden_i, dim=-1, eps=1e-12)
            h_norm = h_norm.to(embed_norm_t.device, dtype=embed_norm_t.dtype)

            # chunked matmul to reduce peak memory usage
            chunk = 64
            max_sim_list = []
            top5_sim_list = []
            for cs in range(0, npos, chunk):
                ce = min(npos, cs + chunk)
                sims = h_norm[cs:ce] @ embed_norm_t
                max_sim_list.append(sims.max(dim=-1).values)
                top5_sim_list.append(sims.topk(k=min(5, sims.shape[-1]), dim=-1).values)
            max_cosine_sim = torch.cat(max_sim_list, dim=0).float().cpu()
            top5_cosine_sim = torch.cat(top5_sim_list, dim=0).float().cpu()

            entropy_np = entropy.float().cpu().numpy()
            top1_np = top1_prob.float().cpu().numpy()
            top5_mass_np = top5_mass.float().cpu().numpy()
            token_metrics: list[dict[str, Any]] = []

            for t in range(npos):
                sample_total_positions += 1
                total_positions += 1
                e = float(entropy_np[t])
                p1 = float(top1_np[t])
                p5m = float(top5_mass_np[t])
                ms = float(max_cosine_sim[t].item())
                case = classify_case(
                    entropy=e,
                    top1_prob=p1,
                    top5_mass=p5m,
                    max_cosine_sim=ms,
                    entropy_threshold=args.entropy_threshold,
                    sim_threshold=args.sim_threshold,
                )
                sample_case_counts[case] += 1
                case_counts_global[case] += 1
                if np.isfinite(e):
                    all_entropies.append(e)
                if np.isfinite(ms):
                    all_max_sims.append(ms)
                token_metrics.append(
                    {
                        "t": t,
                        "entropy": e,
                        "top1_prob": p1,
                        "top5_mass": p5m,
                        "max_cosine_sim": ms,
                    }
                )
                if (not is_correct) and np.isfinite(e):
                    batch_wrong_candidates.append((e, i, t))

            batch_sample_buffers.append(
                {
                    "prompt_preview": prompts[i],
                    "is_correct": is_correct,
                    "total_positions": sample_total_positions,
                    "case_counts": sample_case_counts,
                    "token_metrics": token_metrics,
                    "target_ids": target_ids.detach().cpu(),
                    "top5_ids": top5_ids.detach().cpu(),
                    "top5_vals": top5_vals.detach().cpu(),
                    "top5_cosine_sim": top5_cosine_sim.detach().cpu(),
                    "sample_i": i,
                }
            )

        # 按 batch 在“错误路径”上全局取 top-k 高熵 token
        k = max(0, int(args.high_entropy_topk))
        high_key_set: set[tuple[int, int]] = set()
        if k > 0 and batch_wrong_candidates:
            if len(batch_wrong_candidates) <= k:
                high_key_set = {(si, tt) for _, si, tt in batch_wrong_candidates}
            else:
                ent_arr = np.asarray([x[0] for x in batch_wrong_candidates], dtype=np.float64)
                pick_idx = np.argpartition(ent_arr, -k)[-k:]
                high_key_set = {
                    (batch_wrong_candidates[int(j)][1], batch_wrong_candidates[int(j)][2]) for j in pick_idx
                }

        for sample_buf in batch_sample_buffers:
            sample_i = int(sample_buf["sample_i"])
            sample_high_positions: list[dict[str, Any]] = []
            sample_high_count = 0
            is_correct = bool(sample_buf["is_correct"])
            token_metrics = sample_buf["token_metrics"]
            target_ids = sample_buf["target_ids"]
            top5_ids = sample_buf["top5_ids"]
            top5_vals = sample_buf["top5_vals"]
            top5_cosine_sim = sample_buf["top5_cosine_sim"]

            if target_ids is not None and top5_ids is not None and top5_vals is not None and top5_cosine_sim is not None:
                token_id_list = target_ids.tolist()
                token_strs = tokenizer.convert_ids_to_tokens(token_id_list)

                # 保存所有 case：每个 token 位置一条 JSONL
                for m in token_metrics:
                    t = int(m["t"])
                    e = float(m["entropy"])
                    p1 = float(m["top1_prob"])
                    p5m = float(m["top5_mass"])
                    ms = float(m["max_cosine_sim"])
                    token_id = int(token_id_list[t])
                    token_str = token_strs[t]
                    token_text = tokenizer.decode([token_id], skip_special_tokens=False)
                    top_ids_t = top5_ids[t].tolist()
                    top_probs_t = top5_vals[t].float().tolist()
                    top_tokens_raw = tokenizer.convert_ids_to_tokens(top_ids_t)
                    top_tokens_text = [
                        tokenizer.decode([int(_tid)], skip_special_tokens=False) for _tid in top_ids_t
                    ]
                    case_all = classify_case(
                        entropy=e,
                        top1_prob=p1,
                        top5_mass=p5m,
                        max_cosine_sim=ms,
                        entropy_threshold=args.entropy_threshold,
                        sim_threshold=args.sim_threshold,
                    )
                    if case_all == CASE_NORMAL:
                        continue
                    per_token_json_rows.append(
                        {
                            "sample_idx": st + sample_i,
                            "prompt_prefix": _build_prompt_prefix(
                                tokenizer=tokenizer,
                                prompt_text=sample_buf["prompt_preview"],
                                token_strs=token_strs,
                                t=t,
                            ),
                            "is_correct": is_correct,
                            "t": t,
                            "token_id": token_id,
                            "token": token_text,
                            "token_raw": token_str,
                            "context": _build_token_context(
                                tokenizer=tokenizer,
                                token_ids=token_id_list,
                                t=t,
                                window=6,
                            ),
                            "entropy": e,
                            "top1_prob": p1,
                            "top5_mass": p5m,
                            "max_cosine_sim": ms,
                            "top5_cosine_sim": [float(x) for x in top5_cosine_sim[t].tolist()],
                            "case": case_all,
                            "top5_tokens": top_tokens_text,
                            "top5_tokens_raw": top_tokens_raw,
                            "top5_probs": [float(x) for x in top_probs_t],
                        }
                    )

                for m in token_metrics:
                    t = int(m["t"])
                    if (sample_i, t) not in high_key_set:
                        continue
                    e = float(m["entropy"])
                    p1 = float(m["top1_prob"])
                    p5m = float(m["top5_mass"])
                    ms = float(m["max_cosine_sim"])
                    if not (np.isfinite(e) and np.isfinite(p1) and np.isfinite(p5m) and np.isfinite(ms)):
                        continue
                    case = classify_high_case(
                        top1_prob=p1,
                        top5_mass=p5m,
                        max_cosine_sim=ms,
                        sim_threshold=args.sim_threshold,
                    )
                    sample_high_count += 1
                    high_entropy_count += 1
                    token_id = int(target_ids[t].item())
                    token_str = tokenizer.convert_ids_to_tokens([token_id])[0]
                    top_ids_t = top5_ids[t].tolist()
                    top_probs_t = top5_vals[t].float().tolist()
                    top_tokens_t = tokenizer.convert_ids_to_tokens(top_ids_t)
                    row = {
                        "sample_idx": st + sample_i,
                        "t": t,
                        "token": token_str,
                        "entropy": e,
                        "top1_prob": p1,
                        "top5_mass": p5m,
                        "max_cosine_sim": ms,
                        "top5_cosine_sim": [float(x) for x in top5_cosine_sim[t].tolist()],
                        "case": case,
                        "top5_tokens": top_tokens_t,
                        "top5_probs": [float(x) for x in top_probs_t],
                    }
                    sample_high_positions.append(row)
                    all_high_rows.append(row)
                    if case in case_counts_high:
                        case_counts_high[case] += 1

            _ = sample_high_positions
            _ = sample_high_count

        if torch.cuda.is_available():
            torch.cuda.empty_cache()

    all_ent_arr = np.asarray(all_entropies, dtype=np.float64)
    all_sim_arr = np.asarray(all_max_sims, dtype=np.float64)
    all_ent_arr = all_ent_arr[np.isfinite(all_ent_arr)]
    all_sim_arr = all_sim_arr[np.isfinite(all_sim_arr)]

    print("=" * 50)
    print("【高熵位置类型分析】")
    print("=" * 50)
    print(f"总 token 位置数：{total_positions:>12d}")
    frac = (100.0 * high_entropy_count / total_positions) if total_positions > 0 else 0.0
    print(f"高熵位置数：{high_entropy_count:>14d}  ({frac:.1f}%)")
    print("\n高熵位置中的类型分布：")
    denom_high = max(high_entropy_count, 1)
    denom_total = max(total_positions, 1)
    for c in [CASE_A, CASE_B, CASE_AMBIGUOUS]:
        cnt = case_counts_high[c]
        print(
            f"  {c:<12}: {cnt:>5d}  "
            f"({100.0 * cnt / denom_high:>5.1f}% of 高熵, {100.0 * cnt / denom_total:>5.1f}% of 总数)"
        )

    if all_ent_arr.size > 0:
        ent_stats = _summarize(all_ent_arr, [0.5, 0.9])
        print("\n熵统计：")
        print(
            f"  mean={all_ent_arr.mean():.2f}, median={ent_stats[0.5]:.2f}, "
            f"p90={ent_stats[0.9]:.2f}, max={all_ent_arr.max():.2f}"
        )
    else:
        print("\n熵统计：无数据")

    if all_sim_arr.size > 0:
        sim_stats = _summarize(all_sim_arr, [0.1, 0.5])
        print("\n最大余弦相似度统计（h_t vs 词嵌入）：")
        print(
            f"  mean={all_sim_arr.mean():.2f}, median={sim_stats[0.5]:.2f}, "
            f"p10={sim_stats[0.1]:.2f}, min={all_sim_arr.min():.2f}"
        )
    else:
        print("\n最大余弦相似度统计：无数据")

    examples = _collect_examples(all_high_rows, per_case=2)
    for c in [CASE_A, CASE_B, CASE_AMBIGUOUS]:
        for ex in examples[c]:
            print(
                f"\n[{c}] t={ex['t']}, entropy={ex['entropy']:.2f}, "
                f"top5_mass={ex['top5_mass']:.2f}, max_sim={ex['max_cosine_sim']:.2f}"
            )
            print(f"  top-1 token : {repr(ex['token'])}")
            print(f"  top-5 tokens: {[t for t in ex['top5_tokens']]}")
            print(f"  top-5 probs : {[round(float(x), 4) for x in ex['top5_probs']]}")

    _plot_results(all_high_rows, output_plot)
    print(f"\n[plot] saved: {output_plot.resolve()}")

    with output_jsonl.open("w", encoding="utf-8") as f:
        for row in per_token_json_rows:
            f.write(json.dumps(row, ensure_ascii=False) + "\n")
    print(f"[jsonl] saved: {output_jsonl.resolve()}")


if __name__ == "__main__":
    main()
