#!/usr/bin/env python3
from __future__ import annotations

import argparse
import csv
import json
import multiprocessing as mp
from pathlib import Path
from types import SimpleNamespace
from typing import Any

import matplotlib.pyplot as plt
import numpy as np
import torch
from tqdm import tqdm
from transformers import AutoModelForCausalLM, AutoTokenizer

from analyze_entropy_for_logits import (
    _build_batch_inputs_from_ids,
    _build_model_inputs_for_row,
    answers_match,
    extract_answer,
    resolve_input_rows,
)


def _compute_direction_diversity(
    hidden_states: tuple[torch.Tensor, ...],
    batch_idx: int,
    pos: int,
    layer_indices: list[int] | None = None,
) -> float:
    n_layers = len(hidden_states) - 1
    if n_layers <= 1:
        return float("nan")
    if layer_indices is None:
        layer_indices = list(range(1, n_layers + 1))
    if len(layer_indices) < 2:
        return float("nan")

    deltas: list[torch.Tensor] = []
    for l in layer_indices:
        if l <= 0 or l >= len(hidden_states):
            continue
        delta = hidden_states[l][batch_idx, pos, :] - hidden_states[l - 1][batch_idx, pos, :]
        deltas.append(delta.float())
    if len(deltas) < 2:
        return float("nan")

    ds = torch.stack(deltas, dim=0)
    ds = ds / ds.norm(dim=-1, keepdim=True).clamp(min=1e-8)
    sim = ds @ ds.transpose(0, 1)
    n = sim.shape[0]
    mask = torch.triu(torch.ones((n, n), device=sim.device, dtype=torch.bool), diagonal=1)
    return float(sim[mask].mean().item())


def _compute_segment_diversities(
    hidden_states: tuple[torch.Tensor, ...],
    batch_idx: int,
    pos: int,
) -> dict[str, float]:
    n_layers = len(hidden_states) - 1
    if n_layers <= 1:
        return {"early": float("nan"), "middle": float("nan"), "late": float("nan")}

    seg1_end = max(1, n_layers // 3)
    seg2_end = max(seg1_end + 1, (2 * n_layers) // 3)
    early = list(range(1, seg1_end + 1))
    middle = list(range(seg1_end + 1, seg2_end + 1))
    late = list(range(seg2_end + 1, n_layers + 1))
    return {
        "early": _compute_direction_diversity(hidden_states, batch_idx, pos, early),
        "middle": _compute_direction_diversity(hidden_states, batch_idx, pos, middle),
        "late": _compute_direction_diversity(hidden_states, batch_idx, pos, late),
    }


def _compute_hidden_state_norm(
    hidden_states: tuple[torch.Tensor, ...],
    batch_idx: int,
    pos: int,
    layer_idx: int = -1,
) -> float:
    if len(hidden_states) == 0:
        return float("nan")
    hs = hidden_states[layer_idx][batch_idx, pos, :].float()
    return float(torch.norm(hs, p=2).item())


def _safe_float(x: Any) -> float | None:
    if x is None:
        return None
    v = float(x)
    if np.isfinite(v):
        return v
    return None


def _write_csv(rows: list[dict[str, Any]], path: Path) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    if not rows:
        with path.open("w", encoding="utf-8", newline="") as f:
            writer = csv.writer(f)
            writer.writerow(
                [
                    "sample_idx",
                    "t",
                    "log_sum_exp",
                    "is_correct",
                    "group",
                    "direction_diversity",
                    "div_early",
                    "div_middle",
                    "div_late",
                    "hidden_state_norm",
                ]
            )
        return
    keys = list(rows[0].keys())
    with path.open("w", encoding="utf-8", newline="") as f:
        writer = csv.DictWriter(f, fieldnames=keys)
        writer.writeheader()
        for r in rows:
            writer.writerow(r)


def _build_scatter(rows: list[dict[str, Any]], out_path: Path) -> None:
    out_path.parent.mkdir(parents=True, exist_ok=True)
    fig, ax = plt.subplots(figsize=(8, 5))
    wrong_x, wrong_y, corr_x, corr_y = [], [], [], []
    for r in rows:
        x = _safe_float(r.get("log_sum_exp"))
        y = _safe_float(r.get("direction_diversity"))
        if x is None or y is None:
            continue
        if bool(r.get("is_correct", False)):
            corr_x.append(x)
            corr_y.append(y)
        else:
            wrong_x.append(x)
            wrong_y.append(y)
    if corr_x:
        ax.scatter(corr_x, corr_y, alpha=0.35, s=10, label="correct")
    if wrong_x:
        ax.scatter(wrong_x, wrong_y, alpha=0.35, s=10, label="wrong")
    ax.set_xlabel("log_sum_exp (LSE)")
    ax.set_ylabel("mean pairwise cosine sim of layer deltas")
    ax.set_title("LSE vs Direction Diversity")
    ax.grid(True, alpha=0.3)
    if corr_x or wrong_x:
        ax.legend()
    fig.tight_layout()
    fig.savefig(out_path, dpi=150)
    plt.close(fig)


def _build_violin(rows: list[dict[str, Any]], out_path: Path) -> None:
    out_path.parent.mkdir(parents=True, exist_ok=True)
    low_vals = np.array(
        [_safe_float(r.get("direction_diversity")) for r in rows if r.get("group") == "low_lse"],
        dtype=np.float64,
    )
    high_vals = np.array(
        [_safe_float(r.get("direction_diversity")) for r in rows if r.get("group") == "high_lse"],
        dtype=np.float64,
    )
    low_vals = low_vals[np.isfinite(low_vals)]
    high_vals = high_vals[np.isfinite(high_vals)]

    fig, ax = plt.subplots(figsize=(6, 4))
    data = []
    labels = []
    if low_vals.size > 0:
        data.append(low_vals)
        labels.append("low LSE")
    if high_vals.size > 0:
        data.append(high_vals)
        labels.append("high LSE")
    if data:
        ax.violinplot(data, showmedians=True)
        ax.set_xticks(np.arange(1, len(labels) + 1))
        ax.set_xticklabels(labels)
    else:
        ax.text(0.5, 0.5, "No data", ha="center", va="center", transform=ax.transAxes)
        ax.set_axis_off()
    ax.set_ylabel("mean pairwise cosine sim of layer deltas")
    ax.set_title("Direction Diversity by LSE Group")
    fig.tight_layout()
    fig.savefig(out_path, dpi=150)
    plt.close(fig)


def _mann_whitney_less(low_vals: np.ndarray, high_vals: np.ndarray) -> tuple[float | None, float | None]:
    try:
        from scipy import stats

        if low_vals.size == 0 or high_vals.size == 0:
            return None, None
        stat, p = stats.mannwhitneyu(low_vals, high_vals, alternative="less")
        return float(stat), float(p)
    except Exception:
        return None, None


def _spearman_corr(x: np.ndarray, y: np.ndarray) -> tuple[float | None, float | None]:
    try:
        from scipy import stats

        if x.size == 0 or y.size == 0:
            return None, None
        rho, p = stats.spearmanr(x, y)
        return float(rho), float(p)
    except Exception:
        return None, None


def _run_partial_cancellation_chunk(
    rows_selected: list[dict[str, Any]],
    *,
    model_path: str,
    template: str,
    disable_lf_template: bool,
    batch_size: int,
    low_lse_quantile: float,
    high_lse_top_ratio: float,
    device: torch.device,
) -> tuple[list[dict[str, Any]], int]:
    tokenizer = AutoTokenizer.from_pretrained(model_path, trust_remote_code=True)
    if tokenizer.pad_token_id is None:
        tokenizer.pad_token = tokenizer.eos_token

    lf_template = None
    if not disable_lf_template:
        try:
            from llamafactory.data import get_template_and_fix_tokenizer

            data_args = SimpleNamespace(
                template=template,
                train_on_prompt=False,
                tool_format=None,
                default_system=None,
                enable_thinking=True,
            )
            lf_template = get_template_and_fix_tokenizer(tokenizer, data_args)
        except Exception:
            lf_template = None

    model = AutoModelForCausalLM.from_pretrained(
        model_path,
        torch_dtype=torch.float16,
        device_map=None,
        trust_remote_code=True,
        low_cpu_mem_usage=True,
    ).eval()
    model.to(device)

    result_rows: list[dict[str, Any]] = []
    n_skipped_short = 0
    for st in tqdm(range(0, len(rows_selected), batch_size), desc=f"partial_cancellation[{device}]"):
        batch_rows = rows_selected[st : st + batch_size]
        batch_full_ids: list[list[int]] = []
        prompt_lens: list[int] = []
        for r in batch_rows:
            full_ids_i, prompt_len_i, _ = _build_model_inputs_for_row(
                row=r,
                tokenizer=tokenizer,
                lf_template=lf_template,
            )
            batch_full_ids.append(full_ids_i)
            prompt_lens.append(int(prompt_len_i))

        input_ids, attention_mask = _build_batch_inputs_from_ids(
            token_id_seqs=batch_full_ids,
            pad_token_id=int(tokenizer.pad_token_id),
            device=device,
        )

        with torch.no_grad():
            out = model(input_ids=input_ids, attention_mask=attention_mask, output_hidden_states=True)
            logits = out.logits
            hidden_states = out.hidden_states

        bsz = logits.shape[0]
        for i in range(bsz):
            attn_len = int(attention_mask[i].sum().item())
            prompt_len = int(prompt_lens[i])
            is_correct = bool(batch_rows[i].get("_is_correct", False))
            if attn_len <= 1 or prompt_len >= attn_len:
                n_skipped_short += 1
                continue

            pred_start = max(prompt_len - 1, 0)
            pred_end = attn_len - 1
            if pred_start >= pred_end:
                n_skipped_short += 1
                continue

            logits_i = logits[i, pred_start:pred_end, :]
            target_ids = input_ids[i, prompt_len:attn_len]
            npos = int(min(logits_i.shape[0], target_ids.shape[0]))
            if npos <= 1:
                n_skipped_short += 1
                continue

            logits_i = logits_i[:npos]
            logits_i_fp32 = torch.nan_to_num(logits_i.float(), nan=0.0, posinf=1e4, neginf=-1e4)
            lse_vec = torch.logsumexp(logits_i_fp32, dim=-1).float().cpu().numpy()
            if lse_vec.size <= 1:
                n_skipped_short += 1
                continue

            low_thr = float(np.quantile(lse_vec, low_lse_quantile))
            high_thr = float(np.quantile(lse_vec, 1.0 - high_lse_top_ratio))
            low_idx = np.where(lse_vec <= low_thr)[0]
            high_idx = np.where(lse_vec >= high_thr)[0]
            if low_idx.size == 0 or high_idx.size == 0:
                n_skipped_short += 1
                continue

            low_sorted = low_idx[np.argsort(lse_vec[low_idx])]
            high_sorted = high_idx[np.argsort(-lse_vec[high_idx])]
            low_pick = low_sorted
            high_pick = high_sorted

            for t in low_pick.tolist():
                abs_pos = pred_start + int(t)
                div = _compute_direction_diversity(hidden_states, i, abs_pos)
                seg = _compute_segment_diversities(hidden_states, i, abs_pos)
                hs_norm = _compute_hidden_state_norm(hidden_states, i, abs_pos)
                result_rows.append(
                    {
                        "sample_idx": int(batch_rows[i].get("_global_sample_idx", st + i)),
                        "t": int(t),
                        "log_sum_exp": float(lse_vec[t]),
                        "is_correct": bool(is_correct),
                        "group": "low_lse",
                        "direction_diversity": float(div),
                        "div_early": float(seg["early"]),
                        "div_middle": float(seg["middle"]),
                        "div_late": float(seg["late"]),
                        "hidden_state_norm": float(hs_norm),
                    }
                )
            for t in high_pick.tolist():
                abs_pos = pred_start + int(t)
                div = _compute_direction_diversity(hidden_states, i, abs_pos)
                seg = _compute_segment_diversities(hidden_states, i, abs_pos)
                hs_norm = _compute_hidden_state_norm(hidden_states, i, abs_pos)
                result_rows.append(
                    {
                        "sample_idx": int(batch_rows[i].get("_global_sample_idx", st + i)),
                        "t": int(t),
                        "log_sum_exp": float(lse_vec[t]),
                        "is_correct": bool(is_correct),
                        "group": "high_lse",
                        "direction_diversity": float(div),
                        "div_early": float(seg["early"]),
                        "div_middle": float(seg["middle"]),
                        "div_late": float(seg["late"]),
                        "hidden_state_norm": float(hs_norm),
                    }
                )

        del out, logits, hidden_states, input_ids, attention_mask
        if torch.cuda.is_available():
            torch.cuda.empty_cache()

    del model
    if torch.cuda.is_available():
        torch.cuda.empty_cache()
    return result_rows, n_skipped_short


def _mp_worker(job: dict[str, Any]) -> dict[str, Any]:
    gpu_id = int(job["gpu_id"])
    if torch.cuda.is_available():
        torch.cuda.set_device(gpu_id)
        device = torch.device(f"cuda:{gpu_id}")
    else:
        device = torch.device("cpu")
    rows = job["rows"]
    cfg = job["cfg"]
    result_rows, n_skipped_short = _run_partial_cancellation_chunk(
        rows_selected=rows,
        model_path=cfg["model"],
        template=cfg["template"],
        disable_lf_template=cfg["disable_lf_template"],
        batch_size=int(cfg["batch_size"]),
        low_lse_quantile=float(cfg["low_lse_quantile"]),
        high_lse_top_ratio=float(cfg["high_lse_top_ratio"]),
        device=device,
    )
    return {"result_rows": result_rows, "n_skipped_short": n_skipped_short}


def main() -> None:
    parser = argparse.ArgumentParser(description="Partial cancellation analysis on response tokens.")
    parser.add_argument("--data", type=str, required=True)
    parser.add_argument("--dataset", type=str, default=None)
    parser.add_argument("--dataset_info", type=str, default=None)
    parser.add_argument("--prompt_field", type=str, default=None)
    parser.add_argument("--pred_field", type=str, default=None)
    parser.add_argument("--label_field", type=str, default=None)
    parser.add_argument("--template", type=str, default="llama3")
    parser.add_argument("--disable_lf_template", action="store_true")
    parser.add_argument("--model", type=str, required=True)
    parser.add_argument("--max_samples", type=int, default=2000)
    parser.add_argument("--batch_size", type=int, default=2)
    parser.add_argument(
        "--low_lse_quantile",
        type=float,
        default=0.1,
        help="low_lse 使用的底部分位数（默认 0.1，即最低 10%）。",
    )
    parser.add_argument(
        "--high_lse_top_ratio",
        type=float,
        default=0.2,
        help="high_lse 使用的顶部比例（默认 0.2，即最高 20%）。",
    )
    parser.add_argument("--use_multi_process", action="store_true", help="启用多卡多进程（每卡一个进程）。")
    parser.add_argument(
        "--gpu_ids",
        type=str,
        default=None,
        help="多进程使用的 GPU 列表，如 '0,1,2,3'；默认使用全部可见 GPU。",
    )
    parser.add_argument(
        "--num_processes",
        type=int,
        default=0,
        help="多进程数量（0 表示自动=GPU数量）。",
    )
    parser.add_argument("--only_wrong", action="store_true", help="只分析错误样本。")
    parser.add_argument("--only_correct", action="store_true", help="只分析正确样本。")
    parser.add_argument(
        "--output_dir",
        type=str,
        default="partial_cancellation_outputs",
        help="输出子目录，CSV/JSON/图片都会写到该目录下。",
    )
    args = parser.parse_args()

    if args.only_wrong and args.only_correct:
        raise ValueError("--only_wrong 和 --only_correct 不能同时设置。")
    if not (0.0 < float(args.low_lse_quantile) < 1.0):
        raise ValueError("--low_lse_quantile 必须在 (0,1) 内。")
    if not (0.0 < float(args.high_lse_top_ratio) < 1.0):
        raise ValueError("--high_lse_top_ratio 必须在 (0,1) 内。")

    rows_raw, data_meta = resolve_input_rows(
        data_arg=args.data,
        dataset_name=args.dataset,
        dataset_info_arg=args.dataset_info,
        prompt_field_arg=args.prompt_field,
        pred_field_arg=args.pred_field,
        label_field_arg=args.label_field,
    )
    if not rows_raw:
        raise ValueError("输入数据为空。")

    rows_selected: list[dict[str, Any]] = []
    for row in rows_raw:
        pred = str(row.get("predict", ""))
        label = str(row.get("label", ""))
        is_correct = answers_match(extract_answer(pred), extract_answer(label))
        rr = dict(row)
        rr["_is_correct"] = bool(is_correct)
        if args.only_wrong and is_correct:
            continue
        if args.only_correct and (not is_correct):
            continue
        rows_selected.append(rr)
        if len(rows_selected) >= args.max_samples:
            break

    if not rows_selected:
        raise ValueError("筛选后没有样本可分析。")

    filter_mode = "wrong" if args.only_wrong else ("correct" if args.only_correct else "all")
    print(
        f"[load] mode={data_meta['mode']} data={data_meta['data_file']} selected={len(rows_selected)} filter={filter_mode}"
    )
    print(f"[load] model/tokenizer: {args.model}")

    n_gpu = torch.cuda.device_count() if torch.cuda.is_available() else 0
    use_mp = args.use_multi_process and n_gpu > 1
    if use_mp:
        if args.gpu_ids:
            gpu_ids = [int(x.strip()) for x in args.gpu_ids.split(",") if x.strip() != ""]
        else:
            gpu_ids = list(range(n_gpu))
        if not gpu_ids:
            raise ValueError("启用多进程时 gpu_ids 为空。")
        n_proc = int(args.num_processes) if int(args.num_processes) > 0 else len(gpu_ids)
        n_proc = min(n_proc, len(gpu_ids))
        gpu_ids = gpu_ids[:n_proc]

        print(f"[device] cuda_available=True n_gpu={n_gpu} use_multi_process=True gpu_ids={gpu_ids}")
        shards: list[list[dict[str, Any]]] = [[] for _ in range(n_proc)]
        for idx, row in enumerate(rows_selected):
            rr = dict(row)
            rr["_global_sample_idx"] = int(idx)
            shards[idx % n_proc].append(rr)

        jobs: list[dict[str, Any]] = []
        cfg = {
            "model": args.model,
            "template": args.template,
            "disable_lf_template": bool(args.disable_lf_template),
            "batch_size": int(args.batch_size),
            "low_lse_quantile": float(args.low_lse_quantile),
            "high_lse_top_ratio": float(args.high_lse_top_ratio),
        }
        for wi in range(n_proc):
            jobs.append({"gpu_id": gpu_ids[wi], "rows": shards[wi], "cfg": cfg})

        ctx = mp.get_context("spawn")
        with ctx.Pool(processes=n_proc) as pool:
            worker_outs = pool.map(_mp_worker, jobs)
        result_rows: list[dict[str, Any]] = []
        n_skipped_short = 0
        for wo in worker_outs:
            result_rows.extend(wo["result_rows"])
            n_skipped_short += int(wo["n_skipped_short"])
    else:
        if args.use_multi_process and n_gpu <= 1:
            print("[warn] --use_multi_process 已设置，但可见 GPU 数 <= 1，回退到单进程。")
        device = torch.device("cuda:0" if torch.cuda.is_available() else "cpu")
        print(f"[device] cuda_available={torch.cuda.is_available()} n_gpu={n_gpu} use_multi_process=False device={device}")
        rows_for_single: list[dict[str, Any]] = []
        for idx, row in enumerate(rows_selected):
            rr = dict(row)
            rr["_global_sample_idx"] = int(idx)
            rows_for_single.append(rr)
        result_rows, n_skipped_short = _run_partial_cancellation_chunk(
            rows_selected=rows_for_single,
            model_path=args.model,
            template=args.template,
            disable_lf_template=bool(args.disable_lf_template),
            batch_size=int(args.batch_size),
            low_lse_quantile=float(args.low_lse_quantile),
            high_lse_top_ratio=float(args.high_lse_top_ratio),
            device=device,
        )

    output_dir = Path(args.output_dir)
    output_dir.mkdir(parents=True, exist_ok=True)
    out_csv = output_dir / "partial_cancellation_results.csv"
    out_summary = output_dir / "partial_cancellation_summary.json"
    out_scatter = output_dir / "partial_cancellation_scatter.png"
    out_violin = output_dir / "partial_cancellation_violin.png"
    _write_csv(result_rows, out_csv)
    _build_scatter(result_rows, out_scatter)
    _build_violin(result_rows, out_violin)

    low_vals = np.array(
        [float(r["direction_diversity"]) for r in result_rows if r["group"] == "low_lse"],
        dtype=np.float64,
    )
    high_vals = np.array(
        [float(r["direction_diversity"]) for r in result_rows if r["group"] == "high_lse"],
        dtype=np.float64,
    )
    low_vals = low_vals[np.isfinite(low_vals)]
    high_vals = high_vals[np.isfinite(high_vals)]
    low_norm_vals = np.array(
        [float(r["hidden_state_norm"]) for r in result_rows if r["group"] == "low_lse"],
        dtype=np.float64,
    )
    high_norm_vals = np.array(
        [float(r["hidden_state_norm"]) for r in result_rows if r["group"] == "high_lse"],
        dtype=np.float64,
    )
    low_norm_vals = low_norm_vals[np.isfinite(low_norm_vals)]
    high_norm_vals = high_norm_vals[np.isfinite(high_norm_vals)]
    all_lse = np.array([float(r["log_sum_exp"]) for r in result_rows], dtype=np.float64)
    all_div = np.array([float(r["direction_diversity"]) for r in result_rows], dtype=np.float64)
    mask = np.isfinite(all_lse) & np.isfinite(all_div)
    all_lse = all_lse[mask]
    all_div = all_div[mask]

    mw_stat, mw_p = _mann_whitney_less(low_vals, high_vals)
    mw_norm_stat, mw_norm_p = _mann_whitney_less(low_norm_vals, high_norm_vals)
    rho, rho_p = _spearman_corr(all_lse, all_div)

    summary = {
        "model": args.model,
        "n_rows_raw": len(rows_raw),
        "n_rows_selected": len(rows_selected),
        "filter_mode": filter_mode,
        "low_lse_quantile": float(args.low_lse_quantile),
        "high_lse_top_ratio": float(args.high_lse_top_ratio),
        "n_positions_exported": len(result_rows),
        "n_samples_skipped_short_or_empty": n_skipped_short,
        "group_stats": {
            "low_lse": {
                "n": int(low_vals.size),
                "mean_direction_diversity": (float(np.mean(low_vals)) if low_vals.size > 0 else None),
                "std_direction_diversity": (float(np.std(low_vals, ddof=0)) if low_vals.size > 0 else None),
                "mean_hidden_state_norm": (float(np.mean(low_norm_vals)) if low_norm_vals.size > 0 else None),
                "std_hidden_state_norm": (float(np.std(low_norm_vals, ddof=0)) if low_norm_vals.size > 0 else None),
            },
            "high_lse": {
                "n": int(high_vals.size),
                "mean_direction_diversity": (float(np.mean(high_vals)) if high_vals.size > 0 else None),
                "std_direction_diversity": (float(np.std(high_vals, ddof=0)) if high_vals.size > 0 else None),
                "mean_hidden_state_norm": (float(np.mean(high_norm_vals)) if high_norm_vals.size > 0 else None),
                "std_hidden_state_norm": (float(np.std(high_norm_vals, ddof=0)) if high_norm_vals.size > 0 else None),
            },
        },
        "mann_whitney_u_less": {"stat": mw_stat, "p_value": mw_p},
        "mann_whitney_u_less_hidden_state_norm": {"stat": mw_norm_stat, "p_value": mw_norm_p},
        "spearman_lse_vs_diversity": {"rho": rho, "p_value": rho_p},
        "outputs": {
            "csv": str(out_csv.resolve()),
            "summary_json": str(out_summary.resolve()),
            "scatter_png": str(out_scatter.resolve()),
            "violin_png": str(out_violin.resolve()),
        },
    }
    out_summary.parent.mkdir(parents=True, exist_ok=True)
    out_summary.write_text(json.dumps(summary, ensure_ascii=False, indent=2), encoding="utf-8")

    print(f"[done] csv={out_csv.resolve()}")
    print(f"[done] summary={out_summary.resolve()}")
    print(f"[done] scatter={out_scatter.resolve()}")
    print(f"[done] violin={out_violin.resolve()}")
    print(
        f"[stats] low_mean={summary['group_stats']['low_lse']['mean_direction_diversity']} "
        f"high_mean={summary['group_stats']['high_lse']['mean_direction_diversity']} "
        f"low_norm_mean={summary['group_stats']['low_lse']['mean_hidden_state_norm']} "
        f"high_norm_mean={summary['group_stats']['high_lse']['mean_hidden_state_norm']} "
        f"mw_p={mw_p} mw_norm_p={mw_norm_p} spearman_rho={rho}"
    )


if __name__ == "__main__":
    main()
