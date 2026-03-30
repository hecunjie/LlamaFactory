#!/usr/bin/env python3
from __future__ import annotations

import argparse
import csv
import json
import math
import re
from pathlib import Path
from types import SimpleNamespace
from typing import Any, Optional

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


def load_json_array(path: Path) -> list[dict[str, Any]]:
    obj = json.loads(path.read_text(encoding="utf-8"))
    if isinstance(obj, list):
        return [x for x in obj if isinstance(x, dict)]
    if isinstance(obj, dict):
        # 兼容 {"data": [...]} 这类封装
        data_obj = obj.get("data")
        if isinstance(data_obj, list):
            return [x for x in data_obj if isinstance(x, dict)]
    raise ValueError(f"不支持的 JSON 格式（需要 list[dict] 或 {{'data': [...]}}）：{path}")


def load_data_records(path: Path) -> list[dict[str, Any]]:
    suf = path.suffix.lower()
    if suf == ".jsonl":
        return load_jsonl(path)
    if suf == ".json":
        return load_json_array(path)
    raise ValueError(f"仅支持 .json/.jsonl 文件: {path}")


def _pick_first(row: dict[str, Any], keys: list[str], default: str = "") -> str:
    for k in keys:
        if k in row and row.get(k) is not None:
            return str(row.get(k))
    return default


def _normalize_rows_with_fields(
    rows: list[dict[str, Any]],
    *,
    prompt_field: str,
    pred_field: str,
    label_field: str,
) -> list[dict[str, Any]]:
    out: list[dict[str, Any]] = []
    for row in rows:
        rr = dict(row)
        rr["prompt"] = _pick_first(rr, [prompt_field, "prompt", "instruction", "input", "question"], default="")
        rr["predict"] = _pick_first(rr, [pred_field, "predict", "predictions", "output", "response"], default="")
        rr["label"] = _pick_first(rr, [label_field, "label", "response", "output"], default="")
        out.append(rr)
    return out


def resolve_input_rows(
    *,
    data_arg: str,
    dataset_name: Optional[str],
    dataset_info_arg: Optional[str],
    prompt_field_arg: Optional[str],
    pred_field_arg: Optional[str],
    label_field_arg: Optional[str],
) -> tuple[list[dict[str, Any]], dict[str, Any]]:
    data_path = Path(data_arg)
    is_dataset_info_mode = data_path.is_dir() or data_path.name == "dataset_info.json"

    if is_dataset_info_mode:
        data_dir = data_path if data_path.is_dir() else data_path.parent
        dataset_info_path = Path(dataset_info_arg) if dataset_info_arg else (data_dir / "dataset_info.json")
        if not dataset_info_path.exists():
            raise FileNotFoundError(f"未找到 dataset_info.json: {dataset_info_path}")
        dataset_info = json.loads(dataset_info_path.read_text(encoding="utf-8"))
        if not isinstance(dataset_info, dict):
            raise ValueError(f"dataset_info.json 格式错误（应为 dict）：{dataset_info_path}")
        if not dataset_name:
            raise ValueError("检测到 dataset_info 模式，请通过 --dataset 指定数据集名称。")
        if dataset_name not in dataset_info:
            raise KeyError(f"--dataset={dataset_name} 不在 {dataset_info_path} 中。")
        ds_conf = dataset_info[dataset_name]
        if not isinstance(ds_conf, dict):
            raise ValueError(f"dataset_info['{dataset_name}'] 格式错误（应为 dict）。")
        file_name = ds_conf.get("file_name")
        if not file_name:
            raise ValueError(f"dataset_info['{dataset_name}'] 缺少 file_name，无法定位本地数据文件。")
        file_path = Path(file_name)
        if not file_path.is_absolute():
            file_path = data_dir / file_path
        rows_raw = load_data_records(file_path)

        # dataset_info 模式下按需求默认 output/response（而不是 predict/predictions）
        prompt_field = prompt_field_arg or "prompt"
        pred_field = pred_field_arg or "output"
        label_field = label_field_arg or "response"
        rows = _normalize_rows_with_fields(
            rows_raw,
            prompt_field=prompt_field,
            pred_field=pred_field,
            label_field=label_field,
        )
        meta = {
            "mode": "dataset_info",
            "data_file": str(file_path),
            "dataset_info": str(dataset_info_path),
            "dataset": dataset_name,
            "fields": {"prompt": prompt_field, "predict": pred_field, "label": label_field},
        }
        return rows, meta

    rows_raw = load_data_records(data_path)
    prompt_field = prompt_field_arg or "prompt"
    pred_field = pred_field_arg or "predict"
    label_field = label_field_arg or "label"
    rows = _normalize_rows_with_fields(
        rows_raw,
        prompt_field=prompt_field,
        pred_field=pred_field,
        label_field=label_field,
    )
    meta = {
        "mode": "direct_file",
        "data_file": str(data_path),
        "fields": {"prompt": prompt_field, "predict": pred_field, "label": label_field},
    }
    return rows, meta


def _build_batch_inputs_from_ids(
    token_id_seqs: list[list[int]],
    pad_token_id: int,
    device: torch.device,
) -> tuple[torch.Tensor, torch.Tensor]:
    max_len = max(len(x) for x in token_id_seqs)
    bs = len(token_id_seqs)
    input_ids = torch.full((bs, max_len), int(pad_token_id), dtype=torch.long, device=device)
    attention_mask = torch.zeros((bs, max_len), dtype=torch.long, device=device)
    for i, ids in enumerate(token_id_seqs):
        if not ids:
            continue
        n = len(ids)
        input_ids[i, :n] = torch.tensor(ids, dtype=torch.long, device=device)
        attention_mask[i, :n] = 1
    return input_ids, attention_mask


def _build_model_inputs_for_row(
    row: dict[str, Any],
    tokenizer: Any,
    lf_template: Any = None,
) -> tuple[list[int], int, str]:
    prompt_text = str(row.get("prompt", ""))
    predict_text = str(row.get("predict", ""))
    if lf_template is None:
        full_text = prompt_text + predict_text
        full_ids = tokenizer(full_text, add_special_tokens=False)["input_ids"]
        prompt_ids = tokenizer(prompt_text, add_special_tokens=False)["input_ids"]
        return full_ids, len(prompt_ids), prompt_text

    messages = [
        {"role": "user", "content": prompt_text},
        {"role": "assistant", "content": predict_text},
    ]
    system_text_raw = row.get("system", None)
    tools_text_raw = row.get("tools", None)
    system_text = str(system_text_raw) if system_text_raw is not None else None
    tools_text = str(tools_text_raw) if tools_text_raw is not None else None
    prompt_ids, response_ids = lf_template.encode_oneturn(
        tokenizer,
        messages,
        system=system_text,
        tools=tools_text,
    )
    full_ids = prompt_ids + response_ids
    prompt_rendered = tokenizer.decode(prompt_ids, skip_special_tokens=False)
    return full_ids, len(prompt_ids), prompt_rendered


def classify_case(
    entropy: float,
    top1_prob: float,
    top5_mass: float,
    max_cosine_sim: float,
    entropy_threshold: float,
    sim_threshold: float,
) -> str:
    # 先 B：隐藏状态与词嵌入相似度过低
    if max_cosine_sim < sim_threshold:
        return CASE_B
    if entropy <= entropy_threshold:
        return CASE_NORMAL
    if top5_mass >= 0.6 and top1_prob < 0.4:
        return CASE_A
    return CASE_AMBIGUOUS


def classify_high_case(
    top1_prob: float,
    top5_mass: float,
    max_cosine_sim: float,
    sim_threshold: float,
) -> str:
    # 与 classify_case 一致：先 B，再 A，其余 ambiguous（高熵子集不写 normal）
    if max_cosine_sim < sim_threshold:
        return CASE_B
    if top5_mass >= 0.6 and top1_prob < 0.4:
        return CASE_A
    return CASE_AMBIGUOUS


def _summarize(arr: np.ndarray, keys: list[float]) -> dict[float, float]:
    if arr.size == 0:
        return {k: float("nan") for k in keys}
    return {k: float(np.quantile(arr, k)) for k in keys}


def _plot_log_sum_exp_quantiles(values: np.ndarray, output_path: Path, title_suffix: str = "") -> None:
    """经验分位数曲线：横轴为百分位，纵轴为 log_sum_exp。"""
    values = np.asarray(values, dtype=np.float64)
    values = values[np.isfinite(values)]
    if values.size == 0:
        fig, ax = plt.subplots(figsize=(8, 5))
        ax.text(0.5, 0.5, "No log_sum_exp values.", ha="center", va="center", transform=ax.transAxes)
        ax.set_axis_off()
        fig.tight_layout()
        fig.savefig(output_path, dpi=150)
        plt.close(fig)
        return
    qs = np.linspace(0.0, 1.0, 101)
    y = np.quantile(values, qs)
    fig, ax = plt.subplots(figsize=(8, 5))
    ax.plot(qs * 100.0, y, color="tab:blue", linewidth=1.5)
    ax.fill_between(qs * 100.0, y, alpha=0.15, color="tab:blue")
    ax.set_xlabel("分位数 (百分位)")
    ax.set_ylabel("log_sum_exp")
    ttl = "log_sum_exp 经验分位数曲线"
    if title_suffix:
        ttl = f"{ttl} ({title_suffix})"
    ax.set_title(ttl)
    ax.grid(True, alpha=0.3)
    fig.tight_layout()
    fig.savefig(output_path, dpi=150)
    plt.close(fig)


def _save_log_sum_exp_stats(
    all_lse: np.ndarray,
    lse_group_a: np.ndarray,
    output_json: Path,
) -> dict[str, Any]:
    preset_q = [0.01, 0.05, 0.1, 0.25, 0.5, 0.75, 0.9, 0.95, 0.99]
    out: dict[str, Any] = {
        "n_positions_all": int(all_lse.size),
        "n_positions_case_A": int(lse_group_a.size),
        "mean_log_sum_exp_case_A": float(np.mean(lse_group_a)) if lse_group_a.size > 0 else None,
        "quantiles_all": {},
    }
    if all_lse.size > 0:
        a = all_lse[np.isfinite(all_lse)]
        for p in preset_q:
            pct = int(round(p * 100))
            out["quantiles_all"][f"p{pct:02d}"] = float(np.quantile(a, p))
    with output_json.open("w", encoding="utf-8") as f:
        json.dump(out, f, ensure_ascii=False, indent=2)
    return out


def _accumulate_lse_threshold_one_response(
    lse_vec: np.ndarray,
    npos: int,
    threshold: float,
    out_rl: list[int],
    out_below: list[int],
    out_frac: list[float],
    out_min_lse: list[float],
    out_pos_ratio: list[float],
) -> None:
    """在有效 response 段上累积单样本标量（与 classify 的 token 范围一致）。"""
    if npos <= 0:
        return
    lse_np = np.asarray(lse_vec, dtype=np.float64)
    rl = int(npos)
    below = int(np.sum(lse_np < float(threshold)))
    frac = (below / rl) if rl > 0 else float("nan")
    min_lse = float(np.min(lse_np))
    argmin = int(np.argmin(lse_np))
    pos_ratio = ((argmin + 1) / rl) if rl > 0 else float("nan")
    out_rl.append(rl)
    out_below.append(below)
    out_frac.append(frac)
    out_min_lse.append(min_lse)
    out_pos_ratio.append(pos_ratio)


def _build_lse_threshold_aggregate(
    *,
    threshold: float,
    variant_tag: str,
    filter_mode: str,
    n_samples_total: int,
    response_lengths: list[int],
    below_counts: list[int],
    fracs_below: list[float],
    min_lses: list[float],
    min_pos_ratios: list[float],
) -> dict[str, Any]:
    """由全样本累积列表生成唯一汇总字典。"""
    n_ok = len(response_lengths)
    out: dict[str, Any] = {
        "variant": variant_tag,
        "filter_mode": filter_mode,
        "log_sum_exp_threshold": float(threshold),
        "n_samples_total": int(n_samples_total),
        "n_samples_with_response": int(n_ok),
        "mean_response_length": None,
        "mean_frac_positions_lse_below_threshold": None,
        "global_frac_positions_lse_below_threshold": None,
        "mean_min_lse_position_ratio_in_response": None,
        "mean_min_log_sum_exp": None,
    }
    if n_ok == 0:
        return out
    rl = np.array(response_lengths, dtype=np.float64)
    fr = np.array(fracs_below, dtype=np.float64)
    mins = np.array(min_lses, dtype=np.float64)
    pr = np.array(min_pos_ratios, dtype=np.float64)
    total_below = int(sum(below_counts))
    total_pos = int(sum(response_lengths))
    out["mean_response_length"] = float(np.mean(rl))
    out["mean_frac_positions_lse_below_threshold"] = float(np.mean(fr))
    out["global_frac_positions_lse_below_threshold"] = (total_below / total_pos) if total_pos > 0 else None
    out["mean_min_lse_position_ratio_in_response"] = float(np.mean(pr))
    out["mean_min_log_sum_exp"] = float(np.mean(mins))
    return out


def _save_lse_threshold_summary(agg: dict[str, Any], out_path: Path) -> None:
    out_path.parent.mkdir(parents=True, exist_ok=True)
    text = json.dumps(agg, ensure_ascii=False, indent=2)
    if out_path.suffix.lower() == ".jsonl":
        with out_path.open("w", encoding="utf-8") as f:
            f.write(text + "\n")
    else:
        out_path.write_text(text, encoding="utf-8")


def _resolve_final_norm(model: Any) -> Any:
    """与 CausalLM 在 logits 前使用的 final norm 对齐（Llama: model.model.norm）。"""
    inner = getattr(model, "model", None)
    if inner is not None:
        for name in ("norm", "final_layernorm", "ln_f"):
            if hasattr(inner, name):
                return getattr(inner, name)
    tr = getattr(model, "transformer", None)
    if tr is not None and hasattr(tr, "ln_f"):
        return tr.ln_f
    return None


def _logsumexp_logits_from_pre_norm_hidden(
    hidden_2d: torch.Tensor,
    final_norm: Any,
    lm_head: Any,
) -> np.ndarray:
    """hidden 为某层输出（与 HF hidden_states[k] 一致，未经 final norm）；先 final_norm 再 lm_head，与 base logits 公平可比。"""
    # 与 lm_head.weight 对齐（常见为 fp16 + device_map 时），避免 matmul dtype 不一致
    w_dtype = lm_head.weight.dtype
    h = hidden_2d.to(dtype=w_dtype)
    h = final_norm(h)
    logits = lm_head(h)
    logits_fp32 = torch.nan_to_num(logits.float(), nan=0.0, posinf=1e4, neginf=-1e4)
    return torch.logsumexp(logits_fp32, dim=-1).detach().cpu().numpy()


def _layer_probe_extend_deltas(
    out: Any,
    batch_i: int,
    pred_start: int,
    pred_end: int,
    npos: int,
    lse_base: np.ndarray,
    mask: np.ndarray,
    final_norm: Any,
    lm_head: Any,
    delta_by_layer: list[list[float]],
    lse_by_layer: list[list[float]],
) -> None:
    """在 mask 为 True 的位置上，对各层 hidden 经同一 final_norm+lm_head 算 log_sum_exp，并记录与 base 的差。"""
    idx = np.flatnonzero(mask)
    if idx.size == 0:
        return
    base_sel = lse_base[idx]
    hs = out.hidden_states
    for ell in range(len(hs)):
        h_full = hs[ell][batch_i, pred_start:pred_end, :][:npos]
        h_sel = h_full[idx]
        lse_ell = _logsumexp_logits_from_pre_norm_hidden(h_sel, final_norm, lm_head)
        delta = lse_ell - base_sel
        while len(delta_by_layer) <= ell:
            delta_by_layer.append([])
            lse_by_layer.append([])
        delta_by_layer[ell].extend(delta.astype(np.float64).tolist())
        lse_by_layer[ell].extend(lse_ell.astype(np.float64).tolist())


def _finalize_layer_probe_stats(
    delta_by_layer: list[list[float]],
    lse_by_layer: list[list[float]],
    *,
    variant_tag: str,
    filter_mode: str,
    filter_payload: dict[str, Any],
    n_model_hidden_states: int,
) -> dict[str, Any]:
    per_layer: list[dict[str, Any]] = []
    for ell in range(len(delta_by_layer)):
        d = np.array(delta_by_layer[ell], dtype=np.float64)
        le = np.array(lse_by_layer[ell], dtype=np.float64) if ell < len(lse_by_layer) else np.array([])
        row: dict[str, Any] = {
            "hidden_state_index": ell,
            "n_positions": int(d.size),
        }
        if d.size > 0:
            row["mean_delta_lse_minus_base"] = float(np.mean(d))
            row["std_delta_lse_minus_base"] = float(np.std(d, ddof=0))
            row["mean_log_sum_exp_at_layer"] = float(np.mean(le)) if le.size == d.size else None
        else:
            row["mean_delta_lse_minus_base"] = None
            row["std_delta_lse_minus_base"] = None
            row["mean_log_sum_exp_at_layer"] = None
        per_layer.append(row)
    return {
        "description": (
            "各层：对 hidden_states[ℓ] 先经与 logits 相同的 final LayerNorm/RMSNorm，再过 lm_head 得到 log_sum_exp；"
            "delta = 该值 − base（模型输出 logits 的 log_sum_exp）。最后一层应与 0 接近。"
        ),
        "variant": variant_tag,
        "filter_mode": filter_mode,
        "filter": filter_payload,
        "n_hidden_states_tuple": int(n_model_hidden_states),
        "per_layer": per_layer,
    }


def _plot_lse_layer_probe(summary: dict[str, Any], output_path: Path, title_suffix: str = "") -> None:
    import matplotlib

    matplotlib.use("Agg")
    import matplotlib.pyplot as plt

    per_layer = summary.get("per_layer") or []
    xs: list[int] = []
    means: list[float] = []
    stds: list[float] = []
    for row in per_layer:
        ell = int(row["hidden_state_index"])
        m = row.get("mean_delta_lse_minus_base")
        s = row.get("std_delta_lse_minus_base")
        if m is None:
            continue
        xs.append(ell)
        means.append(float(m))
        stds.append(float(s) if s is not None and not (isinstance(s, float) and math.isnan(s)) else 0.0)
    fig, ax = plt.subplots(figsize=(10, 5))
    if not xs:
        ax.text(0.5, 0.5, "无层探测数据", ha="center", va="center", transform=ax.transAxes)
    else:
        ax.errorbar(xs, means, yerr=stds, fmt="-o", capsize=3, color="tab:blue", ecolor="tab:gray", linewidth=1.2)
        ax.axhline(0.0, color="k", linestyle="--", linewidth=0.8, alpha=0.5)
        ax.set_xlabel("hidden_states 索引（0 常为 embedding 后，末层为最后一层 block 输出）")
        ax.set_ylabel("Δ log_sum_exp = LSE( LN(h_ℓ)→lm_head ) − LSE(base logits)")
        ttl = "各层与 base 的 log_sum_exp 差（仅低 base LSE 位置）"
        if title_suffix:
            ttl = f"{ttl} ({title_suffix})"
        ax.set_title(ttl)
        ax.grid(True, alpha=0.3)
    fig.tight_layout()
    fig.savefig(output_path, dpi=150, format="png")
    plt.close(fig)


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


def _rankdata_average_asc(a: np.ndarray) -> np.ndarray:
    """升序秩（值越小秩越小），并列取平均秩。与 scipy.stats.rankdata(method='average') 一致。"""
    n = a.size
    s = np.argsort(a, kind="mergesort")
    r = np.empty(n, dtype=np.float64)
    r[s] = np.arange(1, n + 1, dtype=np.float64)
    sorted_a = a[s]
    i = 0
    while i < n:
        j = i
        while j + 1 < n and sorted_a[j + 1] == sorted_a[i]:
            j += 1
        if j > i:
            mm = float(r[s[i : j + 1]].mean())
            r[s[i : j + 1]] = mm
        i = j + 1
    return r


def _filter_low_entropy_lse_by_joint_quantile(
    buffer: list[dict[str, Any]],
    q: float,
) -> tuple[list[dict[str, Any]], dict[str, Any]]:
    """用「熵秩与 LSE 秩的平均」作联合分数，取 joint_score ≤ quantile(joint_score, q) 的 token。

    若对熵、LSE **分别**取分位再求交集，在二者负相关时交集可能为空；联合秩可避免 0 条。
    """
    if not buffer:
        meta = {
            "mode": "joint_rank_quantile",
            "joint_quantile": q,
            "joint_score_cutoff": None,
            "n_pool_tokens": 0,
            "n_exported": 0,
            "note": "pool empty",
        }
        return [], meta
    ents = np.array([float(r["entropy"]) for r in buffer], dtype=np.float64)
    lses = np.array([float(r["log_sum_exp"]) for r in buffer], dtype=np.float64)
    rank_e = _rankdata_average_asc(ents)
    rank_l = _rankdata_average_asc(lses)
    joint = (rank_e + rank_l) / 2.0
    joint_cut = float(np.quantile(joint, q))
    # 边际分位仅作参考（与联合筛选无关）
    ent_cut_m = float(np.quantile(ents, q))
    lse_cut_m = float(np.quantile(lses, q))
    meta = {
        "mode": "joint_rank_quantile",
        "joint_quantile": q,
        "joint_score_cutoff": joint_cut,
        "marginal_entropy_cutoff_at_same_q": ent_cut_m,
        "marginal_lse_cutoff_at_same_q": lse_cut_m,
        "n_pool_tokens": len(buffer),
        "description": (
            "joint_score = (rank(entropy)+rank(log_sum_exp))/2，值越小表示熵与 LSE 在池中越偏低；"
            "保留 joint_score ≤ np.quantile(joint_score, q)。"
        ),
    }
    out: list[dict[str, Any]] = []
    for idx, r in enumerate(buffer):
        if float(joint[idx]) <= joint_cut:
            rr = dict(r)
            rr["joint_score"] = float(joint[idx])
            rr["rank_entropy"] = float(rank_e[idx])
            rr["rank_log_sum_exp"] = float(rank_l[idx])
            out.append(rr)
    meta["n_exported"] = len(out)
    for rr in out:
        rr["criteria"] = dict(meta)
    return out, meta


def main() -> None:
    parser = argparse.ArgumentParser(description="Analyze logits entropy and hidden-state confidence.")
    parser.add_argument(
        "--data",
        type=str,
        required=True,
        help=(
            "输入数据路径：可为 .json/.jsonl 文件；"
            "也可传 data 目录或 dataset_info.json（此时需配合 --dataset）。"
        ),
    )
    parser.add_argument(
        "--dataset",
        type=str,
        default=None,
        help="dataset_info 模式下的数据集名（对应 dataset_info.json 的 key）。",
    )
    parser.add_argument(
        "--dataset_info",
        type=str,
        default=None,
        help="可选：显式指定 dataset_info.json 路径；默认使用 {data}/dataset_info.json。",
    )
    parser.add_argument("--prompt_field", type=str, default=None, help="可选：手动指定 prompt 字段名。")
    parser.add_argument("--pred_field", type=str, default=None, help="可选：手动指定预测字段名。")
    parser.add_argument("--label_field", type=str, default=None, help="可选：手动指定标签字段名。")
    parser.add_argument(
        "--template",
        type=str,
        default="llama3",
        help="LlamaFactory 模板名（建议与训练一致）；默认 llama3。",
    )
    parser.add_argument(
        "--disable_lf_template",
        action="store_true",
        help="禁用 LlamaFactory 模板编码，退回旧逻辑（prompt + predict 字符串拼接）。",
    )
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
    parser.add_argument("--only_correct", action="store_true", help="Analyze only correct samples")
    parser.add_argument(
        "--all_samples",
        action="store_true",
        help="Analyze all samples (override only_wrong/only_correct)",
    )
    parser.add_argument("--output_plot", type=str, default="entropy_analysis.png", help="Output plot path")
    parser.add_argument("--output_jsonl", type=str, default="entropy_results.jsonl", help="Output result JSONL path")
    parser.add_argument(
        "--lse_threshold_report",
        action="store_true",
        help=(
            "在全集上汇总：response 上低于阈值的占比、response_length 均值、"
            "最小 log_sum_exp 及其位置占比等，写入单个 JSON（仅汇总，无逐样本行）。"
        ),
    )
    parser.add_argument(
        "--log_sum_exp_threshold",
        type=float,
        default=float("nan"),
        help="低于该值的 token 视为低 log_sum_exp（仅用于 --lse_threshold_report）。",
    )
    parser.add_argument(
        "--output_lse_threshold_jsonl",
        type=str,
        default=None,
        help="阈值汇总输出路径（.json 或单行 .jsonl）；默认 {stem}_lse_threshold_report.json。",
    )
    parser.add_argument(
        "--lse_layer_probe",
        action="store_true",
        help=(
            "在 base log_sum_exp 较低的位置，对各层 hidden_states[ℓ] 先经与 logits 相同的 final LayerNorm/RMSNorm，"
            "再过 lm_head 计算 log_sum_exp，并保存 Δ=LSE_ℓ−LSE_base，汇总并画图。"
        ),
    )
    parser.add_argument(
        "--lse_layer_probe_max_base_lse",
        type=float,
        default=float("nan"),
        help="选中「base LSE ≤ 该值」的 token（与 --lse_layer_probe_bottom_quantile 二选一）。",
    )
    parser.add_argument(
        "--lse_layer_probe_bottom_quantile",
        type=float,
        default=None,
        help="在全集 response token 的 base LSE 上取该分位数为界，≤ 界者为「低」（如 0.2；将额外跑一遍 forward）。与 max_base 二选一。",
    )
    parser.add_argument(
        "--export_low_entropy_lse",
        action="store_true",
        help=(
            "在本轮全部 response token 上，用联合秩导出「同时偏低熵、偏低 LSE」的 token（含上下文与 top5）。"
            "联合分数 joint_score=(rank(熵)+rank(LSE))/2，保留 joint_score≤quantile(joint,q)。"
            "见 --export_low_joint_quantile（边际分位 AND 易为空，已改为联合秩）。"
        ),
    )
    parser.add_argument(
        "--export_low_joint_quantile",
        type=float,
        default=0.2,
        help="与 --export_low_entropy_lse 联用：q∈(0,1)，cutoff=np.quantile(joint_score, q)，约保留比例 q（有并列时略有偏差）。",
    )
    parser.add_argument(
        "--output_low_entropy_lse_jsonl",
        type=str,
        default=None,
        help="低熵+低 LSE 导出路径；默认 {output_plot stem}_low_entropy_low_lse.jsonl。",
    )
    args = parser.parse_args()

    if args.lse_threshold_report and not math.isfinite(args.log_sum_exp_threshold):
        raise ValueError("使用 --lse_threshold_report 时必须指定有限的 --log_sum_exp_threshold")

    if args.export_low_entropy_lse:
        qj = float(args.export_low_joint_quantile)
        if not (0.0 < qj < 1.0):
            raise ValueError("--export_low_joint_quantile 须在 (0, 1) 内")

    if args.lse_layer_probe:
        has_max = math.isfinite(args.lse_layer_probe_max_base_lse)
        has_q = (
            args.lse_layer_probe_bottom_quantile is not None
            and 0 < args.lse_layer_probe_bottom_quantile < 1
        )
        if not has_max and not has_q:
            raise ValueError(
                "--lse_layer_probe 需要指定 --lse_layer_probe_max_base_lse 或 --lse_layer_probe_bottom_quantile（0~1 之间）"
            )
        if has_max and has_q:
            raise ValueError("--lse_layer_probe：请勿同时指定 max_base_lse 与 bottom_quantile")

    if args.only_correct and args.all_samples:
        raise ValueError("--only_correct 与 --all_samples 不能同时设置")

    output_plot = Path(args.output_plot)
    output_jsonl = Path(args.output_jsonl)
    output_plot.parent.mkdir(parents=True, exist_ok=True)
    output_jsonl.parent.mkdir(parents=True, exist_ok=True)

    data, data_meta = resolve_input_rows(
        data_arg=args.data,
        dataset_name=args.dataset,
        dataset_info_arg=args.dataset_info,
        prompt_field_arg=args.prompt_field,
        pred_field_arg=args.pred_field,
        label_field_arg=args.label_field,
    )
    print(f"[load] mode={data_meta['mode']}, data={data_meta['data_file']}")
    if data_meta["mode"] == "dataset_info":
        print(
            f"[load] dataset_info={data_meta['dataset_info']}, dataset={data_meta['dataset']}, "
            f"fields={data_meta['fields']}"
        )
    else:
        print(f"[load] fields={data_meta['fields']}")
    if not data:
        print("输入数据为空，退出。")
        return

    total_count = 0
    correct_count = 0
    wrong_count = 0
    rows_selected: list[dict[str, Any]] = []
    for row in data:
        total_count += 1
        pred = str(row.get("predict", ""))
        label = str(row.get("label", ""))
        is_correct = answers_match(extract_answer(pred), extract_answer(label))
        if is_correct:
            correct_count += 1
        else:
            wrong_count += 1
        if args.all_samples:
            rows_selected.append(row)
        elif args.only_correct:
            if is_correct:
                rows_selected.append(row)
        else:
            if args.only_wrong and (not is_correct):
                rows_selected.append(row)
            elif (not args.only_wrong):
                rows_selected.append(row)
        if len(rows_selected) >= args.max_samples:
            break

    print(
        f"[filter] total={total_count}, correct={correct_count}, wrong={wrong_count}, "
        f"selected={len(rows_selected)}, mode="
        f"{'all' if args.all_samples else ('correct' if args.only_correct else 'wrong')}"
    )

    if not rows_selected:
        print("没有满足筛选条件的样本，退出。")
        return

    filter_mode = "all" if args.all_samples else ("correct" if args.only_correct else "wrong")

    print(f"[load] model/tokenizer: {args.model}")
    tokenizer = AutoTokenizer.from_pretrained(args.model, trust_remote_code=True)
    if tokenizer.pad_token_id is None:
        tokenizer.pad_token = tokenizer.eos_token
    lf_template = None
    if not args.disable_lf_template:
        try:
            from llamafactory.data import get_template_and_fix_tokenizer

            data_args = SimpleNamespace(
                template=args.template,
                train_on_prompt=False,
                tool_format=None,
                default_system=None,
                enable_thinking=True,
            )
            lf_template = get_template_and_fix_tokenizer(tokenizer, data_args)
            print(
                f"[template] enabled: {args.template if args.template else 'auto(tokenizer/default)'} | "
                f"class={type(lf_template).__name__}"
            )
        except Exception as e:
            print(f"[template] failed to init LlamaFactory template, fallback to raw concat. reason: {e}")
            lf_template = None
    else:
        print("[template] disabled, using raw `prompt + predict`.")
    model = AutoModelForCausalLM.from_pretrained(
        args.model,
        torch_dtype=torch.float16,
        device_map="auto",
        trust_remote_code=True,
        low_cpu_mem_usage=True,
    ).eval()

    tied = bool(model.lm_head.weight is model.model.embed_tokens.weight)
    print(f"[check] tied embedding: {tied}")
    final_norm = _resolve_final_norm(model)
    if args.lse_layer_probe:
        if final_norm is None:
            raise RuntimeError(
                "未找到 final LayerNorm/RMSNorm（model.model.norm 等），无法进行 --lse_layer_probe；请换支持的 CausalLM 结构。"
            )
        print(f"[check] final norm for fair LSE probe: {type(final_norm).__name__}")
    embed_matrix = model.model.embed_tokens.weight
    embed_norm = F.normalize(embed_matrix.detach(), dim=1)
    embed_norm_t = embed_norm.transpose(0, 1).contiguous()

    def _print_and_return_case_ratios(
        case_counts_high_in: dict[str, int],
        total_positions_in: int,
        high_entropy_count_in: int,
        variant_tag: str,
    ) -> dict[str, Any]:
        print("=" * 50)
        print(f"【高熵位置类型分析】({variant_tag})")
        print("=" * 50)
        print(f"总 token 位置数：{total_positions_in:>12d}")
        frac = (100.0 * high_entropy_count_in / total_positions_in) if total_positions_in > 0 else 0.0
        print(f"高熵位置数：{high_entropy_count_in:>14d}  ({frac:.1f}%)")

        denom_high = max(high_entropy_count_in, 1)
        denom_total = max(total_positions_in, 1)

        # 控制台“表格”展示 + 同时用于落盘
        print("\n高熵位置中的类型分布：")
        print("| 类别 | 高熵数量 | 占高熵 | 占总数 |")
        print("|---|---:|---:|---:|")

        case_ratios_high: dict[str, float] = {}
        case_ratios_total: dict[str, float] = {}
        for c in [CASE_A, CASE_B, CASE_AMBIGUOUS]:
            cnt = int(case_counts_high_in.get(c, 0))
            ratio_high = cnt / denom_high
            ratio_total = cnt / denom_total
            case_ratios_high[c] = ratio_high
            case_ratios_total[c] = ratio_total
            print(
                f"| {c} | {cnt:>10d} | {ratio_high * 100:>7.1f}% | {ratio_total * 100:>7.1f}% |"
            )

        return {
            "variant": variant_tag,
            "total_positions": int(total_positions_in),
            "high_entropy_count": int(high_entropy_count_in),
            "high_entropy_frac_total": (high_entropy_count_in / total_positions_in) if total_positions_in > 0 else 0.0,
            "case_counts_high": {c: int(case_counts_high_in.get(c, 0)) for c in [CASE_A, CASE_B, CASE_AMBIGUOUS]},
            "case_ratios_high": case_ratios_high,
            "case_ratios_total": case_ratios_total,
        }

    def run_for_rows(
        rows_selected_local: list[dict[str, Any]],
        variant_tag: str,
        output_plot_path: Path,
        output_jsonl_path: Path,
        print_examples: bool = True,
        filter_mode: str = "wrong",
    ) -> dict[str, Any]:
        lse_thr_rl: list[int] = []
        lse_thr_below: list[int] = []
        lse_thr_frac: list[float] = []
        lse_thr_min_lse: list[float] = []
        lse_thr_pos_ratio: list[float] = []
        lse_thr_out_path: Path | None = None
        lse_thr_agg: dict[str, Any] | None = None
        if args.lse_threshold_report:
            if args.output_lse_threshold_jsonl:
                lse_thr_out_path = Path(args.output_lse_threshold_jsonl)
            else:
                lse_thr_out_path = output_plot_path.parent / f"{output_plot_path.stem}_lse_threshold_report.json"

        layer_probe_pass1: list[tuple[int, int, float]] = []
        layer_delta_by_layer: list[list[float]] = []
        layer_lse_by_layer: list[list[float]] = []
        layer_probe_summary: dict[str, Any] | None = None
        use_lse_layer_q = (
            args.lse_layer_probe
            and args.lse_layer_probe_bottom_quantile is not None
            and 0 < float(args.lse_layer_probe_bottom_quantile) < 1
        )

        all_entropies_local: list[float] = []
        all_max_sims_local: list[float] = []
        all_lse_local: list[float] = []
        lse_case_a_local: list[float] = []
        # CASE_NORMAL 目前只用于统计，不参与高熵 token 的聚类输出
        case_counts_global_local = {CASE_NORMAL: 0, CASE_A: 0, CASE_B: 0, CASE_AMBIGUOUS: 0}
        case_counts_high_local = {CASE_A: 0, CASE_B: 0, CASE_AMBIGUOUS: 0}
        all_high_rows_local: list[dict[str, Any]] = []
        per_token_json_rows_local: list[dict[str, Any]] = []
        low_entropy_lse_buffer: list[dict[str, Any]] = []
        low_entropy_lse_rows: list[dict[str, Any]] = []
        low_ent_lse_meta: dict[str, Any] = {}
        low_ent_lse_out_path: Path | None = None
        if args.export_low_entropy_lse:
            if args.output_low_entropy_lse_jsonl:
                low_ent_lse_out_path = Path(args.output_low_entropy_lse_jsonl)
            else:
                low_ent_lse_out_path = output_plot_path.parent / f"{output_plot_path.stem}_low_entropy_low_lse.jsonl"

        total_positions_local = 0
        high_entropy_count_local = 0

        for st in tqdm(range(0, len(rows_selected_local), args.batch_size), desc=f"Analyzing({variant_tag})"):
            batch_rows = rows_selected_local[st : st + args.batch_size]
            batch_full_ids: list[list[int]] = []
            prompts: list[str] = []
            prompt_lens: list[int] = []
            for r in batch_rows:
                full_ids_i, prompt_len_i, prompt_rendered_i = _build_model_inputs_for_row(
                    row=r,
                    tokenizer=tokenizer,
                    lf_template=lf_template,
                )
                batch_full_ids.append(full_ids_i)
                prompt_lens.append(int(prompt_len_i))
                prompts.append(prompt_rendered_i)
            predicts = [str(r.get("predict", "")) for r in batch_rows]
            labels = [str(r.get("label", "")) for r in batch_rows]

            input_ids, attention_mask = _build_batch_inputs_from_ids(
                token_id_seqs=batch_full_ids,
                pad_token_id=int(tokenizer.pad_token_id),
                device=model.lm_head.weight.device,
            )

            with torch.no_grad():
                out = model(input_ids=input_ids, attention_mask=attention_mask, output_hidden_states=True)
                logits = out.logits  # [B, L, V]
                hidden = out.hidden_states[-1]  # [B, L, D]

            bsz, _, vocab_size = logits.shape
            _ = vocab_size

            batch_sample_buffers: list[dict[str, Any]] = []
            batch_topk_candidates: list[tuple[float, int, int]] = []  # (entropy, sample_i, token_t)

            for i in range(bsz):
                attn_len = int(attention_mask[i].sum().item())
                prompt_len = int(prompt_lens[i])
                pred_answer = extract_answer(predicts[i])
                gold_answer = extract_answer(labels[i])
                is_correct = answers_match(pred_answer, gold_answer)

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
                lse_vec = torch.logsumexp(logits_i_fp32, dim=-1).float().cpu().numpy()
                if args.lse_threshold_report:
                    _accumulate_lse_threshold_one_response(
                        lse_vec,
                        npos,
                        float(args.log_sum_exp_threshold),
                        lse_thr_rl,
                        lse_thr_below,
                        lse_thr_frac,
                        lse_thr_min_lse,
                        lse_thr_pos_ratio,
                    )
                if args.lse_layer_probe and final_norm is not None:
                    if use_lse_layer_q:
                        for t in range(npos):
                            layer_probe_pass1.append((st + i, t, float(lse_vec[t])))
                    elif math.isfinite(args.lse_layer_probe_max_base_lse):
                        mask_probe = lse_vec <= float(args.lse_layer_probe_max_base_lse)
                        if np.any(mask_probe):
                            _layer_probe_extend_deltas(
                                out,
                                i,
                                pred_start,
                                pred_end,
                                npos,
                                lse_vec,
                                mask_probe,
                                final_norm,
                                model.lm_head,
                                layer_delta_by_layer,
                                layer_lse_by_layer,
                            )
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
                    total_positions_local += 1
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
                    lse_t = float(lse_vec[t])
                    if np.isfinite(lse_t):
                        all_lse_local.append(lse_t)
                        if case == CASE_A:
                            lse_case_a_local.append(lse_t)
                    sample_case_counts[case] += 1
                    case_counts_global_local[case] += 1
                    if np.isfinite(e):
                        all_entropies_local.append(e)
                    if np.isfinite(ms):
                        all_max_sims_local.append(ms)
                    token_metrics.append(
                        {
                            "t": t,
                            "entropy": e,
                            "top1_prob": p1,
                            "top5_mass": p5m,
                            "max_cosine_sim": ms,
                        }
                    )
                    if np.isfinite(e):
                        batch_topk_candidates.append((e, i, t))

                    if args.export_low_entropy_lse and low_ent_lse_out_path is not None:
                        if np.isfinite(e) and np.isfinite(lse_t):
                            tid_list = target_ids.tolist()
                            tstrs = tokenizer.convert_ids_to_tokens(tid_list)
                            tok_id = int(tid_list[t])
                            top_ids_l = top5_ids[t].tolist()
                            top_p = top5_vals[t].float().tolist()
                            top_raw = tokenizer.convert_ids_to_tokens(top_ids_l)
                            top_txt = [tokenizer.decode([int(x)], skip_special_tokens=False) for x in top_ids_l]
                            low_entropy_lse_buffer.append(
                                {
                                    "sample_idx": st + i,
                                    "variant": variant_tag,
                                    "filter_mode": filter_mode,
                                    "is_correct": is_correct,
                                    "t": t,
                                    "entropy": e,
                                    "log_sum_exp": lse_t,
                                    "prompt_prefix": _build_prompt_prefix(
                                        tokenizer=tokenizer,
                                        prompt_text=prompts[i],
                                        token_strs=tstrs,
                                        t=t,
                                    ),
                                    "context": _build_token_context(
                                        tokenizer=tokenizer,
                                        token_ids=tid_list,
                                        t=t,
                                        window=6,
                                    ),
                                    "token_id": tok_id,
                                    "token": tokenizer.decode([tok_id], skip_special_tokens=False),
                                    "token_raw": tstrs[t],
                                    "top5_tokens": top_txt,
                                    "top5_tokens_raw": top_raw,
                                    "top5_probs": [float(x) for x in top_p],
                                    "top5_token_ids": [int(x) for x in top_ids_l],
                                }
                            )

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

            # 按 batch 在“当前筛选样本”上全局取 top-k 高熵 token
            k = max(0, int(args.high_entropy_topk))
            high_key_set: set[tuple[int, int]] = set()
            if k > 0 and batch_topk_candidates:
                if len(batch_topk_candidates) <= k:
                    high_key_set = {(si, tt) for _, si, tt in batch_topk_candidates}
                else:
                    ent_arr = np.asarray([x[0] for x in batch_topk_candidates], dtype=np.float64)
                    pick_idx = np.argpartition(ent_arr, -k)[-k:]
                    high_key_set = {
                        (batch_topk_candidates[int(j)][1], batch_topk_candidates[int(j)][2]) for j in pick_idx
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

                if (
                    target_ids is not None
                    and top5_ids is not None
                    and top5_vals is not None
                    and top5_cosine_sim is not None
                ):
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
                        per_token_json_rows_local.append(
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
                        high_entropy_count_local += 1
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
                        all_high_rows_local.append(row)
                        if case in case_counts_high_local:
                            case_counts_high_local[case] += 1

                _ = sample_high_positions
                _ = sample_high_count

            if torch.cuda.is_available():
                torch.cuda.empty_cache()

        if args.export_low_entropy_lse and low_ent_lse_out_path is not None:
            low_entropy_lse_rows, low_ent_lse_meta = _filter_low_entropy_lse_by_joint_quantile(
                low_entropy_lse_buffer,
                float(args.export_low_joint_quantile),
            )
            print(
                f"[export_low_entropy_lse] pool_tokens={low_ent_lse_meta.get('n_pool_tokens', 0)} | "
                f"joint_q={args.export_low_joint_quantile} → joint_cutoff={low_ent_lse_meta.get('joint_score_cutoff')} | "
                f"(参考) 边际 entropy_cut@{args.export_low_joint_quantile}={low_ent_lse_meta.get('marginal_entropy_cutoff_at_same_q')} | "
                f"lse_cut={low_ent_lse_meta.get('marginal_lse_cutoff_at_same_q')} | "
                f"exported={low_ent_lse_meta.get('n_exported', 0)}"
            )

        if args.lse_layer_probe and final_norm is not None and use_lse_layer_q and layer_probe_pass1:
            qv = float(args.lse_layer_probe_bottom_quantile)
            arr_p1 = np.array([x[2] for x in layer_probe_pass1], dtype=np.float64)
            thr_q = float(np.quantile(arr_p1, qv))
            qualify = {(a, b) for a, b, c in layer_probe_pass1 if c <= thr_q}
            for st in tqdm(
                range(0, len(rows_selected_local), args.batch_size),
                desc=f"lse_layer_probe_p2({variant_tag})",
            ):
                batch_rows = rows_selected_local[st : st + args.batch_size]
                batch_full_ids_p2: list[list[int]] = []
                prompt_lens_p2: list[int] = []
                for r in batch_rows:
                    full_ids_i, prompt_len_i, _ = _build_model_inputs_for_row(
                        row=r,
                        tokenizer=tokenizer,
                        lf_template=lf_template,
                    )
                    batch_full_ids_p2.append(full_ids_i)
                    prompt_lens_p2.append(int(prompt_len_i))
                input_ids, attention_mask = _build_batch_inputs_from_ids(
                    token_id_seqs=batch_full_ids_p2,
                    pad_token_id=int(tokenizer.pad_token_id),
                    device=model.lm_head.weight.device,
                )
                with torch.no_grad():
                    out_p2 = model(
                        input_ids=input_ids,
                        attention_mask=attention_mask,
                        output_hidden_states=True,
                    )
                logits_p2 = out_p2.logits
                bsz_p2 = logits_p2.shape[0]
                for ii in range(bsz_p2):
                    attn_len = int(attention_mask[ii].sum().item())
                    prompt_len = int(prompt_lens_p2[ii])
                    if attn_len <= 1 or prompt_len >= attn_len:
                        continue
                    pred_start = max(prompt_len - 1, 0)
                    pred_end = attn_len - 1
                    if pred_start >= pred_end:
                        continue
                    logits_i = logits_p2[ii, pred_start:pred_end, :]
                    target_ids = input_ids[ii, prompt_len:attn_len]
                    npos = int(min(logits_i.shape[0], target_ids.shape[0]))
                    if npos <= 0:
                        continue
                    logits_i = logits_i[:npos]
                    logits_i_fp32 = torch.nan_to_num(logits_i.float(), nan=0.0, posinf=1e4, neginf=-1e4)
                    lse_vec_p2 = torch.logsumexp(logits_i_fp32, dim=-1).float().cpu().numpy()
                    mask = np.array([(st + ii, t) in qualify for t in range(npos)], dtype=bool)
                    if np.any(mask):
                        _layer_probe_extend_deltas(
                            out_p2,
                            ii,
                            pred_start,
                            pred_end,
                            npos,
                            lse_vec_p2,
                            mask,
                            final_norm,
                            model.lm_head,
                            layer_delta_by_layer,
                            layer_lse_by_layer,
                        )
                if torch.cuda.is_available():
                    torch.cuda.empty_cache()

        if args.lse_layer_probe and final_norm is not None:
            if use_lse_layer_q and layer_probe_pass1:
                arr_p1 = np.array([x[2] for x in layer_probe_pass1], dtype=np.float64)
                thr_q = float(np.quantile(arr_p1, float(args.lse_layer_probe_bottom_quantile)))
                filter_payload_lp: dict[str, Any] = {
                    "mode": "bottom_quantile",
                    "quantile": float(args.lse_layer_probe_bottom_quantile),
                    "threshold_base_lse": thr_q,
                    "n_response_tokens_total": int(arr_p1.size),
                }
            else:
                filter_payload_lp = {
                    "mode": "max_base_lse",
                    "max_base_lse": float(args.lse_layer_probe_max_base_lse),
                }
            n_hs_tuple = len(layer_delta_by_layer) if layer_delta_by_layer else 0
            layer_probe_summary = _finalize_layer_probe_stats(
                layer_delta_by_layer,
                layer_lse_by_layer,
                variant_tag=variant_tag,
                filter_mode=filter_mode,
                filter_payload=filter_payload_lp,
                n_model_hidden_states=n_hs_tuple,
            )
            probe_json = output_plot_path.parent / f"{output_plot_path.stem}_lse_layer_probe.json"
            probe_png = output_plot_path.parent / f"{output_plot_path.stem}_lse_layer_probe.png"
            probe_json.write_text(json.dumps(layer_probe_summary, ensure_ascii=False, indent=2), encoding="utf-8")
            _plot_lse_layer_probe(layer_probe_summary, probe_png, title_suffix=variant_tag)
            print(f"\n[lse_layer_probe] JSON: {probe_json.resolve()}")
            print(f"[lse_layer_probe] PNG:  {probe_png.resolve()}")
            pl = layer_probe_summary.get("per_layer") or []
            if pl:
                last_mean = pl[-1].get("mean_delta_lse_minus_base")
                n0 = pl[0].get("n_positions", 0)
                print(f"[lse_layer_probe] 末层 mean Δ（应≈0）: {last_mean} | 每层采样位置数（以第 0 层计）: {n0}")

        all_ent_arr = np.asarray(all_entropies_local, dtype=np.float64)
        all_sim_arr = np.asarray(all_max_sims_local, dtype=np.float64)
        all_ent_arr = all_ent_arr[np.isfinite(all_ent_arr)]
        all_sim_arr = all_sim_arr[np.isfinite(all_sim_arr)]

        summary = _print_and_return_case_ratios(
            case_counts_high_in=case_counts_high_local,
            total_positions_in=total_positions_local,
            high_entropy_count_in=high_entropy_count_local,
            variant_tag=variant_tag,
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

        all_lse_arr = np.asarray(all_lse_local, dtype=np.float64)
        lse_a_arr = np.asarray(lse_case_a_local, dtype=np.float64)
        lse_plot_path = output_plot_path.parent / f"{output_plot_path.stem}_log_sum_exp_quantiles.png"
        lse_stats_path = output_plot_path.parent / f"{output_plot_path.stem}_log_sum_exp_stats.json"
        _plot_log_sum_exp_quantiles(all_lse_arr, lse_plot_path, title_suffix=variant_tag)
        lse_stats_payload = _save_log_sum_exp_stats(all_lse_arr, lse_a_arr, lse_stats_path)
        print("\n[log_sum_exp] 全位置经验分位数图:", lse_plot_path.resolve())
        print(f"[log_sum_exp] 统计 JSON: {lse_stats_path.resolve()}")
        mean_a = lse_stats_payload["mean_log_sum_exp_case_A"]
        if mean_a is not None:
            print(
                f"[log_sum_exp] CASE_A（A_multipeak）log_sum_exp 均值: {mean_a:.6f} "
                f"(n={lse_stats_payload['n_positions_case_A']})"
            )
        else:
            print("[log_sum_exp] CASE_A（A_multipeak）无位置，均值不可用。")

        if args.lse_threshold_report and lse_thr_out_path is not None:
            lse_thr_agg = _build_lse_threshold_aggregate(
                threshold=float(args.log_sum_exp_threshold),
                variant_tag=variant_tag,
                filter_mode=filter_mode,
                n_samples_total=len(rows_selected_local),
                response_lengths=lse_thr_rl,
                below_counts=lse_thr_below,
                fracs_below=lse_thr_frac,
                min_lses=lse_thr_min_lse,
                min_pos_ratios=lse_thr_pos_ratio,
            )
            _save_lse_threshold_summary(lse_thr_agg, lse_thr_out_path)
            print(f"\n[lse_threshold_report] 汇总已保存: {lse_thr_out_path.resolve()}")
            print(
                f"[lse_threshold_report] threshold={float(args.log_sum_exp_threshold):.6g} | "
                f"filter_mode={filter_mode} | variant={variant_tag}"
            )
            ns = int(lse_thr_agg.get("n_samples_with_response") or 0)
            if ns > 0:
                print(
                    f"[lse_threshold_report] mean_response_length={lse_thr_agg['mean_response_length']:.6f} | "
                    f"mean_frac_below={lse_thr_agg['mean_frac_positions_lse_below_threshold']:.6f} | "
                    f"global_frac_below={lse_thr_agg['global_frac_positions_lse_below_threshold']!s} | "
                    f"mean_min_lse_pos_ratio={lse_thr_agg['mean_min_lse_position_ratio_in_response']:.6f} | "
                    f"mean_min_log_sum_exp={lse_thr_agg['mean_min_log_sum_exp']:.6f}"
                )
            else:
                print("[lse_threshold_report] 无有效 response 片段，汇总中各均值为空。")

        if print_examples:
            examples = _collect_examples(all_high_rows_local, per_case=2)
            for c in [CASE_A, CASE_B, CASE_AMBIGUOUS]:
                for ex in examples[c]:
                    print(
                        f"\n[{c}] t={ex['t']}, entropy={ex['entropy']:.2f}, "
                        f"top5_mass={ex['top5_mass']:.2f}, max_sim={ex['max_cosine_sim']:.2f}"
                    )
                    print(f"  top-1 token : {repr(ex['token'])}")
                    print(f"  top-5 tokens: {[t for t in ex['top5_tokens']]}")
                    print(f"  top-5 probs : {[round(float(x), 4) for x in ex['top5_probs']]}")

        _plot_results(all_high_rows_local, output_plot_path)
        print(f"\n[plot] saved: {output_plot_path.resolve()}")

        with output_jsonl_path.open("w", encoding="utf-8") as f:
            for row in per_token_json_rows_local:
                f.write(json.dumps(row, ensure_ascii=False) + "\n")
        print(f"[jsonl] saved: {output_jsonl_path.resolve()}")

        if args.export_low_entropy_lse and low_ent_lse_out_path is not None:
            low_ent_lse_out_path.parent.mkdir(parents=True, exist_ok=True)
            with low_ent_lse_out_path.open("w", encoding="utf-8") as f:
                for row in low_entropy_lse_rows:
                    f.write(json.dumps(row, ensure_ascii=False) + "\n")
            print(
                f"[export_low_entropy_lse] 写入 {len(low_entropy_lse_rows)} 条 → {low_ent_lse_out_path.resolve()}"
            )

        out_ret: dict[str, Any] = {**summary, "log_sum_exp": lse_stats_payload}
        if lse_thr_agg is not None:
            out_ret["lse_threshold_report"] = lse_thr_agg
        if layer_probe_summary is not None:
            out_ret["lse_layer_probe"] = layer_probe_summary
        if args.export_low_entropy_lse and low_ent_lse_out_path is not None:
            out_ret["export_low_entropy_lse"] = {
                "path": str(low_ent_lse_out_path.resolve()),
                "n_rows": len(low_entropy_lse_rows),
                "meta": low_ent_lse_meta,
            }
        return out_ret

    # 1) 原始输出分析
    summary_original = run_for_rows(
        rows_selected_local=rows_selected,
        variant_tag="original",
        output_plot_path=output_plot,
        output_jsonl_path=output_jsonl,
        print_examples=True,
        filter_mode=filter_mode,
    )

    # 2) 如果预测输出里包含 <add_think>，再分析一次“删除后的版本”
    need_add_think_clean = any("<add_think>" in str(r.get("predict", "")) for r in rows_selected)
    summaries_by_variant: dict[str, dict[str, Any]] = {"original": summary_original}

    if need_add_think_clean:
        output_plot_no_think = output_plot.parent / f"{output_plot.stem}_no_add_think{output_plot.suffix}"
        output_jsonl_no_think = output_jsonl.parent / f"{output_jsonl.stem}_no_add_think{output_jsonl.suffix}"

        rows_selected_no_think: list[dict[str, Any]] = []
        for r in rows_selected:
            rr = dict(r)
            rr["predict"] = str(r.get("predict", "")).replace("<add_think>", "")
            rows_selected_no_think.append(rr)

        summaries_by_variant["no_add_think"] = run_for_rows(
            rows_selected_local=rows_selected_no_think,
            variant_tag="no_add_think",
            output_plot_path=output_plot_no_think,
            output_jsonl_path=output_jsonl_no_think,
            print_examples=False,
            filter_mode=filter_mode,
        )

    # 3) 汇总保存三类比例表（CSV/JSON）
    summary_json_path = output_jsonl.parent / f"{output_jsonl.stem}_case_ratios.json"
    summary_csv_path = output_jsonl.parent / f"{output_jsonl.stem}_case_ratios.csv"

    csv_rows: list[dict[str, Any]] = []
    for variant_tag, s in summaries_by_variant.items():
        for c in [CASE_A, CASE_B, CASE_AMBIGUOUS]:
            cnt_high = int(s["case_counts_high"][c])
            ratio_high = float(s["case_ratios_high"][c])
            ratio_total = float(s["case_ratios_total"][c])
            csv_rows.append(
                {
                    "variant": variant_tag,
                    "case": c,
                    "count_high": cnt_high,
                    "ratio_high": ratio_high,
                    "ratio_total": ratio_total,
                    "high_entropy_count": int(s["high_entropy_count"]),
                    "total_positions": int(s["total_positions"]),
                }
            )

    summary_payload = {
        "variants": summaries_by_variant,
    }
    summary_json_path.write_text(json.dumps(summary_payload, ensure_ascii=False, indent=2), encoding="utf-8")

    with summary_csv_path.open("w", encoding="utf-8", newline="") as f:
        writer = csv.DictWriter(
            f,
            fieldnames=[
                "variant",
                "case",
                "count_high",
                "ratio_high",
                "ratio_total",
                "high_entropy_count",
                "total_positions",
            ],
        )
        writer.writeheader()
        for row in csv_rows:
            writer.writerow(row)

    print(f"\n[ratio table] saved csv: {summary_csv_path.resolve()}")
    print(f"[ratio table] saved json: {summary_json_path.resolve()}")


if __name__ == "__main__":
    main()
