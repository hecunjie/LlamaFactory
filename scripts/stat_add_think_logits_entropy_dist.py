#!/usr/bin/env python3
from __future__ import annotations

import argparse
import csv
import json
from pathlib import Path
from typing import Any

import matplotlib.pyplot as plt
import numpy as np
import torch
from tqdm import tqdm
from transformers import Seq2SeqTrainingArguments
from transformers import AutoModelForCausalLM, AutoTokenizer
from llamafactory.data import get_dataset, get_template_and_fix_tokenizer
from llamafactory.extras.constants import IGNORE_INDEX
from llamafactory.hparams import get_infer_args
from llamafactory.model import load_tokenizer


def load_jsonl(path: Path) -> list[dict[str, Any]]:
    rows: list[dict[str, Any]] = []
    with path.open("r", encoding="utf-8") as f:
        for line in f:
            line = line.strip()
            if line:
                rows.append(json.loads(line))
    return rows


def build_rows_from_llamafactory(
    model_name_or_path: str,
    dataset: str,
    dataset_dir: str,
    template: str,
    cutoff_len: int,
    max_samples: int | None,
    default_system: str | None,
    enable_thinking: bool,
) -> list[dict[str, str]]:
    model_args, data_args, _, _ = get_infer_args(
        dict(
            model_name_or_path=model_name_or_path,
            dataset=dataset,
            dataset_dir=dataset_dir,
            template=template,
            cutoff_len=cutoff_len,
            max_samples=max_samples,
            preprocessing_num_workers=16,
            default_system=default_system,
            enable_thinking=enable_thinking,
        )
    )
    training_args = Seq2SeqTrainingArguments(output_dir="dummy_dir")
    tokenizer_module = load_tokenizer(model_args)
    tokenizer = tokenizer_module["tokenizer"]
    template_obj = get_template_and_fix_tokenizer(tokenizer, data_args)
    dataset_module = get_dataset(template_obj, model_args, data_args, training_args, "ppo", **tokenizer_module)
    hf_ds = dataset_module["train_dataset"]

    rows: list[dict[str, str]] = []
    for i in range(len(hf_ds)):
        input_ids = hf_ds[i]["input_ids"]
        labels = hf_ds[i]["labels"]
        prompt = tokenizer.decode(input_ids, skip_special_tokens=True)
        label = tokenizer.decode([int(x) for x in labels if int(x) != IGNORE_INDEX], skip_special_tokens=True)
        rows.append({"prompt": prompt, "predict": label})
    return rows


def summarize(values: list[float]) -> dict[str, float | int]:
    if not values:
        return {
            "count": 0,
            "mean": float("nan"),
            "std": float("nan"),
            "min": float("nan"),
            "max": float("nan"),
            "p10": float("nan"),
            "p50": float("nan"),
            "p90": float("nan"),
        }
    arr = np.asarray(values, dtype=np.float64)
    arr = arr[np.isfinite(arr)]
    if arr.size == 0:
        return {
            "count": 0,
            "mean": float("nan"),
            "std": float("nan"),
            "min": float("nan"),
            "max": float("nan"),
            "p10": float("nan"),
            "p50": float("nan"),
            "p90": float("nan"),
        }
    return {
        "count": int(arr.size),
        "mean": float(arr.mean()),
        "std": float(arr.std(ddof=0)),
        "min": float(arr.min()),
        "max": float(arr.max()),
        "p10": float(np.quantile(arr, 0.10)),
        "p50": float(np.quantile(arr, 0.50)),
        "p90": float(np.quantile(arr, 0.90)),
    }


def plot_distributions(entropies: list[float], logits: list[float], output_path: Path, bins: int) -> None:
    ent_arr = np.asarray(entropies, dtype=np.float64)
    logit_arr = np.asarray(logits, dtype=np.float64)
    ent_arr = ent_arr[np.isfinite(ent_arr)]
    logit_arr = logit_arr[np.isfinite(logit_arr)]

    fig, axes = plt.subplots(1, 2, figsize=(14, 5))
    ax0, ax1 = axes

    if ent_arr.size > 0:
        ax0.hist(ent_arr, bins=bins, color="tab:blue", alpha=0.7)
        ax0.axvline(float(ent_arr.mean()), color="tab:blue", linestyle="--", linewidth=1.5, label="mean")
        ax0.legend()
    else:
        ax0.text(0.5, 0.5, "No valid entropy data", ha="center", va="center", transform=ax0.transAxes)
    ax0.set_title("Entropy Distribution")
    ax0.set_xlabel("Entropy")
    ax0.set_ylabel("Count")
    ax0.grid(True, alpha=0.3)

    if logit_arr.size > 0:
        ax1.hist(logit_arr, bins=bins, color="tab:orange", alpha=0.7)
        ax1.axvline(float(logit_arr.mean()), color="tab:orange", linestyle="--", linewidth=1.5, label="mean")
        ax1.legend()
    else:
        ax1.text(0.5, 0.5, "No valid logit data", ha="center", va="center", transform=ax1.transAxes)
    ax1.set_title("Target-Token Logit Distribution")
    ax1.set_xlabel("Logit")
    ax1.set_ylabel("Count")
    ax1.grid(True, alpha=0.3)

    fig.tight_layout()
    fig.savefig(output_path, dpi=160)
    plt.close(fig)


def forward_single(
    model: torch.nn.Module,
    input_ids_1d: torch.Tensor,
    device: torch.device,
) -> torch.Tensor:
    input_ids = input_ids_1d.unsqueeze(0).to(device)
    attention_mask = torch.ones_like(input_ids, device=device)
    with torch.no_grad():
        outputs = model(input_ids=input_ids, attention_mask=attention_mask)
    return outputs.logits[0].float().cpu()


def main() -> None:
    parser = argparse.ArgumentParser(description="统计 <add_think>（或其前一位置）的熵与logit分布。")
    parser.add_argument("--data", type=str, default=None, help="输入 JSONL，需包含 prompt/predict")
    parser.add_argument("--dataset", type=str, default=None, help="LlamaFactory dataset name（来自 dataset_info.json）")
    parser.add_argument("--dataset_dir", type=str, default="data", help="包含 dataset_info.json 的目录")
    parser.add_argument("--template", type=str, default="default", help="LlamaFactory 模板名")
    parser.add_argument("--cutoff_len", type=int, default=2048, help="LlamaFactory 数据构建 cutoff 长度")
    parser.add_argument("--default_system", type=str, default=None, help="模板构建时使用的默认 system prompt")
    parser.add_argument("--enable_thinking", action="store_true", default=True, help="构建数据时启用 thinking 模式")
    parser.add_argument("--model", type=str, required=True, help="本地模型路径（也用于 dataset 模式构建 tokenizer）")
    parser.add_argument("--output_dir", type=str, default="add_think_logits_entropy_dist_outputs", help="输出目录")
    parser.add_argument("--max_samples", type=int, default=2000, help="最多处理样本数")
    parser.add_argument("--hist_bins", type=int, default=60, help="直方图 bins 数")
    parser.add_argument(
        "--drop_think_position",
        action="store_true",
        help=(
            "启用后：对第 n 个 <add_think>，先删除前 n 个 <add_think> 再 forward；"
            "分析位置改为该 <add_think> 之前一个 token。"
        ),
    )
    args = parser.parse_args()

    if args.data is None and args.dataset is None:
        raise ValueError("请至少提供 --data 或 --dataset 之一。")
    if args.data is not None and args.dataset is not None:
        raise ValueError("--data 和 --dataset 不能同时设置，请二选一。")

    data_path_str = args.data if args.data is not None else f"dataset:{args.dataset}"
    out_dir = Path(args.output_dir)
    out_dir.mkdir(parents=True, exist_ok=True)
    out_json = out_dir / "add_think_logits_entropy_summary.json"
    out_csv = out_dir / "add_think_logits_entropy_rows.csv"
    out_plot = out_dir / "add_think_logits_entropy_hist.png"

    if args.data is not None:
        rows = load_jsonl(Path(args.data))
        rows = rows[: args.max_samples]
    else:
        rows = build_rows_from_llamafactory(
            model_name_or_path=args.model,
            dataset=str(args.dataset),
            dataset_dir=args.dataset_dir,
            template=args.template,
            cutoff_len=args.cutoff_len,
            max_samples=args.max_samples,
            default_system=args.default_system,
            enable_thinking=args.enable_thinking,
        )

    if not rows:
        print("输入数据为空，退出。")
        return

    tokenizer = AutoTokenizer.from_pretrained(args.model, trust_remote_code=True)
    if tokenizer.pad_token_id is None:
        tokenizer.pad_token = tokenizer.eos_token

    model = AutoModelForCausalLM.from_pretrained(
        args.model,
        torch_dtype=torch.float16 if torch.cuda.is_available() else torch.float32,
        device_map="auto" if torch.cuda.is_available() else None,
        trust_remote_code=True,
        low_cpu_mem_usage=True,
    ).eval()
    device = next(model.parameters()).device

    add_think_id = tokenizer.convert_tokens_to_ids("<add_think>")
    if isinstance(add_think_id, list):
        add_think_id = add_think_id[0] if add_think_id else -1
    if add_think_id is None or int(add_think_id) < 0:
        raise ValueError("tokenizer 中未找到 <add_think>。")
    add_think_id = int(add_think_id)

    entropy_values: list[float] = []
    target_logits: list[float] = []
    result_rows: list[dict[str, Any]] = []
    sample_with_think = 0

    for sample_idx, r in enumerate(tqdm(rows, desc="Analyzing")):
        prompt = str(r.get("prompt", ""))
        predict = str(r.get("predict", ""))
        full_text = prompt + predict

        tokenized = tokenizer(full_text, add_special_tokens=False)
        ids = tokenized["input_ids"]
        if len(ids) < 2:
            continue

        think_positions = [i for i, tid in enumerate(ids) if int(tid) == add_think_id]
        if not think_positions:
            continue
        sample_with_think += 1

        if not args.drop_think_position:
            logits = forward_single(model=model, input_ids_1d=torch.tensor(ids, dtype=torch.long), device=device)
            vocab = logits.size(-1)
            probs = torch.softmax(torch.nan_to_num(logits, nan=0.0, posinf=1e4, neginf=-1e4), dim=-1)
            ent = -(probs * torch.log(probs.clamp_min(1e-12))).sum(dim=-1).numpy()

            for nth, think_idx in enumerate(think_positions, start=1):
                if think_idx >= len(ids) - 1:
                    continue
                row_idx = think_idx
                target_id = int(ids[think_idx + 1])
                entropy_val = float(ent[row_idx])
                target_logit_val = float(logits[row_idx, target_id].item())

                entropy_values.append(entropy_val)
                target_logits.append(target_logit_val)
                result_rows.append(
                    {
                        "sample_idx": sample_idx,
                        "n_think": nth,
                        "drop_think_position": False,
                        "orig_think_pos": think_idx,
                        "analyze_row_idx": row_idx,
                        "target_token_id": target_id,
                        "target_token_text": tokenizer.decode([target_id], skip_special_tokens=False),
                        "entropy": entropy_val,
                        "target_logit": target_logit_val,
                        "vocab_size": int(vocab),
                    }
                )
        else:
            for k, think_idx in enumerate(think_positions):
                n_think = k + 1
                if think_idx <= 0:
                    continue

                # 删除前 n 个 <add_think>（含第 n 个），再在“原第 n 个 <add_think> 的前一个位置”做分析。
                drop_set = set(think_positions[: n_think])
                new_ids = [tid for idx, tid in enumerate(ids) if idx not in drop_set]
                if len(new_ids) < 2:
                    continue

                orig_prev_idx = think_idx - 1
                removed_before_prev = sum(1 for p in drop_set if p <= orig_prev_idx)
                new_row_idx = orig_prev_idx - removed_before_prev
                if new_row_idx < 0 or new_row_idx >= len(new_ids) - 1:
                    continue

                logits = forward_single(
                    model=model,
                    input_ids_1d=torch.tensor(new_ids, dtype=torch.long),
                    device=device,
                )
                vocab = logits.size(-1)
                probs = torch.softmax(torch.nan_to_num(logits, nan=0.0, posinf=1e4, neginf=-1e4), dim=-1)
                ent = -(probs * torch.log(probs.clamp_min(1e-12))).sum(dim=-1).numpy()

                target_id = int(new_ids[new_row_idx + 1])
                entropy_val = float(ent[new_row_idx])
                target_logit_val = float(logits[new_row_idx, target_id].item())

                entropy_values.append(entropy_val)
                target_logits.append(target_logit_val)
                result_rows.append(
                    {
                        "sample_idx": sample_idx,
                        "n_think": n_think,
                        "drop_think_position": True,
                        "orig_think_pos": think_idx,
                        "analyze_row_idx": int(new_row_idx),
                        "target_token_id": target_id,
                        "target_token_text": tokenizer.decode([target_id], skip_special_tokens=False),
                        "entropy": entropy_val,
                        "target_logit": target_logit_val,
                        "vocab_size": int(vocab),
                    }
                )

    summary = {
        "meta": {
            "data": data_path_str,
            "model": str(args.model),
            "max_samples": int(args.max_samples),
            "drop_think_position": bool(args.drop_think_position),
            "num_samples_loaded": int(len(rows)),
            "num_samples_with_add_think": int(sample_with_think),
            "num_points": int(len(result_rows)),
        },
        "entropy": summarize(entropy_values),
        "target_logit": summarize(target_logits),
    }

    out_json.write_text(json.dumps(summary, ensure_ascii=False, indent=2), encoding="utf-8")

    fieldnames = [
        "sample_idx",
        "n_think",
        "drop_think_position",
        "orig_think_pos",
        "analyze_row_idx",
        "target_token_id",
        "target_token_text",
        "entropy",
        "target_logit",
        "vocab_size",
    ]
    with out_csv.open("w", encoding="utf-8", newline="") as f:
        writer = csv.DictWriter(f, fieldnames=fieldnames)
        writer.writeheader()
        for row in result_rows:
            writer.writerow(row)

    plot_distributions(
        entropies=entropy_values,
        logits=target_logits,
        output_path=out_plot,
        bins=max(5, int(args.hist_bins)),
    )

    print(f"[done] summary json: {out_json.resolve()}")
    print(f"[done] per-point csv: {out_csv.resolve()}")
    print(f"[done] histogram: {out_plot.resolve()}")
    print(f"[stats] points={len(result_rows)}, samples_with_add_think={sample_with_think}")


if __name__ == "__main__":
    main()

