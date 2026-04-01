#!/usr/bin/env python3
# Copyright 2025 — Phase 0: high-entropy hidden state extraction (LlamaFactory dataset_info + llama3 template)
from __future__ import annotations

import argparse
import json
import os

import numpy as np
import torch
import torch.nn.functional as F
from tqdm import tqdm
from transformers import AutoModelForCausalLM, AutoTokenizer

from llamafactory.data import get_template_and_fix_tokenizer
from llamafactory.hparams import DataArguments

from lf_dataset import load_dataset_rows


def _build_full_ids(
    row: dict,
    tokenizer,
    lf_template,
) -> tuple[list[int], int, str]:
    """Align with LlamaFactory SFT: user/assistant via template; returns full_ids and len(prompt_ids)."""
    prompt_text = str(row.get("prompt", ""))
    predict_text = str(row.get("predict", ""))
    already_formatted = (
        ("<|redacted_start_header_id|>" in prompt_text and "<|eot_id|>" in prompt_text)
        or ("<|im_start|>" in prompt_text and "<|redacted_im_end|>" in prompt_text)
        or ("[INST]" in prompt_text and "[/INST]" in prompt_text)
    )
    if lf_template is None or already_formatted:
        full_text = prompt_text + predict_text
        full_ids = tokenizer(full_text, add_special_tokens=False)["input_ids"]
        prompt_ids = tokenizer(prompt_text, add_special_tokens=False)["input_ids"]
        return full_ids, len(prompt_ids), prompt_text

    messages = [
        {"role": "user", "content": prompt_text},
        {"role": "assistant", "content": predict_text},
    ]
    system_text = row.get("system", None)
    tools_text = row.get("tools", None)
    system = str(system_text) if system_text is not None else None
    tools = str(tools_text) if tools_text is not None else None
    prompt_ids, response_ids = lf_template.encode_oneturn(
        tokenizer,
        messages,
        system=system,
        tools=tools,
    )
    full_ids = prompt_ids + response_ids
    prompt_rendered = tokenizer.decode(prompt_ids, skip_special_tokens=False)
    return full_ids, len(prompt_ids), prompt_rendered


def extract(
    model,
    tokenizer,
    lf_template,
    rows: list[dict],
    *,
    top_k_percent: float,
    max_samples: int,
    layer_idx: str,
    cutoff_len: int,
) -> tuple[np.ndarray, list[dict]]:
    all_hs: list[np.ndarray] = []
    all_meta: list[dict] = []

    n_layers = model.config.num_hidden_layers

    def _resolve_layer_tensor_idx(layer_arg: str) -> int:
        s = str(layer_arg).strip().lower()
        if s in {"-1", "last"}:
            return n_layers
        if s in {"middle", "mid"}:
            return max(1, n_layers // 2)
        try:
            user_idx = int(s)
        except ValueError as err:
            raise ValueError(f"Invalid layer_idx={layer_arg!r}, use -1/last/middle or integer.") from err
        return user_idx + 1

    model.eval()
    device = next(model.parameters()).device

    with torch.no_grad():
        for sample_id, row in enumerate(tqdm(rows[:max_samples], desc="samples")):
            full_ids, prompt_len, _ = _build_full_ids(row, tokenizer, lf_template)
            if len(full_ids) > cutoff_len:
                full_ids = full_ids[:cutoff_len]

            input_ids = torch.tensor([full_ids], dtype=torch.long, device=device)
            outputs = model(input_ids=input_ids, output_hidden_states=True)
            hidden_stack = outputs.hidden_states
            # hidden_stack[0] = embeddings; [1..num_hidden_layers] = each block output; [-1] == last layer
            layer_tensor_idx = _resolve_layer_tensor_idx(layer_idx)
            if layer_tensor_idx < 0 or layer_tensor_idx >= len(hidden_stack):
                raise ValueError(
                    f"layer_idx={layer_idx} invalid: need 0..{n_layers - 1}, -1/last, or middle, "
                    f"got hidden_states depth {len(hidden_stack)}."
                )
            h_layer = hidden_stack[layer_tensor_idx][0]

            logits = outputs.logits[0]
            log_probs = F.log_softmax(logits, dim=-1)
            seq_len = len(full_ids)

            token_neg_logprobs: list[float] = []
            for t in range(seq_len - 1):
                next_tok = full_ids[t + 1]
                token_neg_logprobs.append(float(-log_probs[t, next_tok].item()))
            token_neg_logprobs.append(0.0)

            # First logits index that predicts an assistant-content token (skip user / headers).
            assistant_t0 = max(0, prompt_len - 1)
            reason_slice = token_neg_logprobs[assistant_t0 : seq_len - 1]
            if len(reason_slice) == 0:
                continue
            thr = float(np.percentile(reason_slice, 100 - top_k_percent))

            for t in range(assistant_t0, seq_len - 1):
                if token_neg_logprobs[t] < thr:
                    continue
                hs = h_layer[t].detach().float().cpu().numpy()
                ctx_lo = max(0, t - 10)
                ctx_hi = min(seq_len, t + 11)
                ctx = tokenizer.decode(full_ids[ctx_lo:ctx_hi], skip_special_tokens=False)
                # 便于人工审查：在窗口中明确标出该高熵位置对应 token
                left_ctx = tokenizer.decode(full_ids[ctx_lo:t], skip_special_tokens=False)
                tok_str = tokenizer.decode([full_ids[t]], skip_special_tokens=False)
                right_ctx = tokenizer.decode(full_ids[t + 1 : ctx_hi], skip_special_tokens=False)
                ctx_marked = f"{left_ctx}[[{tok_str}]]{right_ctx}"
                all_hs.append(hs)
                all_meta.append(
                    {
                        "sample_id": sample_id,
                        "position": t,
                        "neg_logprob": token_neg_logprobs[t],
                        "token_id": full_ids[t],
                        "token_str": tok_str,
                        "context_window": ctx,
                        "context_window_marked": ctx_marked,
                    }
                )

    if not all_hs:
        return np.zeros((0, model.config.hidden_size), dtype=np.float32), []
    return np.stack(all_hs, axis=0), all_meta


def main() -> None:
    parser = argparse.ArgumentParser(description="Extract high-entropy hidden states (LlamaFactory dataset + llama3).")
    parser.add_argument("--model_name_or_path", type=str, required=True, help="HF model id or local path (e.g. Llama-3).")
    parser.add_argument("--dataset_dir", type=str, default="data", help="Folder containing dataset_info.json.")
    parser.add_argument("--dataset", type=str, required=True, help="Name in dataset_info.json (same as training --dataset).")
    parser.add_argument("--template", type=str, default="llama3", help="Chat template name (default: llama3).")
    parser.add_argument("--output_dir", type=str, default="outputs/hidden_states_run", help="Directory for .npy / .json.")
    parser.add_argument("--top_k_percent", type=float, default=10.0, help="Top fraction of neg_logprob within assistant span.")
    parser.add_argument("--max_samples", type=int, default=2000, help="Max dataset rows (use 200 for smoke test).")
    parser.add_argument(
        "--layer_idx",
        type=str,
        default="-1",
        help="Layer selector: -1/last (final layer), middle, or integer block index.",
    )
    parser.add_argument("--cutoff_len", type=int, default=4096, help="Truncate tokenized length.")
    parser.add_argument("--dtype", type=str, default="bfloat16", choices=["float16", "bfloat16", "float32"])
    args = parser.parse_args()

    dataset_dir = os.path.abspath(args.dataset_dir)
    rows, ds_attr = load_dataset_rows(dataset_dir, args.dataset)
    print(f"Loaded {len(rows)} rows from dataset={args.dataset!r} ({ds_attr})")

    tokenizer = AutoTokenizer.from_pretrained(args.model_name_or_path, trust_remote_code=True)
    dtype_map = {"float16": torch.float16, "bfloat16": torch.bfloat16, "float32": torch.float32}
    torch_dtype = dtype_map[args.dtype]
    model = AutoModelForCausalLM.from_pretrained(
        args.model_name_or_path,
        torch_dtype=torch_dtype,
        device_map="auto",
        trust_remote_code=True,
    )

    data_args = DataArguments(template=args.template, dataset_dir=dataset_dir)
    lf_template = get_template_and_fix_tokenizer(tokenizer, data_args)

    hs, meta = extract(
        model,
        tokenizer,
        lf_template,
        rows,
        top_k_percent=args.top_k_percent,
        max_samples=args.max_samples,
        layer_idx=args.layer_idx,
        cutoff_len=args.cutoff_len,
    )

    os.makedirs(args.output_dir, exist_ok=True)
    np_path = os.path.join(args.output_dir, "hidden_states.npy")
    meta_path = os.path.join(args.output_dir, "hidden_states_meta.json")
    np.save(np_path, hs)
    with open(meta_path, "w", encoding="utf-8") as f:
        json.dump(meta, f, ensure_ascii=False, indent=2)

    cfg_path = os.path.join(args.output_dir, "extract_config.json")
    with open(cfg_path, "w", encoding="utf-8") as f:
        json.dump(vars(args) | {"dataset_attr": str(ds_attr), "n_extracted": len(meta)}, f, ensure_ascii=False, indent=2)

    print(f"Saved {len(meta)} vectors -> {np_path}")
    print(f"Meta -> {meta_path}")


if __name__ == "__main__":
    main()
