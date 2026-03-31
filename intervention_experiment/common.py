from __future__ import annotations

import json
import math
import os
import random
import re
import time
from dataclasses import dataclass
from pathlib import Path
from typing import Any

import torch
from datasets import load_dataset
from transformers import AutoModelForCausalLM, AutoTokenizer


ANSWER_RE = re.compile(r"####\s*(.+?)(?:\n|$)")
ANSWER_RE_3 = re.compile(r"(?<!#)###(?![#])\s*(.+?)(?:\n|$)")
ANSWER_IS_RE = re.compile(r"[Tt]he\s+answer\s+is\s+[:\-]?\s*([^\.\n,]+)")
FALLBACK_NUM_RE = re.compile(r"-?\d[\d,]*(?:\.\d+)?")


@dataclass
class ExperimentArgs:
    model_name_or_path: str
    output_path: str
    dataset_name: str = "openai/gsm8k"
    dataset_config: str = "main"
    split: str = "test"
    dataset_path: str | None = None
    lf_dataset_name: str | None = None
    dataset_info_path: str = "data/dataset_info.json"
    lf_split: str | None = None
    max_samples: int | None = None
    seed: int = 42
    max_new_tokens: int = 512
    top_p: float = 0.95
    base_temp: float = 1.0
    threshold: float | None = None
    local_temp: float | None = None
    device_map: str = "single"  # single|auto
    dtype: str = "bfloat16"  # bfloat16|float16|float32


def _safe_float(x: str) -> float | None:
    try:
        return float(x)
    except Exception:
        return None


def get_rank_info() -> tuple[int, int, int]:
    rank = int(os.environ.get("RANK", "0"))
    world_size = int(os.environ.get("WORLD_SIZE", "1"))
    local_rank = int(os.environ.get("LOCAL_RANK", str(rank)))
    return rank, world_size, local_rank


def set_seed(seed: int) -> None:
    random.seed(seed)
    torch.manual_seed(seed)
    if torch.cuda.is_available():
        torch.cuda.manual_seed_all(seed)


def _normalize_number(text: str) -> str | None:
    text = text.strip().replace(",", "").replace("$", "").replace("%", "").rstrip(".")
    m = re.fullmatch(r"-?\d+(?:\.\d+)?", text)
    if m:
        try:
            val = float(text)
            if not math.isfinite(val):
                return None
            return str(int(val)) if val == int(val) else str(val)
        except Exception:
            return None

    m2 = re.match(r"^(-?\d+(?:\.\d+)?)", text)
    if m2:
        try:
            val = float(m2.group(1))
            if not math.isfinite(val):
                return None
            return str(int(val)) if val == int(val) else str(val)
        except Exception:
            return None
    return None


def parse_answer(text: str) -> str | None:
    text = (text or "").strip()
    matches = list(ANSWER_RE.finditer(text))
    if matches:
        raw = matches[-1].group(1).strip()
        return _normalize_number(raw) or raw

    matches = list(ANSWER_RE_3.finditer(text))
    if matches:
        raw = matches[-1].group(1).strip()
        return _normalize_number(raw) or raw

    matches = list(ANSWER_IS_RE.finditer(text))
    if matches:
        raw = matches[-1].group(1).strip().rstrip(".")
        return _normalize_number(raw) or raw

    numbers = FALLBACK_NUM_RE.findall(text)
    if numbers:
        raw = numbers[-1].replace(",", "")
        return _normalize_number(raw) or raw
    return None


def is_correct(pred: str | None, gold: str | None) -> bool:
    if pred is None or gold is None:
        return False
    try:
        return float(pred) == float(gold)
    except Exception:
        return pred.strip().lower() == gold.strip().lower()


def build_prompt_ids(tokenizer, question: str) -> torch.Tensor:
    messages = [{"role": "user", "content": question}]
    try:
        ids = tokenizer.apply_chat_template(
            messages,
            tokenize=True,
            add_generation_prompt=True,
            return_tensors="pt",
        )
    except Exception:
        prompt = f"Question: {question}\nLet's think step by step.\n"
        ids = tokenizer(prompt, return_tensors="pt").input_ids
    return ids


def sample_top_p(logits: torch.Tensor, top_p: float, temperature: float) -> tuple[int, bool]:
    t = max(temperature, 1e-5)
    logits = logits / t
    probs = torch.softmax(logits, dim=-1)
    sorted_probs, sorted_ids = torch.sort(probs, descending=True)
    cumsum = torch.cumsum(sorted_probs, dim=-1)
    keep = cumsum <= top_p
    keep[..., 0] = True
    filtered = sorted_probs * keep
    denom = filtered.sum(dim=-1, keepdim=True).clamp_min(1e-12)
    filtered = filtered / denom
    sampled_idx = torch.multinomial(filtered, num_samples=1).item()
    token_id = sorted_ids[sampled_idx].item()
    top1_id = torch.argmax(logits).item()
    return token_id, token_id == top1_id


def _load_model_tokenizer(args: ExperimentArgs):
    rank, world_size, local_rank = get_rank_info()
    dtype_map = {
        "bfloat16": torch.bfloat16,
        "float16": torch.float16,
        "float32": torch.float32,
    }
    torch_dtype = dtype_map.get(args.dtype, torch.bfloat16)

    tokenizer = AutoTokenizer.from_pretrained(args.model_name_or_path, trust_remote_code=True)
    if tokenizer.pad_token_id is None:
        tokenizer.pad_token = tokenizer.eos_token

    use_auto_map = args.device_map == "auto" and world_size == 1 and torch.cuda.is_available()
    model = AutoModelForCausalLM.from_pretrained(
        args.model_name_or_path,
        torch_dtype=torch_dtype,
        trust_remote_code=True,
        device_map="auto" if use_auto_map else None,
        low_cpu_mem_usage=use_auto_map,
    ).eval()
    if not use_auto_map:
        if torch.cuda.is_available():
            device = torch.device(f"cuda:{local_rank}")
        else:
            device = torch.device("cpu")
        model = model.to(device)

    if rank == 0:
        print(
            f"[init] device_map={args.device_map}, world_size={world_size}, "
            f"cuda={torch.cuda.is_available()}"
        )
    return model, tokenizer


def _model_input_device(model) -> torch.device:
    try:
        return model.get_input_embeddings().weight.device
    except Exception:
        return next(model.parameters()).device


def _decode_tokens(tokenizer, token_ids: list[int]) -> str:
    return tokenizer.decode(token_ids, skip_special_tokens=True, clean_up_tokenization_spaces=True)


def _generate_one(
    model,
    tokenizer,
    prompt_ids: torch.Tensor,
    method: str,
    top_p: float,
    base_temp: float,
    threshold: float | None,
    local_temp: float | None,
    max_new_tokens: int,
) -> dict[str, Any]:
    device = _model_input_device(model)
    eos_id = tokenizer.eos_token_id
    emb_matrix = model.get_input_embeddings().weight if method == "soft_step" else None

    prompt_ids = prompt_ids.to(device)
    all_ids = prompt_ids.clone()
    cur_input = prompt_ids
    past_key_values = None

    generated: list[int] = []
    intervention_positions: list[int] = []
    intervention_lse_values: list[float] = []
    step_records: list[dict[str, Any]] = []

    with torch.no_grad():
        for step_idx in range(max_new_tokens):
            outputs = model(
                input_ids=cur_input,
                past_key_values=past_key_values,
                use_cache=True,
                return_dict=True,
            )
            logits = outputs.logits[0, -1, :]
            past_key_values = outputs.past_key_values
            lse = torch.logsumexp(logits, dim=-1).item()
            intervene = threshold is not None and lse < threshold and method != "baseline"
            final_logits = logits
            used_temp = base_temp

            if intervene and method == "local_temp":
                used_temp = local_temp if local_temp is not None else base_temp

            if intervene and method == "soft_step":
                probs = torch.softmax(logits / max(base_temp, 1e-5), dim=-1)
                soft_emb = probs @ emb_matrix  # (hidden,)
                current_embeds = model.get_input_embeddings()(all_ids)
                extended_embeds = torch.cat(
                    [current_embeds, soft_emb.unsqueeze(0).unsqueeze(0)], dim=1
                )
                outputs_extra = model(
                    inputs_embeds=extended_embeds,
                    use_cache=False,
                    return_dict=True,
                )
                final_logits = outputs_extra.logits[0, -1, :]

            next_token, picked_is_top1 = sample_top_p(
                final_logits,
                top_p=top_p,
                temperature=used_temp,
            )
            generated.append(next_token)
            if intervene:
                intervention_positions.append(int(all_ids.shape[1]))
                intervention_lse_values.append(float(lse))

            step_records.append(
                {
                    "position": int(all_ids.shape[1]),
                    "lse": float(lse),
                    "intervened": bool(intervene),
                    "picked_is_top1": bool(picked_is_top1),
                }
            )

            next_token_tensor = torch.tensor([[next_token]], device=device, dtype=all_ids.dtype)
            all_ids = torch.cat([all_ids, next_token_tensor], dim=1)
            cur_input = next_token_tensor
            if eos_id is not None and next_token == eos_id:
                break

    response = _decode_tokens(tokenizer, generated)
    return {
        "response": response,
        "generated_tokens": generated,
        "n_interventions": len(intervention_positions),
        "intervention_positions": intervention_positions,
        "intervention_lse_values": intervention_lse_values,
        "step_records": step_records,
    }


def _load_gsm8k(args: ExperimentArgs):
    if args.lf_dataset_name:
        return _load_from_lf_dataset_info(args)

    if args.dataset_path:
        path = Path(args.dataset_path)
        rows: list[dict[str, Any]] = []
        if path.suffix.lower() == ".jsonl":
            with path.open("r", encoding="utf-8") as f:
                for line in f:
                    line = line.strip()
                    if line:
                        obj = json.loads(line)
                        if isinstance(obj, dict):
                            rows.append(obj)
        else:
            obj = json.loads(path.read_text(encoding="utf-8"))
            if isinstance(obj, list):
                rows = [x for x in obj if isinstance(x, dict)]
            elif isinstance(obj, dict):
                rows = [obj]

        norm_rows = []
        for r in rows:
            q = r.get("question", r.get("instruction", r.get("prompt")))
            a = r.get("answer", r.get("output", r.get("label")))
            if q is None or a is None:
                continue
            norm_rows.append({"question": str(q), "answer": str(a)})
        if args.max_samples is not None:
            norm_rows = norm_rows[: args.max_samples]
        return norm_rows

    ds = load_dataset(args.dataset_name, args.dataset_config, split=args.split)
    if args.max_samples is not None:
        ds = ds.select(range(min(args.max_samples, len(ds))))
    return ds


def _load_from_lf_dataset_info(args: ExperimentArgs):
    info_path = Path(args.dataset_info_path)
    if not info_path.is_absolute():
        info_path = (Path.cwd() / info_path).resolve()
    if not info_path.exists():
        raise FileNotFoundError(f"dataset_info not found: {info_path}")

    info = json.loads(info_path.read_text(encoding="utf-8"))
    if args.lf_dataset_name not in info:
        raise ValueError(f"Dataset `{args.lf_dataset_name}` not found in {info_path}")

    spec = info[args.lf_dataset_name]
    columns = spec.get("columns", {})
    prompt_col = columns.get("prompt", "instruction")
    response_col = columns.get("response", "output")
    split = args.lf_split or spec.get("split", args.split)
    subset = spec.get("subset", None)

    # Supports the common "alpaca-like" datasets for this experiment.
    # sharegpt/multimodal configs are out of scope for numeric-answer eval.
    if spec.get("formatting", "alpaca") != "alpaca":
        raise ValueError(
            f"Dataset `{args.lf_dataset_name}` formatting={spec.get('formatting')} is not supported "
            "by intervention_experiment (requires prompt/response style)."
        )

    if "file_name" in spec:
        file_path = (info_path.parent / spec["file_name"]).resolve()
        if not file_path.exists():
            raise FileNotFoundError(f"Dataset file from dataset_info not found: {file_path}")
        rows = _load_local_rows(file_path)
    else:
        hub_id = spec.get("hf_hub_url") or spec.get("ms_hub_url") or spec.get("om_hub_url")
        if not hub_id:
            raise ValueError(
                f"Dataset `{args.lf_dataset_name}` has no file_name or hub url in dataset_info."
            )
        ds = load_dataset(hub_id, subset, split=split)
        rows = [dict(x) for x in ds]

    norm_rows = []
    for r in rows:
        q = r.get(prompt_col, r.get("question", r.get("instruction")))
        a = r.get(response_col, r.get("answer", r.get("output", r.get("label"))))
        if q is None or a is None:
            continue
        norm_rows.append({"question": str(q), "answer": str(a)})

    if args.max_samples is not None:
        norm_rows = norm_rows[: args.max_samples]
    print(
        f"[data] loaded {len(norm_rows)} samples from dataset_info "
        f"`{args.lf_dataset_name}` ({info_path})"
    )
    return norm_rows


def _load_local_rows(path: Path) -> list[dict[str, Any]]:
    if path.suffix.lower() == ".jsonl":
        rows: list[dict[str, Any]] = []
        with path.open("r", encoding="utf-8") as f:
            for line in f:
                line = line.strip()
                if not line:
                    continue
                obj = json.loads(line)
                if isinstance(obj, dict):
                    rows.append(obj)
        return rows

    obj = json.loads(path.read_text(encoding="utf-8"))
    if isinstance(obj, list):
        return [x for x in obj if isinstance(x, dict)]
    if isinstance(obj, dict):
        return [obj]
    return []


def _get_shard_indices(total: int, rank: int, world_size: int) -> list[int]:
    return [i for i in range(total) if i % world_size == rank]


def run_experiment(args: ExperimentArgs, method: str) -> str:
    assert method in {"baseline", "local_temp", "soft_step"}
    set_seed(args.seed)
    rank, world_size, _ = get_rank_info()

    model, tokenizer = _load_model_tokenizer(args)
    dataset = _load_gsm8k(args)
    shard_indices = _get_shard_indices(len(dataset), rank, world_size)

    out_path = Path(args.output_path)
    out_path.parent.mkdir(parents=True, exist_ok=True)
    part_path = out_path.with_suffix(out_path.suffix + f".rank{rank}")
    part_tmp_path = out_path.with_suffix(out_path.suffix + f".rank{rank}.tmp")

    started = time.time()
    with part_tmp_path.open("w", encoding="utf-8") as fout:
        for done_idx, sample_idx in enumerate(shard_indices, start=1):
            item = dataset[int(sample_idx)]
            question = str(item["question"])
            gold_answer_text = str(item["answer"])
            gold = parse_answer(gold_answer_text)

            prompt_ids = build_prompt_ids(tokenizer, question)
            gen = _generate_one(
                model=model,
                tokenizer=tokenizer,
                prompt_ids=prompt_ids,
                method=method,
                top_p=args.top_p,
                base_temp=args.base_temp,
                threshold=args.threshold,
                local_temp=args.local_temp,
                max_new_tokens=args.max_new_tokens,
            )
            pred = parse_answer(gen["response"])
            row = {
                "sample_idx": int(sample_idx),
                "question": question,
                "response": gen["response"],
                "predicted_answer": pred,
                "gold_answer": gold,
                "is_correct": is_correct(pred, gold),
                "n_interventions": gen["n_interventions"],
                "intervention_positions": gen["intervention_positions"],
                "intervention_lse_values": gen["intervention_lse_values"],
                "response_length": len(gen["generated_tokens"]),
                "step_records": gen["step_records"],
            }
            fout.write(json.dumps(row, ensure_ascii=False) + "\n")
            if done_idx % 20 == 0:
                elapsed = time.time() - started
                print(
                    f"[{method}][rank={rank}] {done_idx}/{len(shard_indices)} done "
                    f"(elapsed={elapsed:.1f}s)"
                )
    # Atomic finalize for distributed merge safety:
    # rank0 should only merge files that are fully written.
    part_tmp_path.replace(part_path)

    _merge_parts_when_ready(out_path, world_size)
    return str(out_path)


def _merge_parts_when_ready(out_path: Path, world_size: int, timeout_sec: int = 3600) -> None:
    rank, _, _ = get_rank_info()
    if world_size <= 1 or rank != 0:
        return

    part_paths = [out_path.with_suffix(out_path.suffix + f".rank{i}") for i in range(world_size)]
    t0 = time.time()
    while True:
        if all(p.exists() for p in part_paths):
            break
        if time.time() - t0 > timeout_sec:
            raise TimeoutError(f"Timeout waiting shard files: {part_paths}")
        time.sleep(1.0)

    rows = []
    for p in part_paths:
        with p.open("r", encoding="utf-8") as f:
            for line in f:
                line = line.strip()
                if line:
                    rows.append(json.loads(line))
    rows.sort(key=lambda x: int(x["sample_idx"]))

    with out_path.open("w", encoding="utf-8") as fout:
        for row in rows:
            fout.write(json.dumps(row, ensure_ascii=False) + "\n")
    for p in part_paths:
        p.unlink(missing_ok=True)
    print(f"[merge] merged {len(rows)} rows -> {out_path}")
