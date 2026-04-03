import math
import re
from typing import Optional, Sequence

import torch
import torch.distributed as dist

from inference import generate_with_pause


def _normalize_number(text: str) -> Optional[str]:
    text = text.strip().replace(",", "").replace("$", "").replace("%", "").rstrip(".")
    m = re.fullmatch(r"-?\d+(?:\.\d+)?", text)
    if m:
        try:
            val = float(text)
            if not math.isfinite(val):
                return None
            return str(int(val)) if val == int(val) else str(val)
        except (ValueError, OverflowError):
            pass
    m2 = re.match(r"^(-?\d+(?:\.\d+)?)", text)
    if m2:
        try:
            val = float(m2.group(1))
            if not math.isfinite(val):
                return None
            return str(int(val)) if val == int(val) else str(val)
        except (ValueError, OverflowError):
            pass
    return None


def extract_gsm8k_answer(text: str) -> Optional[str]:
    text = text.strip()
    matches = list(re.finditer(r"####\s*(.+?)(?:\n|$)", text))
    if matches:
        raw = matches[-1].group(1).strip()
        return _normalize_number(raw) or raw
    numbers = re.findall(r"-?\d[\d,]*(?:\.\d+)?", text)
    if numbers:
        raw = numbers[-1].replace(",", "")
        return _normalize_number(raw) or raw
    return None


def _match(pred: Optional[str], gold: Optional[str]) -> bool:
    if pred is None or gold is None:
        return False
    try:
        return float(pred) == float(gold)
    except (ValueError, TypeError):
        return pred.strip().lower() == gold.strip().lower()


def _eval_one_shard(
    model,
    tokenizer,
    samples: Sequence[dict],
    pause_token_id: int,
    max_new_tokens: int,
    device: str,
) -> tuple[int, int, int, int]:
    """Returns (correct_count, total_pause, pause_nonzero_count, n_samples)."""
    correct = 0
    total_pause = 0
    pause_nonzero = 0
    n = 0
    for sample in samples:
        prompt = sample["prompt"]
        label = sample["response"]
        pred = generate_with_pause(
            model=model,
            tokenizer=tokenizer,
            prompt=prompt,
            pause_token_id=pause_token_id,
            max_new_tokens=max_new_tokens,
            device=device,
        )
        pred_ans = extract_gsm8k_answer(pred["text"])
        gold_ans = extract_gsm8k_answer(label)
        if _match(pred_ans, gold_ans):
            correct += 1
        pause_count = int(pred["pause_count"])
        total_pause += pause_count
        if pause_count > 0:
            pause_nonzero += 1
        n += 1
    return correct, total_pause, pause_nonzero, n


def evaluate_gsm8k(
    model,
    tokenizer,
    samples: Sequence[dict],
    pause_token_id: int,
    max_new_tokens: int = 256,
    device: str = "cuda",
    distributed: bool = False,
):
    """
    If distributed=True and process group is initialized, each rank evaluates a
    strided shard of `samples`, then aggregates via all_reduce (NCCL-friendly).
    """
    if distributed and dist.is_available() and dist.is_initialized():
        rank = dist.get_rank()
        ws = dist.get_world_size()
        shard = list(samples[rank::ws])
        correct, total_pause, pause_nonzero, n_local = _eval_one_shard(
            model,
            tokenizer,
            shard,
            pause_token_id,
            max_new_tokens,
            device,
        )
        stats = torch.tensor(
            [correct, total_pause, pause_nonzero, n_local],
            dtype=torch.float64,
            device=device,
        )
        dist.all_reduce(stats, op=dist.ReduceOp.SUM)
        tot_n = int(stats[3].item())
        tot_n = max(tot_n, 1)
        return {
            "accuracy": float(stats[0].item() / tot_n),
            "avg_pause_count": float(stats[1].item() / tot_n),
            "pause_nonzero_rate": float(stats[2].item() / tot_n),
            "num_samples": tot_n,
        }

    correct, total_pause, pause_nonzero, n = _eval_one_shard(
        model,
        tokenizer,
        samples,
        pause_token_id,
        max_new_tokens,
        device,
    )
    tot_n = max(n, 1)
    return {
        "accuracy": correct / tot_n,
        "avg_pause_count": total_pause / tot_n,
        "pause_nonzero_rate": pause_nonzero / tot_n,
        "num_samples": n,
    }
