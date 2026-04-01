import math
import re
from typing import Optional, Sequence

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


def evaluate_gsm8k(
    model,
    tokenizer,
    samples: Sequence[dict],
    pause_token_id: int,
    max_new_tokens: int = 256,
    device: str = "cuda",
):
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

    accuracy = correct / max(n, 1)
    avg_pause = total_pause / max(n, 1)
    return {
        "accuracy": accuracy,
        "avg_pause_count": avg_pause,
        "pause_nonzero_rate": pause_nonzero / max(n, 1),
        "num_samples": n,
    }
