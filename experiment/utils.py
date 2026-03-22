"""Utility helpers for answer parsing, matching, and vector ops."""

from __future__ import annotations

import math
import re
from typing import Optional

import torch


def model_input_device(model: torch.nn.Module) -> torch.device:
    """
    Device for `input_ids` / `inputs_embeds` when using multi-GPU `device_map='auto'`.
    `model.device` is unreliable under sharding; tie inputs to the embedding module.
    """
    return model.get_input_embeddings().weight.device


def parse_answer(text: str) -> Optional[str]:
    """
    Extract answer string from:
      1) #### <ans>
      2) The answer is <ans>
      3) fallback: last number
    """
    if not isinstance(text, str):
        return None
    s = text.strip()

    m_all = list(re.finditer(r"####\s*(.+?)(?:\n|$)", s, flags=re.IGNORECASE))
    if m_all:
        ans = m_all[-1].group(1).strip()
        return _normalize_answer_token(ans)

    m_all = list(
        re.finditer(
            r"[Tt]he\s+answer\s+is\s*[:\-]?\s*([^\n]+?)(?:\.?\s*$|\n|,)",
            s,
            flags=re.IGNORECASE,
        )
    )
    if m_all:
        ans = m_all[-1].group(1).strip().rstrip(".")
        return _normalize_answer_token(ans)

    nums = re.findall(r"-?\d[\d,]*(?:\.\d+)?", s)
    if nums:
        return _normalize_answer_token(nums[-1])
    return None


def answers_equal(pred: Optional[str], gold: Optional[str]) -> bool:
    """Compare answers numerically (tol=1e-3) then string-wise."""
    if pred is None or gold is None:
        return False
    p = _normalize_answer_token(pred)
    g = _normalize_answer_token(gold)
    if p is None or g is None:
        return False
    try:
        return abs(float(p) - float(g)) <= 1e-3
    except (ValueError, OverflowError):
        return p.strip().lower() == g.strip().lower()


def find_pre_answer_position(input_ids: torch.Tensor, tokenizer) -> Optional[int]:
    """
    Find the position right before phrase "The answer is" appears.

    Returns index of the token before the phrase start.
    """
    if input_ids.ndim != 1:
        input_ids = input_ids.view(-1)

    seq_len = input_ids.shape[0]
    target = "the answer is"
    if seq_len < 2:
        return None

    # Sliding-window by decode; robust to tokenizer boundary variations.
    # Use the LAST matched occurrence so that system prompt mentions like
    # "put the final answer after The answer is" do not hijack the position.
    matched_pre_pos = None
    for start in range(seq_len):
        for end in range(start + 1, min(seq_len, start + 12) + 1):
            chunk = tokenizer.decode(input_ids[start:end], skip_special_tokens=False)
            norm = _normalize_ws(chunk.lower())
            if target in norm:
                pre_pos = start - 1
                if pre_pos >= 0:
                    matched_pre_pos = pre_pos
    return matched_pre_pos


def l2_norm(v: torch.Tensor) -> torch.Tensor:
    """L2 normalization with epsilon for numerical stability."""
    return v / (torch.norm(v, p=2) + 1e-8)


def compute_esoft(
    logits: torch.Tensor,
    embedding_matrix: torch.Tensor,
    top_k: int = 20,
) -> torch.Tensor:
    """
    Compute probability-weighted soft embedding vector from logits.
    """
    probs = torch.softmax(logits, dim=-1)
    k = min(top_k, probs.shape[-1])
    top_probs, top_ids = torch.topk(probs, k=k, dim=-1)
    top_probs = top_probs / (top_probs.sum() + 1e-12)
    top_embeds = embedding_matrix[top_ids]  # (k, hidden_dim)
    return (top_probs.unsqueeze(-1) * top_embeds).sum(dim=0)


def split_system_user_from_prompt(prompt: str) -> tuple[Optional[str], Optional[str], bool]:
    """Split prompt into system/user blocks by markers."""
    if not isinstance(prompt, str):
        return None, None, False
    m = re.match(
        r"^\s*system\s*\n(?P<system>.*?)\nuser\s*\n(?P<user>.*?)(?:\nassistant\b.*)?$",
        prompt.strip(),
        flags=re.IGNORECASE | re.DOTALL,
    )
    if not m:
        return None, None, False
    sys_s = m.group("system").strip()
    user_s = m.group("user").strip()
    if not sys_s or not user_s:
        return None, None, False
    return sys_s, user_s, True


def _normalize_answer_token(text: str) -> Optional[str]:
    """
    Normalize a single answer span to a canonical string for comparison.

    Handles GSM8K-style outputs like \"540 meters\", \"$42\", \"12,345 dollars\"
    by parsing the leading numeric literal when the whole string is not a float.
    """
    s = text.strip().replace(",", "").replace("$", "").replace("%", "").rstrip(".")
    if not s:
        return None
    try:
        v = float(s)
        if math.isfinite(v):
            if abs(v - int(v)) <= 1e-12:
                return str(int(v))
            return str(v)
    except (ValueError, OverflowError):
        pass
    # e.g. "540 meters", "42 apples" — compare numeric part to gold "540" / "42"
    m = re.match(r"^\s*(-?\d+(?:\.\d+)?)", s)
    if m:
        try:
            v = float(m.group(1))
            if math.isfinite(v):
                if abs(v - int(v)) <= 1e-12:
                    return str(int(v))
                return str(v)
        except (ValueError, OverflowError):
            pass
    return s


def _normalize_ws(text: str) -> str:
    return " ".join(text.split())
