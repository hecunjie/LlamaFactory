from typing import Dict

import torch
from transformers import LogitsProcessor


class PauseLimiter(LogitsProcessor):
    def __init__(self, pause_token_id: int, max_consecutive_pause: int = 3) -> None:
        self.pause_token_id = pause_token_id
        self.max_consecutive_pause = max_consecutive_pause

    def __call__(self, input_ids: torch.LongTensor, scores: torch.FloatTensor):
        batch_size = input_ids.size(0)
        for b in range(batch_size):
            consecutive = 0
            for tok in reversed(input_ids[b].tolist()):
                if tok == self.pause_token_id:
                    consecutive += 1
                else:
                    break
            if consecutive >= self.max_consecutive_pause:
                scores[b, self.pause_token_id] = -float("inf")
        return scores


def generate_with_pause(
    model,
    tokenizer,
    prompt: str,
    pause_token_id: int,
    max_new_tokens: int = 256,
    device: str = "cuda",
) -> Dict[str, object]:
    inputs = tokenizer(prompt, return_tensors="pt")
    inputs = {k: v.to(device) for k, v in inputs.items()}

    outputs = model.generate(
        **inputs,
        max_new_tokens=max_new_tokens,
        do_sample=False,
        logits_processor=[PauseLimiter(pause_token_id=pause_token_id, max_consecutive_pause=3)],
        pad_token_id=tokenizer.pad_token_id,
        eos_token_id=tokenizer.eos_token_id,
    )

    generated = outputs[0][inputs["input_ids"].shape[1] :].tolist()
    pause_count = sum(1 for t in generated if t == pause_token_id)
    filtered = [t for t in generated if t != pause_token_id]
    text = tokenizer.decode(filtered, skip_special_tokens=True)
    return {"text": text, "pause_count": pause_count}
