import torch
from transformers import AutoModelForCausalLM, AutoTokenizer


def load_model_and_tokenizer(model_name: str):
    tokenizer = AutoTokenizer.from_pretrained(model_name, use_fast=True)
    if tokenizer.pad_token is None:
        tokenizer.pad_token = tokenizer.eos_token

    if torch.cuda.is_available():
        # Prefer bf16 when available for better numerical stability.
        dtype = torch.bfloat16 if torch.cuda.is_bf16_supported() else torch.float16
    else:
        # fp16 on CPU is unstable/unsupported in many kernels and can produce NaNs.
        dtype = torch.float32

    model = AutoModelForCausalLM.from_pretrained(model_name, torch_dtype=dtype)
    return model, tokenizer
