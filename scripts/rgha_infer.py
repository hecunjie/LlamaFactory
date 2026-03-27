#!/usr/bin/env python3
from __future__ import annotations

import argparse
import json
from pathlib import Path
from types import MethodType
from typing import Any

import torch
import torch.nn.functional as F
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
            if not line:
                continue
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
        label = tokenizer.decode(
            [int(x) for x in labels if int(x) != IGNORE_INDEX],
            skip_special_tokens=True,
        )
        rows.append({"prompt": prompt, "label": label})
    return rows


def patch_rgha_forward(
    model: torch.nn.Module,
    rgha_hidden_size: int = 256,
    rgha_entropy_alpha: float = 0.4,
    rgha_sim_beta: float = 0.6,
    rgha_threshold: float = 0.58,
) -> torch.nn.Module:
    unwrapped = model
    while hasattr(unwrapped, "module"):
        unwrapped = unwrapped.module

    hidden_size = int(getattr(unwrapped.config, "hidden_size", 0))
    if hidden_size <= 0:
        raise ValueError("Cannot resolve hidden_size from model config.")
    ref_param = next(unwrapped.parameters())
    module_device = ref_param.device
    module_dtype = ref_param.dtype

    if not hasattr(unwrapped, "rgha_ln"):
        unwrapped.rgha_ln = torch.nn.LayerNorm(hidden_size, device=module_device, dtype=module_dtype)
    if not hasattr(unwrapped, "rgha_mlp"):
        unwrapped.rgha_mlp = torch.nn.Sequential(
            torch.nn.Linear(hidden_size, rgha_hidden_size, device=module_device, dtype=module_dtype),
            torch.nn.SiLU(),
            torch.nn.Linear(rgha_hidden_size, hidden_size, device=module_device, dtype=module_dtype),
        )
    if not hasattr(unwrapped, "rgha_gate"):
        unwrapped.rgha_gate = torch.nn.Linear(2, 1, device=module_device, dtype=module_dtype)
    else:
        unwrapped.rgha_ln = unwrapped.rgha_ln.to(device=module_device, dtype=module_dtype)
        unwrapped.rgha_mlp = unwrapped.rgha_mlp.to(device=module_device, dtype=module_dtype)
        unwrapped.rgha_gate = unwrapped.rgha_gate.to(device=module_device, dtype=module_dtype)

    unwrapped.use_rgha = True
    unwrapped.rgha_entropy_alpha = float(rgha_entropy_alpha)
    unwrapped.rgha_sim_beta = float(rgha_sim_beta)
    unwrapped.rgha_threshold = float(rgha_threshold)
    unwrapped._rgha_original_forward = getattr(unwrapped, "_rgha_original_forward", unwrapped.forward)

    def _forward_with_rgha(self, *args, **kwargs):
        if not kwargs.get("output_hidden_states", False):
            kwargs["output_hidden_states"] = True

        outputs = self._rgha_original_forward(*args, **kwargs)
        if getattr(outputs, "hidden_states", None) is None or getattr(outputs, "logits", None) is None:
            return outputs

        hidden_states = outputs.hidden_states[-1]
        logits = outputs.logits
        input_embeds = self.get_input_embeddings()
        output_embeds = self.get_output_embeddings()
        if input_embeds is None or output_embeds is None:
            return outputs

        logits_fp32 = torch.nan_to_num(logits.float(), nan=0.0, posinf=1e4, neginf=-1e4)
        probs = torch.softmax(logits_fp32, dim=-1)
        entropy = -(probs * torch.log(probs.clamp_min(1e-12))).sum(dim=-1)
        vocab_size = max(int(logits.size(-1)), 2)
        norm_entropy = (entropy / torch.log(torch.tensor(float(vocab_size), device=entropy.device))).clamp(0.0, 1.0)

        embed_norm_t = F.normalize(input_embeds.weight.float(), dim=-1, eps=1e-12).transpose(0, 1).contiguous()
        h_norm = F.normalize(hidden_states.float(), dim=-1, eps=1e-12).to(embed_norm_t.dtype)
        max_sim = torch.matmul(h_norm, embed_norm_t).max(dim=-1).values.float()
        one_minus_max_sim = (1.0 - max_sim).clamp(0.0, 2.0)
        risk = self.rgha_entropy_alpha * norm_entropy + self.rgha_sim_beta * one_minus_max_sim
        risk_mask = risk > self.rgha_threshold

        feat = torch.stack((norm_entropy, one_minus_max_sim), dim=-1).to(hidden_states.dtype)
        gate = torch.sigmoid(self.rgha_gate(feat)).squeeze(-1) * risk_mask.to(hidden_states.dtype)
        delta = self.rgha_mlp(self.rgha_ln(hidden_states))
        refined_hidden = hidden_states + gate.unsqueeze(-1) * delta
        outputs.logits = output_embeds(refined_hidden)
        return outputs

    unwrapped.forward = MethodType(_forward_with_rgha, unwrapped)
    return model


def main() -> None:
    parser = argparse.ArgumentParser(description="HF inference with RGHA forward patch.")
    parser.add_argument("--model_name_or_path", type=str, required=True, help="Checkpoint path")
    parser.add_argument("--data", type=str, default=None, help="Input JSONL with fields prompt/label")
    parser.add_argument("--dataset", type=str, default=None, help="LlamaFactory dataset name in dataset_info.json")
    parser.add_argument("--dataset_dir", type=str, default="data", help="Directory that contains dataset_info.json data files")
    parser.add_argument("--template", type=str, default="default", help="LlamaFactory template name")
    parser.add_argument("--cutoff_len", type=int, default=2048, help="Cutoff length for LlamaFactory dataset build")
    parser.add_argument("--default_system", type=str, default=None, help="Default system prompt for template build")
    parser.add_argument("--enable_thinking", action="store_true", default=True, help="Enable thinking mode when parsing dataset")
    parser.add_argument("--output", type=str, required=True, help="Output JSONL path")
    parser.add_argument("--batch_size", type=int, default=8)
    parser.add_argument("--max_samples", type=int, default=None)
    parser.add_argument("--max_new_tokens", type=int, default=256)
    parser.add_argument("--temperature", type=float, default=0.0)
    parser.add_argument("--top_p", type=float, default=1.0)
    parser.add_argument("--do_sample", action="store_true", default=False)
    parser.add_argument("--num_generations", type=int, default=1, help="Number of generations per prompt")
    parser.add_argument(
        "--device",
        type=str,
        default="auto",
        help="Device for inference: auto/cuda/cuda:0/cpu. `auto` will use multi-GPU device_map when available.",
    )

    parser.add_argument("--use_rgha", action="store_true", default=False)
    parser.add_argument("--rgha_hidden_size", type=int, default=256)
    parser.add_argument("--rgha_entropy_alpha", type=float, default=0.4)
    parser.add_argument("--rgha_sim_beta", type=float, default=0.6)
    parser.add_argument("--rgha_threshold", type=float, default=0.58)
    args = parser.parse_args()

    output_path = Path(args.output)
    output_path.parent.mkdir(parents=True, exist_ok=True)

    if args.data is not None:
        rows = load_jsonl(Path(args.data))
        if args.max_samples is not None:
            rows = rows[: args.max_samples]
    elif args.dataset is not None:
        rows = build_rows_from_llamafactory(
            model_name_or_path=args.model_name_or_path,
            dataset=args.dataset,
            dataset_dir=args.dataset_dir,
            template=args.template,
            cutoff_len=args.cutoff_len,
            max_samples=args.max_samples,
            default_system=args.default_system,
            enable_thinking=args.enable_thinking,
        )
    else:
        raise ValueError("Provide either --data or --dataset.")

    if not rows:
        print("No data rows.")
        return
    if args.num_generations < 1:
        raise ValueError("`--num_generations` must be >= 1.")
    if args.num_generations > 1 and (not args.do_sample or args.temperature <= 0):
        raise ValueError("When `--num_generations > 1`, set `--do_sample` and `--temperature > 0`.")

    tokenizer = AutoTokenizer.from_pretrained(args.model_name_or_path, trust_remote_code=True)
    if tokenizer.pad_token_id is None:
        tokenizer.pad_token = tokenizer.eos_token
    # Decoder-only models should use left padding during generation.
    tokenizer.padding_side = "left"

    torch_dtype = torch.bfloat16 if torch.cuda.is_available() else torch.float32
    use_multi_gpu = args.device == "auto" and torch.cuda.is_available() and torch.cuda.device_count() > 1
    if use_multi_gpu:
        model = AutoModelForCausalLM.from_pretrained(
            args.model_name_or_path,
            torch_dtype=torch_dtype,
            trust_remote_code=True,
            low_cpu_mem_usage=True,
            device_map="auto",
        )
        # Pick one concrete CUDA device for input tensors (first CUDA shard).
        input_device = "cuda:0"
        hf_map = getattr(model, "hf_device_map", None)
        if isinstance(hf_map, dict):
            for dev in hf_map.values():
                if isinstance(dev, str) and dev.startswith("cuda"):
                    input_device = dev
                    break
    else:
        if args.device == "auto":
            target_device = "cuda:0" if torch.cuda.is_available() else "cpu"
        else:
            target_device = args.device
        model = AutoModelForCausalLM.from_pretrained(
            args.model_name_or_path,
            torch_dtype=torch_dtype,
            trust_remote_code=True,
            low_cpu_mem_usage=True,
        ).to(target_device)
        input_device = target_device
    model.eval()

    if args.use_rgha:
        model = patch_rgha_forward(
            model=model,
            rgha_hidden_size=args.rgha_hidden_size,
            rgha_entropy_alpha=args.rgha_entropy_alpha,
            rgha_sim_beta=args.rgha_sim_beta,
            rgha_threshold=args.rgha_threshold,
        )
        print("[RGHA] forward patch enabled.")
    else:
        print("[RGHA] disabled.")

    with output_path.open("w", encoding="utf-8") as fout:
        pbar = tqdm(range(0, len(rows), args.batch_size), desc="Infer", total=(len(rows) + args.batch_size - 1) // args.batch_size)
        for st in pbar:
            batch = rows[st : st + args.batch_size]
            prompts = [str(r.get("prompt", "")) for r in batch]
            labels = [str(r.get("label", "")) for r in batch]

            enc = tokenizer(
                prompts,
                return_tensors="pt",
                padding=True,
                truncation=False,
                add_special_tokens=False,
            ).to(input_device)

            with torch.no_grad():
                out_ids = model.generate(
                    **enc,
                    do_sample=args.do_sample,
                    temperature=args.temperature,
                    top_p=args.top_p,
                    max_new_tokens=args.max_new_tokens,
                    num_return_sequences=args.num_generations,
                    pad_token_id=tokenizer.pad_token_id,
                    eos_token_id=tokenizer.eos_token_id,
                )

            input_lens = enc["attention_mask"].sum(dim=1).tolist()
            bsz = len(batch)
            for i in range(bsz):
                pred_list: list[str] = []
                for k in range(args.num_generations):
                    out_idx = i * args.num_generations + k
                    ids = out_ids[out_idx]
                    gen_ids = ids[int(input_lens[i]) :].tolist()
                    pred_text = tokenizer.decode(gen_ids, skip_special_tokens=True)
                    pred_list.append(pred_text)
                rec = {
                    "prompt": prompts[i],
                    "predict": pred_list[0],
                    "predicts": pred_list,
                    "label": labels[i],
                }
                fout.write(json.dumps(rec, ensure_ascii=False) + "\n")
            processed = min(st + len(batch), len(rows))
            pbar.set_postfix_str(f"samples={processed}/{len(rows)}")

    print(f"Saved predictions to: {output_path}")


if __name__ == "__main__":
    main()

