#!/usr/bin/env python3
from __future__ import annotations

import argparse
import json
import os
import re
import shutil
import subprocess
import sys
import tempfile
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


def _legacy_prompt_to_messages(prompt: str, system: str | None) -> list[dict[str, str]]:
    """Turn export-style ``user\\n\\n...assistant`` text into chat ``messages`` (user content only)."""
    messages: list[dict[str, str]] = []
    if system and str(system).strip():
        messages.append({"role": "system", "content": str(system).strip()})
    p = prompt.strip()
    m = re.search(r"(?is)^\s*user\s*\n+(.*?)(?:\n\s*assistant\s*)$", p)
    if m:
        messages.append({"role": "user", "content": m.group(1).strip()})
    else:
        messages.append({"role": "user", "content": p})
    return messages


def _encode_batch_with_chat_template(
    tokenizer: AutoTokenizer,
    rows: list[dict[str, Any]],
    default_system: str | None,
    device: torch.device | str,
) -> tuple[dict[str, torch.Tensor], list[str]]:
    """Left-pad token ids from ``apply_chat_template(..., add_generation_prompt=True)``."""
    pad_id = tokenizer.pad_token_id
    if pad_id is None:
        pad_id = tokenizer.eos_token_id
    seqs: list[list[int]] = []
    display_prompts: list[str] = []
    for r in rows:
        if "messages" in r and r["messages"] is not None:
            messages = r["messages"]
            if not isinstance(messages, list):
                raise ValueError("Each row's 'messages' must be a list of {role, content} dicts.")
        else:
            sys = r.get("system", default_system)
            if isinstance(sys, str) and not sys.strip():
                sys = None
            messages = _legacy_prompt_to_messages(str(r.get("prompt", "")), sys)

        out = tokenizer.apply_chat_template(
            messages,
            tokenize=True,
            add_generation_prompt=True,
            return_tensors=None,
        )
        if isinstance(out, torch.Tensor):
            ids = out.squeeze(0).tolist()
        else:
            ids = list(out)
        seqs.append(ids)
        display_prompts.append(tokenizer.decode(ids, skip_special_tokens=True))

    max_len = max(len(s) for s in seqs)
    batch_n = len(seqs)
    input_ids = torch.full((batch_n, max_len), int(pad_id), dtype=torch.long)
    attention_mask = torch.zeros((batch_n, max_len), dtype=torch.long)
    for i, s in enumerate(seqs):
        L = len(s)
        input_ids[i, -L:] = torch.tensor(s, dtype=torch.long)
        attention_mask[i, -L:] = 1
    enc = {"input_ids": input_ids, "attention_mask": attention_mask}
    enc = {k: v.to(device) for k, v in enc.items()}
    return enc, display_prompts


def _unwrap_causal_lm(model: torch.nn.Module) -> torch.nn.Module:
    unwrapped = model
    while hasattr(unwrapped, "module"):
        unwrapped = unwrapped.module
    return unwrapped


def _resolve_checkpoint_dir(model_name_or_path: str) -> Path:
    p = Path(model_name_or_path).expanduser()
    if p.is_dir():
        return p.resolve()
    try:
        from huggingface_hub import snapshot_download

        return Path(snapshot_download(repo_id=model_name_or_path))
    except Exception:
        return p.resolve()


def _load_raw_state_dict(ckpt_dir: Path) -> dict[str, torch.Tensor]:
    """Load full checkpoint tensors from a HF-style model directory (single or sharded safetensors)."""
    single = ckpt_dir / "model.safetensors"
    if single.is_file():
        from safetensors.torch import load_file

        return dict(load_file(str(single)))

    idx = ckpt_dir / "model.safetensors.index.json"
    if idx.is_file():
        from safetensors.torch import load_file

        with idx.open("r", encoding="utf-8") as f:
            weight_map: dict[str, str] = json.load(f)["weight_map"]
        merged: dict[str, torch.Tensor] = {}
        shard_files = sorted(set(weight_map.values()))
        for sf in shard_files:
            shard_path = ckpt_dir / sf
            if not shard_path.is_file():
                continue
            merged.update(load_file(str(shard_path)))
        return merged

    bin_path = ckpt_dir / "pytorch_model.bin"
    if bin_path.is_file():
        try:
            return torch.load(bin_path, map_location="cpu", weights_only=True)
        except TypeError:
            return torch.load(bin_path, map_location="cpu")

    return {}


def _strip_known_prefixes(key: str) -> str:
    k = key
    changed = True
    while changed:
        changed = False
        for pref in ("module.", "base_model.model.", "model."):
            if k.startswith(pref):
                k = k[len(pref) :]
                changed = True
    return k


def load_rgha_weights_from_checkpoint(model: torch.nn.Module, model_name_or_path: str) -> int:
    r"""Load ``rgha_*`` tensors saved by LlamaFactory training into patched submodules.

    ``AutoModelForCausalLM.from_pretrained`` ignores these keys (unexpected in checkpoint), so we
    must load them after ``patch_rgha_forward`` creates ``rgha_ln`` / ``rgha_mlp`` / ``rgha_gate``.
    """
    unwrapped = _unwrap_causal_lm(model)
    ckpt_dir = _resolve_checkpoint_dir(model_name_or_path)
    raw = _load_raw_state_dict(ckpt_dir)
    if not raw:
        return 0

    rgha_ckpt = {k: v for k, v in raw.items() if "rgha_" in k}
    if not rgha_ckpt:
        return 0

    model_sd = unwrapped.state_dict()
    to_load: dict[str, torch.Tensor] = {}
    for ck, tensor in rgha_ckpt.items():
        candidates = {_strip_known_prefixes(ck), ck}
        for cand in candidates:
            if cand not in model_sd:
                continue
            if model_sd[cand].shape != tensor.shape:
                continue
            to_load[cand] = tensor.to(dtype=model_sd[cand].dtype, device=model_sd[cand].device)
            break

    if not to_load:
        return 0
    unwrapped.load_state_dict(to_load, strict=False)
    return len(to_load)




def _build_eos_token_ids(tokenizer: AutoTokenizer, think_token: str = "<add_think>") -> int | list[int]:
    eos_ids: list[int] = []
    if tokenizer.eos_token_id is not None and tokenizer.eos_token_id >= 0:
        eos_ids.append(int(tokenizer.eos_token_id))

    extra_ids = getattr(tokenizer, "additional_special_tokens_ids", None)
    if isinstance(extra_ids, list):
        eos_ids.extend(int(i) for i in extra_ids if isinstance(i, int) and i >= 0)

    for tok in ("<|eot_id|>", "<|im_end|>", "<|end|>"):
        tid = tokenizer.convert_tokens_to_ids(tok)
        if isinstance(tid, int) and tid >= 0:
            eos_ids.append(tid)

    think_id = tokenizer.convert_tokens_to_ids(think_token)
    if isinstance(think_id, int) and think_id >= 0:
        eos_ids = [i for i in eos_ids if i != think_id]

    eos_ids = list(dict.fromkeys(eos_ids))
    if not eos_ids:
        raise ValueError("No valid eos_token_id found from tokenizer.")
    return eos_ids[0] if len(eos_ids) == 1 else eos_ids

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
) -> list[dict[str, Any]]:
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
    dataset_module = get_dataset(template_obj, model_args, data_args, training_args, "sft", **tokenizer_module)
    hf_ds = dataset_module["train_dataset"]

    rows: list[dict[str, Any]] = []
    for i in range(len(hf_ds)):
        input_ids = hf_ds[i]["input_ids"]
        labels = hf_ds[i]["labels"]
        ids_list = [int(x) for x in input_ids]
        lab_list = [int(x) for x in labels]
        n = min(len(ids_list), len(lab_list))
        ids_list = ids_list[:n]
        lab_list = lab_list[:n]
        first_sup = next((j for j in range(n) if lab_list[j] != IGNORE_INDEX), n)
        prompt = tokenizer.decode(ids_list[:first_sup], skip_special_tokens=True)
        label = tokenizer.decode(
            [lab_list[j] for j in range(n) if lab_list[j] != IGNORE_INDEX],
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
    unwrapped = _unwrap_causal_lm(model)

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
        # Keep gate features unchanged, but trigger mask uses only low cosine similarity.
        risk_mask = one_minus_max_sim > self.rgha_threshold

        feat = torch.stack((norm_entropy, one_minus_max_sim), dim=-1).to(hidden_states.dtype)
        gate = torch.sigmoid(self.rgha_gate(feat)).squeeze(-1) * risk_mask.to(hidden_states.dtype)
        delta = self.rgha_mlp(self.rgha_ln(hidden_states))
        refined_hidden = hidden_states + gate.unsqueeze(-1) * delta
        pre_norm = hidden_states.norm(p=2, dim=-1, keepdim=True).clamp_min(1e-12)
        post_norm = refined_hidden.norm(p=2, dim=-1, keepdim=True).clamp_min(1e-12)
        refined_hidden = refined_hidden * (pre_norm / post_norm)
        outputs.logits = output_embeds(refined_hidden)
        return outputs

    unwrapped.forward = MethodType(_forward_with_rgha, unwrapped)
    return model


def _split_rows(rows: list[dict[str, Any]], num_shards: int) -> list[list[dict[str, Any]]]:
    if num_shards <= 0:
        raise ValueError("num_shards must be positive.")
    n = len(rows)
    chunk = (n + num_shards - 1) // num_shards
    return [rows[i * chunk : min((i + 1) * chunk, n)] for i in range(num_shards)]


def _run_data_parallel_inference(args: argparse.Namespace, rows: list[dict[str, Any]]) -> None:
    """Spawn one process per GPU; each loads full model on a single card (true data parallel)."""
    n_gpu = int(args.data_parallel_gpus)
    if n_gpu < 2:
        return
    if not torch.cuda.is_available() or torch.cuda.device_count() < n_gpu:
        raise ValueError(
            f"--data_parallel_gpus={n_gpu} requires at least {n_gpu} visible CUDA devices "
            f"(got {0 if not torch.cuda.is_available() else torch.cuda.device_count()})."
        )

    out_base = Path(args.output)
    out_base.parent.mkdir(parents=True, exist_ok=True)
    tmpdir = tempfile.mkdtemp(prefix="rgha_dp_")
    try:
        shards = _split_rows(rows, n_gpu)
        shard_outs = [out_base.with_suffix(out_base.suffix + f".shard{i}") for i in range(n_gpu)]

        procs: list[subprocess.Popen] = []
        script_path = Path(__file__).resolve()
        py = sys.executable

        for i in range(n_gpu):
            chunk = shards[i]
            if not chunk:
                continue
            sf = Path(tmpdir) / f"shard_{i}.jsonl"
            with sf.open("w", encoding="utf-8") as f:
                for r in chunk:
                    f.write(json.dumps(r, ensure_ascii=False) + "\n")

            env = os.environ.copy()
            env["CUDA_VISIBLE_DEVICES"] = str(i)
            outp = shard_outs[i]

            cmd = [
                py,
                "-u",
                str(script_path),
                "--model_name_or_path",
                args.model_name_or_path,
                "--data",
                str(sf),
                "--output",
                str(outp),
                "--batch_size",
                str(args.batch_size),
                "--max_new_tokens",
                str(args.max_new_tokens),
                "--temperature",
                str(args.temperature),
                "--top_p",
                str(args.top_p),
                "--num_generations",
                str(args.num_generations),
                "--repetition_penalty",
                str(args.repetition_penalty),
                "--no_repeat_ngram_size",
                str(args.no_repeat_ngram_size),
                "--device",
                "cuda:0",
                "--rgha_hidden_size",
                str(args.rgha_hidden_size),
                "--rgha_entropy_alpha",
                str(args.rgha_entropy_alpha),
                "--rgha_sim_beta",
                str(args.rgha_sim_beta),
                "--rgha_threshold",
                str(args.rgha_threshold),
                "--_dp_worker",
            ]
            if args.do_sample:
                cmd.append("--do_sample")
            if args.use_rgha:
                cmd.append("--use_rgha")
            if args.no_chat_template:
                cmd.append("--no_chat_template")
            if args.default_system is not None:
                cmd.extend(["--default_system", args.default_system])

            procs.append(subprocess.Popen(cmd, env=env))

        for p in procs:
            rc = p.wait()
            if rc != 0:
                raise RuntimeError(f"Data-parallel worker exited with code {rc}.")

        with out_base.open("w", encoding="utf-8") as fout:
            for outp in shard_outs:
                if not outp.exists():
                    continue
                with outp.open("r", encoding="utf-8") as fin:
                    shutil.copyfileobj(fin, fout)
        print(f"Merged {n_gpu} shard slot(s) into: {out_base.resolve()}")
    finally:
        shutil.rmtree(tmpdir, ignore_errors=True)


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
    parser.add_argument("--repetition_penalty", type=float, default=1.0)
    parser.add_argument("--no_repeat_ngram_size", type=int, default=0)
    parser.add_argument("--num_generations", type=int, default=1, help="Number of generations per prompt")
    parser.add_argument(
        "--device",
        type=str,
        default="auto",
        help="Device for inference: auto/cuda/cuda:0/cpu. `auto` will use multi-GPU device_map when available.",
    )
    parser.add_argument(
        "--data_parallel_gpus",
        type=int,
        default=0,
        help=(
            "If >=2, split samples across this many processes, each with CUDA_VISIBLE_DEVICES=i "
            "and full model on one GPU (true data parallel). Output is merged to --output. "
            "Use 0 or 1 to disable."
        ),
    )
    parser.add_argument("--_dp_worker", action="store_true", help=argparse.SUPPRESS)

    parser.add_argument("--use_rgha", action="store_true", default=False)
    parser.add_argument("--rgha_hidden_size", type=int, default=256)
    parser.add_argument("--rgha_entropy_alpha", type=float, default=0.4)
    parser.add_argument("--rgha_sim_beta", type=float, default=0.6)
    parser.add_argument("--rgha_threshold", type=float, default=0.58)
    parser.add_argument(
        "--no_chat_template",
        action="store_true",
        help=(
            "Do not use tokenizer.apply_chat_template; tokenize raw 'prompt' strings as before "
            "(legacy; breaks alignment with LlamaFactory llama3 / Llama 3 chat training)."
        ),
    )
    parser.add_argument(
        "--default_system",
        type=str,
        default=None,
        help="Optional system message when using chat template and a row has no 'system' field.",
    )
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

    if args.data_parallel_gpus >= 2 and not args._dp_worker:
        _run_data_parallel_inference(args, rows)
        return

    tokenizer = AutoTokenizer.from_pretrained(args.model_name_or_path, trust_remote_code=True)
    if tokenizer.pad_token_id is None:
        tokenizer.pad_token = tokenizer.eos_token
    # Decoder-only models should use left padding during generation.
    tokenizer.padding_side = "left"

    if args.use_rgha:
        print(
            "[RGHA] Note: from_pretrained will warn that rgha_* keys in the checkpoint are unused. "
            "That is normal — LlamaForCausalLM has no RGHA slots until we patch; weights are loaded right after."
        )

    eos_token_id = _build_eos_token_ids(tokenizer)

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
        n_rgha = load_rgha_weights_from_checkpoint(model, args.model_name_or_path)
        if n_rgha > 0:
            print(
                f"[RGHA] forward patch enabled; loaded {n_rgha} RGHA tensor(s) from checkpoint "
                f"(safe to ignore the earlier 'not used when initializing' message for those keys)."
            )
        else:
            print(
                "[RGHA] forward patch enabled but no rgha_* weights found in checkpoint — "
                "RGHA modules are randomly initialized (expect HF 'unused / not initialized' warnings on load)."
            )
    else:
        print("[RGHA] disabled.")

    use_chat_template = not args.no_chat_template
    if use_chat_template and getattr(tokenizer, "chat_template", None) is None:
        print(
            "[chat] Tokenizer has no chat_template; falling back to raw prompt tokenization "
            "(pass --no_chat_template to silence this)."
        )
        use_chat_template = False
    elif use_chat_template:
        print(
            "[chat] Using tokenizer.apply_chat_template(add_generation_prompt=True) — aligned with "
            "HF Llama 3 / LlamaFactory 'llama3' (special tokens + assistant header)."
        )

    with output_path.open("w", encoding="utf-8") as fout:
        pbar = tqdm(range(0, len(rows), args.batch_size), desc="Infer", total=(len(rows) + args.batch_size - 1) // args.batch_size)
        for st in pbar:
            batch = rows[st : st + args.batch_size]
            labels = [str(r.get("label", "")) for r in batch]

            if use_chat_template:
                enc, prompts_for_out = _encode_batch_with_chat_template(
                    tokenizer, batch, args.default_system, input_device
                )
            else:
                prompts_for_out = [str(r.get("prompt", "")) for r in batch]
                enc = tokenizer(
                    prompts_for_out,
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
                    repetition_penalty=args.repetition_penalty,
                    no_repeat_ngram_size=max(0, int(args.no_repeat_ngram_size)),
                    pad_token_id=tokenizer.pad_token_id,
                    eos_token_id=eos_token_id,
                )

            # With left padding, non-pad length != tensor width. HF appends new tokens after the
            # full padded row (width = input_ids.shape[1]); using attention_mask.sum() slices wrong
            # and leaks prompt tail into "predict" (often mid-sentence) and looks like repetition.
            prompt_width = int(enc["input_ids"].shape[1])
            bsz = len(batch)
            for i in range(bsz):
                pred_list: list[str] = []
                for k in range(args.num_generations):
                    out_idx = i * args.num_generations + k
                    ids = out_ids[out_idx]
                    gen_ids = ids[prompt_width:].tolist()
                    pred_text = tokenizer.decode(gen_ids, skip_special_tokens=True)
                    pred_list.append(pred_text)
                rec = {
                    "prompt": prompts_for_out[i],
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

