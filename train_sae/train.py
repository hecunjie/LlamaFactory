#!/usr/bin/env python3
"""
SAE 训练主入口。

用法（在 LlamaFactory 根目录）:
  python -m train_sae.train --data_path ./data/samples.jsonl --model_name Qwen/Qwen2.5-3B-Instruct
"""

from __future__ import annotations

import argparse
import random
import sys
from pathlib import Path

import numpy as np
import torch
from transformers import AutoModelForCausalLM, AutoTokenizer

_ROOT = Path(__file__).resolve().parent.parent
if str(_ROOT) not in sys.path:
    sys.path.insert(0, str(_ROOT))

from experiment.utils import model_input_device  # noqa: E402

from train_sae import config as cfg
from train_sae.dataset import HiddenStateDataset, get_dataloader
from train_sae.model import QualityClassifier, SparseAutoencoder
from train_sae.trainer import SAETrainer


def _set_seed(seed: int) -> None:
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    if torch.cuda.is_available():
        torch.cuda.manual_seed_all(seed)


def main() -> None:
    parser = argparse.ArgumentParser(description="训练 SAE（稀疏自编码器）")
    parser.add_argument("--data_path", type=str, default=None, help="JSONL 轨迹数据路径")
    parser.add_argument("--model_name", type=str, default=None, help="HF LLM 名称或本地路径")
    parser.add_argument("--resume", type=str, default=None, help="从 checkpoint 继续训练")
    parser.add_argument("--no_cache", action="store_true", help="删除缓存并强制重新提取 hidden states")
    parser.add_argument("--device", type=str, default=None, choices=["cuda", "cpu"])
    args = parser.parse_args()

    if args.data_path:
        cfg.DATA_PATH = args.data_path
    if args.model_name:
        cfg.LLM_NAME = args.model_name
    if args.device:
        cfg.DEVICE = args.device

    _set_seed(int(cfg.SEED))

    cache_pt = Path(cfg.CACHE_DIR) / "hidden_states.pt"
    meta_json = Path(cfg.CACHE_DIR) / "metadata.json"

    if args.no_cache:
        for p in (cache_pt, meta_json):
            if p.is_file():
                p.unlink()
                print(f"[train] 已删除缓存文件: {p}")

    need_llm = (not cache_pt.is_file()) or args.no_cache

    llm_model = None
    tokenizer = None
    if need_llm:
        print(f"[train] 加载 LLM: {cfg.LLM_NAME}")
        tokenizer = AutoTokenizer.from_pretrained(cfg.LLM_NAME, trust_remote_code=True)
        if tokenizer.pad_token_id is None:
            tokenizer.pad_token = tokenizer.eos_token

        use_cuda = cfg.DEVICE == "cuda" and torch.cuda.is_available()
        dtype = torch.bfloat16 if str(cfg.DTYPE).lower() == "bfloat16" else torch.float16
        llm_model = AutoModelForCausalLM.from_pretrained(
            cfg.LLM_NAME,
            torch_dtype=dtype,
            trust_remote_code=True,
            device_map="auto" if use_cuda else None,
            low_cpu_mem_usage=use_cuda,
        ).eval()
        if not use_cuda:
            llm_model = llm_model.to("cpu")
        for p in llm_model.parameters():
            p.requires_grad_(False)
        print(f"[train] LLM 输入设备: {model_input_device(llm_model)}")
    else:
        print("[train] 检测到 hidden states 缓存，跳过加载 LLM。")

    dataset = HiddenStateDataset(
        cfg.DATA_PATH,
        llm_model,
        tokenizer,
        config=cfg,
        force_extract=False,
    )
    if len(dataset) == 0:
        print("[train] 数据集为空，退出。")
        return

    loader = get_dataloader(dataset, config=cfg, shuffle=True)

    hidden_dim = int(cfg.HIDDEN_DIM)
    sae_dim = int(cfg.HIDDEN_DIM) * int(cfg.SAE_EXPANSION)
    top_k = int(cfg.TOP_K)

    train_device = torch.device(
        cfg.DEVICE if torch.cuda.is_available() and cfg.DEVICE == "cuda" else "cpu"
    )
    sae = SparseAutoencoder(hidden_dim, sae_dim, top_k).to(train_device)
    clf = QualityClassifier(sae_dim).to(train_device)

    if args.resume:
        trainer, _ = SAETrainer.load_checkpoint(
            args.resume,
            sae,
            clf,
            loader,
            config=cfg,
        )
    else:
        trainer = SAETrainer(sae, clf, loader, config=cfg)

    trainer.train()

    print("[train] 最终统计：训练循环已结束；详细指标见 TensorBoard 与 checkpoint。")


if __name__ == "__main__":
    main()
