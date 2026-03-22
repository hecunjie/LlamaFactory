#!/usr/bin/env python3
"""
在缓存的 hidden states 上评估训练好的 SAE：重建、稀疏度、质量分类与可视化。

用法:
  python -m train_sae.evaluate --checkpoint ./sae_checkpoints/epoch_019.pt --cache_path ./sae_cache/hidden_states.pt
"""

from __future__ import annotations

import argparse
import json
import sys
from pathlib import Path
from typing import Any

import matplotlib.pyplot as plt
import numpy as np
import torch
import torch.nn.functional as F

_ROOT = Path(__file__).resolve().parent.parent
if str(_ROOT) not in sys.path:
    sys.path.insert(0, str(_ROOT))

from train_sae import config as cfg
from train_sae.model import QualityClassifier, SparseAutoencoder


def _torch_load(path: Path, map_location: str = "cpu") -> dict[str, Any]:
    try:
        return torch.load(path, map_location=map_location, weights_only=False)
    except TypeError:
        return torch.load(path, map_location=map_location)


def _pca_2d(x: np.ndarray) -> np.ndarray:
    """x: (n, d) -> (n, 2)，中心化后取前两个主成分。"""
    x = x.astype(np.float64)
    x = x - x.mean(axis=0, keepdims=True)
    # SVD: x = U S Vt
    _, _, vt = np.linalg.svd(x, full_matrices=False)
    comp = vt[:2].T  # (d, 2)
    return (x @ comp).astype(np.float32)


def _explained_variance(y_true: torch.Tensor, y_pred: torch.Tensor) -> float:
    """标量 explained variance: 1 - Var(y_true - y_pred) / Var(y_true)。"""
    diff = y_true - y_pred
    v_true = y_true.var(unbiased=False)
    if float(v_true) <= 1e-12:
        return 1.0
    v_diff = diff.var(unbiased=False)
    return float((1.0 - v_diff / v_true).clamp(0.0, 1.0).item())


def main() -> None:
    parser = argparse.ArgumentParser(description="评估 SAE checkpoint")
    parser.add_argument("--checkpoint", type=str, required=True)
    parser.add_argument(
        "--cache_path",
        type=str,
        default=None,
        help="hidden_states.pt 路径，默认 CACHE_DIR/hidden_states.pt",
    )
    parser.add_argument("--output_dir", type=str, default="./sae_eval")
    parser.add_argument("--device", type=str, default=None, choices=["cuda", "cpu"])
    args = parser.parse_args()

    out_dir = Path(args.output_dir)
    out_dir.mkdir(parents=True, exist_ok=True)

    cache_path = Path(args.cache_path) if args.cache_path else Path(cfg.CACHE_DIR) / "hidden_states.pt"
    if not cache_path.is_file():
        raise FileNotFoundError(f"未找到缓存: {cache_path.resolve()}，请先运行训练提取或指定 --cache_path")

    device = torch.device(
        args.device
        if args.device
        else ("cuda" if torch.cuda.is_available() and cfg.DEVICE == "cuda" else "cpu")
    )

    blob = _torch_load(cache_path)
    h_all: torch.Tensor = blob["hidden_states"].float()
    labels: torch.Tensor = blob["labels"].long()

    ckpt = _torch_load(Path(args.checkpoint))
    ck_cfg = ckpt.get("config") or {}
    hidden_dim = int(ck_cfg.get("HIDDEN_DIM", cfg.HIDDEN_DIM))
    expansion = int(ck_cfg.get("SAE_EXPANSION", cfg.SAE_EXPANSION))
    top_k = int(ck_cfg.get("TOP_K", cfg.TOP_K))
    sae_dim = hidden_dim * expansion

    sae = SparseAutoencoder(hidden_dim, sae_dim, top_k).to(device)
    clf = QualityClassifier(sae_dim).to(device)
    sae.load_state_dict(ckpt["sae"])
    clf.load_state_dict(ckpt["classifier"])
    sae.eval()
    clf.eval()

    n = h_all.shape[0]
    bs = min(512, max(1, n))
    mse_list: list[float] = []
    cos_list: list[float] = []
    ev_list: list[float] = []
    z_chunks: list[torch.Tensor] = []
    h_hat_chunks: list[torch.Tensor] = []
    l0_list: list[float] = []
    logits_all: list[torch.Tensor] = []

    with torch.no_grad():
        for i in range(0, n, bs):
            h = h_all[i : i + bs].to(device)
            out = sae(h)
            h_hat = out["h_hat"]
            z = out["z"]
            mse_list.append(F.mse_loss(h_hat, h).item())
            cos = F.cosine_similarity(h_hat, h, dim=-1).mean().item()
            cos_list.append(cos)
            ev_list.append(_explained_variance(h, h_hat))
            l0_list.append(float((z > 0).float().sum(dim=-1).mean().item()))
            z_chunks.append(z.cpu())
            h_hat_chunks.append(h_hat.cpu())
            logits_all.append(clf(z).cpu())

    h_hat_all = torch.cat(h_hat_chunks, dim=0)
    z_all = torch.cat(z_chunks, dim=0)
    logits_cat = torch.cat(logits_all, dim=0)
    preds = (torch.sigmoid(logits_cat) >= 0.5).long()
    acc = float((preds == labels).float().mean().item())

    mean_mse = float(np.mean(mse_list))
    mean_cos = float(np.mean(cos_list))
    mean_ev = float(np.mean(ev_list))
    mean_l0 = float(np.mean(l0_list))

    # 特征激活频率（每个特征在多少比例的样本上 >0）
    active = (z_all > 0).float().numpy()
    freq = active.mean(axis=0)

    z_np = z_all.numpy()
    cor = labels.numpy() == 1
    wro = ~cor
    if cor.any() and wro.any():
        importance = np.abs(z_np[cor].mean(0) - z_np[wro].mean(0))
    elif cor.any():
        importance = np.abs(z_np[cor].mean(0))
    elif wro.any():
        importance = np.abs(z_np[wro].mean(0))
    else:
        importance = np.zeros(z_np.shape[1], dtype=np.float64)
    top_idx = np.argsort(-importance)[:20]

    results = {
        "mean_mse": mean_mse,
        "mean_cosine_similarity": mean_cos,
        "mean_explained_variance": mean_ev,
        "mean_l0": mean_l0,
        "classifier_accuracy": acc,
        "top20_feature_indices": top_idx.tolist(),
        "top20_importance": importance[top_idx].tolist(),
        "n_samples": int(n),
        "sae_dim": int(sae_dim),
        "top_k_config": int(top_k),
    }
    with (out_dir / "eval_results.json").open("w", encoding="utf-8") as f:
        json.dump(results, f, ensure_ascii=False, indent=2)
    print(f"[eval] 结果已写入: {(out_dir / 'eval_results.json').resolve()}")

    # 直方图：特征激活频率分布
    fig, ax = plt.subplots(figsize=(7, 4))
    ax.hist(freq, bins=50, color="steelblue", edgecolor="white", alpha=0.85)
    ax.set_xlabel("特征在样本上的激活频率")
    ax.set_ylabel("特征数量")
    ax.set_title("SAE 特征激活频率分布")
    fig.tight_layout()
    p_freq = out_dir / "feature_freq_hist.png"
    fig.savefig(p_freq, dpi=150)
    plt.close(fig)
    print(f"[eval] 图: {p_freq.resolve()}")

    # PCA：h 与 h_hat
    h_np = h_all.numpy()
    hh_np = h_hat_all.numpy()
    # 子采样以加速
    rng = np.random.default_rng(42)
    sub = min(8000, n)
    idx = rng.choice(n, size=sub, replace=False)
    h2 = _pca_2d(h_np[idx])
    hat2 = _pca_2d(hh_np[idx])
    lab = labels.numpy()[idx]

    fig, axes = plt.subplots(1, 2, figsize=(10, 4))
    for ax, data, title in zip(
        axes,
        [h2, hat2],
        ["原始 h (PCA-2D)", "重建 h_hat (PCA-2D)"],
    ):
        ax.scatter(data[lab == 1, 0], data[lab == 1, 1], s=4, alpha=0.35, c="blue", label="正确")
        ax.scatter(data[lab == 0, 0], data[lab == 0, 1], s=4, alpha=0.35, c="red", label="错误")
        ax.set_title(title)
        ax.legend(markerscale=2)
    fig.tight_layout()
    p_pca = out_dir / "pca_h_vs_hhat.png"
    fig.savefig(p_pca, dpi=150)
    plt.close(fig)
    print(f"[eval] 图: {p_pca.resolve()}")

    # 热图：随机 20 条样本的稀疏激活
    rng2 = np.random.default_rng(0)
    k_show = min(20, n)
    si = rng2.choice(n, size=k_show, replace=False)
    # 特征太多，只显示非零列并限制列数便于可视化
    z_sub = z_np[si]
    nz_cols = np.where(np.abs(z_sub).max(axis=0) > 0)[0]
    if nz_cols.size > 200:
        nz_cols = nz_cols[np.argsort(-np.abs(z_sub[:, nz_cols]).max(axis=0))[:200]]
    z_plot = z_sub[:, nz_cols]

    fig, ax = plt.subplots(figsize=(12, 5))
    im = ax.imshow(z_plot, aspect="auto", cmap="magma", interpolation="nearest")
    ax.set_xlabel("特征索引（子集）")
    ax.set_ylabel("样本")
    ax.set_title("随机样本的 SAE 激活热图")
    fig.colorbar(im, ax=ax, fraction=0.02, pad=0.02)
    fig.tight_layout()
    p_hm = out_dir / "activation_heatmap.png"
    fig.savefig(p_hm, dpi=150)
    plt.close(fig)
    print(f"[eval] 图: {p_hm.resolve()}")

    print("\n[eval] 摘要")
    print(f"  平均 MSE: {mean_mse:.6f}")
    print(f"  平均 cosine similarity: {mean_cos:.6f} （目标 >0.99）")
    print(f"  平均 explained variance: {mean_ev:.6f} （目标 >0.95）")
    print(f"  平均 L0: {mean_l0:.2f} （配置 TOP_K={top_k}）")
    print(f"  分类器准确率: {acc:.4f}")
    print("  Top-20 推理相关特征 (索引, importance):")
    for j, t in enumerate(top_idx):
        print(f"    {j+1:2d}. dim={int(t):5d}  |diff mean z|={importance[t]:.6f}")


if __name__ == "__main__":
    main()
