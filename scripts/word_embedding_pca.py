#!/usr/bin/env python3
"""
对指定 LLM 的词嵌入矩阵做 PCA，并画 scree plot。

功能：
- 从 HuggingFace / 本地路径加载模型；
- 读取输入嵌入矩阵（word embedding）；
- 对 embedding 做 PCA；
- 保存：
  - scree plot（特征值曲线 + 累积贡献率）PNG 图片；
  - 所有主成分特征值（explained_variance_）；
  - 各主成分解释方差占比（explained_variance_ratio_）；
  - 累积解释方差占比（cumulative_explained_variance_ratio_）；
  - 一个简单的推荐 k（达到设定阈值的最小主成分数量）。

示例用法：
  python scripts/word_embedding_pca.py \
      --model_name_or_path Qwen/Qwen2-1.5B-Instruct \
      --output_dir outputs/embedding_pca \
      --device cuda
"""

import argparse
import json
import os
from typing import Tuple

import numpy as np
import torch
from matplotlib import pyplot as plt
from sklearn.decomposition import PCA
from transformers import AutoModel


def load_word_embeddings(model_name_or_path: str, device: str = "cpu") -> torch.Tensor:
    """加载模型并返回输入 embedding 权重张量，shape: (vocab_size, hidden_dim)."""
    print(f"Loading model from {model_name_or_path} ...")
    model = AutoModel.from_pretrained(
        model_name_or_path,
        trust_remote_code=True,
    )

    model.to(device)
    model.eval()

    embedding = model.get_input_embeddings()
    if embedding is None:
        raise RuntimeError("模型不包含 input embeddings（get_input_embeddings() 返回 None）")

    weight = embedding.weight.detach().to("cpu")  # PCA 放在 CPU 上即可
    print(f"Embedding weight shape: {tuple(weight.shape)} (vocab_size, hidden_dim)")
    return weight


def run_pca_on_embeddings(weight: torch.Tensor) -> Tuple[np.ndarray, np.ndarray]:
    """对词嵌入做 PCA，返回 (eigenvalues, explained_variance_ratio)."""
    # weight: (vocab_size, hidden_dim)，我们希望主成分最多为 hidden_dim
    embeddings = weight.numpy()
    vocab_size, hidden_dim = embeddings.shape
    print(f"Running PCA on matrix of shape (vocab_size={vocab_size}, hidden_dim={hidden_dim}) ...")

    # n_components 不要超过特征维度
    n_components = min(vocab_size, hidden_dim)
    pca = PCA(n_components=n_components, svd_solver="full")
    pca.fit(embeddings)

    eigenvalues = pca.explained_variance_  # 特征值
    explained_ratio = pca.explained_variance_ratio_

    print(f"PCA done. Got {len(eigenvalues)} components.")
    return eigenvalues, explained_ratio


def plot_scree(
    eigenvalues: np.ndarray,
    explained_ratio: np.ndarray,
    output_path: str,
) -> None:
    """画 scree plot（特征值 + 累积解释方差），保存为 PNG。"""
    ks = np.arange(1, len(eigenvalues) + 1)
    cum_ratio = np.cumsum(explained_ratio)

    plt.figure(figsize=(10, 6))

    # 特征值曲线
    plt.plot(ks, eigenvalues, marker="o", label="Eigenvalues")
    # 累积解释方差（用第二轴更清晰）
    ax1 = plt.gca()
    ax2 = ax1.twinx()
    ax2.plot(ks, cum_ratio, color="orange", marker="x", label="Cumulative explained variance ratio")

    ax1.set_xlabel("Principal Component Index (k)")
    ax1.set_ylabel("Eigenvalue")
    ax2.set_ylabel("Cumulative Explained Variance Ratio")

    ax1.grid(True, linestyle="--", alpha=0.5)
    plt.title("Scree Plot of Word Embedding PCA")

    # 合并图例
    lines_1, labels_1 = ax1.get_legend_handles_labels()
    lines_2, labels_2 = ax2.get_legend_handles_labels()
    ax1.legend(lines_1 + lines_2, labels_1 + labels_2, loc="best")

    plt.tight_layout()
    plt.savefig(output_path, dpi=200)
    plt.close()

    print(f"Scree plot saved to {output_path}")


def suggest_k(cum_ratio: np.ndarray, threshold: float = 0.95) -> int:
    """根据累积解释方差比例，给出达到 threshold 的最小 k。"""
    idx = np.searchsorted(cum_ratio, threshold)
    k = int(idx + 1) if idx < len(cum_ratio) else len(cum_ratio)
    return k


def save_results(
    eigenvalues: np.ndarray,
    explained_ratio: np.ndarray,
    output_dir: str,
    threshold: float,
) -> None:
    """保存 PCA 结果到指定目录。"""
    os.makedirs(output_dir, exist_ok=True)

    cum_ratio = np.cumsum(explained_ratio)
    k_suggested = suggest_k(cum_ratio, threshold=threshold)

    # 保存数值数据
    np.save(os.path.join(output_dir, "eigenvalues.npy"), eigenvalues)
    np.save(os.path.join(output_dir, "explained_variance_ratio.npy"), explained_ratio)
    np.save(os.path.join(output_dir, "cumulative_explained_variance_ratio.npy"), cum_ratio)

    # 保存 JSON 概览
    summary = {
        "num_components": int(len(eigenvalues)),
        "eigenvalues": eigenvalues.tolist(),
        "explained_variance_ratio": explained_ratio.tolist(),
        "cumulative_explained_variance_ratio": cum_ratio.tolist(),
        "threshold": float(threshold),
        "suggested_k": int(k_suggested),
    }
    with open(os.path.join(output_dir, "pca_summary.json"), "w", encoding="utf-8") as f:
        json.dump(summary, f, ensure_ascii=False, indent=2)

    print(f"PCA eigenvalues and ratios saved to {output_dir}")
    print(f"Suggested k (cumulative >= {threshold:.2%}): {k_suggested}")


def main():
    parser = argparse.ArgumentParser(
        description="对 LLM 词嵌入矩阵做 PCA 并绘制 scree plot，用于选择主成分数量 k。",
    )
    parser.add_argument(
        "--model_name_or_path",
        type=str,
        required=True,
        help="HuggingFace 模型名称或本地模型路径，例如：`Qwen/Qwen2-1.5B-Instruct` 或 `./checkpoints/qwen`。",
    )
    parser.add_argument(
        "--output_dir",
        type=str,
        required=True,
        help="结果保存目录，将保存 PNG 图和 numpy / json 文件。",
    )
    parser.add_argument(
        "--device",
        type=str,
        default="cpu",
        choices=("cpu", "cuda"),
        help="加载模型所用设备，默认 cpu。",
    )
    parser.add_argument(
        "--variance_threshold",
        type=float,
        default=0.95,
        help="用于推荐 k 的累积解释方差阈值，默认 0.95（95%%）。",
    )
    args = parser.parse_args()

    # 保证有 GPU 时可以自动用上
    device = args.device
    if device == "cuda" and not torch.cuda.is_available():
        print("Warning: CUDA 不可用，自动回退到 CPU。")
        device = "cpu"

    embeddings = load_word_embeddings(args.model_name_or_path, device=device)

    eigenvalues, explained_ratio = run_pca_on_embeddings(embeddings)

    os.makedirs(args.output_dir, exist_ok=True)
    scree_path = os.path.join(args.output_dir, "scree_plot.png")
    plot_scree(eigenvalues, explained_ratio, scree_path)

    save_results(
        eigenvalues=eigenvalues,
        explained_ratio=explained_ratio,
        output_dir=args.output_dir,
        threshold=args.variance_threshold,
    )


if __name__ == "__main__":
    main()

