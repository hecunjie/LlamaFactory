"""TopK 稀疏自编码器 + 推理质量线性分类头。"""

from __future__ import annotations

import torch
import torch.nn as nn
import torch.nn.functional as F


class SparseAutoencoder(nn.Module):
    """
    TopK Sparse Autoencoder。

    Encoder: Linear(hidden_dim -> sae_dim, bias=True) + TopK + ReLU
    Decoder: Linear(sae_dim -> hidden_dim, bias=False)，列（特征维）单位化在 optimizer.step 之后执行。
    """

    def __init__(self, hidden_dim: int, sae_dim: int, top_k: int):
        super().__init__()
        self.hidden_dim = hidden_dim
        self.sae_dim = sae_dim
        self.top_k = min(int(top_k), sae_dim)

        self.encoder = nn.Linear(hidden_dim, sae_dim, bias=True)
        self.decoder = nn.Linear(sae_dim, hidden_dim, bias=False)

    def encode(self, h: torch.Tensor) -> torch.Tensor:
        """TopK 激活：每行仅保留最大的 top_k 个 pre-activation，再 ReLU。"""
        z_pre = self.encoder(h)
        topk_vals, topk_idx = z_pre.topk(self.top_k, dim=-1)
        z = torch.zeros_like(z_pre)
        z.scatter_(-1, topk_idx, topk_vals)
        return F.relu(z)

    def decode(self, z: torch.Tensor) -> torch.Tensor:
        return self.decoder(z)

    def forward(self, h: torch.Tensor) -> dict[str, torch.Tensor]:
        z = self.encode(h)
        h_hat = self.decode(z)
        recon_loss = F.mse_loss(h_hat, h)
        l0 = (z > 0).float().sum(dim=-1).mean()
        l2_ratio = (h_hat.norm(dim=-1) / h.norm(dim=-1).clamp(min=1e-8)).mean()
        return {
            "z": z,
            "h_hat": h_hat,
            "recon_loss": recon_loss,
            "l0": l0,
            "l2_ratio": l2_ratio,
        }

    @torch.no_grad()
    def normalize_decoder(self) -> None:
        """将 decoder.weight 的每一列（每个字典原子）归一化为单位范数。"""
        # decoder.weight: (hidden_dim, sae_dim)，列 j 对应第 j 个特征
        norms = self.decoder.weight.norm(dim=0, keepdim=True).clamp(min=1e-8)
        self.decoder.weight.div_(norms)


class QualityClassifier(nn.Module):
    """单层线性分类器：在稀疏特征 z 上预测是否来自正确推理轨迹。"""

    def __init__(self, sae_dim: int):
        super().__init__()
        self.linear = nn.Linear(sae_dim, 1)

    def forward(self, z: torch.Tensor) -> torch.Tensor:
        return self.linear(z).squeeze(-1)
