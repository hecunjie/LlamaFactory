"""SAE + 质量分类头的联合训练循环。"""

from __future__ import annotations

from pathlib import Path
from typing import Any, Optional

import torch
import torch.nn as nn
from torch.utils.data import DataLoader
from torch.utils.tensorboard import SummaryWriter
from transformers import get_cosine_schedule_with_warmup

from . import config as cfg
from .model import QualityClassifier, SparseAutoencoder


class SAETrainer:
    """
    总损失: L = L_recon + LAMBDA_QUALITY * L_quality
    L_quality 使用 z.detach()，避免分类梯度回传到 SAE。
    每个 step: optimizer.step() -> sae.normalize_decoder() -> scheduler.step()
    """

    def __init__(
        self,
        sae: SparseAutoencoder,
        classifier: QualityClassifier,
        train_loader: DataLoader,
        config: Any = None,
    ):
        self.config = config or cfg
        self.sae = sae
        self.classifier = classifier
        self.train_loader = train_loader

        params = list(sae.parameters()) + list(classifier.parameters())
        self.optimizer = torch.optim.AdamW(
            params,
            lr=float(self.config.LEARNING_RATE),
            weight_decay=float(self.config.WEIGHT_DECAY),
        )

        steps_per_epoch = max(len(train_loader), 1)
        total_steps = int(self.config.NUM_EPOCHS) * steps_per_epoch
        warmup = min(int(self.config.WARMUP_STEPS), total_steps)
        self.scheduler = get_cosine_schedule_with_warmup(
            self.optimizer,
            num_warmup_steps=warmup,
            num_training_steps=total_steps,
        )

        self.bce = nn.BCEWithLogitsLoss()
        self.global_step = 0
        self.start_epoch = 0

        log_root = Path(self.config.SAVE_DIR) / "logs"
        log_root.mkdir(parents=True, exist_ok=True)
        self.writer = SummaryWriter(log_dir=str(log_root))

    def train(self) -> None:
        """完整训练；start_epoch 可由 load_checkpoint 设置。"""
        device = torch.device(
            self.config.DEVICE if torch.cuda.is_available() and self.config.DEVICE == "cuda" else "cpu"
        )
        self.sae.to(device)
        self.classifier.to(device)

        if self.start_epoch >= int(self.config.NUM_EPOCHS):
            print(f"[trainer] start_epoch={self.start_epoch} >= NUM_EPOCHS，跳过训练。")
            self.writer.close()
            return

        for epoch in range(self.start_epoch, int(self.config.NUM_EPOCHS)):
            metrics = self.train_epoch(epoch, device)
            if (epoch + 1) % int(self.config.SAVE_EVERY) == 0:
                self.save_checkpoint(epoch, metrics)

        self.writer.close()
        print("[trainer] 训练结束。")

    def train_epoch(self, epoch: int, device: torch.device) -> dict[str, float]:
        self.sae.train()
        self.classifier.train()

        sum_recon = 0.0
        sum_quality = 0.0
        sum_total = 0.0
        sum_l0 = 0.0
        sum_l2r = 0.0
        sum_acc = 0.0
        n_batches = 0

        for batch in self.train_loader:
            h = batch["hidden_state"].to(device, dtype=torch.float32)
            y = batch["is_correct"].to(device, dtype=torch.float32)

            self.optimizer.zero_grad(set_to_none=True)
            out = self.sae(h)
            z = out["z"]
            h_hat = out["h_hat"]
            recon_loss = out["recon_loss"]

            logits_q = self.classifier(z.detach())
            quality_loss = self.bce(logits_q, y)

            total = recon_loss + float(self.config.LAMBDA_QUALITY) * quality_loss
            total.backward()

            if float(self.config.GRAD_CLIP) > 0:
                torch.nn.utils.clip_grad_norm_(
                    list(self.sae.parameters()) + list(self.classifier.parameters()),
                    float(self.config.GRAD_CLIP),
                )

            self.optimizer.step()
            self.sae.normalize_decoder()
            self.scheduler.step()

            with torch.no_grad():
                preds = (torch.sigmoid(logits_q) >= 0.5).float()
                acc = (preds == y).float().mean()

            sum_recon += float(recon_loss.item())
            sum_quality += float(quality_loss.item())
            sum_total += float(total.item())
            sum_l0 += float(out["l0"].item())
            sum_l2r += float(out["l2_ratio"].item())
            sum_acc += float(acc.item())
            n_batches += 1
            self.global_step += 1

            if self.global_step % int(self.config.LOG_EVERY) == 0:
                msg = (
                    f"step={self.global_step} epoch={epoch} "
                    f"L_recon={recon_loss.item():.6f} L_quality={quality_loss.item():.6f} "
                    f"L_total={total.item():.6f} L0={out['l0'].item():.2f} "
                    f"l2_ratio={out['l2_ratio'].item():.4f} cls_acc={acc.item():.4f}"
                )
                print(msg)
                self.writer.add_scalar("train/recon_loss", recon_loss.item(), self.global_step)
                self.writer.add_scalar("train/quality_loss", quality_loss.item(), self.global_step)
                self.writer.add_scalar("train/total_loss", total.item(), self.global_step)
                self.writer.add_scalar("train/l0", out["l0"].item(), self.global_step)
                self.writer.add_scalar("train/l2_ratio", out["l2_ratio"].item(), self.global_step)
                self.writer.add_scalar("train/cls_acc", acc.item(), self.global_step)
                self.writer.add_scalar("train/lr", self.scheduler.get_last_lr()[0], self.global_step)

        denom = max(n_batches, 1)
        ep = {
            "epoch_recon": sum_recon / denom,
            "epoch_quality": sum_quality / denom,
            "epoch_total": sum_total / denom,
            "epoch_l0": sum_l0 / denom,
            "epoch_l2_ratio": sum_l2r / denom,
            "epoch_cls_acc": sum_acc / denom,
        }
        print(
            f"[epoch {epoch}] mean L_recon={ep['epoch_recon']:.6f} L_quality={ep['epoch_quality']:.6f} "
            f"L_total={ep['epoch_total']:.6f} L0={ep['epoch_l0']:.2f} l2_ratio={ep['epoch_l2_ratio']:.4f} "
            f"cls_acc={ep['epoch_cls_acc']:.4f}"
        )
        self.writer.add_scalar("epoch/recon_loss", ep["epoch_recon"], epoch)
        self.writer.add_scalar("epoch/quality_loss", ep["epoch_quality"], epoch)
        self.writer.add_scalar("epoch/cls_acc", ep["epoch_cls_acc"], epoch)
        return ep

    def save_checkpoint(self, epoch: int, metrics: dict[str, float]) -> None:
        """保存到 SAVE_DIR/epoch_{epoch:03d}.pt。"""
        save_dir = Path(self.config.SAVE_DIR)
        save_dir.mkdir(parents=True, exist_ok=True)
        path = save_dir / f"epoch_{epoch:03d}.pt"
        cfg_dict = {
            k: getattr(self.config, k)
            for k in dir(self.config)
            if k.isupper() and not k.startswith("_")
        }
        payload = {
            "epoch": epoch,
            "sae": self.sae.state_dict(),
            "classifier": self.classifier.state_dict(),
            "optimizer": self.optimizer.state_dict(),
            "scheduler": self.scheduler.state_dict(),
            "metrics": metrics,
            "config": cfg_dict,
            "global_step": self.global_step,
        }
        torch.save(payload, path)
        print(f"[trainer] checkpoint 已保存: {path.resolve()}")

    @classmethod
    def load_checkpoint(
        cls,
        path: str,
        sae: SparseAutoencoder,
        classifier: QualityClassifier,
        train_loader: DataLoader,
        config: Any = None,
    ) -> tuple["SAETrainer", int]:
        """
        从 checkpoint 恢复优化器/调度器状态。
        返回 (trainer, start_epoch)：下一轮训练从 start_epoch 开始（已训练完 epoch 则从下一条继续）。
        """
        config = config or cfg
        try:
            ckpt = torch.load(path, map_location="cpu", weights_only=False)
        except TypeError:
            ckpt = torch.load(path, map_location="cpu")
        trainer = cls(sae, classifier, train_loader, config=config)
        trainer.sae.load_state_dict(ckpt["sae"])
        trainer.classifier.load_state_dict(ckpt["classifier"])
        trainer.optimizer.load_state_dict(ckpt["optimizer"])
        trainer.scheduler.load_state_dict(ckpt["scheduler"])
        trainer.global_step = int(ckpt.get("global_step", 0))
        done_epoch = int(ckpt.get("epoch", -1))
        trainer.start_epoch = done_epoch + 1
        print(f"[trainer] 已从 checkpoint 恢复: {path}，将从 epoch={trainer.start_epoch} 继续")
        return trainer, trainer.start_epoch
