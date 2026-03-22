#!/usr/bin/env python3
"""
分析训练好的 SAE 特征：重要性排序、语义样例（若缓存含 token/context）、可视化。

用法（在 LlamaFactory 根目录）:
  python -m train_sae.analyze_features --checkpoint ./sae_checkpoints/epoch_020.pt
"""

from __future__ import annotations

import argparse
import json
import random
import sys
from pathlib import Path
from typing import Any, TextIO

import matplotlib.pyplot as plt
import numpy as np
import torch
from tqdm import tqdm

try:
    import seaborn as sns
except ImportError:
    sns = None  # type: ignore[assignment]

try:
    from sklearn.decomposition import PCA
    from sklearn.metrics import silhouette_score
except ImportError as e:
    raise ImportError(
        "analyze_features 需要 sklearn，请安装: pip install scikit-learn"
    ) from e

_ROOT = Path(__file__).resolve().parent.parent
if str(_ROOT) not in sys.path:
    sys.path.insert(0, str(_ROOT))

from train_sae.model import QualityClassifier, SparseAutoencoder  # noqa: E402


class _Tee(TextIO):
    """同时写入多个流（终端 + summary 文件）。"""

    def __init__(self, *streams: TextIO) -> None:
        self._streams = streams

    def write(self, s: str) -> int:
        for f in self._streams:
            f.write(s)
            f.flush()
        return len(s)

    def flush(self) -> None:
        for f in self._streams:
            f.flush()

    def isatty(self) -> bool:
        return False


def _torch_load(path: Path, map_location: str = "cpu") -> dict[str, Any]:
    try:
        return torch.load(path, map_location=map_location, weights_only=False)
    except TypeError:
        return torch.load(path, map_location=map_location)


def _metadata_has_context(meta: list[dict[str, Any]]) -> bool:
    if not meta:
        return False
    m0 = meta[0]
    return "token_str" in m0 and "context" in m0


def _encode_all_z(
    sae: SparseAutoencoder,
    h: torch.Tensor,
    device: torch.device,
    batch_size: int = 1024,
) -> torch.Tensor:
    """对 hidden states 批量 encode，返回 CPU float16 (N, sae_dim)。"""
    n = h.shape[0]
    sae.eval()
    chunks: list[torch.Tensor] = []
    with torch.inference_mode():
        for start in tqdm(range(0, n, batch_size), desc="SAE encode"):
            hb = h[start : start + batch_size].to(device, dtype=torch.float32)
            zb = sae.encode(hb).cpu().half()
            chunks.append(zb)
    return torch.cat(chunks, dim=0)


def _classifier_accuracy(
    clf: QualityClassifier,
    z_f32: torch.Tensor,
    labels: torch.Tensor,
    device: torch.device,
    batch_size: int = 1024,
) -> float:
    clf.eval()
    n = z_f32.shape[0]
    correct = 0
    with torch.inference_mode():
        for start in range(0, n, batch_size):
            zb = z_f32[start : start + batch_size].to(device)
            lb = labels[start : start + batch_size].to(device).float()
            logits = clf(zb)
            pred = (torch.sigmoid(logits) >= 0.5).float()
            correct += int((pred == lb).sum().item())
    return correct / max(n, 1)


def _silhouette_subsample(
    X: np.ndarray,
    y: np.ndarray,
    max_samples: int = 2000,
    seed: int = 42,
) -> float:
    """标签需为 0/1；每类至少 2 个样本才计算。"""
    rng = np.random.default_rng(seed)
    n = X.shape[0]
    if n < 4:
        return float("nan")
    idx_c = np.flatnonzero(y == 1)
    idx_w = np.flatnonzero(y == 0)
    if len(idx_c) < 2 or len(idx_w) < 2:
        return float("nan")
    half = max_samples // 2
    take_c = rng.choice(idx_c, size=min(half, len(idx_c)), replace=False)
    take_w = rng.choice(idx_w, size=min(half, len(idx_w)), replace=False)
    idx = np.concatenate([take_c, take_w])
    Xs = X[idx]
    ys = y[idx]
    if len(np.unique(ys)) < 2:
        return float("nan")
    return float(silhouette_score(Xs, ys, metric="euclidean"))


def main() -> None:
    parser = argparse.ArgumentParser(description="Analyze SAE features for reasoning quality")
    parser.add_argument("--checkpoint", type=str, required=True)
    parser.add_argument("--cache_path", type=str, default="./sae_cache/hidden_states.pt")
    parser.add_argument("--output_dir", type=str, default="./feature_analysis")
    parser.add_argument("--top_n", type=int, default=20)
    parser.add_argument("--context_examples", type=int, default=10)
    parser.add_argument("--device", type=str, default="cuda", choices=["cuda", "cpu"])
    parser.add_argument("--batch_size", type=int, default=1024)
    parser.add_argument("--seed", type=int, default=42)
    args = parser.parse_args()

    out_dir = Path(args.output_dir)
    out_dir.mkdir(parents=True, exist_ok=True)
    summary_path = out_dir / "analysis_summary.txt"
    summary_f = summary_path.open("w", encoding="utf-8")
    tee = _Tee(sys.stdout, summary_f)
    # 将 print 重定向到 tee（仅本脚本内显式使用 tee.write 更清晰；这里用 print 的 file 参数不通用）
    _orig_stdout = sys.stdout
    sys.stdout = tee  # type: ignore[assignment]

    try:
        device = torch.device(
            args.device if torch.cuda.is_available() and args.device == "cuda" else "cpu"
        )
        random.seed(args.seed)
        np.random.seed(args.seed)
        torch.manual_seed(args.seed)

        cache_path = Path(args.cache_path)
        ckpt_path = Path(args.checkpoint)
        blob = _torch_load(cache_path)
        h_all: torch.Tensor = blob["hidden_states"].float()
        labels_t: torch.Tensor = blob["labels"].long()
        meta: list[dict[str, Any]] = blob.get("metadata", [])
        n_total = int(h_all.shape[0])
        y_np = labels_t.numpy().astype(np.int64)
        has_ctx = _metadata_has_context(meta)

        ckpt = _torch_load(ckpt_path)
        ck_cfg = ckpt.get("config") or {}
        hidden_dim = int(ck_cfg.get("HIDDEN_DIM", 2048))
        expansion = int(ck_cfg.get("SAE_EXPANSION", 8))
        top_k = int(ck_cfg.get("TOP_K", 64))
        sae_dim = hidden_dim * expansion

        sae = SparseAutoencoder(hidden_dim, sae_dim, top_k)
        sae.load_state_dict(ckpt["sae"])
        sae.to(device).eval()

        clf = QualityClassifier(sae_dim)
        clf.load_state_dict(ckpt["classifier"])
        clf.to(device).eval()

        print("=== SAE 特征分析 ===")
        print(f"checkpoint: {ckpt_path.resolve()}")
        print(f"cache:      {cache_path.resolve()}")
        print(f"output_dir: {out_dir.resolve()}")
        print()

        # ---------- Part 1 ----------
        n_cor = int((labels_t == 1).sum().item())
        n_wro = n_total - n_cor
        p_cor = n_cor / max(n_total, 1)
        p_wro = n_wro / max(n_total, 1)
        majority = max(p_cor, p_wro)

        print("=== SAE 特征基础统计 ===")
        print(f"总样本数：{n_total}")
        print(f"  正确轨迹：{n_cor} ({100 * p_cor:.1f}%)")
        print(f"  错误轨迹：{n_wro} ({100 * p_wro:.1f}%)")
        print(f"Majority baseline：{majority:.1%}")
        print()

        all_z = _encode_all_z(sae, h_all, device, batch_size=int(args.batch_size))
        z_np_f = all_z.float().numpy()
        active = z_np_f > 0
        l0 = active.sum(axis=1).mean()
        nz_vals = z_np_f[active]
        mean_nz = float(nz_vals.mean()) if nz_vals.size > 0 else 0.0
        ever_active = active.any(axis=0)
        n_dead = int((~ever_active).sum())
        freq_all = active.mean(axis=0)
        n_high = int((freq_all > 0.5).sum())

        print("特征激活统计（全量）：")
        print(f"  平均 L0：{l0:.2f}")
        print(f"  平均激活强度（非零）：{mean_nz:.4f}")
        print(f"  死特征数（从未激活）：{n_dead} / {sae_dim} ({100 * n_dead / sae_dim:.2f}%)")
        print(
            f"  高频特征数（激活频率 > 50%）：{n_high} / {sae_dim} ({100 * n_high / sae_dim:.2f}%)"
        )
        print()

        if not has_ctx:
            print(
                "[Note] metadata lacks token_str/context; Part 3 prints use position/question_id only.\n"
            )

        # ---------- Part 2: importance ----------
        mask_c = y_np == 1
        mask_w = y_np == 0
        mean_c = np.zeros(sae_dim, dtype=np.float64)
        mean_w = np.zeros(sae_dim, dtype=np.float64)
        freq_c = np.zeros(sae_dim, dtype=np.float64)
        freq_w = np.zeros(sae_dim, dtype=np.float64)
        for i in range(sae_dim):
            col = z_np_f[:, i]
            if mask_c.any():
                mean_c[i] = col[mask_c].mean()
                freq_c[i] = (col[mask_c] > 0).mean()
            if mask_w.any():
                mean_w[i] = col[mask_w].mean()
                freq_w[i] = (col[mask_w] > 0).mean()

        importance = np.abs(mean_c - mean_w)
        order = np.argsort(-importance)
        top_n = min(int(args.top_n), sae_dim)
        top_ids = order[:top_n].tolist()

        clf_acc = _classifier_accuracy(clf, all_z.float(), labels_t, device, args.batch_size)

        print(f"=== Top-{top_n} 推理相关特征 ===")
        print(
            f"{'排名':>4} {'特征ID':>8} {'importance':>12} {'方向':>10} "
            f"{'mean(C)':>10} {'mean(W)':>10} {'freq(C)':>10} {'freq(W)':>10}"
        )
        top_rows: list[dict[str, Any]] = []
        directions: list[str] = []
        for rank, j in enumerate(top_ids, start=1):
            imp = float(importance[j])
            mc, mw = float(mean_c[j]), float(mean_w[j])
            fc, fw = float(freq_c[j]), float(freq_w[j])
            direction = "correct↑" if mc > mw else "wrong↑"
            directions.append(direction)
            print(
                f"{rank:4d} {j:8d} {imp:12.4f} {direction:>10} "
                f"{mc:10.4f} {mw:10.4f} {fc:10.4f} {fw:10.4f}"
            )
            top_rows.append(
                {
                    "rank": rank,
                    "feature_id": int(j),
                    "importance": imp,
                    "direction": direction,
                    "mean_correct": mc,
                    "mean_wrong": mw,
                    "freq_correct": fc,
                    "freq_wrong": fw,
                }
            )
        print()

        importance_payload = {
            "top_features": top_rows,
            "majority_baseline": float(majority),
            "classifier_accuracy": float(clf_acc),
            "total_features": int(sae_dim),
            "dead_features": int(n_dead),
            "mean_l0": float(l0),
            "mean_activation_nonzero": mean_nz,
        }
        with (out_dir / "importance.json").open("w", encoding="utf-8") as f:
            json.dump(importance_payload, f, indent=2, ensure_ascii=False)
        print(f"Saved: {(out_dir / 'importance.json').resolve()}\n")

        # ---------- Part 3 ----------
        ctx_n = int(args.context_examples)
        for rank, feat_idx in enumerate(top_ids, start=1):
            col = all_z[:, feat_idx].float()
            vals, top_idx = torch.topk(col, k=min(ctx_n, n_total))
            zero_mask = col == 0
            zero_positions = torch.nonzero(zero_mask, as_tuple=False).flatten()
            zero_take = zero_positions[:5].tolist()

            direction = directions[rank - 1]
            print(f"=== 特征 {feat_idx} 分析（重要性排名 #{rank}，方向：{direction}）===\n")
            print("激活最强的样例：")
            top_list: list[dict[str, Any]] = []
            for k in range(vals.numel()):
                sid = int(top_idx[k].item())
                act = float(vals[k].item())
                ic = bool(labels_t[sid].item() == 1)
                m = meta[sid] if sid < len(meta) else {}
                line = f"  [{k+1}] 激活强度={act:.4f} | {'正确' if ic else '错误'}轨迹"
                if has_ctx:
                    tok = m.get("token_str", "")
                    ctx = m.get("context", "")
                    line += f" | token={tok!r} | 上下文：{ctx}"
                    top_list.append(
                        {
                            "sample_idx": sid,
                            "activation": act,
                            "is_correct": ic,
                            "token_str": tok,
                            "context": ctx,
                        }
                    )
                else:
                    line += (
                        f" | question_id={m.get('question_id')} traj_id={m.get('traj_id')} "
                        f"pos={m.get('position')} entropy={m.get('entropy')}"
                    )
                    top_list.append(
                        {
                            "sample_idx": sid,
                            "activation": act,
                            "is_correct": ic,
                            "question_id": m.get("question_id"),
                            "traj_id": m.get("traj_id"),
                            "position": m.get("position"),
                            "entropy": m.get("entropy"),
                        }
                    )
                print(line)

            print("\n不激活的反例（激活=0），最多 5 条：")
            zero_list: list[dict[str, Any]] = []
            if not zero_take:
                print("  （该特征在所有样本上均无精确为 0 的激活，或无可列样本）")
            for k, sid in enumerate(zero_take):
                ic = bool(labels_t[sid].item() == 1)
                m = meta[sid] if sid < len(meta) else {}
                line = f"  [{k+1}] {'正确' if ic else '错误'}轨迹"
                if has_ctx:
                    tok = m.get("token_str", "")
                    ctx = m.get("context", "")
                    line += f" | token={tok!r} | 上下文：{ctx}"
                    zero_list.append(
                        {
                            "sample_idx": sid,
                            "is_correct": ic,
                            "token_str": tok,
                            "context": ctx,
                        }
                    )
                else:
                    line += (
                        f" | question_id={m.get('question_id')} traj_id={m.get('traj_id')} "
                        f"pos={m.get('position')}"
                    )
                    zero_list.append(
                        {
                            "sample_idx": sid,
                            "is_correct": ic,
                            "question_id": m.get("question_id"),
                            "traj_id": m.get("traj_id"),
                            "position": m.get("position"),
                        }
                    )
                print(line)
            print()

            feat_json = {
                "feature_id": int(feat_idx),
                "rank": rank,
                "direction": direction,
                "importance": float(importance[feat_idx]),
                "top_activations": top_list,
                "zero_activation_examples": zero_list,
            }
            with (out_dir / f"feature_{feat_idx}_examples.json").open("w", encoding="utf-8") as f:
                json.dump(feat_json, f, indent=2, ensure_ascii=False)

        # ---------- Part 4: figures ----------
        imp_thresh = float(np.sort(importance)[-top_n]) if top_n > 0 else 0.0

        fig, axes = plt.subplots(1, 2, figsize=(12, 4.5))
        ax = axes[0]
        ax.hist(importance, bins=80, color="steelblue", edgecolor="white", alpha=0.85)
        ax.axvline(imp_thresh, color="red", linestyle="--", linewidth=1.5, label=f"top-{top_n} threshold")
        ax.set_xlabel("Importance |mean(C)-mean(W)|")
        ax.set_ylabel("Count")
        ax.set_title("Distribution of feature importance")
        ax.legend()

        ax2 = axes[1]
        colors = ["#1f77b4" if d.startswith("correct") else "#d62728" for d in directions]
        ids_plot = top_ids
        imps_plot = [float(importance[i]) for i in ids_plot]
        y_pos = np.arange(len(ids_plot))
        ax2.barh(y_pos, imps_plot, color=colors)
        ax2.set_yticks(y_pos, labels=[str(i) for i in ids_plot])
        ax2.invert_yaxis()
        ax2.set_xlabel("Importance")
        ax2.set_ylabel("Feature ID")
        ax2.set_title(f"Top-{top_n} features (blue=correct↑, red=wrong↑)")
        fig.tight_layout()
        p1 = out_dir / "importance_distribution.png"
        fig.savefig(p1, dpi=150)
        plt.close(fig)
        print(f"Saved figure: {p1.resolve()}")

        # PCA triple（同一子样本 idx，便于对比）
        h_np = h_all.numpy()
        rng = np.random.default_rng(args.seed)
        n = h_np.shape[0]
        m = min(2000, n)
        idx = rng.choice(n, size=m, replace=False)
        Xz = z_np_f[idx]
        Xzt = z_np_f[idx][:, top_ids]
        Xh = h_np[idx]
        y_sub = y_np[idx]

        def _one_pca_plot(ax: Any, X: np.ndarray, title_prefix: str) -> None:
            pca = PCA(n_components=2, random_state=args.seed)
            Z2 = pca.fit_transform(X)
            sil = _silhouette_subsample(X, y_sub, max_samples=min(3000, X.shape[0]), seed=args.seed)
            r1, r2 = pca.explained_variance_ratio_
            c1 = y_sub == 1
            ax.scatter(Z2[c1, 0], Z2[c1, 1], s=6, alpha=0.35, c="blue", label="correct")
            ax.scatter(Z2[~c1, 0], Z2[~c1, 1], s=6, alpha=0.35, c="red", label="wrong")
            ax.set_xlabel(f"PC1 ({r1:.1%} var)")
            ax.set_ylabel(f"PC2 ({r2:.1%} var)")
            ax.set_title(f"{title_prefix}\nsilhouette={sil:.3f}")
            ax.legend(markerscale=2, fontsize=8)

        _one_pca_plot(axes[0], Xz, "z (all features)")
        _one_pca_plot(axes[1], Xzt, f"z (top-{top_n})")
        _one_pca_plot(axes[2], Xh, "h (raw hidden)")
        fig.tight_layout()
        p2 = out_dir / "feature_pca.png"
        fig.savefig(p2, dpi=150)
        plt.close(fig)
        print(f"Saved figure: {p2.resolve()}")

        # Heatmap 100 samples: 50 correct + 50 wrong
        rng = np.random.default_rng(args.seed + 1)
        idx_c = np.flatnonzero(y_np == 1)
        idx_w = np.flatnonzero(y_np == 0)
        take_c = rng.choice(idx_c, size=min(50, len(idx_c)), replace=False)
        take_w = rng.choice(idx_w, size=min(50, len(idx_w)), replace=False)
        idx_hm = np.concatenate([take_c, take_w])
        z_hm = z_np_f[idx_hm][:, top_ids]

        fig, ax = plt.subplots(figsize=(10, 12))
        if sns is not None:
            sns.heatmap(
                z_hm,
                ax=ax,
                cmap="RdBu_r",
                center=0.0,
                xticklabels=[str(i) for i in top_ids],
                yticklabels=False,
                cbar_kws={"label": "activation"},
            )
        else:
            im = ax.imshow(z_hm, aspect="auto", cmap="RdBu_r", interpolation="nearest")
            ax.set_xticks(range(len(top_ids)))
            ax.set_xticklabels([str(i) for i in top_ids], rotation=45, ha="right")
            plt.colorbar(im, ax=ax, fraction=0.02, pad=0.02)
        ax.axhline(len(take_c), color="black", linewidth=2)
        ax.set_xlabel(f"Top-{top_n} features (by importance)")
        ax.set_ylabel("Samples (correct top, wrong bottom)")
        ax.set_title("Activation heatmap (50 correct + 50 wrong)")
        fig.tight_layout()
        p3 = out_dir / "activation_heatmap.png"
        fig.savefig(p3, dpi=150)
        plt.close(fig)
        print(f"Saved figure: {p3.resolve()}")

        p4 = out_dir / "feature_freq_scatter.png"
        fig, ax = plt.subplots(figsize=(7, 7))
        ax.scatter(freq_c, freq_w, s=4, alpha=0.2, c="gray")
        ax.scatter(
            freq_c[top_ids],
            freq_w[top_ids],
            s=100,
            c="darkorange",
            edgecolors="black",
            linewidths=0.6,
            zorder=5,
            label=f"top-{top_n}",
        )
        ax.plot([0, 1], [0, 1], "k--", alpha=0.5, label="y=x")
        ax.set_xlabel("freq(C)")
        ax.set_ylabel("freq(W)")
        ax.set_title("Feature frequency scatter (deviation from diagonal = separability)")
        ax.legend()
        ax.set_xlim(-0.02, 1.02)
        ax.set_ylim(-0.02, 1.02)
        fig.tight_layout()
        fig.savefig(p4, dpi=150)
        plt.close(fig)
        print(f"Saved figure: {p4.resolve()}")

    finally:
        sys.stdout = _orig_stdout
        summary_f.close()

    print(f"Summary written: {summary_path.resolve()}")


if __name__ == "__main__":
    main()
