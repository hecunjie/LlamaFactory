"""
HiddenStateDataset：从轨迹 JSONL 中提取「整题统一分位数」下的高熵位置 hidden states。

实现说明见 cursor_prompt/train_sqe.md；tokenize 策略遵循文档「重要实现细节」：
对已有格式化字符串 `prompt` 直接 encode（add_special_tokens=False），再拼接 response 的 token ids。
"""

from __future__ import annotations

import json
import sys
import warnings
from pathlib import Path
from typing import Any, Optional

import numpy as np
import torch
from torch.utils.data import DataLoader, Dataset, WeightedRandomSampler

from . import config as cfg

# 复用实验工具：设备与答案比较
_ROOT = Path(__file__).resolve().parent.parent
if str(_ROOT) not in sys.path:
    sys.path.insert(0, str(_ROOT))
from experiment.utils import answers_equal, model_input_device  # noqa: E402


def _parse_boxed(text: str) -> Optional[str]:
    """提取 \\boxed{X} 内容，支持嵌套花括号。"""
    if not isinstance(text, str) or not text:
        return None
    key = r"\boxed{"
    idx = 0
    while True:
        start = text.find(key, idx)
        if start < 0:
            return None
        i = start + len(key)
        depth = 1
        j = i
        while j < len(text) and depth > 0:
            c = text[j]
            if c == "{":
                depth += 1
            elif c == "}":
                depth -= 1
            j += 1
        if depth == 0:
            inner = text[i : j - 1].strip()
            return inner if inner else None
        idx = start + 1


def _answers_equal(pred: Optional[str], gold: Optional[str]) -> bool:
    """数值比较（容差 1e-3）+ 字符串比较。"""
    return answers_equal(pred, gold)


def _forward_batch_entropy_hidden(
    llm_model: torch.nn.Module,
    device: torch.device,
    layer_index: int,
    full_id_rows: list[torch.Tensor],
    prompt_len: int,
) -> list[tuple[torch.Tensor, torch.Tensor]]:
    """
    同一道题内多条轨迹一次 batch forward。

    full_id_rows: 每条为 shape (1, L) 的 LongTensor（CPU 即可，内部一次性拷到 device）。
    返回与 full_id_rows 等长的列表，每项为 (ent_cpu, h_resp_cpu)，无效序列为空张量。
    """
    hidden_dim = int(llm_model.config.hidden_size)
    empty_ent = torch.tensor([], dtype=torch.float32)
    empty_h = torch.empty(0, hidden_dim, dtype=torch.float32)

    if not full_id_rows:
        return []

    B = len(full_id_rows)
    lens = [int(t.shape[1]) for t in full_id_rows]
    max_len = max(lens)
    input_ids = torch.zeros(B, max_len, dtype=torch.long, device=device)
    attention_mask = torch.zeros(B, max_len, dtype=torch.long, device=device)
    for i, row in enumerate(full_id_rows):
        L = int(row.shape[1])
        input_ids[i, :L] = row.view(-1).to(device, non_blocking=True)
        attention_mask[i, :L] = 1

    with torch.inference_mode():
        out = llm_model(
            input_ids=input_ids,
            attention_mask=attention_mask,
            output_hidden_states=True,
            return_dict=True,
        )
    logits = out.logits
    hs_tuple = out.hidden_states
    if hs_tuple is None:
        raise RuntimeError("模型未返回 hidden_states，请确认 forward 参数。")
    layer_h = hs_tuple[layer_index]

    results: list[tuple[torch.Tensor, torch.Tensor]] = []
    P = int(prompt_len)
    for i, L in enumerate(lens):
        if L <= P:
            results.append((empty_ent, empty_h))
            continue
        # 仅使用真实长度 L 内的 logits / hidden，忽略右侧 padding
        resp_logits = logits[i, P - 1 : L - 1, :].float()
        resp_len = L - P

        probs = torch.softmax(resp_logits, dim=-1)
        log_probs = torch.log(probs + 1e-10)
        ent = -(probs * log_probs).sum(dim=-1).cpu().float()

        h_resp = layer_h[i, P : P + resp_len, :].float().cpu()
        n = min(int(ent.shape[0]), int(h_resp.shape[0]))
        results.append((ent[:n], h_resp[:n]))

    return results


def _tokenize_prompt_and_full_rows(
    tokenizer: Any,
    prompt: str,
    predicts: list[Any],
) -> tuple[int, list[Optional[torch.Tensor]]]:
    """
    对同一 prompt 只 tokenize 一次；为每条 predict 构造 full_ids (1, L)（CPU LongTensor）。

    返回 (prompt_len, full_rows)。full_rows 与 predicts 等长：
      - 非 str 的 predict：占位 None（调用方应 continue，不写入 per_traj）
      - 无法构成有效序列：None（调用方写入空轨迹）
      - 否则：shape (1, L) 的 LongTensor
    """
    prompt_enc = tokenizer(
        prompt,
        add_special_tokens=False,
        return_tensors="pt",
    )
    prompt_t = prompt_enc["input_ids"].cpu()
    if prompt_t.numel() == 0:
        return 0, [None] * len(predicts)
    P = int(prompt_t.shape[1])
    rows: list[Optional[torch.Tensor]] = []
    for pred_text in predicts:
        if not isinstance(pred_text, str):
            rows.append(None)
            continue
        if not pred_text:
            rows.append(None)
            continue
        resp = tokenizer(
            pred_text,
            add_special_tokens=False,
            return_tensors="pt",
        )["input_ids"].cpu()
        if resp.numel() == 0:
            rows.append(None)
            continue
        full = torch.cat([prompt_t, resp], dim=1)
        if int(full.shape[1]) <= P:
            rows.append(None)
        else:
            rows.append(full)
    return P, rows


class HiddenStateDataset(Dataset):
    """
    每个缓存样本对应一个高熵位置的 L2 归一化 hidden vector。

    __getitem__ 返回：
      hidden_state: (hidden_dim,) float32
      is_correct: 0/1
      entropy: float
    """

    def __init__(
        self,
        data_path: str,
        llm_model: Optional[torch.nn.Module],
        tokenizer: Optional[Any],
        config: Any = None,
        force_extract: bool = False,
    ):
        self.config = config or cfg
        self.data_path = Path(data_path)
        cache_root = Path(self.config.CACHE_DIR)
        cache_root.mkdir(parents=True, exist_ok=True)
        self._cache_pt = cache_root / "hidden_states.pt"
        self._cache_meta_json = cache_root / "metadata.json"

        self.hidden_states: torch.Tensor
        self.labels: torch.Tensor
        self.entropies: torch.Tensor
        self._metadata: list[dict[str, Any]]

        use_cache = self._cache_pt.is_file() and not force_extract

        if use_cache:
            print(f"[dataset] 从缓存加载: {self._cache_pt.resolve()}")
            try:
                blob = torch.load(self._cache_pt, map_location="cpu", weights_only=False)
            except TypeError:
                blob = torch.load(self._cache_pt, map_location="cpu")
            self.hidden_states = blob["hidden_states"]
            self.labels = blob["labels"]
            self.entropies = blob["entropies"]
            self._metadata = blob.get("metadata", [])
        else:
            if llm_model is None or tokenizer is None:
                raise ValueError("无缓存且未提供 LLM/tokenizer，无法提取 hidden states。")
            self._extract_all(llm_model, tokenizer)
            self._save_cache()

        self.print_stats()

    def _save_cache(self) -> None:
        torch.save(
            {
                "hidden_states": self.hidden_states,
                "labels": self.labels,
                "entropies": self.entropies,
                "metadata": self._metadata,
            },
            self._cache_pt,
        )
        with self._cache_meta_json.open("w", encoding="utf-8") as f:
            json.dump(self._metadata, f, ensure_ascii=False, indent=2)
        print(f"[dataset] 已写入缓存: {self._cache_pt.resolve()}")
        print(f"[dataset] 元数据 JSON: {self._cache_meta_json.resolve()}")

    def _extract_all(self, llm_model: torch.nn.Module, tokenizer: Any) -> None:
        """遍历 JSONL，按题聚合熵分位数后提取高熵位置 hidden states。"""
        rows = self._load_jsonl(self.data_path)
        dev = model_input_device(llm_model)
        layer_index = self.config.HIDDEN_LAYER_INDEX
        pctl = float(self.config.ENTROPY_PERCENTILE)
        max_pred = int(self.config.MAX_PREDICTS_PER_SAMPLE)

        all_hs: list[torch.Tensor] = []
        all_labels: list[int] = []
        all_entropies: list[float] = []
        all_meta: list[dict[str, Any]] = []

        hidden_dim = int(llm_model.config.hidden_size)
        empty_every = int(getattr(self.config, "EMPTY_CACHE_EVERY", 50))

        for qid, row in enumerate(rows):
            if qid > 0 and qid % int(self.config.EXTRACT_PROGRESS_EVERY) == 0:
                print(f"[dataset] 已处理题目进度: {qid}/{len(rows)}")

            try:
                prompt = row.get("prompt", "")
                label = row.get("label", "")
                gold = _parse_boxed(label)
                predicts = row.get("predicts") or []
                if not isinstance(predicts, list):
                    predicts = []
                predicts = predicts[:max_pred]

                prompt_len_p, full_rows = _tokenize_prompt_and_full_rows(
                    tokenizer, prompt, predicts
                )
                if prompt_len_p == 0:
                    continue

                # 本题所有有效序列一次 batch forward（通常远快于逐条 forward）
                batch_pairs: list[tuple[int, torch.Tensor]] = []
                for tid, pred_text in enumerate(predicts):
                    if not isinstance(pred_text, str):
                        continue
                    fr = full_rows[tid]
                    if fr is not None:
                        batch_pairs.append((tid, fr))

                tid_to_eh: dict[int, tuple[torch.Tensor, torch.Tensor]] = {}
                if batch_pairs:
                    outs = _forward_batch_entropy_hidden(
                        llm_model,
                        dev,
                        layer_index,
                        [p[1] for p in batch_pairs],
                        prompt_len_p,
                    )
                    for j, (tid, _) in enumerate(batch_pairs):
                        tid_to_eh[tid] = outs[j]

                pool: list[float] = []
                per_traj: list[dict[str, Any]] = []
                for tid, pred_text in enumerate(predicts):
                    if not isinstance(pred_text, str):
                        continue
                    pa = _parse_boxed(pred_text)
                    correct = _answers_equal(pa, gold)
                    fr = full_rows[tid]
                    if fr is None:
                        per_traj.append(
                            {
                                "tid": tid,
                                "text": pred_text,
                                "ent": torch.tensor([], dtype=torch.float32),
                                "h": torch.empty(0, hidden_dim, dtype=torch.float32),
                                "correct": correct,
                            }
                        )
                        continue

                    ent, h_resp = tid_to_eh[tid]
                    if ent.numel() == 0:
                        per_traj.append(
                            {
                                "tid": tid,
                                "text": pred_text,
                                "ent": torch.tensor([], dtype=torch.float32),
                                "h": h_resp,
                                "correct": correct,
                            }
                        )
                        continue

                    pool.extend(ent.numpy().astype(np.float64, copy=False).ravel().tolist())
                    per_traj.append(
                        {
                            "tid": tid,
                            "text": pred_text,
                            "ent": ent,
                            "h": h_resp,
                            "correct": correct,
                        }
                    )

                if not pool:
                    continue

                thr = float(np.percentile(np.asarray(pool, dtype=np.float64), pctl))

                # 取高熵位置的 hidden，L2 归一化（向量化筛选）
                for info in per_traj:
                    ent = info["ent"]
                    h_resp = info["h"]
                    if ent.numel() == 0 or h_resp.numel() == 0:
                        continue
                    n = min(int(ent.shape[0]), int(h_resp.shape[0]))
                    e_np = ent[:n].numpy()
                    mask = e_np >= thr
                    if not np.any(mask):
                        continue
                    idxs = np.flatnonzero(mask)
                    h_slice = h_resp[:n]
                    for pos in idxs:
                        e = float(e_np[pos])
                        h = h_slice[int(pos)].clone()
                        h = h / (h.norm(p=2) + 1e-8)
                        all_hs.append(h)
                        all_labels.append(1 if info["correct"] else 0)
                        all_entropies.append(e)
                        all_meta.append(
                            {
                                "question_id": qid,
                                "traj_id": int(info["tid"]),
                                "position": int(pos),
                                "entropy": e,
                                "threshold": thr,
                                "is_correct": bool(info["correct"]),
                            }
                        )
            except Exception as e:  # noqa: BLE001
                warnings.warn(f"题目 {qid} 提取失败，已跳过: {e}", UserWarning, stacklevel=1)
            finally:
                if (
                    torch.cuda.is_available()
                    and str(dev).startswith("cuda")
                    and empty_every > 0
                    and qid > 0
                    and qid % empty_every == 0
                ):
                    torch.cuda.empty_cache()

        if not all_hs:
            self.hidden_states = torch.empty(0, 0, dtype=torch.float32)
            self.labels = torch.empty(0, dtype=torch.long)
            self.entropies = torch.empty(0, dtype=torch.float32)
            self._metadata = []
            return

        self.hidden_states = torch.stack(all_hs, dim=0).to(dtype=torch.float32)
        self.labels = torch.tensor(all_labels, dtype=torch.long)
        self.entropies = torch.tensor(all_entropies, dtype=torch.float32)
        self._metadata = all_meta

    @staticmethod
    def _load_jsonl(path: Path) -> list[dict[str, Any]]:
        out: list[dict[str, Any]] = []
        with path.open("r", encoding="utf-8") as f:
            for line in f:
                line = line.strip()
                if not line:
                    continue
                out.append(json.loads(line))
        return out

    def print_stats(self) -> None:
        """打印数据集规模、正误比例、熵统计等。"""
        n = len(self)
        print("\n[HiddenStateDataset] 统计信息")
        print(f"  总样本数: {n}")
        if n == 0:
            return
        n_cor = int((self.labels == 1).sum().item())
        n_wro = n - n_cor
        print(f"  正确样本: {n_cor} ({100.0 * n_cor / n:.2f}%)")
        print(f"  错误样本: {n_wro} ({100.0 * n_wro / n:.2f}%)")

        qids = {m["question_id"] for m in self._metadata}
        traj_keys = {(m["question_id"], m["traj_id"]) for m in self._metadata}
        print(f"  题目数: {len(qids)}")
        print(f"  轨迹条数（题内去重）: {len(traj_keys)}")
        per_q: dict[int, int] = {}
        for m in self._metadata:
            per_q[m["question_id"]] = per_q.get(m["question_id"], 0) + 1
        if per_q:
            avg_sp = float(np.mean(list(per_q.values())))
            print(f"  每道题平均样本数: {avg_sp:.2f}")

        e = self.entropies.numpy()
        print(f"  熵 mean/median/std: {e.mean():.4f} / {np.median(e):.4f} / {e.std():.4f}")
        print()

    def __len__(self) -> int:
        return int(self.hidden_states.shape[0])

    def __getitem__(self, idx: int) -> dict[str, torch.Tensor]:
        return {
            "hidden_state": self.hidden_states[idx],
            "is_correct": self.labels[idx],
            "entropy": self.entropies[idx],
        }


def get_dataloader(
    dataset: HiddenStateDataset,
    config: Any = None,
    shuffle: bool = True,
) -> DataLoader:
    """
    DataLoader；shuffle=True 时使用 WeightedRandomSampler 近似平衡正负样本。
    """
    config = config or cfg
    if len(dataset) == 0:
        raise RuntimeError("数据集为空，无法构建 DataLoader。")

    labels = dataset.labels
    n_cor = int((labels == 1).sum().item())
    n_wro = int((labels == 0).sum().item())
    w_cor = 1.0 / max(n_cor, 1)
    w_wro = 1.0 / max(n_wro, 1)
    w_list = [w_cor if int(labels[i]) == 1 else w_wro for i in range(len(dataset))]
    weights = torch.tensor(w_list, dtype=torch.double)

    if shuffle:
        sampler = WeightedRandomSampler(
            weights,
            num_samples=len(dataset),
            replacement=True,
        )
        return DataLoader(
            dataset,
            batch_size=int(config.BATCH_SIZE),
            sampler=sampler,
            num_workers=0,
            pin_memory=torch.cuda.is_available(),
        )

    return DataLoader(
        dataset,
        batch_size=int(config.BATCH_SIZE),
        shuffle=False,
        num_workers=0,
        pin_memory=torch.cuda.is_available(),
    )
