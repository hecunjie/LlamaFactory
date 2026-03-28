# Copyright 2025 the LlamaFactory team.
#
# Licensed under the Apache License, Version 2.0.

from __future__ import annotations

import json
import os
from typing import TYPE_CHECKING, Any, Optional

import torch
import torch.nn.functional as F
from transformers import TrainerCallback, TrainerControl, TrainerState
from transformers.training_args import TrainingArguments

from ...extras.constants import IGNORE_INDEX
from ...extras.logging import get_logger


if TYPE_CHECKING:
    from ...hparams import FinetuningArguments


logger = get_logger(__name__)


def _unwrap_model(model: "torch.nn.Module") -> "torch.nn.Module":
    m = model
    while hasattr(m, "module"):
        m = m.module
    return m


def _resolve_final_norm(unwrapped: "torch.nn.Module") -> Optional["torch.nn.Module"]:
    # PEFT: underlying CausalLM lives at base_model.model
    if hasattr(unwrapped, "base_model") and unwrapped.base_model is not None:
        bm = unwrapped.base_model
        if hasattr(bm, "model") and bm.model is not None:
            inner = _resolve_final_norm(bm.model)
            if inner is not None:
                return inner
    if hasattr(unwrapped, "model") and unwrapped.model is not None:
        inner = unwrapped.model
        if hasattr(inner, "norm"):
            return inner.norm
        if hasattr(inner, "model") and inner.model is not None and hasattr(inner.model, "norm"):
            return inner.model.norm
    if hasattr(unwrapped, "transformer") and unwrapped.transformer is not None:
        if hasattr(unwrapped.transformer, "ln_f"):
            return unwrapped.transformer.ln_f
    if hasattr(unwrapped, "language_model") and unwrapped.language_model is not None:
        lm = unwrapped.language_model
        if hasattr(lm, "model") and lm.model is not None and hasattr(lm.model, "norm"):
            return lm.model.norm
    return None


def _compute_metrics(
    logits: "torch.Tensor",
    max_cos: "torch.Tensor",
    eps: float = 1e-12,
) -> tuple["torch.Tensor", "torch.Tensor", "torch.Tensor"]:
    logits_fp = torch.nan_to_num(logits.float(), nan=0.0, posinf=1e4, neginf=-1e4)
    p = F.softmax(logits_fp, dim=-1)
    H = -(p * torch.log(p.clamp_min(eps))).sum(dim=-1)
    max_logit = logits_fp.max(dim=-1).values
    return H, max_cos.to(dtype=torch.float32), max_logit


def _max_cosine_vs_vocab(
    hidden_ln: "torch.Tensor",
    lm_weight: "torch.Tensor",
    seq_chunk: int,
) -> "torch.Tensor":
    W = lm_weight.float()
    W_norm = F.normalize(W, dim=-1, eps=1e-12)
    b, s, _ = hidden_ln.shape
    out = torch.empty(b, s, device=hidden_ln.device, dtype=torch.float32)
    h_float = hidden_ln.float()
    for start in range(0, s, seq_chunk):
        sl = slice(start, min(start + seq_chunk, s))
        h_norm = F.normalize(h_float[:, sl, :], dim=-1, eps=1e-12)
        cos_blk = torch.matmul(h_norm, W_norm.T)
        out[:, sl] = cos_blk.max(dim=-1).values
    return out


def _group_stats(
    delta_h: "torch.Tensor",
    delta_cos: "torch.Tensor",
    delta_logit: "torch.Tensor",
    group_mask: "torch.Tensor",
) -> dict[str, Any]:
    if not group_mask.any():
        return {"count": 0, "delta_H": {"mean": 0.0, "std": 0.0}, "delta_cos": {"mean": 0.0, "std": 0.0}, "delta_logit": {"mean": 0.0, "std": 0.0}}
    dh = delta_h[group_mask].float()
    dc = delta_cos[group_mask].float()
    dl = delta_logit[group_mask].float()
    return {
        "count": int(group_mask.sum().item()),
        "delta_H": {"mean": round(dh.mean().item(), 6), "std": round(dh.std(unbiased=False).item(), 6)},
        "delta_cos": {"mean": round(dc.mean().item(), 6), "std": round(dc.std(unbiased=False).item(), 6)},
        "delta_logit": {"mean": round(dl.mean().item(), 6), "std": round(dl.std(unbiased=False).item(), 6)},
    }


class LogitsAnalysisCallback(TrainerCallback):
    r"""Snapshots token-wise H, max cosine (vs lm_head), max logit before/after optimizer.step."""

    def __init__(self, finetuning_args: "FinetuningArguments") -> None:
        self.finetuning_args = finetuning_args
        self._trainer: Any = None
        self._hook_handle: Any = None
        self._hidden_cache: dict[str, "torch.Tensor"] = {}
        self._metrics_before: Optional[dict[str, "torch.Tensor"]] = None
        self._warned_no_norm: bool = False

    def attach_trainer(self, trainer: Any) -> None:
        self._trainer = trainer

    def _hook_fn(self, module: Any, inp: Any, output: Any) -> None:
        if isinstance(output, tuple):
            t = output[0]
        else:
            t = output
        self._hidden_cache["last_ln"] = t.detach()

    def on_train_begin(
        self,
        args: TrainingArguments,
        state: TrainerState,
        control: TrainerControl,
        **kwargs: Any,
    ) -> TrainerControl:
        if self._trainer is None:
            logger.warning_rank0("[logits_analysis] Callback has no trainer reference; analysis disabled.")
            return control
        model = self._trainer.accelerator.unwrap_model(self._trainer.model, keep_torch_compile=False)
        unwrapped = _unwrap_model(model)
        norm_mod = _resolve_final_norm(unwrapped)
        if norm_mod is None:
            if not self._warned_no_norm:
                logger.warning_rank0("[logits_analysis] Could not resolve final LayerNorm/RMSNorm; analysis disabled.")
                self._warned_no_norm = True
            return control
        self._hook_handle = norm_mod.register_forward_hook(self._hook_fn)
        return control

    def on_train_end(
        self,
        args: TrainingArguments,
        state: TrainerState,
        control: TrainerControl,
        **kwargs: Any,
    ) -> TrainerControl:
        if self._hook_handle is not None:
            self._hook_handle.remove()
            self._hook_handle = None
        return control

    def on_pre_optimizer_step(
        self,
        args: TrainingArguments,
        state: TrainerState,
        control: TrainerControl,
        **kwargs: Any,
    ) -> TrainerControl:
        self._metrics_before = None
        if self._trainer is None:
            return control
        batch = getattr(self._trainer, "_logits_analysis_batch", None)
        if batch is None:
            return control
        model = self._trainer.accelerator.unwrap_model(self._trainer.model, keep_torch_compile=False)
        unwrapped = _unwrap_model(model)
        out_emb = unwrapped.get_output_embeddings()
        if out_emb is None:
            return control
        lm_w = out_emb.weight

        self._hidden_cache.clear()
        fwd: dict[str, Any] = {
            "input_ids": batch["input_ids"],
            "attention_mask": batch["attention_mask"],
            "use_cache": False,
        }
        with torch.no_grad():
            outputs = self._trainer.model(**fwd)
            logits = outputs.logits
            hidden = self._hidden_cache.get("last_ln")
            if hidden is None:
                return control
            max_cos = _max_cosine_vs_vocab(
                hidden,
                lm_w,
                seq_chunk=max(1, int(self.finetuning_args.logits_analysis_cosine_seq_chunk)),
            )
            H, mc, ml = _compute_metrics(logits, max_cos)
        self._metrics_before = {"H": H, "max_cos": mc, "max_logit": ml}
        return control

    def on_optimizer_step(
        self,
        args: TrainingArguments,
        state: TrainerState,
        control: TrainerControl,
        **kwargs: Any,
    ) -> TrainerControl:
        if self._trainer is None:
            return control
        if getattr(self._trainer.accelerator, "optimizer_step_was_skipped", False):
            self._metrics_before = None
            return control
        before = self._metrics_before
        self._metrics_before = None
        if before is None:
            return control
        batch = getattr(self._trainer, "_logits_analysis_batch", None)
        if batch is None:
            return control

        model = self._trainer.accelerator.unwrap_model(self._trainer.model, keep_torch_compile=False)
        unwrapped = _unwrap_model(model)
        out_emb = unwrapped.get_output_embeddings()
        if out_emb is None:
            return control
        lm_w = out_emb.weight

        self._hidden_cache.clear()
        fwd = {
            "input_ids": batch["input_ids"],
            "attention_mask": batch["attention_mask"],
            "use_cache": False,
        }
        with torch.no_grad():
            outputs = self._trainer.model(**fwd)
            logits = outputs.logits
            hidden = self._hidden_cache.get("last_ln")
            if hidden is None:
                return control
            max_cos = _max_cosine_vs_vocab(
                hidden,
                lm_w,
                seq_chunk=max(1, int(self.finetuning_args.logits_analysis_cosine_seq_chunk)),
            )
            H_a, mc_a, ml_a = _compute_metrics(logits, max_cos)

        H_b, mc_b, ml_b = before["H"], before["max_cos"], before["max_logit"]
        attn = batch["attention_mask"].bool()
        labels = batch.get("labels")
        mask = attn
        if labels is not None:
            mask = mask & labels.ne(IGNORE_INDEX)

        dH = H_a - H_b
        dcos = mc_a - mc_b
        dl = ml_a - ml_b

        th_e = float(self.finetuning_args.logits_analysis_threshold_entropy)
        th_c = float(self.finetuning_args.logits_analysis_threshold_cosine)
        high_H = H_b > th_e
        low_cos = mc_b < th_c
        g_a = high_H & low_cos & mask
        g_b = high_H & (~low_cos) & mask
        g_c = (~high_H) & low_cos & mask
        g_d = (~high_H) & (~low_cos) & mask

        completed_step = int(state.global_step) + 1
        log_every = max(1, int(self.finetuning_args.logits_analysis_log_every_n_steps))
        if completed_step % log_every != 0:
            return control
        if not self._trainer.is_world_process_zero():
            return control

        record = {
            "step": completed_step,
            "group_A": _group_stats(dH, dcos, dl, g_a),
            "group_B": _group_stats(dH, dcos, dl, g_b),
            "group_C": _group_stats(dH, dcos, dl, g_c),
            "group_D": _group_stats(dH, dcos, dl, g_d),
        }
        out_dir = self.finetuning_args.logits_analysis_output_path
        os.makedirs(out_dir, exist_ok=True)
        path = os.path.join(out_dir, f"analysis_step_{completed_step}_{completed_step}.jsonl")
        with open(path, "a", encoding="utf-8") as f:
            f.write(json.dumps(record, ensure_ascii=False) + "\n")
        return control
