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

# Cap total elements cloned for param delta (avoid OOM on full finetuning).
_MAX_PARAM_SNAPSHOT_NUMEL = 150_000_000


def _ordered_trainable_named_params(unwrapped: "torch.nn.Module") -> list[tuple[str, "torch.nn.Parameter"]]:
    r"""LoRA / adapter tensors first so snapshot budget is not eaten by huge embeddings."""
    named = [(n, p) for n, p in unwrapped.named_parameters() if p.requires_grad]
    lora = [(n, p) for n, p in named if "lora" in n.lower()]
    non = [(n, p) for n, p in named if "lora" not in n.lower()]
    return lora + non


def _build_param_snapshot_pairs(
    ordered: list[tuple[str, "torch.nn.Parameter"]],
) -> list[tuple[str, "torch.Tensor"]]:
    r"""Clone trainable tensors up to global numel budget.

    Skip any single tensor larger than the budget (e.g. full ``embed_tokens``), otherwise the
    first huge param caused ``break`` with an **empty** list and ``max_abs_trainable_param_delta``
    stayed ``null`` while LoRA weights were never snapshotted.
    """
    pairs: list[tuple[str, "torch.Tensor"]] = []
    total_numel = 0
    for name, p in ordered:
        n = p.numel()
        if n > _MAX_PARAM_SNAPSHOT_NUMEL:
            continue
        if total_numel + n > _MAX_PARAM_SNAPSHOT_NUMEL:
            break
        pairs.append((name, p.detach().cpu().clone()))
        total_numel += n
    return pairs


def _max_abs_param_delta(
    unwrapped: "torch.nn.Module",
    snapshot_pairs: list[tuple[str, "torch.Tensor"]],
) -> float:
    name_to_p = dict(unwrapped.named_parameters())
    mx = 0.0
    for name, cold in snapshot_pairs:
        p = name_to_p.get(name)
        if p is None:
            continue
        d = (p.detach().float() - cold.float().to(device=p.device)).abs().max().item()
        mx = max(mx, float(d))
    return mx


def _sync_device_after_optimizer(device: "torch.device") -> None:
    r"""Ensure optimizer.step() has finished updating parameters before the next forward.

    Without this, CUDA/MPS async execution can make the post-step forward read stale
    weights, yielding zero delta_H / delta_cos / delta_logit.
    """
    if device.type == "cuda":
        torch.cuda.synchronize(device)
    elif device.type == "mps" and getattr(torch.backends, "mps", None) is not None and torch.backends.mps.is_available():
        torch.mps.synchronize()


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


def _stat_mean_std(t: "torch.Tensor") -> dict[str, float]:
    r"""Serialize mean/std without squashing tiny updates: round(..., 6) was turning ~1e-7 into 0.0."""
    m = float(t.mean().item())
    s = float(t.std(unbiased=False).item())
    return {"mean": m, "std": s}


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
        "delta_H": _stat_mean_std(dh),
        "delta_cos": _stat_mean_std(dc),
        "delta_logit": _stat_mean_std(dl),
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
        self._param_snapshot_pairs: list[tuple[str, "torch.Tensor"]] = []
        self._optimizer_lr_before_step: Optional[float] = None

    def attach_trainer(self, trainer: Any) -> None:
        self._trainer = trainer

    def _run_analysis_forward(
        self,
        fwd: dict[str, Any],
        lm_w: "torch.Tensor",
    ) -> Optional[tuple["torch.Tensor", "torch.Tensor", "torch.Tensor", float, float]]:
        r"""Eval + no_grad: deterministic in weight (no dropout), then restore train mode.

        Also returns fp64 logits fingerprints (sum, max abs) to verify weight updates are visible
        even when bf16 weights round-trip identically.
        """
        model = self._trainer.model
        was_training = model.training
        model.eval()
        try:
            self._hidden_cache.clear()
            with torch.no_grad():
                outputs = model(**fwd)
                logits = outputs.logits
                logits_fp64_sum = float(logits.detach().double().sum().item())
                logits_fp64_absmax = float(logits.detach().double().abs().max().item())
                hidden = self._hidden_cache.get("last_ln")
                if hidden is None:
                    return None
                max_cos = _max_cosine_vs_vocab(
                    hidden,
                    lm_w,
                    seq_chunk=max(1, int(self.finetuning_args.logits_analysis_cosine_seq_chunk)),
                )
                H, mc, ml = _compute_metrics(logits, max_cos)
                return H, mc, ml, logits_fp64_sum, logits_fp64_absmax
        finally:
            if was_training:
                model.train()

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

        opt = getattr(self._trainer, "optimizer", None)
        if opt is not None and len(opt.param_groups) > 0:
            self._optimizer_lr_before_step = float(opt.param_groups[0]["lr"])
        else:
            self._optimizer_lr_before_step = None

        model = self._trainer.accelerator.unwrap_model(self._trainer.model, keep_torch_compile=False)
        unwrapped = _unwrap_model(model)
        out_emb = unwrapped.get_output_embeddings()
        if out_emb is None:
            return control
        lm_w = out_emb.weight

        ordered = _ordered_trainable_named_params(unwrapped)
        self._param_snapshot_pairs = _build_param_snapshot_pairs(ordered)

        fwd: dict[str, Any] = {
            "input_ids": batch["input_ids"],
            "attention_mask": batch["attention_mask"],
            "use_cache": False,
        }
        out = self._run_analysis_forward(fwd, lm_w)
        if out is None:
            return control
        H, mc, ml, ls0, lmx0 = out
        self._metrics_before = {
            "H": H.detach().clone(),
            "max_cos": mc.detach().clone(),
            "max_logit": ml.detach().clone(),
            "logits_fp64_sum": ls0,
            "logits_fp64_absmax": lmx0,
        }
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

        _sync_device_after_optimizer(batch["input_ids"].device)

        model = self._trainer.accelerator.unwrap_model(self._trainer.model, keep_torch_compile=False)
        unwrapped = _unwrap_model(model)
        out_emb = unwrapped.get_output_embeddings()
        if out_emb is None:
            return control
        lm_w = out_emb.weight

        n_param_snapshots = len(self._param_snapshot_pairs)
        param_delta_max: Optional[float] = None
        if self._param_snapshot_pairs:
            param_delta_max = _max_abs_param_delta(unwrapped, self._param_snapshot_pairs)
        self._param_snapshot_pairs = []

        fwd = {
            "input_ids": batch["input_ids"],
            "attention_mask": batch["attention_mask"],
            "use_cache": False,
        }
        out2 = self._run_analysis_forward(fwd, lm_w)
        if out2 is None:
            return control
        H_a, mc_a, ml_a, ls1, lmx1 = out2

        H_b, mc_b, ml_b = before["H"], before["max_cos"], before["max_logit"]
        ls0 = float(before["logits_fp64_sum"])
        lmx0 = float(before["logits_fp64_absmax"])
        attn = batch["attention_mask"].bool()
        labels = batch.get("labels")
        mask = attn
        if labels is not None:
            mask = mask & labels.ne(IGNORE_INDEX)

        dH = H_a - H_b
        dcos = mc_a - mc_b
        dl = ml_a - ml_b

        if mask.any():
            mean_abs_dl = float(dl[mask].float().abs().mean().item())
        else:
            mean_abs_dl = 0.0

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

        trainable_n = sum(1 for _, p in unwrapped.named_parameters() if p.requires_grad)
        record: dict[str, Any] = {
            "step": completed_step,
            "optimizer_lr": self._optimizer_lr_before_step,
            "trainable_named_param_count": trainable_n,
            "param_snapshot_tensor_count": n_param_snapshots,
            "max_abs_trainable_param_delta": param_delta_max,
            "mean_abs_delta_logit_masked": mean_abs_dl,
            "logits_fp64_sum_before": ls0,
            "logits_fp64_sum_after": ls1,
            "logits_fp64_sum_delta": ls1 - ls0,
            "logits_fp64_absmax_before": lmx0,
            "logits_fp64_absmax_after": lmx1,
            "logits_fp64_absmax_delta": lmx1 - lmx0,
            "group_A": _group_stats(dH, dcos, dl, g_a),
            "group_B": _group_stats(dH, dcos, dl, g_b),
            "group_C": _group_stats(dH, dcos, dl, g_c),
            "group_D": _group_stats(dH, dcos, dl, g_d),
        }
        self._optimizer_lr_before_step = None
        out_dir = self.finetuning_args.logits_analysis_output_path
        os.makedirs(out_dir, exist_ok=True)
        path = os.path.join(out_dir, f"analysis_step_{completed_step}_{completed_step}.jsonl")
        with open(path, "a", encoding="utf-8") as f:
            f.write(json.dumps(record, ensure_ascii=False) + "\n")
        return control
