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
    r"""Synchronize the device stream (CUDA/MPS).

    Call **after** ``optimizer.step()`` so the next forward reads updated weights; also call
    **before** the pre-step analysis forward so backward kernels have finished before we
    snapshot parameters and run extra eval forwards.

    Without this, async execution can make forwards read stale weights or gradients, yielding
    zero ``delta_*`` metrics that look like "no training effect".
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


def _select_focus_token_ids(
    logits_row: "torch.Tensor",
    mode: str,
    topk: int,
    prob_threshold: float,
    max_tokens: int,
) -> "torch.Tensor":
    r"""Return vocab ids to track at one position: top-k by prob or prob >= threshold (capped)."""
    logits_fp = logits_row.float()
    probs = F.softmax(logits_fp, dim=-1)
    v = int(probs.numel())
    if mode == "topk":
        k = max(1, min(int(topk), v))
        _, ids = probs.topk(k)
        return ids.long()
    # prob_threshold
    mask = probs >= prob_threshold
    ids = mask.nonzero(as_tuple=True)[0]
    if ids.numel() == 0:
        _, j = probs.max(dim=-1)
        return j.long().unsqueeze(0)
    if ids.numel() > max_tokens:
        sub = probs[ids]
        _, order = sub.topk(max_tokens)
        ids = ids[order]
    return ids.long()


def _fp32_logits_dots_at_ids(
    h: "torch.Tensor",
    W: "torch.Tensor",
    ids: "torch.Tensor",
    bias: Optional["torch.Tensor"],
) -> "torch.Tensor":
    r"""Replicate LM head at token ids: dot(h, W[k]) + bias[k] in float32 (no bf16 matmul rounding)."""
    dots = (W[ids].float() * h.float().unsqueeze(0)).sum(dim=-1)
    if bias is not None:
        dots = dots + bias[ids].float()
    return dots


def _build_logits_focus_snapshot(
    logits: "torch.Tensor",
    mask: "torch.Tensor",
    hidden: "torch.Tensor",
    lm_w: "torch.Tensor",
    bias: Optional["torch.Tensor"],
    finetuning_args: Any,
) -> list[dict[str, Any]]:
    r"""Per masked position: pick token ids, store model logits and fp32 LM-head dots (pre-step h, W)."""
    max_pos = int(getattr(finetuning_args, "logits_analysis_max_positions", 256))
    mode = str(getattr(finetuning_args, "logits_analysis_focus_mode", "topk"))
    topk = int(getattr(finetuning_args, "logits_analysis_focus_topk", 10))
    pth = float(getattr(finetuning_args, "logits_analysis_focus_prob_threshold", 0.01))
    max_tok = int(getattr(finetuning_args, "logits_analysis_focus_max_tokens", 32))
    idx = mask.nonzero(as_tuple=False)
    if max_pos > 0 and idx.size(0) > max_pos:
        idx = idx[:max_pos]
    out: list[dict[str, Any]] = []
    for i in range(idx.size(0)):
        b, s = int(idx[i, 0].item()), int(idx[i, 1].item())
        row = logits[b, s, :]
        ids = _select_focus_token_ids(row, mode, topk, pth, max_tok)
        ids_d = ids.to(device=logits.device)
        lb_model = row[ids_d].detach().float().cpu()
        h_vec = hidden[b, s]
        dots_b = _fp32_logits_dots_at_ids(h_vec, lm_w, ids_d, bias).detach().cpu()
        out.append(
            {
                "b": b,
                "s": s,
                "token_ids": ids.cpu().tolist(),
                "logits_model_before": lb_model.tolist(),
                "logits_fp32_dot_before": dots_b.tolist(),
            }
        )
    return out


def _complete_logits_focus_after(
    logits_after: "torch.Tensor",
    hidden_after: "torch.Tensor",
    lm_w: "torch.Tensor",
    bias: Optional["torch.Tensor"],
    focus_before: list[dict[str, Any]],
) -> tuple[list[dict[str, Any]], Optional[float], Optional[float], Optional[float], Optional[float]]:
    r"""After step: model logits + fp32 dots; compare to stored before. Returns (per, mean_abs_model, max_abs_model, mean_abs_fp32, max_abs_fp32)."""
    per: list[dict[str, Any]] = []
    flat_abs_model: list["torch.Tensor"] = []
    flat_abs_fp32: list["torch.Tensor"] = []
    for ent in focus_before:
        b, s = ent["b"], ent["s"]
        ids = torch.tensor(ent["token_ids"], device=logits_after.device, dtype=torch.long)
        la_model = logits_after[b, s, ids].detach().float()
        lb_model = torch.tensor(ent["logits_model_before"], dtype=torch.float32, device=logits_after.device)
        delta_model = la_model - lb_model
        flat_abs_model.append(delta_model.abs())

        h_a = hidden_after[b, s]
        dots_a = _fp32_logits_dots_at_ids(h_a, lm_w, ids, bias)
        dots_b = torch.tensor(ent["logits_fp32_dot_before"], dtype=torch.float32, device=logits_after.device)
        delta_fp32 = dots_a - dots_b
        flat_abs_fp32.append(delta_fp32.abs())

        per.append(
            {
                "b": b,
                "s": s,
                "token_ids": ent["token_ids"],
                "logits_model_before": ent["logits_model_before"],
                "logits_model_after": la_model.cpu().tolist(),
                "delta_logits_model": delta_model.cpu().tolist(),
                "logits_fp32_dot_before": ent["logits_fp32_dot_before"],
                "logits_fp32_dot_after": dots_a.detach().cpu().tolist(),
                "delta_logits_fp32_dot": delta_fp32.detach().cpu().tolist(),
            }
        )
    if not flat_abs_model:
        return per, None, None, None, None
    cat_m = torch.cat([x.flatten() for x in flat_abs_model])
    cat_f = torch.cat([x.flatten() for x in flat_abs_fp32])
    return (
        per,
        float(cat_m.mean().item()),
        float(cat_m.max().item()),
        float(cat_f.mean().item()),
        float(cat_f.max().item()),
    )


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
        self._pre_step_diag: dict[str, Any] = {}

    def attach_trainer(self, trainer: Any) -> None:
        self._trainer = trainer

    def _run_analysis_forward(
        self,
        fwd: dict[str, Any],
        lm_w: "torch.Tensor",
    ) -> Optional[tuple["torch.Tensor", "torch.Tensor", "torch.Tensor", float, float, "torch.Tensor", "torch.Tensor"]]:
        r"""Eval + no_grad: deterministic in weight (no dropout), then restore train mode.

        Returns ``logits`` and final-norm ``hidden`` (for fp32 LM-head dot checks vs bf16 ``outputs.logits``).
        """
        # Always forward through the *wrapped* model (trainer.model), not the bare inner module.
        #
        # Reason: framework wrappers (DeepSpeed, FSDP) own the "live" parameter tensors.
        #   • With DeepSpeed ZeRO, gradients are stored internally (p.grad is set to None after
        #     accumulation) and optimizer.step() writes back to the module through DS internals.
        #     Calling unwrapped_inner(**fwd) bypasses DS, so the forward sees pre-step weights
        #     BOTH before and after optimizer.step() → delta is always 0.
        #   • With FSDP ZeRO-3, full parameters are only materialised during a forward that goes
        #     through the FSDP wrapper; calling the bare model directly uses the local shard.
        # The forward hook registered on norm_mod fires regardless because trainer.model internally
        # calls through to the bare submodule where the hook lives.
        #
        # torch.compile: wrapped is the compiled model. Hooks on submodules still fire in eager
        # fallback; compile mode generally preserves forward hooks unless the module is inlined.
        # If hooks stop firing (hidden_cache["last_ln"] is None), this returns None and analysis
        # is skipped for that step (no crash).
        wrapped = self._trainer.model
        was_training = wrapped.training
        wrapped.eval()
        try:
            self._hidden_cache.clear()
            with torch.no_grad():
                outputs = wrapped(**fwd)
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
                return H, mc, ml, logits_fp64_sum, logits_fp64_absmax, logits, hidden
        finally:
            if was_training:
                wrapped.train()

    def _hook_fn(self, module: Any, inp: Any, output: Any) -> None:
        if isinstance(output, tuple):
            t = output[0]
        else:
            t = output
        # Clone so we never alias activations that may be reused in-place on the next forward.
        self._hidden_cache["last_ln"] = t.detach().clone()

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

        _sync_device_after_optimizer(batch["input_ids"].device)

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
        lm_bias: Optional["torch.Tensor"] = getattr(out_emb, "bias", None)

        ordered = _ordered_trainable_named_params(unwrapped)
        self._param_snapshot_pairs = _build_param_snapshot_pairs(ordered)

        # ── Diagnostics: snapshot multiple param elements in fp32 before step ──
        _sample_grad_norm: Optional[float] = None
        _sample_param_mean_abs: Optional[float] = None
        _direct_param_name: Optional[str] = None
        _direct_param_fp32_before: Optional[float] = None
        _multi_param_fp32_before: list[tuple[str, int, float]] = []
        for name, p in ordered[:min(5, len(ordered))]:
            if _direct_param_name is None:
                _direct_param_name = name
                _direct_param_fp32_before = float(p.detach().float().flatten()[0].item())
            n_elem = min(3, p.numel())
            flat = p.detach().float().flatten()
            for ei in range(n_elem):
                _multi_param_fp32_before.append((name, ei, float(flat[ei].item())))
            if p.grad is not None and _sample_grad_norm is None:
                _sample_grad_norm = float(p.grad.detach().float().norm().item())
                _sample_param_mean_abs = float(p.detach().float().abs().mean().item())

        # Probe DeepSpeed engine structure to find fp32 master weights.
        # HF Trainer: self.model = bare model, self.model_wrapped = DS engine.
        _ds_fp32_before: Optional[float] = None
        _ds_engine_type: Optional[str] = None
        _ds_optimizer_type: Optional[str] = None
        _ds_fp32_attr_found: Optional[str] = None
        _model_wrapped_type: Optional[str] = None
        _model_is_model_wrapped: Optional[bool] = None

        _ds_engine = getattr(self._trainer, "model_wrapped", self._trainer.model)
        _ds_engine_type = type(_ds_engine).__name__
        _model_wrapped_type = _ds_engine_type
        _model_is_model_wrapped = (_ds_engine is self._trainer.model)

        ds_opt = getattr(_ds_engine, "optimizer", None)
        if ds_opt is not None:
            _ds_optimizer_type = type(ds_opt).__name__
            for attr_name in [
                "fp32_partitioned_groups_flat",
                "single_partition_of_fp32_groups",
                "_fp32_partitioned_groups_flat",
                "fp32_groups_flat",
                "bit16_groups",
                "fp32_groups",
                "param_groups_fp32",
            ]:
                obj = getattr(ds_opt, attr_name, None)
                if obj is not None:
                    _ds_fp32_attr_found = attr_name
                    if isinstance(obj, (list, tuple)) and len(obj) > 0:
                        first = obj[0]
                        if hasattr(first, "flatten"):
                            _ds_fp32_before = float(first.flatten()[0].item())
                    break

        self._pre_step_diag = {
            "sample_grad_norm": _sample_grad_norm,
            "sample_param_mean_abs": _sample_param_mean_abs,
            "direct_param_name": _direct_param_name,
            "direct_param_fp32_before": _direct_param_fp32_before,
            "multi_param_fp32_before": _multi_param_fp32_before,
            "ds_fp32_before": _ds_fp32_before,
            "ds_engine_type": _ds_engine_type,
            "ds_optimizer_type": _ds_optimizer_type,
            "ds_fp32_attr_found": _ds_fp32_attr_found,
            "model_wrapped_type": _model_wrapped_type,
            "model_is_model_wrapped": _model_is_model_wrapped,
        }

        # CRITICAL: Do NOT run a forward pass here (before optimizer.step).
        # Under DeepSpeed, calling engine.forward() in on_pre_optimizer_step
        # resets internal gradient buffers, causing optimizer.step() to see
        # zero gradients and produce zero parameter updates.
        # Instead, we only snapshot parameters here and run BOTH "before" and
        # "after" forward passes in on_optimizer_step (after the step is done).
        # The "before" forward uses the param snapshot to reconstruct logits
        # via fp32 dot products; the "after" forward uses live (updated) params.
        self._metrics_before = {"placeholder": True}
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
        lm_bias = getattr(out_emb, "bias", None)

        n_param_snapshots = len(self._param_snapshot_pairs)
        param_delta_max: Optional[float] = None
        if self._param_snapshot_pairs:
            param_delta_max = _max_abs_param_delta(unwrapped, self._param_snapshot_pairs)
        self._param_snapshot_pairs = []

        completed_step = int(state.global_step) + 1
        log_every = max(1, int(self.finetuning_args.logits_analysis_log_every_n_steps))
        if completed_step % log_every != 0:
            return control
        if not self._trainer.is_world_process_zero():
            return control

        trainable_n = sum(1 for _, p in unwrapped.named_parameters() if p.requires_grad)

        diag = self._pre_step_diag
        sample_grad_norm = diag.get("sample_grad_norm")
        sample_param_mean_abs = diag.get("sample_param_mean_abs")
        bf16_ulp_estimate: Optional[float] = None
        if sample_param_mean_abs is not None:
            bf16_ulp_estimate = sample_param_mean_abs / 128.0

        # ── Re-read params after optimizer.step() and compare ──────────────────
        _direct_param_name = diag.get("direct_param_name")
        _direct_param_fp32_before = diag.get("direct_param_fp32_before")
        _name_to_p = dict(unwrapped.named_parameters())

        _direct_param_fp32_after: Optional[float] = None
        if _direct_param_name is not None:
            _p_direct = _name_to_p.get(_direct_param_name)
            if _p_direct is not None:
                _direct_param_fp32_after = float(_p_direct.detach().float().flatten()[0].item())
        _direct_param_fp32_delta: Optional[float] = None
        if _direct_param_fp32_before is not None and _direct_param_fp32_after is not None:
            _direct_param_fp32_delta = _direct_param_fp32_after - _direct_param_fp32_before

        _multi_before = diag.get("multi_param_fp32_before", [])
        _multi_deltas: list[dict[str, Any]] = []
        for pname, eidx, val_before in _multi_before:
            _pp = _name_to_p.get(pname)
            if _pp is not None and eidx < _pp.numel():
                val_after = float(_pp.detach().float().flatten()[eidx].item())
                _multi_deltas.append({
                    "name": pname, "idx": eidx,
                    "before": val_before, "after": val_after,
                    "delta": val_after - val_before,
                })

        # DeepSpeed fp32 master weight check — use model_wrapped (the real DS engine)
        _ds_fp32_before = diag.get("ds_fp32_before")
        _ds_fp32_attr_found = diag.get("ds_fp32_attr_found")
        _ds_fp32_after: Optional[float] = None
        _ds_fp32_delta: Optional[float] = None
        _ds_engine = getattr(self._trainer, "model_wrapped", self._trainer.model)
        if _ds_fp32_attr_found is not None:
            ds_opt = getattr(_ds_engine, "optimizer", None)
            if ds_opt is not None:
                obj = getattr(ds_opt, _ds_fp32_attr_found, None)
                if obj is not None and isinstance(obj, (list, tuple)) and len(obj) > 0:
                    first = obj[0]
                    if hasattr(first, "flatten"):
                        _ds_fp32_after = float(first.flatten()[0].item())
        if _ds_fp32_before is not None and _ds_fp32_after is not None:
            _ds_fp32_delta = _ds_fp32_after - _ds_fp32_before

        _callback_called_at = {
            "pre_optimizer_step": True,
            "optimizer_step": True,
            "global_step_at_post": int(state.global_step),
            "no_pre_step_forward": True,
        }

        # Run a single "after" forward to get post-step logits (no before/after comparison
        # this round — we first need to confirm params actually change without the extra forward).
        fwd = {
            "input_ids": batch["input_ids"],
            "attention_mask": batch["attention_mask"],
            "use_cache": False,
        }
        out_after = self._run_analysis_forward(fwd, lm_w)
        ls1: Optional[float] = None
        lmx1: Optional[float] = None
        if out_after is not None:
            _, _, _, ls1, lmx1, _, _ = out_after

        self._pre_step_diag = {}

        record: dict[str, Any] = {
            "step": completed_step,
            "optimizer_lr": self._optimizer_lr_before_step,
            "trainable_named_param_count": trainable_n,
            "param_snapshot_tensor_count": n_param_snapshots,
            "max_abs_trainable_param_delta": param_delta_max,
            "diag_sample_grad_norm": sample_grad_norm,
            "diag_sample_param_mean_abs": sample_param_mean_abs,
            "diag_bf16_ulp_estimate": bf16_ulp_estimate,
            "diag_direct_param_name": _direct_param_name,
            "diag_direct_param_fp32_before": _direct_param_fp32_before,
            "diag_direct_param_fp32_after": _direct_param_fp32_after,
            "diag_direct_param_fp32_delta": _direct_param_fp32_delta,
            "diag_multi_param_deltas": _multi_deltas[:10],
            "diag_model_wrapped_type": diag.get("model_wrapped_type"),
            "diag_model_is_model_wrapped": diag.get("model_is_model_wrapped"),
            "diag_ds_engine_type": diag.get("ds_engine_type"),
            "diag_ds_optimizer_type": diag.get("ds_optimizer_type"),
            "diag_ds_fp32_attr_found": diag.get("ds_fp32_attr_found"),
            "diag_ds_fp32_master_before": _ds_fp32_before,
            "diag_ds_fp32_master_after": _ds_fp32_after,
            "diag_ds_fp32_master_delta": _ds_fp32_delta,
            "diag_callback_timing": _callback_called_at,
            "diag_note": (
                "HYPOTHESIS TEST: pre-step forward REMOVED. If multi_param_deltas now show "
                "non-zero values, the extra engine.forward() in on_pre_optimizer_step was "
                "destroying DeepSpeed's accumulated gradients before optimizer.step(). "
                "If deltas are STILL 0.0, the problem is elsewhere."
            ),
            "logits_fp64_sum_after": ls1,
            "logits_fp64_absmax_after": lmx1,
        }
        self._optimizer_lr_before_step = None
        out_dir = self.finetuning_args.logits_analysis_output_path
        os.makedirs(out_dir, exist_ok=True)
        path = os.path.join(out_dir, f"analysis_step_{completed_step}_{completed_step}.jsonl")
        with open(path, "a", encoding="utf-8") as f:
            f.write(json.dumps(record, ensure_ascii=False) + "\n")
        return control
