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


def _logits_analysis_phase_dir(base: str, phase: str) -> str:
    r"""``logits_analysis_output_path``/train or .../eval (``phase`` is ``\"train\"`` or ``\"eval\"``)."""
    root = os.path.normpath(os.path.expanduser(base.strip()))
    return os.path.join(root, phase)


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
    r"""Per masked position: pick token ids, store **fp32** logits at ids and fp32 LM-head dots.

    ``logits`` must be float32 (cast from ``outputs.logits``) so deltas are not bf16-grid quantized.
    """
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
        row = logits[b, s, :].float()
        ids = _select_focus_token_ids(row, mode, topk, pth, max_tok)
        ids_d = ids.to(device=logits.device)
        lb_fp32 = row[ids_d].detach().cpu()
        h_vec = hidden[b, s]
        dots_b = _fp32_logits_dots_at_ids(h_vec, lm_w, ids_d, bias).detach().cpu()
        out.append(
            {
                "b": b,
                "s": s,
                "token_ids": ids.cpu().tolist(),
                "logits_fp32_before": lb_fp32.tolist(),
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
    r"""After step: **fp32** logits at ids vs stored before; LM-head dots unchanged.

    ``logits_after`` must be float32. Returns (per, mean_abs_logits, max_abs_logits, mean_abs_dot, max_abs_dot).
    """
    per: list[dict[str, Any]] = []
    flat_abs_logits: list["torch.Tensor"] = []
    flat_abs_dot: list["torch.Tensor"] = []
    for ent in focus_before:
        b, s = ent["b"], ent["s"]
        ids = torch.tensor(ent["token_ids"], device=logits_after.device, dtype=torch.long)
        row_a = logits_after[b, s, ids].detach().float()
        raw_before = ent.get("logits_fp32_before")
        if raw_before is None:
            raw_before = ent.get("logits_model_before")
        lb = torch.tensor(raw_before, dtype=torch.float32, device=logits_after.device)
        delta_logits = row_a - lb
        flat_abs_logits.append(delta_logits.abs())

        h_a = hidden_after[b, s]
        dots_a = _fp32_logits_dots_at_ids(h_a, lm_w, ids, bias)
        dots_b = torch.tensor(ent["logits_fp32_dot_before"], dtype=torch.float32, device=logits_after.device)
        delta_dot = dots_a - dots_b
        flat_abs_dot.append(delta_dot.abs())

        per.append(
            {
                "b": b,
                "s": s,
                "token_ids": ent["token_ids"],
                "logits_fp32_before": ent.get("logits_fp32_before", ent.get("logits_model_before")),
                "logits_fp32_after": row_a.cpu().tolist(),
                "delta_logits_fp32": delta_logits.cpu().tolist(),
                "logits_fp32_dot_before": ent["logits_fp32_dot_before"],
                "logits_fp32_dot_after": dots_a.detach().cpu().tolist(),
                "delta_logits_fp32_dot": delta_dot.detach().cpu().tolist(),
            }
        )
    if not flat_abs_logits:
        return per, None, None, None, None
    cat_l = torch.cat([x.flatten() for x in flat_abs_logits])
    cat_d = torch.cat([x.flatten() for x in flat_abs_dot])
    return (
        per,
        float(cat_l.mean().item()),
        float(cat_l.max().item()),
        float(cat_d.mean().item()),
        float(cat_d.max().item()),
    )


def _group_stats(
    delta_h: "torch.Tensor",
    delta_cos: "torch.Tensor",
    delta_logit: "torch.Tensor",
    group_mask: "torch.Tensor",
) -> dict[str, Any]:
    if not group_mask.any():
        z = {"mean": 0.0, "std": 0.0, "mean_abs": 0.0}
        return {
            "count": 0,
            "delta_entropy": z.copy(),
            "delta_max_cosine": z.copy(),
            "delta_max_logit": z.copy(),
        }
    dh = delta_h[group_mask].float()
    dc = delta_cos[group_mask].float()
    dl = delta_logit[group_mask].float()
    return {
        "count": int(group_mask.sum().item()),
        "delta_entropy": {**_stat_mean_std(dh), "mean_abs": float(dh.abs().mean().item())},
        "delta_max_cosine": {**_stat_mean_std(dc), "mean_abs": float(dc.abs().mean().item())},
        "delta_max_logit": {**_stat_mean_std(dl), "mean_abs": float(dl.abs().mean().item())},
    }


def _masked_delta_summary(
    delta_h: "torch.Tensor",
    delta_cos: "torch.Tensor",
    delta_logit: "torch.Tensor",
    mask: "torch.Tensor",
) -> dict[str, Any]:
    r"""All supervised positions: mean/std/mean_abs of ΔH, Δcos, Δmax_logit."""
    if not mask.any():
        z = {"mean": 0.0, "std": 0.0, "mean_abs": 0.0}
        return {"token_count": 0, "delta_entropy": z, "delta_max_cosine": z, "delta_max_logit": z}
    dh = delta_h[mask].float()
    dc = delta_cos[mask].float()
    dl = delta_logit[mask].float()
    return {
        "token_count": int(mask.sum().item()),
        "delta_entropy": {**_stat_mean_std(dh), "mean_abs": float(dh.abs().mean().item())},
        "delta_max_cosine": {**_stat_mean_std(dc), "mean_abs": float(dc.abs().mean().item())},
        "delta_max_logit": {**_stat_mean_std(dl), "mean_abs": float(dl.abs().mean().item())},
    }


def _group_value_stats(
    H: "torch.Tensor",
    max_cos: "torch.Tensor",
    max_logit: "torch.Tensor",
    group_mask: "torch.Tensor",
) -> dict[str, Any]:
    r"""Per cluster: mean/std/mean_abs of **static** H, max_cos, max_logit (eval snapshot)."""
    if not group_mask.any():
        z = {"mean": 0.0, "std": 0.0, "mean_abs": 0.0}
        return {
            "count": 0,
            "entropy": z.copy(),
            "max_cosine": z.copy(),
            "max_logit": z.copy(),
        }
    h = H[group_mask].float()
    c = max_cos[group_mask].float()
    l = max_logit[group_mask].float()
    return {
        "count": int(group_mask.sum().item()),
        "entropy": {**_stat_mean_std(h), "mean_abs": float(h.abs().mean().item())},
        "max_cosine": {**_stat_mean_std(c), "mean_abs": float(c.abs().mean().item())},
        "max_logit": {**_stat_mean_std(l), "mean_abs": float(l.abs().mean().item())},
    }


def _masked_value_summary(
    H: "torch.Tensor",
    max_cos: "torch.Tensor",
    max_logit: "torch.Tensor",
    mask: "torch.Tensor",
) -> dict[str, Any]:
    r"""All supervised positions: mean/std/mean_abs of H, max_cos, max_logit (no delta)."""
    if not mask.any():
        z = {"mean": 0.0, "std": 0.0, "mean_abs": 0.0}
        return {"token_count": 0, "entropy": z, "max_cosine": z, "max_logit": z}
    h = H[mask].float()
    c = max_cos[mask].float()
    l = max_logit[mask].float()
    return {
        "token_count": int(mask.sum().item()),
        "entropy": {**_stat_mean_std(h), "mean_abs": float(h.abs().mean().item())},
        "max_cosine": {**_stat_mean_std(c), "mean_abs": float(c.abs().mean().item())},
        "max_logit": {**_stat_mean_std(l), "mean_abs": float(l.abs().mean().item())},
    }


# Cluster labels (high/low entropy × high/low max-cosine vs vocab), for jsonl output.
_CLUSTER_DEFS: tuple[tuple[str, str, str], ...] = (
    ("A", "high_entropy_low_cos", "H > th_e AND max_cos < th_c (hard positions)"),
    ("B", "high_entropy_high_cos", "H > th_e AND max_cos >= th_c"),
    ("C", "low_entropy_low_cos", "H <= th_e AND max_cos < th_c"),
    ("D", "low_entropy_high_cos", "H <= th_e AND max_cos >= th_c (baseline-like)"),
)


class LogitsAnalysisCallback(TrainerCallback):
    r"""Before/after ``accelerator.backward`` snapshots: per-token Δentropy, Δmax_cosine, Δmax_logit.

    Model ``outputs.logits`` are cast to **float32** before entropy, max-logit, and focused-id snapshots
    so deltas are not stuck on the bf16 quantization grid.

    Writes jsonl records keyed by four clusters (A–D). Optional per-position focus when
    ``logits_analysis_save_focus_per_position`` is True.

    When ``logits_analysis_on_eval`` is True, ``record_eval_snapshot`` appends **eval** lines
    (static H / max_cos / max_logit per cluster; no optimizer step, so not train-style Δ).
    """

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
    ) -> Optional[tuple["torch.Tensor", "torch.Tensor", "torch.Tensor", float, float, "torch.Tensor", "torch.Tensor"]]:
        r"""Eval + no_grad: deterministic in weight (no dropout), then restore train mode.

        Casts ``outputs.logits`` to **float32** before entropy / max-logit / focus snapshots so analysis
        is not limited to bf16 representable steps.
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
                logits = outputs.logits.detach().float()
                logits_fp64_sum = float(logits.double().sum().item())
                logits_fp64_absmax = float(logits.double().abs().max().item())
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

    # ── Public API called from training_step (NOT from HF callbacks) ──────────
    #
    # With DeepSpeed + Accelerate, ``accelerator.backward(loss)`` internally calls
    # ``engine.step()`` which runs the optimizer AND zeros gradients.  By the time
    # HF Trainer's ``on_pre_optimizer_step`` / ``on_optimizer_step`` fire, the
    # parameter update has ALREADY happened.  So those callbacks always see
    # identical before/after values.
    #
    # Fix: call ``snapshot_before_backward`` right before ``accelerator.backward``
    # and ``analyze_after_backward`` right after it, directly from
    # ``CustomSeq2SeqTrainer.training_step``.

    def snapshot_before_backward(self, batch: dict[str, Any]) -> None:
        r"""Run the "before" analysis forward and snapshot parameters.

        Must be called **before** ``accelerator.backward(loss)`` in ``training_step``.
        """
        if self._trainer is None:
            return
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
            return
        lm_w = out_emb.weight
        lm_bias: Optional["torch.Tensor"] = getattr(out_emb, "bias", None)

        ordered = _ordered_trainable_named_params(unwrapped)
        self._param_snapshot_pairs = _build_param_snapshot_pairs(ordered)

        fwd: dict[str, Any] = {
            "input_ids": batch["input_ids"],
            "attention_mask": batch["attention_mask"],
            "use_cache": False,
        }
        out = self._run_analysis_forward(fwd, lm_w)
        if out is None:
            self._metrics_before = None
            return
        H, mc, ml, ls0, lmx0, logits, hidden = out
        attn = batch["attention_mask"].bool()
        labels = batch.get("labels")
        mask = attn
        if labels is not None:
            mask = mask & labels.ne(IGNORE_INDEX)
        save_focus = bool(getattr(self.finetuning_args, "logits_analysis_save_focus_per_position", False))
        logits_focus: list[dict[str, Any]] = []
        if save_focus:
            logits_focus = _build_logits_focus_snapshot(logits, mask, hidden, lm_w, lm_bias, self.finetuning_args)

        self._metrics_before = {
            "H": H.detach().clone(),
            "max_cos": mc.detach().clone(),
            "max_logit": ml.detach().clone(),
            "logits_fp64_sum": ls0,
            "logits_fp64_absmax": lmx0,
            "logits_focus": logits_focus,
            "_save_focus_per_position": save_focus,
        }

    def analyze_after_backward(self, batch: dict[str, Any]) -> None:
        r"""Run the "after" analysis forward, compare with snapshot, and write JSON.

        Must be called **after** ``accelerator.backward(loss)`` in ``training_step``.
        Under DeepSpeed, ``accelerator.backward`` also runs ``engine.step()`` so
        parameters are already updated when this method executes.
        """
        if self._trainer is None:
            return
        before = self._metrics_before
        self._metrics_before = None
        if before is None:
            return

        _sync_device_after_optimizer(batch["input_ids"].device)

        model = self._trainer.accelerator.unwrap_model(self._trainer.model, keep_torch_compile=False)
        unwrapped = _unwrap_model(model)
        out_emb = unwrapped.get_output_embeddings()
        if out_emb is None:
            return
        lm_w = out_emb.weight
        lm_bias = getattr(out_emb, "bias", None)

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
            return
        H_a, mc_a, ml_a, ls1, lmx1, logits_a, hidden_a = out2

        H_b, mc_b, ml_b = before["H"], before["max_cos"], before["max_logit"]
        ls0 = float(before["logits_fp64_sum"])
        lmx0 = float(before["logits_fp64_absmax"])
        attn = batch["attention_mask"].bool()
        labels = batch.get("labels")
        mask = attn
        if labels is not None:
            mask = mask & labels.ne(IGNORE_INDEX)

        save_focus = bool(before.get("_save_focus_per_position", False))
        focus_before = before.get("logits_focus") or []
        logits_focus_per_position: list[dict[str, Any]] = []
        logits_focus_mean_abs: Optional[float] = None
        logits_focus_max_abs: Optional[float] = None
        logits_focus_fp32_mean_abs: Optional[float] = None
        logits_focus_fp32_max_abs: Optional[float] = None
        if save_focus and focus_before:
            (
                logits_focus_per_position,
                logits_focus_mean_abs,
                logits_focus_max_abs,
                logits_focus_fp32_mean_abs,
                logits_focus_fp32_max_abs,
            ) = _complete_logits_focus_after(logits_a, hidden_a, lm_w, lm_bias, focus_before)

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

        state = self._trainer.state
        completed_step = int(state.global_step) + 1
        log_every = max(1, int(self.finetuning_args.logits_analysis_log_every_n_steps))
        if completed_step % log_every != 0:
            return
        if not self._trainer.is_world_process_zero():
            return

        trainable_n = sum(1 for _, p in unwrapped.named_parameters() if p.requires_grad)

        groups_clustered: dict[str, Any] = {}
        for (cid, slug, desc), gm in zip(_CLUSTER_DEFS, (g_a, g_b, g_c, g_d)):
            st = _group_stats(dH, dcos, dl, gm)
            groups_clustered[cid] = {
                "name": slug,
                "description": desc,
                **st,
            }

        masked_summary = _masked_delta_summary(dH, dcos, dl, mask)

        record: dict[str, Any] = {
            "step": completed_step,
            "optimizer_lr": self._optimizer_lr_before_step,
            "analysis": "clustered_token_deltas",
            "thresholds": {
                "entropy_high_if_gt": th_e,
                "max_cosine_low_if_lt": th_c,
            },
            "masked_summary": masked_summary,
            "groups": groups_clustered,
            "changes_explained": {
                "delta_entropy": "H_after - H_before; H from softmax of **fp32** logits",
                "delta_max_cosine": "max_k cos(h,w_k) after - before (LN hidden vs lm_head rows)",
                "delta_max_logit": "max_k **fp32** logits after - before (cast from model output)",
            },
            "logits_analysis_dtype": "float32",
            "trainable_named_param_count": trainable_n,
            "param_snapshot_tensor_count": n_param_snapshots,
            "max_abs_trainable_param_delta": param_delta_max,
            "mean_abs_delta_max_logit_masked": mean_abs_dl,
            "logits_fp64_sum_before": ls0,
            "logits_fp64_sum_after": ls1,
            "logits_fp64_sum_delta": ls1 - ls0,
            "logits_fp64_absmax_before": lmx0,
            "logits_fp64_absmax_after": lmx1,
            "logits_fp64_absmax_delta": lmx1 - lmx0,
        }
        if save_focus:
            record["focused_logits"] = {
                "mode": getattr(self.finetuning_args, "logits_analysis_focus_mode", "topk"),
                "positions_n": len(logits_focus_per_position),
                "mean_abs_delta_logits_fp32": logits_focus_mean_abs,
                "max_abs_delta_logits_fp32": logits_focus_max_abs,
                "mean_abs_delta_lm_head_dot_fp32": logits_focus_fp32_mean_abs,
                "max_abs_delta_lm_head_dot_fp32": logits_focus_fp32_max_abs,
                "per_position": logits_focus_per_position,
            }
            record["note_focused_logit_metrics"] = (
                "logits_fp32_* use outputs.logits cast to float32. "
                "lm_head_dot_* is dot(h,W)+bias in float32 (same as before)."
            )
        self._optimizer_lr_before_step = None
        out_dir = _logits_analysis_phase_dir(self.finetuning_args.logits_analysis_output_path, "train")
        os.makedirs(out_dir, exist_ok=True)
        path = os.path.join(out_dir, f"analysis_step_{completed_step}_{completed_step}.jsonl")
        with open(path, "a", encoding="utf-8") as f:
            f.write(json.dumps(record, ensure_ascii=False) + "\n")

    def record_eval_snapshot(
        self,
        batch: dict[str, Any],
        *,
        global_step: int,
        eval_batch_index: int,
        metric_key_prefix: str,
    ) -> None:
        r"""Single eval forward: static H / max_cos / max_logit by A–D (no optimizer step ⇒ no train-style Δ).

        Called from ``CustomSeq2SeqTrainer.prediction_step`` for standard loss eval (not ``predict_with_generate``).
        Appends one line under ``logits_analysis_output_path/eval/`` to
        ``eval_logits_analysis_{metric_key_prefix}_step_{global_step}.jsonl``.
        """
        if self._trainer is None:
            return
        if not bool(getattr(self.finetuning_args, "logits_analysis_on_eval", False)):
            return
        if not self._trainer.is_world_process_zero():
            return
        _sync_device_after_optimizer(batch["input_ids"].device)

        model = self._trainer.accelerator.unwrap_model(self._trainer.model, keep_torch_compile=False)
        unwrapped = _unwrap_model(model)
        out_emb = unwrapped.get_output_embeddings()
        if out_emb is None:
            return
        lm_w = out_emb.weight

        fwd: dict[str, Any] = {
            "input_ids": batch["input_ids"],
            "attention_mask": batch["attention_mask"],
            "use_cache": False,
        }
        out = self._run_analysis_forward(fwd, lm_w)
        if out is None:
            return
        H, mc, ml, ls0, lmx0, _logits, _hidden = out
        del _logits, _hidden

        attn = batch["attention_mask"].bool()
        labels = batch.get("labels")
        mask = attn
        if labels is not None:
            mask = mask & labels.ne(IGNORE_INDEX)

        th_e = float(self.finetuning_args.logits_analysis_threshold_entropy)
        th_c = float(self.finetuning_args.logits_analysis_threshold_cosine)
        high_H = H > th_e
        low_cos = mc < th_c
        g_a = high_H & low_cos & mask
        g_b = high_H & (~low_cos) & mask
        g_c = (~high_H) & low_cos & mask
        g_d = (~high_H) & (~low_cos) & mask

        groups_clustered: dict[str, Any] = {}
        for (cid, slug, desc), gm in zip(_CLUSTER_DEFS, (g_a, g_b, g_c, g_d)):
            st = _group_value_stats(H, mc, ml, gm)
            groups_clustered[cid] = {
                "name": slug,
                "description": desc,
                **st,
            }

        masked_summary = _masked_value_summary(H, mc, ml, mask)

        record: dict[str, Any] = {
            "step": global_step,
            "phase": "eval",
            "metric_key_prefix": metric_key_prefix,
            "eval_batch_index": eval_batch_index,
            "analysis": "eval_cluster_snapshot",
            "thresholds": {
                "entropy_high_if_gt": th_e,
                "max_cosine_low_if_lt": th_c,
            },
            "masked_summary": masked_summary,
            "groups": groups_clustered,
            "values_explained": {
                "entropy": "Static H from softmax of **fp32** logits (single eval forward; not train Δ).",
                "max_cosine": "Static max_k cos(h,w_k) (LN hidden vs lm_head rows).",
                "max_logit": "Static max_k **fp32** logits.",
            },
            "logits_analysis_dtype": "float32",
            "logits_fp64_sum": ls0,
            "logits_fp64_absmax": lmx0,
        }
        out_dir = _logits_analysis_phase_dir(self.finetuning_args.logits_analysis_output_path, "eval")
        os.makedirs(out_dir, exist_ok=True)
        path = os.path.join(
            out_dir,
            f"eval_logits_analysis_{metric_key_prefix}_step_{global_step}.jsonl",
        )
        with open(path, "a", encoding="utf-8") as f:
            f.write(json.dumps(record, ensure_ascii=False) + "\n")

    # ── HF Trainer callbacks (kept as no-ops for compatibility) ─────────────

    def on_pre_optimizer_step(
        self,
        args: TrainingArguments,
        state: TrainerState,
        control: TrainerControl,
        **kwargs: Any,
    ) -> TrainerControl:
        return control

    def on_optimizer_step(
        self,
        args: TrainingArguments,
        state: TrainerState,
        control: TrainerControl,
        **kwargs: Any,
    ) -> TrainerControl:
        return control
