# Copyright 2025 HuggingFace Inc. and the LlamaFactory team.
#
# This code is inspired by the HuggingFace's transformers library.
# https://github.com/huggingface/transformers/blob/v4.40.0/src/transformers/trainer_seq2seq.py
#
# Licensed under the Apache License, Version 2.0 (the "License");
# you may not use this file except in compliance with the License.
# You may obtain a copy of the License at
#
#     http://www.apache.org/licenses/LICENSE-2.0
#
# Unless required by applicable law or agreed to in writing, software
# distributed under the License is distributed on an "AS IS" BASIS,
# WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
# See the License for the specific language governing permissions and
# limitations under the License.

import json
import os
from types import MethodType
from typing import TYPE_CHECKING, Any, Optional, Union

import numpy as np
import torch
from transformers import Seq2SeqTrainer
from typing_extensions import override

from ...extras import logging
from ...extras.constants import IGNORE_INDEX
from ..callbacks import SaveProcessorCallback
from ..fp8_utils import configure_fp8_environment, patch_accelerator_for_fp8, verify_fp8_status
from ..trainer_utils import create_custom_optimizer, create_custom_scheduler


if TYPE_CHECKING:
    from torch.utils.data import Dataset
    from transformers import ProcessorMixin
    from transformers.trainer import PredictionOutput

    from ...hparams import DataArguments, FinetuningArguments, ModelArguments, TrainingArguments


logger = logging.get_logger(__name__)


class CustomSeq2SeqTrainer(Seq2SeqTrainer):
    r"""Inherits Seq2SeqTrainer to compute generative metrics such as BLEU and ROUGE."""

    def __init__(
        self,
        finetuning_args: "FinetuningArguments",
        processor: Optional["ProcessorMixin"],
        model_args: Optional["ModelArguments"] = None,
        gen_kwargs: Optional[dict[str, Any]] = None,
        data_args: Optional["DataArguments"] = None,
        **kwargs,
    ) -> None:
        kwargs["processing_class"] = kwargs.pop("tokenizer")
        # Configure FP8 environment if enabled
        training_args: TrainingArguments = kwargs.get("args")
        if training_args.fp8:
            configure_fp8_environment(training_args)
            if getattr(training_args, "fp8_backend", "auto") == "te":
                patch_accelerator_for_fp8()

        super().__init__(**kwargs)
        if processor is not None:
            # avoid wrong loss under gradient accumulation
            # https://github.com/huggingface/transformers/pull/36044#issuecomment-2746657112
            self.model_accepts_loss_kwargs = False

        self.finetuning_args = finetuning_args
        self.data_args = data_args
        if gen_kwargs is not None:
            # https://github.com/huggingface/transformers/blob/v4.45.0/src/transformers/trainer_seq2seq.py#L287
            self._gen_kwargs = gen_kwargs

        if processor is not None:
            self.add_callback(SaveProcessorCallback(processor))

        if finetuning_args.use_badam:
            from badam import BAdamCallback, clip_grad_norm_old_version  # type: ignore

            self.accelerator.clip_grad_norm_ = MethodType(clip_grad_norm_old_version, self.accelerator)
            self.add_callback(BAdamCallback)

        if finetuning_args.use_dft_loss:
            from ..trainer_utils import dft_loss_func

            self.compute_loss_func = dft_loss_func

        elif finetuning_args.use_eaft_loss:
            from ..trainer_utils import eaft_loss_func

            self.compute_loss_func = lambda outputs, labels, num_items_in_batch=None: eaft_loss_func(
                outputs, labels, num_items_in_batch, finetuning_args.eaft_alpha
            )

        if training_args.fp8 and hasattr(self, "accelerator"):  # verify FP8 status after trainer initialization
            verify_fp8_status(self.accelerator, training_args)

    @override
    def create_optimizer(self) -> "torch.optim.Optimizer":
        if self.optimizer is None:
            self.optimizer = create_custom_optimizer(self.model, self.args, self.finetuning_args)
        return super().create_optimizer()

    @override
    def create_scheduler(
        self, num_training_steps: int, optimizer: Optional["torch.optim.Optimizer"] = None
    ) -> "torch.optim.lr_scheduler.LRScheduler":
        create_custom_scheduler(self.args, num_training_steps, optimizer)
        return super().create_scheduler(num_training_steps, optimizer)

    @override
    def _get_train_sampler(self, *args, **kwargs) -> Optional["torch.utils.data.Sampler"]:
        if self.finetuning_args.disable_shuffling:
            return torch.utils.data.SequentialSampler(self.train_dataset)

        return super()._get_train_sampler(*args, **kwargs)

    def _build_reasoning_inputs(
        self, model, hidden_states, special_token_mask, reasoning_input_ids, reasoning_labels
    ):
        r"""Build inputs for reasoning forward pass from special token hidden states.

        Takes the hidden states at special token positions (WITHOUT detach, so
        gradient flows back) and concatenates them as prefix context with
        reasoning token embeddings. The model then predicts the ENTIRE reasoning
        sequence. Gradient flows: reasoning_loss → spec_h → first forward → params.

        Returns (inputs_embeds, attention_mask, labels) or None if no valid samples.
        """
        from torch.nn.utils.rnn import pad_sequence

        unwrapped = model
        while hasattr(unwrapped, "module"):
            unwrapped = unwrapped.module
        embed_fn = unwrapped.get_input_embeddings()

        batch_size = hidden_states.size(0)
        embeds_list, labels_list = [], []

        for i in range(batch_size):
            mask = special_token_mask[i]
            n_special = mask.sum().item()
            if n_special == 0:
                continue

            # Keep gradient flow: reasoning loss must update how model produces
            # hidden states at special token positions (the core training signal).
            spec_h = hidden_states[i][mask == 1]  # (n_special, dim) — gradient flows through!

            r_ids = reasoning_input_ids[i]
            r_lbls = reasoning_labels[i]
            valid_len = (r_ids != self.processing_class.pad_token_id).sum().item()
            if valid_len == 0:
                continue
            r_ids = r_ids[:valid_len]
            r_lbls = r_lbls[:valid_len]

            r_embeds = embed_fn(r_ids)  # (valid_len, dim)

            # [special_token_hidden_states, reasoning_embeddings]
            combined = torch.cat([spec_h, r_embeds], dim=0)
            prefix_labels = torch.full(
                (n_special,), IGNORE_INDEX, device=hidden_states.device, dtype=torch.long
            )
            combined_labels = torch.cat([prefix_labels, r_lbls], dim=0)

            embeds_list.append(combined)
            labels_list.append(combined_labels)

        if not embeds_list:
            return None

        input_embeds = pad_sequence(embeds_list, batch_first=True, padding_value=0.0)
        labels_padded = pad_sequence(labels_list, batch_first=True, padding_value=IGNORE_INDEX)
        seq_lens = [e.size(0) for e in embeds_list]
        max_len = max(seq_lens)
        attn_mask = torch.zeros((len(embeds_list), max_len), dtype=torch.long, device=hidden_states.device)
        for idx, l in enumerate(seq_lens):
            attn_mask[idx, :l] = 1

        return input_embeds, attn_mask, labels_padded

    def _forward_with_latent_chain(self, model, input_ids, attention_mask, labels, special_token_mask):
        r"""Sequential forward: each latent token uses the previous token's
        output hidden state as its input (instead of its word embedding).

        Process (per sample):
          1. Forward prefix (before first latent) → KV cache + last hidden state
          2. latent_0: input = latent_hidden_norm(h_prefix_last) → forward → h_0
             latent_i (i>0): input = latent_hidden_norm(h_{i-1}) → forward → h_i
             ALL latent tokens use previous token's hidden state as input.
             (dedicated LayerNorm maps hidden states to embedding scale)
          3. Forward suffix (after last latent) → logits → CE loss on answer

        Phase 1 and Phase 2 run under torch.no_grad() to avoid DeepSpeed
        gradient double-reduction (model parameters must only appear in ONE
        backward-reachable forward pass). Only Phase 3 carries gradients.
        Latent hidden states are returned detached; latent_hidden_norm still
        receives gradient through the reasoning forward path.

        Returns:
            loss_answer: averaged CE loss on answer tokens across batch.
            latent_hs_list: list of (num_latent, dim) tensors per sample
                (detached hidden states). None for samples without latent tokens.
            suffix_preds_list: list of (suffix_start, argmax_ids) per sample
                for metric computation. None for samples without suffix.
        """
        batch_size, seq_len = input_ids.shape
        device = input_ids.device

        unwrapped = model
        while hasattr(unwrapped, "module"):
            unwrapped = unwrapped.module
        embed_fn = unwrapped.get_input_embeddings()
        latent_norm = getattr(unwrapped, "latent_hidden_norm", None)  # dedicated LayerNorm

        # Disable gradient checkpointing: Phase 3 uses KV cache (past_key_values),
        # but gradient checkpointing forces use_cache=False in transformers,
        # causing attention mask / KV length mismatch.
        _base_model = unwrapped
        while hasattr(_base_model, "model"):
            _base_model = _base_model.model
        _gc_was_enabled = getattr(_base_model, "gradient_checkpointing", False)
        if _gc_was_enabled:
            _base_model.gradient_checkpointing_disable()

        total_loss = torch.tensor(0.0, device=device)
        latent_hs_list: list[torch.Tensor | None] = []
        suffix_preds_list: list[tuple[int, torch.Tensor] | None] = []  # (suffix_start, argmax_ids) per sample
        valid_count = 0

        for b in range(batch_size):
            sample_ids = input_ids[b]
            sample_attn = attention_mask[b]
            sample_labels = labels[b]
            sample_mask = special_token_mask[b]

            valid_len = int(sample_attn.sum().item())
            latent_pos = (sample_mask[:valid_len] == 1).nonzero(as_tuple=True)[0]

            if len(latent_pos) == 0:
                # No latent tokens → standard forward for this sample
                out = model(
                    input_ids=sample_ids[:valid_len].unsqueeze(0),
                    attention_mask=torch.ones(1, valid_len, device=device, dtype=sample_attn.dtype),
                    labels=sample_labels[:valid_len].unsqueeze(0),
                )
                total_loss = total_loss + out.loss
                valid_count += 1
                latent_hs_list.append(None)
                suffix_preds_list.append((0, out.logits[0].argmax(dim=-1).detach()))
                del out
                continue

            first_pos = latent_pos[0].item()
            last_pos = latent_pos[-1].item()
            num_latent = len(latent_pos)

            # ---- Phase 1 & 2: no_grad (avoids DeepSpeed double-reduction) ----
            with torch.no_grad():
                # Phase 1: Forward prefix (before first latent)
                if first_pos > 0:
                    prefix_out = model(
                        input_ids=sample_ids[:first_pos].unsqueeze(0),
                        attention_mask=torch.ones(1, first_pos, device=device, dtype=sample_attn.dtype),
                        use_cache=True,
                        output_hidden_states=True,
                    )
                    past_kv = prefix_out.past_key_values
                    prefix_last_h = prefix_out.hidden_states[-1][0, -1, :]  # (dim,)
                    del prefix_out
                else:
                    past_kv = None
                    prefix_last_h = None

                # Phase 2: Sequential latent tokens
                latent_hs: list[torch.Tensor] = []
                for i in range(num_latent):
                    pos = latent_pos[i].item()
                    if i == 0:
                        # First latent: use prefix last hidden state
                        if prefix_last_h is not None:
                            prev_h = prefix_last_h  # (dim,)
                            if latent_norm is not None:
                                prev_h = latent_norm(prev_h.unsqueeze(0)).squeeze(0)
                            inp = prev_h.unsqueeze(0).unsqueeze(0)  # (1,1,dim)
                        else:
                            # No prefix (edge case) → fallback to word embedding
                            inp = embed_fn(sample_ids[pos : pos + 1]).unsqueeze(0)  # (1,1,dim)
                    else:
                        # Subsequent: apply latent_hidden_norm to h_{i-1}
                        prev_h = latent_hs[-1]  # (dim,)
                        if latent_norm is not None:
                            prev_h = latent_norm(prev_h.unsqueeze(0)).squeeze(0)
                        inp = prev_h.unsqueeze(0).unsqueeze(0)  # (1,1,dim)

                    cum_len = first_pos + i + 1
                    step_attn = torch.ones(1, cum_len, device=device, dtype=sample_attn.dtype)

                    lat_out = model(
                        inputs_embeds=inp,
                        attention_mask=step_attn,
                        past_key_values=past_kv,
                        use_cache=True,
                        output_hidden_states=True,
                    )
                    past_kv = lat_out.past_key_values
                    h_i = lat_out.hidden_states[-1][0, -1, :]  # (dim,)
                    latent_hs.append(h_i)
                    del lat_out

            # latent_hs are already detached (computed under no_grad)
            latent_hs_list.append(torch.stack(latent_hs))  # (num_latent, dim)

            # ---- Debug: log L2 norms of hidden states vs word embeddings ----
            _gs = self.state.global_step if hasattr(self, "state") and self.state is not None else 0
            if b == 0 and (_gs <= 5 or _gs % 100 == 0):
                try:
                    with torch.no_grad():
                        stacked_hs = torch.stack(latent_hs)  # (num_latent, dim)
                        hs_norms = stacked_hs.norm(dim=-1)  # (num_latent,)
                        # Word embedding norms for the same latent token positions
                        latent_ids = sample_ids[latent_pos]  # (num_latent,)
                        word_embs = embed_fn(latent_ids)  # (num_latent, dim)
                        emb_norms = word_embs.norm(dim=-1)  # (num_latent,)
                        # Prefix last hidden state norm
                        prefix_h_norm = prefix_last_h.norm().item() if prefix_last_h is not None else 0.0
                        # Normed input norms (what actually goes into the model)
                        if latent_norm is not None:
                            normed_prefix = latent_norm(prefix_last_h.unsqueeze(0)).squeeze(0) if prefix_last_h is not None else None
                            normed_prefix_norm = normed_prefix.norm().item() if normed_prefix is not None else 0.0
                            normed_hs = latent_norm(stacked_hs)
                            normed_hs_norms = normed_hs.norm(dim=-1)
                        else:
                            normed_prefix_norm = prefix_h_norm
                            normed_hs_norms = hs_norms
                        logger.info_rank0(
                            f"[latent_chain] step={_gs}\n"
                            f"  raw hidden_states L2 norms:  {[f'{n:.4f}' for n in hs_norms.tolist()]}\n"
                            f"  normed hidden_states L2 norms (input to next): {[f'{n:.4f}' for n in normed_hs_norms.tolist()]}\n"
                            f"  word embedding L2 norms:     {[f'{n:.4f}' for n in emb_norms.tolist()]}\n"
                            f"  prefix_last_h raw L2 norm:   {prefix_h_norm:.4f}\n"
                            f"  prefix_last_h normed L2 (input to latent_0): {normed_prefix_norm:.4f}"
                        )
                except Exception as e:
                    logger.warning_rank0(f"[latent_chain] norm logging failed: {e}")

            # ---- Phase 3: Forward suffix (answer portion) — WITH gradients ----
            suffix_start = last_pos + 1
            suffix_end = valid_len
            suffix_len = suffix_end - suffix_start

            if suffix_len > 1:
                full_ctx_len = first_pos + num_latent + suffix_len
                suffix_out = model(
                    input_ids=sample_ids[suffix_start:suffix_end].unsqueeze(0),
                    attention_mask=torch.ones(
                        1, full_ctx_len, device=device, dtype=sample_attn.dtype
                    ),
                    past_key_values=past_kv,
                )
                # CE loss: logits[j] predicts token at suffix_start + j + 1
                logits = suffix_out.logits[0]  # (suffix_len, vocab)
                shift_logits = logits[:-1, :]  # (suffix_len-1, vocab)
                shift_labels = sample_labels[suffix_start + 1 : suffix_end]
                loss_fn = torch.nn.CrossEntropyLoss(ignore_index=IGNORE_INDEX)
                sample_loss = loss_fn(
                    shift_logits.contiguous().view(-1, shift_logits.size(-1)),
                    shift_labels.contiguous().view(-1),
                )
                total_loss = total_loss + sample_loss
                valid_count += 1
                suffix_preds_list.append((suffix_start, logits.argmax(dim=-1).detach()))
                del suffix_out
            else:
                suffix_preds_list.append(None)

            del past_kv

        avg_loss = total_loss / max(valid_count, 1)

        # Re-enable gradient checkpointing if it was disabled
        if _gc_was_enabled:
            _base_model.gradient_checkpointing_enable({"use_reentrant": False})

        return avg_loss, latent_hs_list, suffix_preds_list

    def _build_reasoning_inputs_from_latent_hs(
        self, model, latent_hs_list, reasoning_input_ids, reasoning_labels
    ):
        r"""Build reasoning forward inputs from pre-computed latent hidden states.

        Similar to _build_reasoning_inputs but takes a list of per-sample latent
        hidden state tensors directly (already extracted via the sequential chain).
        """
        from torch.nn.utils.rnn import pad_sequence

        unwrapped = model
        while hasattr(unwrapped, "module"):
            unwrapped = unwrapped.module
        embed_fn = unwrapped.get_input_embeddings()

        embeds_list, labels_list = [], []

        for i, spec_h in enumerate(latent_hs_list):
            if spec_h is None:
                continue

            n_special = spec_h.size(0)
            r_ids = reasoning_input_ids[i]
            r_lbls = reasoning_labels[i]
            valid_len = (r_ids != self.processing_class.pad_token_id).sum().item()
            if valid_len == 0:
                continue
            r_ids = r_ids[:valid_len]
            r_lbls = r_lbls[:valid_len]

            r_embeds = embed_fn(r_ids)
            combined = torch.cat([spec_h, r_embeds], dim=0)
            prefix_labels = torch.full(
                (n_special,), IGNORE_INDEX, device=spec_h.device, dtype=torch.long
            )
            combined_labels = torch.cat([prefix_labels, r_lbls], dim=0)

            embeds_list.append(combined)
            labels_list.append(combined_labels)

        if not embeds_list:
            return None

        input_embeds = pad_sequence(embeds_list, batch_first=True, padding_value=0.0)
        labels_padded = pad_sequence(labels_list, batch_first=True, padding_value=IGNORE_INDEX)
        seq_lens = [e.size(0) for e in embeds_list]
        max_len = max(seq_lens)
        attn_mask = torch.zeros(
            (len(embeds_list), max_len), dtype=torch.long, device=embeds_list[0].device
        )
        for idx, l in enumerate(seq_lens):
            attn_mask[idx, :l] = 1

        return input_embeds, attn_mask, labels_padded

    @staticmethod
    def _reset_ds_reduction_state(model):
        r"""Reset DeepSpeed ZeRO's gradient reduction flags to allow a second backward pass."""
        # ZeRO Stage 1/2
        if hasattr(model, "optimizer") and hasattr(model.optimizer, "params_already_reduced"):
            if isinstance(model.optimizer.params_already_reduced, list):
                model.optimizer.params_already_reduced = [
                    False
                ] * len(model.optimizer.params_already_reduced)
        # ZeRO Stage 3 (uses a different mechanism)
        if hasattr(model, "optimizer") and hasattr(model.optimizer, "parameter_offload"):
            pass  # Stage 3 doesn't use params_already_reduced

    # ---- Training ----

    @override
    def training_step(
        self,
        model: "torch.nn.Module",
        inputs: dict[str, Union["torch.Tensor", Any]],
        num_items_in_batch: Optional[int] = None,
    ) -> "torch.Tensor":
        r"""Combined training step: Sequential latent chain → Reasoning → SINGLE backward.

        Latent tokens are processed SEQUENTIALLY: each latent token uses the
        previous token's output hidden state as its input embedding (not its
        word embedding). This creates a recurrent chain:
          latent_0: input = h_prefix_last      →  h_0   (prefix's last hidden state)
          latent_i: input = h_{i-1}            →  h_i   (for i > 0)

        The suffix (answer) is forwarded with KV cache from the latent chain.
        loss_answer is CE on answer tokens only (think block = IGNORE_INDEX).
        loss_reasoning uses the latent hidden states (with latent_hidden_norm) to
        predict the ground-truth reasoning chain.

        Loss = loss_answer + w * loss_reasoning.
        """
        model.train()

        # Pop reasoning fields
        reasoning_input_ids = inputs.pop("reasoning_input_ids", None)
        reasoning_labels = inputs.pop("reasoning_labels", None)
        special_token_mask = inputs.pop("special_token_mask", None)

        w = self.finetuning_args.reasoning_loss_weight
        # do_latent_chain: data has latent tokens → must use sequential forward for answer loss
        do_latent_chain = (
            special_token_mask is not None
            and special_token_mask.sum() > 0
        )
        # do_reasoning_forward: additionally need w > 0 to run Forward 2 (reasoning loss)
        do_reasoning_forward = (
            do_latent_chain
            and w > 0
            and reasoning_input_ids is not None
        )

        if do_latent_chain:
            # ---- Forward 1: Sequential latent chain + answer CE ----
            with self.compute_loss_context_manager():
                loss_answer, latent_hs_list, _ = self._forward_with_latent_chain(
                    model,
                    inputs["input_ids"],
                    inputs["attention_mask"],
                    inputs["labels"],
                    special_token_mask,
                )

            # ---- Forward 2: Reasoning (only when w > 0) ----
            loss_reasoning = torch.tensor(0.0, device=loss_answer.device)
            if do_reasoning_forward:
                # Apply latent_hidden_norm to latent hidden states for reasoning forward.
                # latent_hs are detached (computed under no_grad in Phase 2), so
                # latent_norm gets gradient from reasoning loss but backward does
                # NOT flow through Phase 2's model calls → no DeepSpeed conflict.
                unwrapped = model
                while hasattr(unwrapped, "module"):
                    unwrapped = unwrapped.module
                latent_norm = getattr(unwrapped, "latent_hidden_norm", None)

                normed_hs_list: list[torch.Tensor | None] = []
                for hs in latent_hs_list:
                    if hs is not None and latent_norm is not None:
                        if getattr(self, "_log_latent_norm_once", True):
                            with torch.no_grad():
                                pre_mean = hs.norm(dim=-1).mean().item()
                            logger.info_rank0(
                                f"[reasoning] latent hidden_states L2 mean before latent_hidden_norm: {pre_mean:.4f}"
                            )
                        normed = latent_norm(hs)
                        if getattr(self, "_log_latent_norm_once", True):
                            with torch.no_grad():
                                post_mean = normed.norm(dim=-1).mean().item()
                            logger.info_rank0(
                                f"[reasoning] latent_hidden_norm L2 mean after: {post_mean:.4f}"
                            )
                            self._log_latent_norm_once = False
                        normed_hs_list.append(normed)
                    else:
                        normed_hs_list.append(hs)

                reasoning_inputs = self._build_reasoning_inputs_from_latent_hs(
                    model, normed_hs_list, reasoning_input_ids, reasoning_labels
                )
                if reasoning_inputs is not None:
                    input_embeds, attn_mask, labels_padded = reasoning_inputs
                    with self.compute_loss_context_manager():
                        outputs_r = model(
                            inputs_embeds=input_embeds,
                            attention_mask=attn_mask,
                            labels=labels_padded,
                        )
                        loss_reasoning = outputs_r.loss
                    del outputs_r

            total_loss = loss_answer + w * loss_reasoning
        else:
            # ---- Standard SFT forward (no latent tokens in data) ----
            with self.compute_loss_context_manager():
                loss_answer, outputs = super().compute_loss(
                    model, inputs, return_outputs=True, num_items_in_batch=num_items_in_batch
                )
            del outputs
            loss_reasoning = torch.tensor(0.0, device=loss_answer.device)
            total_loss = loss_answer

        # ---- GA scaling + backward ----
        # Replicate HF Trainer's exact logic: divide loss by gradient_accumulation_steps
        # BEFORE backward (so gradients are correctly scaled) and BEFORE return (so
        # HF's _tr_loss accumulation gives the correct mean when divided by num_steps).
        # Condition matches HF: only scale when model doesn't handle num_items_in_batch
        # internally and no custom compute_loss_func is used.
        if self.args.n_gpu > 1:
            total_loss = total_loss.mean()  # DataParallel multi-GPU averaging

        if (not self.model_accepts_loss_kwargs or num_items_in_batch is None) and self.compute_loss_func is None:
            total_loss = total_loss / self.args.gradient_accumulation_steps

        kwargs = {}
        if hasattr(self, "_grad_norm_kwargs"):
            kwargs = self._grad_norm_kwargs
        self.accelerator.backward(total_loss, **kwargs)

        # Log split losses — store GA-scaled values to match HF Trainer's _tr_loss
        # accumulation logic exactly. HF stores `loss/GA` per micro-batch, sums them,
        # then divides by num_global_steps. We do the same so loss_sft + w*loss_reasoning
        # == train/loss at every logging point, even at epoch-boundary partial steps.
        ga = self.args.gradient_accumulation_steps
        if not hasattr(self, "_custom_loss_buffer"):
            self._custom_loss_buffer = {"sft": [], "reasoning": []}
        self._custom_loss_buffer["sft"].append(loss_answer.detach().float() / ga)
        self._custom_loss_buffer["reasoning"].append(loss_reasoning.detach().float() / ga)

        return total_loss.detach()

    @override
    def log(self, logs: dict[str, float], start_time: Optional[float] = None) -> None:
        # Inject train-time split losses
        # Use the EXACT same formula as HF Trainer's train/loss:
        #   sum(loss_i / GA) / num_global_steps_since_last_log
        # This ensures loss_sft + w * loss_reasoning == train/loss at every point,
        # including epoch-boundary partial accumulation steps.
        if hasattr(self, "_custom_loss_buffer") and self._custom_loss_buffer:
            num_steps = self.state.global_step - getattr(self, "_custom_loss_last_logged_step", 0)
            if num_steps > 0:
                for k, v in self._custom_loss_buffer.items():
                    if v:
                        # Sum of GA-scaled values, then divide by num_global_steps
                        local_loss = torch.stack(v).sum() / num_steps

                        # Sync across GPUs (All-Reduce Mean) to match HF's nested_gather().mean()
                        if self.args.world_size > 1:
                            import torch.distributed as dist
                            dist.all_reduce(local_loss, op=dist.ReduceOp.SUM)
                            local_loss = local_loss / self.args.world_size

                        logs[f"loss_{k}"] = round(local_loss.item(), 4)
            self._custom_loss_last_logged_step = self.state.global_step
            self._custom_loss_buffer = {"sft": [], "reasoning": []}

        # Inject eval-time split losses
        if hasattr(self, "_eval_loss_buffer") and self._eval_loss_buffer:
            for k, v in self._eval_loss_buffer.items():
                if v:
                    # Eval is usually already gathered or run on rank 0, but if distributed eval:
                    local_loss = torch.stack(v).mean()
                    # (Optional) If you run eval in distributed mode without gather, sync here clearly.
                    # Standard HF Helper handles eval metrics gather. 
                    # For safety if this buffer is per-gpu:
                    if self.args.world_size > 1:
                        import torch.distributed as dist
                        dist.all_reduce(local_loss, op=dist.ReduceOp.SUM)
                        local_loss = local_loss / self.args.world_size
                        
                    logs[f"eval_loss_{k}"] = round(local_loss.item(), 4)

        if start_time is not None:
            super().log(logs, start_time)
        else:
            super().log(logs)

    # ---- Evaluation ----

    @override
    def prediction_step(
        self,
        model: "torch.nn.Module",
        inputs: dict[str, Union["torch.Tensor", Any]],
        prediction_loss_only: bool,
        ignore_keys: Optional[list[str]] = None,
        **gen_kwargs,
    ) -> tuple[Optional[float], Optional["torch.Tensor"], Optional["torch.Tensor"]]:
        r"""Compute eval loss (with reasoning split) or generate tokens."""
        if self.args.predict_with_generate:
            inputs.pop("reasoning_input_ids", None)
            inputs.pop("reasoning_labels", None)
            inputs.pop("reasoning_attention_mask", None)
            inputs.pop("special_token_mask", None)
            labels = inputs.pop("labels", None)
            loss, generated_tokens, _ = super().prediction_step(
                model, inputs, prediction_loss_only=prediction_loss_only, ignore_keys=ignore_keys, **gen_kwargs
            )
            if generated_tokens is not None:
                generated_tokens[:, : inputs["input_ids"].size(-1)] = self.processing_class.pad_token_id
                generated_tokens = generated_tokens.contiguous()
            return loss, generated_tokens, labels

        # Non-generate eval: sequential latent chain under no_grad
        reasoning_input_ids = inputs.pop("reasoning_input_ids", None)
        reasoning_labels = inputs.pop("reasoning_labels", None)
        inputs.pop("reasoning_attention_mask", None)
        special_token_mask = inputs.pop("special_token_mask", None)
        labels = inputs.get("labels")

        w = self.finetuning_args.reasoning_loss_weight
        do_latent_chain = (
            special_token_mask is not None
            and special_token_mask.sum() > 0
        )
        do_reasoning_forward = (
            do_latent_chain
            and w > 0
            and reasoning_input_ids is not None
        )

        with torch.no_grad():
            if do_latent_chain:
                # Sequential latent chain + answer CE
                loss_answer, latent_hs_list, suffix_preds_list = self._forward_with_latent_chain(
                    model,
                    inputs["input_ids"],
                    inputs["attention_mask"],
                    inputs["labels"],
                    special_token_mask,
                )

                # Reasoning forward (only when w > 0)
                loss_reasoning = torch.tensor(0.0, device=loss_answer.device)
                if do_reasoning_forward:
                    unwrapped = model
                    while hasattr(unwrapped, "module"):
                        unwrapped = unwrapped.module
                    latent_norm = getattr(unwrapped, "latent_hidden_norm", None)

                    normed_hs_list: list[torch.Tensor | None] = []
                    for hs in latent_hs_list:
                        if hs is not None and latent_norm is not None:
                            normed_hs_list.append(latent_norm(hs))
                        else:
                            normed_hs_list.append(hs)

                    reasoning_inputs = self._build_reasoning_inputs_from_latent_hs(
                        model, normed_hs_list, reasoning_input_ids, reasoning_labels
                    )
                    if reasoning_inputs is not None:
                        input_embeds, attn_mask, labels_padded = reasoning_inputs
                        outputs_r = model(
                            inputs_embeds=input_embeds,
                            attention_mask=attn_mask,
                            labels=labels_padded,
                        )
                        loss_reasoning = outputs_r.loss

                total_loss = loss_answer + w * loss_reasoning

                # Construct full-sequence argmax preds for metric computation (exact_match)
                batch_size_eval, seq_len_eval = inputs["input_ids"].shape
                full_preds = torch.full(
                    (batch_size_eval, seq_len_eval),
                    self.processing_class.pad_token_id,
                    dtype=torch.long,
                    device=inputs["input_ids"].device,
                )
                for b_idx, sp in enumerate(suffix_preds_list):
                    if sp is not None:
                        s_start, s_argmax = sp
                        end = min(s_start + len(s_argmax), seq_len_eval)
                        full_preds[b_idx, s_start:end] = s_argmax[: end - s_start]
                logits = full_preds  # 2D argmax tensor, handled by eval_logit_processor
            else:
                loss_answer, outputs = super().compute_loss(model, inputs, return_outputs=True)
                loss_reasoning = torch.tensor(0.0, device=loss_answer.device)
                total_loss = loss_answer
                logits = outputs.logits

        if not hasattr(self, "_eval_loss_buffer"):
            self._eval_loss_buffer = {"sft": [], "reasoning": []}
        self._eval_loss_buffer["sft"].append(loss_answer.detach().float().cpu())
        self._eval_loss_buffer["reasoning"].append(loss_reasoning.detach().float().cpu())

        return total_loss.detach(), logits, labels

    @override
    def evaluation_loop(self, *args, **kwargs):
        self._eval_loss_buffer = {"sft": [], "reasoning": []}
        output = super().evaluation_loop(*args, **kwargs)
        for k, v in self._eval_loss_buffer.items():
            if v:
                output.metrics[f"eval_loss_{k}"] = round(torch.stack(v).mean().item(), 4)
        logger.info_rank0(
            f"[eval] eval_loss_sft={output.metrics.get('eval_loss_sft', 'N/A')}, "
            f"eval_loss_reasoning={output.metrics.get('eval_loss_reasoning', 'N/A')}"
        )
        self._eval_loss_buffer = {"sft": [], "reasoning": []}
        return output

    def _recover_reasoning_for_sample(
        self,
        model,
        prompt_ids: torch.Tensor,
        generated_ids_np: "np.ndarray",
        special_positions_in_gen: list[int] = None,
        special_positions_in_full: list[int] = None,
        positions_are_absolute: bool = False,
        max_new_tokens: int = 512,
        sample_idx: int = -1,
    ) -> str:
        r"""Recover reasoning from latent hidden states using sequential chain.

        Processes latent tokens sequentially (each using previous output as input),
        applies final norm, then greedy-decodes the reasoning chain.

        Args:
            model: unwrapped model (no DeepSpeed / DDP wrapper).
            prompt_ids: (PromptLen,) – the input_ids of the prompt (from dataset).
            generated_ids_np: (GenLen,) np array – the generated part (answer tokens).
            special_positions_in_gen: positions of special tokens *within generated_ids_np*
                (legacy mode, used when positions_are_absolute=False).
            special_positions_in_full: absolute positions of special tokens in the full
                sequence [prompt_ids + generated_ids] (new mode, positions_are_absolute=True).
            positions_are_absolute: if True, use special_positions_in_full directly.
            max_new_tokens: max reasoning tokens to generate.
            sample_idx: sample index for diagnostic logging.

        Returns:
            Decoded reasoning text string.
        """
        device = next(model.parameters()).device
        do_log = sample_idx < 3

        # 1. Build FULL sequence: [prompt] + [generated]
        gen_tensor = torch.tensor(generated_ids_np, dtype=torch.long)
        full_ids = torch.cat([prompt_ids.to("cpu"), gen_tensor], dim=0)
        prompt_len = prompt_ids.shape[0]

        if positions_are_absolute and special_positions_in_full is not None:
            abs_positions = special_positions_in_full
        elif special_positions_in_gen is not None:
            abs_positions = [prompt_len + p for p in special_positions_in_gen]
        else:
            return ""

        if not abs_positions:
            return ""

        first_pos = abs_positions[0]
        num_latent = len(abs_positions)
        embed_fn = model.get_input_embeddings()
        latent_norm = getattr(model, "latent_hidden_norm", None)

        with torch.no_grad():
            # ---- Phase 1: Forward prefix (before first latent) ----
            if first_pos > 0:
                prefix_ids = full_ids[:first_pos].unsqueeze(0).to(device)
                prefix_out = model(input_ids=prefix_ids, use_cache=True, output_hidden_states=True)
                past_kv = prefix_out.past_key_values
                prefix_last_h = prefix_out.hidden_states[-1][0, -1, :]  # (dim,)
                del prefix_out
            else:
                past_kv = None
                prefix_last_h = None

            # ---- Phase 2: Sequential latent tokens ----
            latent_hs = []
            for i in range(num_latent):
                pos = abs_positions[i]
                if i == 0:
                    # First latent: use prefix last hidden state
                    if prefix_last_h is not None:
                        prev_h = prefix_last_h
                        if latent_norm is not None:
                            prev_h = latent_norm(prev_h.unsqueeze(0)).squeeze(0)
                        inp = prev_h.unsqueeze(0).unsqueeze(0)
                    else:
                        # No prefix (edge case) → fallback to word embedding
                        inp = embed_fn(full_ids[pos : pos + 1].to(device)).unsqueeze(0)
                else:
                    prev_h = latent_hs[-1]
                    if latent_norm is not None:
                        prev_h = latent_norm(prev_h.unsqueeze(0)).squeeze(0)
                    inp = prev_h.unsqueeze(0).unsqueeze(0)

                cum_len = first_pos + i + 1
                step_attn = torch.ones(1, cum_len, device=device, dtype=torch.long)

                lat_out = model(
                    inputs_embeds=inp,
                    attention_mask=step_attn,
                    past_key_values=past_kv,
                    use_cache=True,
                    output_hidden_states=True,
                )
                past_kv = lat_out.past_key_values
                h_i = lat_out.hidden_states[-1][0, -1, :]
                latent_hs.append(h_i)
                del lat_out

            del past_kv  # free context KV cache

            spec_h = torch.stack(latent_hs)  # (num_latent, dim)

            # Apply latent_hidden_norm for reasoning recovery
            if latent_norm is not None:
                if do_log:
                    pre_mean = spec_h.norm(dim=-1).mean().item()
                    logger.info_rank0(
                        f"[recover_reasoning] latent hs L2 mean before latent_hidden_norm: {pre_mean:.4f}"
                    )
                spec_h = latent_norm(spec_h)
                if do_log:
                    post_mean = spec_h.norm(dim=-1).mean().item()
                    logger.info_rank0(
                        f"[recover_reasoning] latent hs L2 mean after latent_hidden_norm: {post_mean:.4f}"
                    )

            spec_h = spec_h.unsqueeze(0)  # (1, num_latent, dim)

            if do_log:
                norms = spec_h.squeeze(0).norm(dim=-1).tolist()
                logger.info_rank0(
                    f"  [sample {sample_idx}] spec_h norms={[f'{n:.2f}' for n in norms]}"
                )

            # ---- Greedy decode reasoning from latent hidden states ----
            result_ids = []
            out = model(inputs_embeds=spec_h, use_cache=True)
            decode_kv = out.past_key_values
            logits_first = out.logits[0, -1, :]
            next_id = logits_first.argmax(dim=-1)
            result_ids.append(next_id.item())

            if do_log:
                probs = torch.softmax(logits_first.float(), dim=-1)
                topk_probs, topk_ids = probs.topk(5)
                top_tokens = [
                    (self.processing_class.decode([tid.item()]), f"{p.item():.4f}")
                    for tid, p in zip(topk_ids, topk_probs)
                ]
                logger.info_rank0(f"  [sample {sample_idx}] first-step top5: {top_tokens}")

            for _ in range(max_new_tokens - 1):
                if next_id.item() == self.processing_class.eos_token_id:
                    break
                next_embed = embed_fn(next_id.reshape(1, 1))
                out = model(
                    inputs_embeds=next_embed,
                    past_key_values=decode_kv,
                    use_cache=True,
                )
                decode_kv = out.past_key_values
                next_id = out.logits[0, -1, :].argmax(dim=-1)
                result_ids.append(next_id.item())

            del decode_kv

        if not result_ids:
            return ""

        if do_log:
            logger.info_rank0(
                f"  [sample {sample_idx}] decoded {len(result_ids)} tokens, "
                f"first 10: {result_ids[:10]}"
            )

        return self.processing_class.decode(result_ids, skip_special_tokens=False)

    def save_predictions(
        self, dataset: "Dataset", predict_results: "PredictionOutput", skip_special_tokens: bool = True
    ) -> None:
        r"""Save model predictions to `output_dir`.

        When num_latent_thinking_token > 0:
        - input_ids (from dataset) includes the think block as prompt context
        - generated preds contain only the answer (model doesn't generate think block)
        - Latent token positions are found in input_ids, not in preds
        - For reasoning recovery: forward [input_ids + answer], extract latent hidden states
        """
        if not self.is_world_process_zero():
            return

        output_prediction_file = os.path.join(self.args.output_dir, "generated_predictions.jsonl")
        logger.info_rank0(f"Saving prediction results to {output_prediction_file}")

        labels = np.where(
            predict_results.label_ids != IGNORE_INDEX, predict_results.label_ids, self.processing_class.pad_token_id
        )
        preds = np.where(
            predict_results.predictions != IGNORE_INDEX,
            predict_results.predictions,
            self.processing_class.pad_token_id,
        )

        for i in range(len(preds)):
            pad_len = np.nonzero(preds[i] != self.processing_class.pad_token_id)[0]
            if len(pad_len):  # move pad token to last
                preds[i] = np.concatenate((preds[i][pad_len[0] :], preds[i][: pad_len[0]]), axis=-1)

        decoded_inputs = self.processing_class.batch_decode(dataset["input_ids"], skip_special_tokens=False)
        decoded_labels = self.processing_class.batch_decode(labels, skip_special_tokens=skip_special_tokens)

        # ---- Reasoning Recovery ----
        num_latent = getattr(self.data_args, "num_latent_thinking_token", 0) if self.data_args else 0

        # Decode predicted answers (generated part is answer only, no think block)
        decoded_answers = []
        for i in range(len(preds)):
            seq = preds[i]
            valid_len = (seq != self.processing_class.pad_token_id).sum()
            seq = seq[:valid_len]
            ans_text = self.processing_class.decode(seq, skip_special_tokens=False)
            decoded_answers.append(ans_text)

        # Debug: log first sample
        if len(preds) > 0:
            seq0 = preds[0]
            valid_len0 = int((seq0 != self.processing_class.pad_token_id).sum())
            seq0 = seq0[:valid_len0]
            logger.info_rank0(f"[save_predictions] Sample 0: valid_len={valid_len0}, first 20 token IDs: {seq0[:20].tolist()}")
            logger.info_rank0(f"[save_predictions] Sample 0 decoded: {self.processing_class.decode(seq0[:20], skip_special_tokens=False)}")
            logger.info_rank0(f"[save_predictions] Sample 0 input_ids len: {len(dataset[0]['input_ids'])}")

        decoded_reasonings = [""] * len(preds)

        if num_latent > 0:
            # ---- Latent thinking: find latent positions in input_ids (prompt context) ----
            # At prediction time, input_ids = [prompt + think_block], preds = [answer]
            # Latent token positions are in the input_ids, NOT in the generated output.
            latent_token_ids = set()
            for j in range(num_latent):
                name = f"<latent_{j}>"
                tok_ids = self.processing_class.encode(name, add_special_tokens=False)
                if len(tok_ids) == 1:
                    latent_token_ids.add(tok_ids[0])

            logger.info_rank0(f"[save_predictions] Latent token IDs for reasoning recovery: {latent_token_ids}")

            if not latent_token_ids:
                logger.warning("[save_predictions] Cannot resolve latent token IDs. Skipping reasoning recovery.")
            else:
                unwrapped = self.model
                while hasattr(unwrapped, "module"):
                    unwrapped = unwrapped.module
                unwrapped.eval()

                n_recovered = 0
                n_has_latent = 0
                for i in range(len(preds)):
                    try:
                        # input_ids includes prompt + think block
                        input_ids_i = dataset[i]["input_ids"]

                        # Find latent token positions in input_ids
                        latent_positions_in_input = [
                            j for j in range(len(input_ids_i))
                            if int(input_ids_i[j]) in latent_token_ids
                        ]

                        if not latent_positions_in_input:
                            continue

                        n_has_latent += 1

                        # Get generated answer tokens
                        seq = preds[i]
                        valid_len = int((seq != self.processing_class.pad_token_id).sum())
                        answer_ids = seq[:valid_len]

                        if i == 0:
                            logger.info_rank0(
                                f"[save_predictions] Sample 0: input_len={len(input_ids_i)}, "
                                f"answer_len={len(answer_ids)}, "
                                f"latent_positions_in_input={latent_positions_in_input}"
                            )

                        # Build full sequence: [input_ids (with think block)] + [answer]
                        # and forward to get hidden states at latent positions
                        prompt_ids = torch.tensor(input_ids_i, dtype=torch.long)
                        reasoning_text = self._recover_reasoning_for_sample(
                            unwrapped,
                            prompt_ids=prompt_ids,
                            generated_ids_np=answer_ids,
                            special_positions_in_full=latent_positions_in_input,
                            positions_are_absolute=True,
                            max_new_tokens=512,
                            sample_idx=i,
                        )
                        decoded_reasonings[i] = reasoning_text
                        if reasoning_text:
                            n_recovered += 1

                        if (i + 1) % 50 == 0:
                            logger.info_rank0(
                                f"[save_predictions] Processed {i+1}/{len(preds)}, "
                                f"has_latent={n_has_latent}, recovered={n_recovered}"
                            )
                    except Exception as e:
                        import traceback
                        logger.warning(f"Reasoning recovery failed for sample {i}: {e}\n{traceback.format_exc()}")

                logger.info_rank0(
                    f"[save_predictions] Reasoning recovery done: "
                    f"recovered={n_recovered}, has_latent={n_has_latent}, total={len(preds)}"
                )

        else:
            # ---- Legacy approach: discover special token IDs from tokenizer vocabulary ----
            special_token_ids = set()
            added_vocab = self.processing_class.get_added_vocab()  # {token_str: token_id}
            for tok_str, tok_id in added_vocab.items():
                tok_lower = str(tok_str).lower()
                if ("latent" in tok_lower or "thinking" in tok_lower or "reason" in tok_lower) and "|" not in tok_lower:
                    special_token_ids.add(tok_id)

            logger.info_rank0(f"[save_predictions] Special token IDs for reasoning recovery: {special_token_ids}")
            id_to_tok = {v: k for k, v in added_vocab.items() if v in special_token_ids}
            logger.info_rank0(f"[save_predictions] Special token mapping: {id_to_tok}")

            if special_token_ids:
                logger.info_rank0("[save_predictions] Starting reasoning recovery (legacy special tokens)...")
                unwrapped = self.model
                while hasattr(unwrapped, "module"):
                    unwrapped = unwrapped.module
                unwrapped.eval()

                n_recovered = 0
                n_has_positions = 0
                for i in range(len(preds)):
                    try:
                        seq = preds[i]
                        valid_len = int((seq != self.processing_class.pad_token_id).sum())
                        seq = seq[:valid_len]

                        special_positions_in_gen = [j for j in range(len(seq)) if int(seq[j]) in special_token_ids]
                        if not special_positions_in_gen:
                            continue

                        n_has_positions += 1
                        prompt_ids = torch.tensor(dataset[i]["input_ids"], dtype=torch.long)

                        reasoning_text = self._recover_reasoning_for_sample(
                            unwrapped, prompt_ids, seq, special_positions_in_gen,
                            max_new_tokens=512, sample_idx=i
                        )
                        decoded_reasonings[i] = reasoning_text
                        if reasoning_text:
                            n_recovered += 1

                        if (i + 1) % 50 == 0:
                            logger.info_rank0(
                                f"[save_predictions] Processed {i+1}/{len(preds)}, "
                                f"has_positions={n_has_positions}, recovered={n_recovered}"
                            )
                    except Exception as e:
                        import traceback
                        logger.warning(f"Reasoning recovery failed for sample {i}: {e}\n{traceback.format_exc()}")

                logger.info_rank0(
                    f"[save_predictions] Reasoning recovery done: "
                    f"recovered={n_recovered}, has_positions={n_has_positions}, total={len(preds)}"
                )
            else:
                logger.warning("[save_predictions] No special token IDs found in tokenizer. Skipping reasoning recovery.")

        with open(output_prediction_file, "w", encoding="utf-8") as f:
            for text, ans, rea, label in zip(decoded_inputs, decoded_answers, decoded_reasonings, decoded_labels):
                f.write(json.dumps({
                    "prompt": text,
                    "predict_answer": ans,
                    "predict_reasoning": rea,
                    "label": label
                }, ensure_ascii=False) + "\n")
