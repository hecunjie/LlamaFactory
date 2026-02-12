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

    from ...hparams import FinetuningArguments, ModelArguments, TrainingArguments


logger = logging.get_logger(__name__)


class CustomSeq2SeqTrainer(Seq2SeqTrainer):
    r"""Inherits Seq2SeqTrainer to compute generative metrics such as BLEU and ROUGE."""

    def __init__(
        self,
        finetuning_args: "FinetuningArguments",
        processor: Optional["ProcessorMixin"],
        model_args: Optional["ModelArguments"] = None,
        gen_kwargs: Optional[dict[str, Any]] = None,
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

        Takes the hidden states at special token positions (detached) and
        concatenates them as prefix context with reasoning token embeddings.
        The model then predicts the ENTIRE reasoning sequence using only
        special token representations as context.

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

            # Detach: break gradient flow back to first forward
            spec_h = hidden_states[i][mask == 1].detach()  # (n_special, dim)

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
        r"""Two-pass training step: SFT forward+backward, then reasoning forward+backward.

        This avoids DeepSpeed's 'parameter already reduced' error by resetting
        the reduction state between the two backward passes.
        """
        model.train()

        # Pop reasoning fields
        reasoning_input_ids = inputs.pop("reasoning_input_ids", None)
        reasoning_labels = inputs.pop("reasoning_labels", None)
        special_token_mask = inputs.pop("special_token_mask", None)

        do_reasoning = (
            reasoning_input_ids is not None
            and special_token_mask is not None
            and special_token_mask.sum() > 0
        )

        if do_reasoning:
            inputs["output_hidden_states"] = True

        # ---- Pass 1: SFT forward + backward ----
        with self.compute_loss_context_manager():
            loss_sft, outputs = super().compute_loss(
                model, inputs, return_outputs=True, num_items_in_batch=num_items_in_batch
            )

        # Save hidden states before backward
        hidden_states = None
        if do_reasoning:
            hs = getattr(outputs, "hidden_states", None)
            if isinstance(outputs, dict):
                hs = outputs.get("hidden_states", hs)
            if hs is not None:
                hidden_states = hs[-1].detach().clone()

        del outputs

        kwargs = {}
        if hasattr(self, "_grad_norm_kwargs"):
            kwargs = self._grad_norm_kwargs
        self.accelerator.backward(loss_sft, **kwargs)

        # ---- Pass 2: Reasoning forward + backward ----
        loss_reasoning = torch.tensor(0.0, device=loss_sft.device)
        if do_reasoning and hidden_states is not None:
            reasoning_inputs = self._build_reasoning_inputs(
                model, hidden_states, special_token_mask, reasoning_input_ids, reasoning_labels
            )
            if reasoning_inputs is not None:
                input_embeds, attn_mask, labels_padded = reasoning_inputs

                # Reset DeepSpeed reduction state so second backward is allowed
                self._reset_ds_reduction_state(model)

                with self.compute_loss_context_manager():
                    outputs_r = model(
                        inputs_embeds=input_embeds,
                        attention_mask=attn_mask,
                        labels=labels_padded,
                    )
                    loss_reasoning = outputs_r.loss

                del outputs_r
                self.accelerator.backward(loss_reasoning, **kwargs)

        # Log split losses
        if not hasattr(self, "_custom_loss_buffer"):
            self._custom_loss_buffer = {"sft": [], "reasoning": []}
        self._custom_loss_buffer["sft"].append(loss_sft.detach().float())
        self._custom_loss_buffer["reasoning"].append(loss_reasoning.detach().float())

        return (loss_sft + loss_reasoning).detach()

    @override
    def log(self, logs: dict[str, float], start_time: Optional[float] = None) -> None:
        # Inject train-time split losses
        if hasattr(self, "_custom_loss_buffer") and self._custom_loss_buffer:
            for k, v in self._custom_loss_buffer.items():
                if v:
                    # 1. Compute local mean for this buffer flush (e.g. logging_steps=10)
                    local_loss = torch.stack(v).mean()
                    
                    # 2. Sync across GPUs (All-Reduce Mean) to match main 'loss'
                    if self.args.world_size > 1:
                        import torch.distributed as dist
                        dist.all_reduce(local_loss, op=dist.ReduceOp.SUM)
                        local_loss = local_loss / self.args.world_size
                    
                    logs[f"loss_{k}"] = round(local_loss.item(), 4)
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
            # Preserve references needed for reasoning recovery
            # Clone to ensure we don't lose data if inputs is modified
            gt_labels = inputs.get("labels")
            gt_mask = inputs.get("special_token_mask")
            input_ids = inputs.get("input_ids")
            
            inputs.pop("reasoning_input_ids", None)
            inputs.pop("reasoning_labels", None)
            inputs.pop("reasoning_attention_mask", None)
            inputs.pop("special_token_mask", None)
            labels = inputs.pop("labels", None)
            
            loss, generated_tokens, _ = super().prediction_step(
                model, inputs, prediction_loss_only=prediction_loss_only, ignore_keys=ignore_keys, **gen_kwargs
            )

            # Attempt to recover reasoning if special tokens can be found in generation
            if generated_tokens is not None and gt_mask is not None and gt_labels is not None:
                 try:
                    batch_size = generated_tokens.size(0)
                    reasoning_seqs = []
                    
                    # We need the model to output hidden states on the concatenated inputs
                    # Construct batch inputs
                    full_input_ids_list = []
                    signatures = []
                    
                    for i in range(batch_size):
                        # Identify target signature from GT
                        # Mask is 1 where special tokens are
                        mask_i = gt_mask[i]
                        lbl_i = gt_labels[i]
                        # Verify valid labels at mask
                        if mask_i.sum() > 0:
                            sig = lbl_i[mask_i == 1]
                            # Remove IGNORE_INDEX if any (shouldn't be, but safety)
                            sig = sig[sig != IGNORE_INDEX]
                            signatures.append(sig)
                        else:
                            signatures.append(None)
                            
                        # generated_tokens usually includes the prompt for causal models (checked via post-processing logic below)
                        g_ids = generated_tokens[i]
                        valid_g = g_ids[g_ids != self.processing_class.pad_token_id]
                        
                        full_input_ids_list.append(valid_g)

                    # Pad full inputs
                    from torch.nn.utils.rnn import pad_sequence
                    full_batch = pad_sequence(full_input_ids_list, batch_first=True, padding_value=self.processing_class.pad_token_id)
                    full_batch = full_batch.to(model.device)
                    
                    # Forward pass
                    with torch.no_grad():
                        outputs = model(full_batch, output_hidden_states=True)
                        hidden_states_all = outputs.hidden_states[-1]
                        
                    # Extract hidden states and Generate Reasoning
                    for i in range(batch_size):
                        sig = signatures[i]
                        if sig is None or len(sig) == 0:
                             reasoning_seqs.append(torch.tensor([], dtype=torch.long, device=model.device))
                             continue
                             
                        # Find signature in valid_g (inside full_batch)
                        # Offset by len(valid_p)
                        full_seq = full_batch[i] # Padded
                        # Re-calculate valid length to avoid padding issues
                        valid_len = (full_seq != self.processing_class.pad_token_id).sum()
                        # Searching in the whole sequence is safer
                        
                        # Search for sequence 'sig' in 'full_seq[:valid_len]'
                        # Simplified search: match only the FIRST token of the signature (the primary special token)
                        # This avoids issues where '\n' is part of the signature but handled differently in generation
                        found_idx = -1
                        match_len = 0
                        
                        seq_np = full_seq[:valid_len].cpu().numpy()
                        
                        if len(sig) > 0:
                            # Use only the first token of signature as anchor
                            anchor_token = sig[0].item()
                            
                            # Find anchor in sequence
                            anchor_indices = np.where(seq_np == anchor_token)[0]
                            if len(anchor_indices) > 0:
                                found_idx = anchor_indices[0] # Take the first occurrence
                                match_len = 1 # We primarily use 1 token's hidden state
                                
                                # Check subsequent matches up to sig length
                                # (Optional, but good for robustness if we want to extract multi-token contexts)
                                k = 0
                                while found_idx + k < len(seq_np) and k < len(sig) and seq_np[found_idx + k] == sig[k].item():
                                    k += 1
                                match_len = k
                        
                        if found_idx != -1:
                            # Found special tokens. Hidden states are at these positions.
                            # We want to use these hidden states to generate reasoning.
                            # Extract hidden states
                            spec_h = hidden_states_all[i, found_idx : found_idx+match_len] # (L, D)
                            
                            if spec_h.size(0) == 0:
                                reasoning_seqs.append(torch.tensor([], dtype=torch.long, device=model.device))
                                continue
                            
                            # Generate
                            # We need to wrap model.generate to make it accept inputs_embeds primarily
                            # and start generation.
                            # Usually model.generate takes inputs_embeds.
                            # We must provide inputs_embeds in shape (1, L, D).
                            spec_h = spec_h.unsqueeze(0) # (1, L, D)
                            
                            # Generate
                            r_out = model.generate(
                                inputs_embeds=spec_h, 
                                max_new_tokens=512, # Reasonable limit
                                do_sample=False # Greedy for deterministic reasoning
                            ) 
                            # r_out includes the "input" usually? 
                            # For inputs_embeds, it outputs only new tokens usually or full?
                            # HF generate with inputs_embeds: usually returns new tokens if decoder-only?
                            # Let's check: generated sequence usually includes prompt.
                            
                            # Flatten
                            r_seq = r_out[0]
                            reasoning_seqs.append(r_seq)
                        else:
                            reasoning_seqs.append(torch.tensor([], dtype=torch.long, device=model.device))
                    
                    # Pad reasoning seqs
                    max_r_len = max([len(x) for x in reasoning_seqs]) if reasoning_seqs else 0
                    if max_r_len > 0:
                        # Append to generated_tokens
                        # We need to rebuild generated_tokens to accommodate the extra length
                        # generated_tokens is already padded.
                        # We will construct a new list of concatenated tensors
                        new_gen_list = []
                        for i in range(batch_size):
                             g_ids = generated_tokens[i]
                             valid_g = g_ids[g_ids != self.processing_class.pad_token_id]
                             r_ids = reasoning_seqs[i]
                             # Concatenate
                             combined = torch.cat([valid_g, r_ids])
                             new_gen_list.append(combined)
                        
                        generated_tokens = pad_sequence(new_gen_list, batch_first=True, padding_value=self.processing_class.pad_token_id)
                        
                 except Exception as e:
                     logger.warning(f"Reasoning recovery failed: {e}")

            if generated_tokens is not None:
                generated_tokens[:, : inputs["input_ids"].size(-1)] = self.processing_class.pad_token_id
                generated_tokens = generated_tokens.contiguous()
            return loss, generated_tokens, labels

        # Non-generate eval: two forwards under no_grad (no DeepSpeed issues)
        reasoning_input_ids = inputs.pop("reasoning_input_ids", None)
        reasoning_labels = inputs.pop("reasoning_labels", None)
        inputs.pop("reasoning_attention_mask", None)
        special_token_mask = inputs.pop("special_token_mask", None)
        labels = inputs.get("labels")

        do_reasoning = (
            reasoning_input_ids is not None
            and special_token_mask is not None
            and special_token_mask.sum() > 0
        )

        if do_reasoning:
            inputs["output_hidden_states"] = True

        with torch.no_grad():
            loss_sft, outputs = super().compute_loss(model, inputs, return_outputs=True)

            loss_reasoning = torch.tensor(0.0, device=loss_sft.device)
            if do_reasoning:
                hs = getattr(outputs, "hidden_states", None)
                if isinstance(outputs, dict):
                    hs = outputs.get("hidden_states", hs)
                if hs is not None:
                    hidden_states = hs[-1]
                    reasoning_inputs = self._build_reasoning_inputs(
                        model, hidden_states, special_token_mask,
                        reasoning_input_ids, reasoning_labels
                    )
                    if reasoning_inputs is not None:
                        input_embeds, attn_mask, labels_padded = reasoning_inputs
                        outputs_r = model(
                            inputs_embeds=input_embeds,
                            attention_mask=attn_mask,
                            labels=labels_padded,
                        )
                        loss_reasoning = outputs_r.loss

        total_loss = loss_sft + loss_reasoning

        if not hasattr(self, "_eval_loss_buffer"):
            self._eval_loss_buffer = {"sft": [], "reasoning": []}
        self._eval_loss_buffer["sft"].append(loss_sft.detach().float().cpu())
        self._eval_loss_buffer["reasoning"].append(loss_reasoning.detach().float().cpu())

        logits = None
        if not prediction_loss_only:
            logits = outputs.get("logits") if isinstance(outputs, dict) else getattr(outputs, "logits", None)

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

    def save_predictions(
        self, dataset: "Dataset", predict_results: "PredictionOutput", skip_special_tokens: bool = True
    ) -> None:
        r"""Save model predictions to `output_dir`.

        A custom behavior that not contained in Seq2SeqTrainer.
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

        # Custom split logic for Answer and Reasoning
        # The structure of preds[i] is [Answer ... EOS ... Reasoning ... EOS]
        # We try to split by the first EOS token
        
        decoded_answers = []
        decoded_reasonings = []
        
        eos_id = self.processing_class.eos_token_id
        
        for i in range(len(preds)):
            seq = preds[i]
            # Strip padding first (which is at the end)
            valid_len = (seq != self.processing_class.pad_token_id).sum()
            seq = seq[:valid_len]
            
            # Find first EOS
            # We assume the first EOS separates Answer and Reasoning.
            # If no reasoning was generated (because no special tokens were found), it will be just Answer+EOS.
            
            eos_indices = np.where(seq == eos_id)[0]
            if len(eos_indices) > 0:
                first_eos = eos_indices[0]
                answer_seq = seq[:first_eos] # Exclude EOS
                
                # Check for reasoning part
                # If there are tokens after first EOS, they are reasoning
                if len(seq) > first_eos + 1:
                    reasoning_seq = seq[first_eos + 1:]
                    # Strip subsequent EOS if present in reasoning
                    # (Reasoning generation might have produced its own EOS)
                else:
                    reasoning_seq = np.array([], dtype=seq.dtype)
            else:
                # No EOS found? Just treat whole thing as answer
                answer_seq = seq
                reasoning_seq = np.array([], dtype=seq.dtype)
                
            ans_text = self.processing_class.decode(answer_seq, skip_special_tokens=skip_special_tokens)
            rea_text = self.processing_class.decode(reasoning_seq, skip_special_tokens=skip_special_tokens)
            
            decoded_answers.append(ans_text)
            decoded_reasonings.append(rea_text)

        with open(output_prediction_file, "w", encoding="utf-8") as f:
            for text, ans, rea, label in zip(decoded_inputs, decoded_answers, decoded_reasonings, decoded_labels):
                f.write(json.dumps({
                    "prompt": text, 
                    "predict_answer": ans, 
                    "predict_reasoning": rea, 
                    "label": label
                }, ensure_ascii=False) + "\n")
