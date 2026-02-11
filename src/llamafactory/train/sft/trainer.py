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

    @override
    def compute_loss(self, model, inputs, return_outputs=False, num_items_in_batch=None):
        # 1. Pop custom fields
        reasoning_input_ids = inputs.pop("reasoning_input_ids", None)
        reasoning_labels = inputs.pop("reasoning_labels", None)
        special_token_mask = inputs.pop("special_token_mask", None)
        
        # 2. Add hidden states request if needed for reasoning
        # Check if we should do the reasoning loss pass
        do_reasoning = (
            reasoning_input_ids is not None 
            and special_token_mask is not None
        )
        
        if do_reasoning:
            inputs["output_hidden_states"] = True

        # 3. Call original compute_loss (which runs the model)
        # We use super() which eventually calls model(**inputs)
        # Note: self.compute_loss_func might be set by CustomSeq2SeqTrainer init.
        # super().compute_loss handles calling it if it handles it (HF Trainer allows overriding compute_loss, 
        # but to use a custom loss func inside standard compute_loss, one usually overrides `compute_loss` or assigns `self.compute_loss_func` if HF supports it. 
        # HF Trainer supports `self.compute_loss_func`? No, it's a recent-ish addition or custom?
        # Checking HF source: Trainer.compute_loss checks `if self.compute_loss_func is not None`.
        # So we can safely call super().
        
        if return_outputs:
             loss, outputs = super().compute_loss(model, inputs, return_outputs=True, num_items_in_batch=num_items_in_batch)
        else:
             # We force return_outputs=True to get hidden states if we need them
             loss, outputs = super().compute_loss(model, inputs, return_outputs=True, num_items_in_batch=num_items_in_batch)

        # 4. Compute Reasoning Loss
        if do_reasoning:
            # outputs is (loss, logits, hidden_states, ...) or Seq2SeqLMOutput
            if isinstance(outputs, dict):
                hidden_states = outputs.get("hidden_states")
            elif hasattr(outputs, "hidden_states"):
                hidden_states = outputs.hidden_states
            else:
                # Tuple: loss, logits, hidden_states (if output_hidden_states=True)
                # HF standard: if output_hidden_states=True, hidden_states is usually 3rd or inside.
                # For CausalLM: (loss, logits, past_key_values, hidden_states, attentions)
                # Let's hope it's index 2 or 3 depending on return_dict.
                # Usually return_dict=True by default in TrainingArguments?
                hidden_states = outputs.hidden_states if hasattr(outputs, "hidden_states") else None

            if hidden_states is not None:
                # visible_hidden_states: Tuple of tensor (one for each layer). We want last layer.
                last_hidden_state = hidden_states[-1] # (batch, seq, dim)
                
                # Construct inputs for reasoning pass
                # We need to gather the hidden states corresponding to special tokens.
                # special_token_mask: (batch, seq) - 1 for special tokens.
                
                # We iterate over batch to handle potentially different lengths or positions
                batch_size = last_hidden_state.size(0)
                reasoning_loss = 0.0
                valid_samples = 0
                
                proj_inputs_list = []
                reasoning_labels_list = []
                
                for i in range(batch_size):
                    mask = special_token_mask[i]
                    if mask.sum() == 0:
                        continue
                        
                    # Extract hidden states
                    # Logic: We take all hidden states marked by mask.
                    # These form the "prompt" for the reasoning.
                    spec_tokens_h = last_hidden_state[i][mask == 1] # (k, dim)
                    
                    # Reasoning input ids
                    # These are standard token ids [r1, r2, ..., rn, eos]
                    # We need embeddings.
                    r_ids = reasoning_input_ids[i]
                    r_lbls = reasoning_labels[i]
                    
                    # Filter padding from reasoning if any (labels=IGNORE_INDEX or input_ids=PAD)
                    # Assuming standard padding (right)
                    valid_len = (r_ids != self.processing_class.pad_token_id).sum()
                    r_ids = r_ids[:valid_len]
                    r_lbls = r_lbls[:valid_len]

                    # Get embeddings
                    # model.get_input_embeddings() works for CausalLM
                    # For Seq2Seq (Encoder-Decoder), this logic is different.
                    # Assuming CausalLM (LlamaFactory mostly SFTs Decoder models like Llama, Qwen).
                    model_embeds = model.get_input_embeddings()
                    r_embeds = model_embeds(r_ids) # (len, dim)
                    
                    # Concat
                    # Input: [SpecialTokensH, ReasoningEmbeds]
                    combined_embeds = torch.cat([spec_tokens_h, r_embeds], dim=0) # (k + len, dim)
                    
                    # Labels
                    # Target: [IGNORE... (for SpecialTokens), ReasoningLabels]
                    # Note: CausalLM shifts labels internally. 
                    # If we pass labels matching input length, it calculates loss.
                    # We want to predict Reasoning.
                    # Labels for SpecialTokens part should be IGNORE_INDEX
                    prefix_labels = torch.full((spec_tokens_h.size(0),), IGNORE_INDEX, device=last_hidden_state.device, dtype=torch.long)
                    combined_labels = torch.cat([prefix_labels, r_lbls], dim=0)
                    
                    proj_inputs_list.append(combined_embeds)
                    reasoning_labels_list.append(combined_labels)
                    valid_samples += 1

                if valid_samples > 0:
                    # Pad sequences to batch them
                    from torch.nn.utils.rnn import pad_sequence
                    
                    # Pad inputs_embeds
                    # We can't use pad_sequence on embeddings directly easily if used with forward?
                    # Yes we can. Pad with 0.
                    input_embeds_padded = pad_sequence(proj_inputs_list, batch_first=True, padding_value=0.0)
                    labels_padded = pad_sequence(reasoning_labels_list, batch_first=True, padding_value=IGNORE_INDEX)
                    
                    # Attention mask
                    # 1 for real tokens, 0 for pad
                    # lengths are in [x.size(0) for x in proj_inputs_list]
                    # Create mask
                    seq_lens = [x.size(0) for x in proj_inputs_list]
                    max_len = max(seq_lens)
                    att_mask = torch.zeros((valid_samples, max_len), dtype=torch.long, device=last_hidden_state.device)
                    for idx, l in enumerate(seq_lens):
                        att_mask[idx, :l] = 1
                        
                    # Forward Pass 2
                    # Note: We must NOT pass input_ids, only inputs_embeds
                    outputs_reasoning = model(
                        inputs_embeds=input_embeds_padded,
                        attention_mask=att_mask,
                        labels=labels_padded
                    )
                    
                    batch_reasoning_loss = outputs_reasoning.loss
                    loss += batch_reasoning_loss

        return loss if not return_outputs else (loss, outputs)

    @override
    def prediction_step(
        self,
        model: "torch.nn.Module",
        inputs: dict[str, Union["torch.Tensor", Any]],
        prediction_loss_only: bool,
        ignore_keys: Optional[list[str]] = None,
        **gen_kwargs,
    ) -> tuple[Optional[float], Optional["torch.Tensor"], Optional["torch.Tensor"]]:
        r"""Remove the prompt part in the generated tokens.

        Subclass and override to inject custom behavior.
        """
        if self.args.predict_with_generate:  # do not pass labels to model when generate
            labels = inputs.pop("labels", None)
        else:
            labels = inputs.get("labels")

        loss, generated_tokens, _ = super().prediction_step(
            model, inputs, prediction_loss_only=prediction_loss_only, ignore_keys=ignore_keys, **gen_kwargs
        )
        if generated_tokens is not None and self.args.predict_with_generate:
            generated_tokens[:, : inputs["input_ids"].size(-1)] = self.processing_class.pad_token_id
            generated_tokens = generated_tokens.contiguous()

        return loss, generated_tokens, labels

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
        decoded_preds = self.processing_class.batch_decode(preds, skip_special_tokens=skip_special_tokens)
        decoded_labels = self.processing_class.batch_decode(labels, skip_special_tokens=skip_special_tokens)

        with open(output_prediction_file, "w", encoding="utf-8") as f:
            for text, pred, label in zip(decoded_inputs, decoded_preds, decoded_labels):
                f.write(json.dumps({"prompt": text, "predict": pred, "label": label}, ensure_ascii=False) + "\n")
