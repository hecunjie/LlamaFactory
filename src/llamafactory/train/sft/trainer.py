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

        # 2. Check if we should compute reasoning loss
        do_reasoning = (
            reasoning_input_ids is not None
            and special_token_mask is not None
        )

        if do_reasoning:
            inputs["output_hidden_states"] = True

        # 3. Single forward pass
        loss, outputs = super().compute_loss(model, inputs, return_outputs=True, num_items_in_batch=num_items_in_batch)

        # 4. Compute Reasoning Loss using LM Head projection (no second forward)
        added_reasoning_loss = torch.tensor(0.0, device=loss.device)
        if do_reasoning:
            hidden_states = getattr(outputs, "hidden_states", None)
            if isinstance(outputs, dict):
                hidden_states = outputs.get("hidden_states", hidden_states)

            if hidden_states is not None:
                last_hidden_state = hidden_states[-1]  # (batch, seq, dim)
                batch_size = last_hidden_state.size(0)

                # Get LM Head from the model
                unwrapped_model = model
                while hasattr(unwrapped_model, "module"):
                    unwrapped_model = unwrapped_model.module
                lm_head = unwrapped_model.lm_head

                total_reasoning_loss = torch.tensor(0.0, device=loss.device)
                valid_samples = 0

                for i in range(batch_size):
                    mask = special_token_mask[i]  # (seq,)
                    n_special = mask.sum().item()
                    if n_special == 0:
                        continue

                    # Get hidden states at special token positions
                    spec_h = last_hidden_state[i][mask == 1]  # (n_special, dim)

                    # Get reasoning labels for this sample
                    r_lbls = reasoning_labels[i]
                    valid_len = (reasoning_input_ids[i] != self.processing_class.pad_token_id).sum().item()
                    r_lbls = r_lbls[:valid_len]  # (n_reasoning,)

                    # Match lengths: each special token predicts a reasoning token
                    n_predict = min(n_special, len(r_lbls))
                    if n_predict == 0:
                        continue

                    # Project special token hidden states through LM Head to get logits
                    spec_logits = lm_head(spec_h[:n_predict])  # (n_predict, vocab_size)

                    # Cross-entropy loss: predict reasoning tokens from special token hidden states
                    reasoning_loss_i = torch.nn.functional.cross_entropy(
                        spec_logits, r_lbls[:n_predict], ignore_index=IGNORE_INDEX
                    )
                    total_reasoning_loss = total_reasoning_loss + reasoning_loss_i
                    valid_samples += 1

                if valid_samples > 0:
                    added_reasoning_loss = total_reasoning_loss / valid_samples
                    loss = loss + added_reasoning_loss

        # Log Losses
        if self.model.training:
            if not hasattr(self, "_custom_loss_buffer"):
                self._custom_loss_buffer = {"reasoning": [], "sft": []}

            sft_loss = loss - added_reasoning_loss
            self._custom_loss_buffer["sft"].append(sft_loss.detach().float())
            self._custom_loss_buffer["reasoning"].append(added_reasoning_loss.detach().float())

        return loss if not return_outputs else (loss, outputs)

    @override
    def log(self, logs: dict[str, float], start_time: Optional[float] = None) -> None:
        if hasattr(self, "_custom_loss_buffer") and self._custom_loss_buffer:
             # Average and add to logs
             for k, v in self._custom_loss_buffer.items():
                 if v: # Check if list is not empty
                     avg = torch.stack(v).mean().item()
                     logs[f"loss_{k}"] = round(avg, 4)
             self._custom_loss_buffer = {"sft": [], "reasoning": []}
             
        if start_time is not None:
             super().log(logs, start_time)
        else:
             super().log(logs)

    @override
    def prediction_step(
        self,
        model: "torch.nn.Module",
        inputs: dict[str, Union["torch.Tensor", Any]],
        prediction_loss_only: bool,
        ignore_keys: Optional[list[str]] = None,
        **gen_kwargs,
    ) -> tuple[Optional[float], Optional["torch.Tensor"], Optional["torch.Tensor"]]:
        r"""Override to compute reasoning loss during eval and remove custom fields for generate."""
        # Extract custom fields
        reasoning_input_ids = inputs.pop("reasoning_input_ids", None)
        reasoning_labels = inputs.pop("reasoning_labels", None)
        inputs.pop("reasoning_attention_mask", None)
        special_token_mask = inputs.pop("special_token_mask", None)

        if self.args.predict_with_generate:
            # Generation mode: no reasoning loss, just generate
            labels = inputs.pop("labels", None)
            loss, generated_tokens, _ = super().prediction_step(
                model, inputs, prediction_loss_only=prediction_loss_only, ignore_keys=ignore_keys, **gen_kwargs
            )
            if generated_tokens is not None:
                generated_tokens[:, : inputs["input_ids"].size(-1)] = self.processing_class.pad_token_id
                generated_tokens = generated_tokens.contiguous()
            return loss, generated_tokens, labels
        else:
            # Non-generate eval: compute loss with reasoning split
            labels = inputs.get("labels")

            do_reasoning = (
                reasoning_input_ids is not None
                and special_token_mask is not None
            )

            if do_reasoning:
                inputs["output_hidden_states"] = True

            # Standard forward for SFT loss
            with torch.no_grad():
                loss, outputs = super().compute_loss(model, inputs, return_outputs=True)

            # Compute reasoning loss (same logic as compute_loss)
            added_reasoning_loss = torch.tensor(0.0, device=loss.device)
            if do_reasoning:
                hidden_states = getattr(outputs, "hidden_states", None)
                if isinstance(outputs, dict):
                    hidden_states = outputs.get("hidden_states", hidden_states)

                if hidden_states is not None:
                    last_hidden_state = hidden_states[-1]
                    batch_size = last_hidden_state.size(0)

                    unwrapped_model = model
                    while hasattr(unwrapped_model, "module"):
                        unwrapped_model = unwrapped_model.module
                    lm_head = unwrapped_model.lm_head

                    total_reasoning_loss = torch.tensor(0.0, device=loss.device)
                    valid_samples = 0

                    for i in range(batch_size):
                        mask = special_token_mask[i]
                        n_special = mask.sum().item()
                        if n_special == 0:
                            continue

                        spec_h = last_hidden_state[i][mask == 1]
                        r_lbls = reasoning_labels[i]
                        valid_len = (reasoning_input_ids[i] != self.processing_class.pad_token_id).sum().item()
                        r_lbls = r_lbls[:valid_len]

                        n_predict = min(n_special, len(r_lbls))
                        if n_predict == 0:
                            continue

                        spec_logits = lm_head(spec_h[:n_predict])
                        reasoning_loss_i = torch.nn.functional.cross_entropy(
                            spec_logits, r_lbls[:n_predict], ignore_index=IGNORE_INDEX
                        )
                        total_reasoning_loss = total_reasoning_loss + reasoning_loss_i
                        valid_samples += 1

                    if valid_samples > 0:
                        added_reasoning_loss = total_reasoning_loss / valid_samples

            # Buffer eval losses for later aggregation
            if not hasattr(self, "_eval_loss_buffer"):
                self._eval_loss_buffer = {"sft": [], "reasoning": []}
            self._eval_loss_buffer["sft"].append(loss.detach().float())
            self._eval_loss_buffer["reasoning"].append(added_reasoning_loss.detach().float())

            total_loss = loss + added_reasoning_loss

            # Get logits for metrics (e.g. exact match via argmax)
            logits = None
            if not prediction_loss_only:
                if isinstance(outputs, dict):
                    logits = outputs.get("logits")
                elif hasattr(outputs, "logits"):
                    logits = outputs.logits

            return total_loss.detach(), logits, labels

    @override
    def evaluation_loop(self, *args, **kwargs):
        # Reset eval loss buffer before eval loop
        self._eval_loss_buffer = {"sft": [], "reasoning": []}
        output = super().evaluation_loop(*args, **kwargs)
        # Inject split losses into metrics BEFORE evaluate() logs them
        if self._eval_loss_buffer:
            for k, v in self._eval_loss_buffer.items():
                if v:
                    avg = torch.stack(v).mean().item()
                    output.metrics[f"eval_loss_{k}"] = round(avg, 4)
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
        decoded_preds = self.processing_class.batch_decode(preds, skip_special_tokens=skip_special_tokens)
        decoded_labels = self.processing_class.batch_decode(labels, skip_special_tokens=skip_special_tokens)

        with open(output_prediction_file, "w", encoding="utf-8") as f:
            for text, pred, label in zip(decoded_inputs, decoded_preds, decoded_labels):
                f.write(json.dumps({"prompt": text, "predict": pred, "label": label}, ensure_ascii=False) + "\n")
