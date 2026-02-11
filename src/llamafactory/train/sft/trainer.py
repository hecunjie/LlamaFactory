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

    def _compute_sft_and_reasoning_loss(
        self,
        model: "torch.nn.Module",
        inputs: dict[str, Union["torch.Tensor", Any]],
        num_items_in_batch: Optional[int] = None,
    ) -> tuple["torch.Tensor", "torch.Tensor", "torch.Tensor", Any]:
        r"""Shared loss computation for both training and evaluation.

        Returns:
            total_loss: loss_sft + loss_reasoning
            loss_sft: standard SFT cross-entropy loss
            loss_reasoning: reasoning recovery loss (0 if no reasoning data)
            outputs: model outputs (with logits, hidden_states, etc.)
        """
        # Pop custom fields
        reasoning_input_ids = inputs.pop("reasoning_input_ids", None)
        reasoning_labels = inputs.pop("reasoning_labels", None)
        inputs.pop("reasoning_attention_mask", None)
        special_token_mask = inputs.pop("special_token_mask", None)

        do_reasoning = reasoning_input_ids is not None and special_token_mask is not None
        if do_reasoning:
            inputs["output_hidden_states"] = True

        # Single forward pass — SFT loss
        loss_sft, outputs = super().compute_loss(
            model, inputs, return_outputs=True, num_items_in_batch=num_items_in_batch
        )

        # Reasoning loss via LM Head projection (no second forward)
        loss_reasoning = torch.tensor(0.0, device=loss_sft.device)
        if do_reasoning:
            hidden_states = getattr(outputs, "hidden_states", None)
            if isinstance(outputs, dict):
                hidden_states = outputs.get("hidden_states", hidden_states)

            if hidden_states is not None:
                last_hidden = hidden_states[-1]  # (batch, seq, dim)
                batch_size = last_hidden.size(0)

                unwrapped = model
                while hasattr(unwrapped, "module"):
                    unwrapped = unwrapped.module
                lm_head = unwrapped.lm_head

                total, valid = torch.tensor(0.0, device=loss_sft.device), 0
                for i in range(batch_size):
                    mask = special_token_mask[i]
                    n_special = mask.sum().item()
                    if n_special == 0:
                        continue

                    spec_h = last_hidden[i][mask == 1]  # (n_special, dim)
                    r_lbls = reasoning_labels[i]
                    valid_len = (reasoning_input_ids[i] != self.processing_class.pad_token_id).sum().item()
                    r_lbls = r_lbls[:valid_len]
                    n_predict = min(n_special, len(r_lbls))
                    if n_predict == 0:
                        continue

                    logits = lm_head(spec_h[:n_predict])
                    total = total + torch.nn.functional.cross_entropy(
                        logits, r_lbls[:n_predict], ignore_index=IGNORE_INDEX
                    )
                    valid += 1

                if valid > 0:
                    loss_reasoning = total / valid

        total_loss = loss_sft + loss_reasoning
        return total_loss, loss_sft, loss_reasoning, outputs

    # ---- Training ----

    @override
    def compute_loss(self, model, inputs, return_outputs=False, num_items_in_batch=None):
        total_loss, loss_sft, loss_reasoning, outputs = self._compute_sft_and_reasoning_loss(
            model, inputs, num_items_in_batch
        )

        if self.model.training:
            if not hasattr(self, "_custom_loss_buffer"):
                self._custom_loss_buffer = {"sft": [], "reasoning": []}
            self._custom_loss_buffer["sft"].append(loss_sft.detach().float())
            self._custom_loss_buffer["reasoning"].append(loss_reasoning.detach().float())

        return total_loss if not return_outputs else (total_loss, outputs)

    @override
    def log(self, logs: dict[str, float], start_time: Optional[float] = None) -> None:
        # Inject train-time split losses
        if hasattr(self, "_custom_loss_buffer") and self._custom_loss_buffer:
            for k, v in self._custom_loss_buffer.items():
                if v:
                    logs[f"loss_{k}"] = round(torch.stack(v).mean().item(), 4)
            self._custom_loss_buffer = {"sft": [], "reasoning": []}

        # Inject eval-time split losses (belt-and-suspenders with evaluation_loop)
        if hasattr(self, "_eval_loss_buffer") and self._eval_loss_buffer:
            for k, v in self._eval_loss_buffer.items():
                if v:
                    logs[f"eval_loss_{k}"] = round(torch.stack(v).mean().item(), 4)

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
            # Generation mode — pop custom fields and delegate
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

        # Non-generate eval: reuse the shared loss function
        labels = inputs.get("labels")
        with torch.no_grad():
            total_loss, loss_sft, loss_reasoning, outputs = self._compute_sft_and_reasoning_loss(model, inputs)

        # Buffer for aggregation in evaluation_loop
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
        # Inject split losses into metrics before evaluate() calls self.log()
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
        decoded_preds = self.processing_class.batch_decode(preds, skip_special_tokens=skip_special_tokens)
        decoded_labels = self.processing_class.batch_decode(labels, skip_special_tokens=skip_special_tokens)

        with open(output_prediction_file, "w", encoding="utf-8") as f:
            for text, pred, label in zip(decoded_inputs, decoded_preds, decoded_labels):
                f.write(json.dumps({"prompt": text, "predict": pred, "label": label}, ensure_ascii=False) + "\n")
