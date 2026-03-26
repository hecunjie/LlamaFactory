# Copyright 2025 HuggingFace Inc. and the LlamaFactory team.
#
# This code is inspired by the HuggingFace's transformers library.
# https://github.com/huggingface/transformers/blob/v4.40.0/examples/pytorch/summarization/run_summarization.py
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

from typing import TYPE_CHECKING, Optional

from ...data import SFTDataCollatorWith4DAttentionMask, get_dataset, get_template_and_fix_tokenizer
from ...extras.constants import IGNORE_INDEX
from ...extras.logging import get_logger
from ...extras.misc import calculate_tps
from ...extras.packages import is_transformers_version_greater_than
from ...extras.ploting import plot_loss
from ...model import load_model, load_tokenizer
from ..trainer_utils import create_modelcard_and_push
from .metric import ComputeAccuracy, ComputeExactMatch, ComputeSimilarity, eval_logit_processor
from .trainer import CustomSeq2SeqTrainer


if TYPE_CHECKING:
    from transformers import Seq2SeqTrainingArguments, TrainerCallback

    from ...hparams import DataArguments, FinetuningArguments, GeneratingArguments, ModelArguments


logger = get_logger(__name__)


def run_sft(
    model_args: "ModelArguments",
    data_args: "DataArguments",
    training_args: "Seq2SeqTrainingArguments",
    finetuning_args: "FinetuningArguments",
    generating_args: "GeneratingArguments",
    callbacks: Optional[list["TrainerCallback"]] = None,
):
    tokenizer_module = load_tokenizer(model_args)
    tokenizer = tokenizer_module["tokenizer"]

    # Auto-add latent thinking tokens when num_latent_thinking_token > 0
    if data_args.num_latent_thinking_token > 0:
        latent_tokens = [f"<latent_{i}>" for i in range(data_args.num_latent_thinking_token)]
        num_added = tokenizer.add_tokens(latent_tokens, special_tokens=True)
        if num_added > 0:
            logger.info(f"Added {num_added} latent thinking tokens to vocabulary: {latent_tokens}")

    # Add <add_think> to tokenizer before dataset is built so it is tokenized as one token.
    # - `add_think_token`: normal SFT (standard token embeddings; no recurrent forward).
    # - `recurrent_add_think_training`: same registration + recurrent hidden-as-input path.
    # For offline analysis only, reuse tokenizer from a saved checkpoint that already has
    # <add_think> to avoid duplicate add_tokens / id drift.
    add_think_added = 0
    if (
        getattr(finetuning_args, "add_think_token", False)
        or getattr(finetuning_args, "recurrent_add_think_training", False)
        or getattr(finetuning_args, "use_align_loss", False)
        or getattr(finetuning_args, "use_ortho_loss", False)
    ):
        add_think_added = tokenizer.add_tokens(["<add_think>"], special_tokens=True)
        if add_think_added > 0:
            logger.info(
                "Added '<add_think>' to tokenizer "
                f"(new vocab size={len(tokenizer)}; "
                f"add_think_token={getattr(finetuning_args, 'add_think_token', False)}, "
                f"recurrent_add_think_training={getattr(finetuning_args, 'recurrent_add_think_training', False)}, "
                f"use_align_loss={getattr(finetuning_args, 'use_align_loss', False)}, "
                f"use_ortho_loss={getattr(finetuning_args, 'use_ortho_loss', False)})"
            )

    template = get_template_and_fix_tokenizer(tokenizer, data_args)
    dataset_module = get_dataset(template, model_args, data_args, training_args, stage="sft", **tokenizer_module)
    model = load_model(tokenizer, model_args, finetuning_args, training_args.do_train)

    # Resize embeddings when tokenizer vocab != model embedding rows (e.g. new <add_think> or
    # tokenizer already had the token but base checkpoint did not).
    if (
        getattr(finetuning_args, "add_think_token", False)
        or getattr(finetuning_args, "recurrent_add_think_training", False)
        or getattr(finetuning_args, "use_align_loss", False)
        or getattr(finetuning_args, "use_ortho_loss", False)
    ):
        unwrapped = model
        while hasattr(unwrapped, "module"):
            unwrapped = unwrapped.module
        vocab_size = len(tokenizer)
        emb = unwrapped.get_input_embeddings()
        if emb is not None and emb.num_embeddings != vocab_size:
            unwrapped.resize_token_embeddings(vocab_size)
            if getattr(unwrapped.config, "tie_word_embeddings", False) and hasattr(unwrapped, "tie_weights"):
                unwrapped.tie_weights()
            logger.info_rank0(
                f"Resized model token embeddings to {vocab_size} for '<add_think>' "
                f"(add_think_added={add_think_added})."
            )

    # Add a dedicated LayerNorm for mapping latent hidden states → embedding-scale inputs.
    # The model's final_norm (RMSNorm) is trained for the LM head and does NOT produce
    # outputs with the same distribution as word embeddings. This learnable LayerNorm
    # bridges that gap. Attached to the model so DeepSpeed / optimizer manage it.
    if data_args.num_latent_thinking_token > 0:
        import torch.nn as nn
        hidden_size = model.config.hidden_size
        model.latent_hidden_norm = nn.LayerNorm(hidden_size)
        logger.info(f"Added latent_hidden_norm (LayerNorm, dim={hidden_size}) to model for latent chain.")

    if getattr(model, "is_quantized", False) and not training_args.do_train:
        setattr(model, "_hf_peft_config_loaded", True)  # hack here: make model compatible with prediction

    data_collator = SFTDataCollatorWith4DAttentionMask(
        template=template,
        model=model if not training_args.predict_with_generate else None,
        pad_to_multiple_of=8 if training_args.do_train else None,  # for shift short attention
        label_pad_token_id=IGNORE_INDEX if data_args.ignore_pad_token_for_loss else tokenizer.pad_token_id,
        block_diag_attn=model_args.block_diag_attn,
        attn_implementation=getattr(model.config, "_attn_implementation", None),
        compute_dtype=model_args.compute_dtype,
        **tokenizer_module,
    )

    # Metric utils
    metric_module = {}
    if model_args.use_kt:
        if training_args.predict_with_generate:
            raise NotImplementedError("`predict_with_generate` is not supported in KTransformers SFT yet.")
        elif finetuning_args.compute_accuracy:
            raise NotImplementedError("`compute_accuracy` is not supported in KTransformers SFT yet.")

    if training_args.predict_with_generate:
        if finetuning_args.compute_exact_match:
             metric_module["compute_metrics"] = ComputeExactMatch(tokenizer=tokenizer)
        else:
             metric_module["compute_metrics"] = ComputeSimilarity(tokenizer=tokenizer)
    elif finetuning_args.compute_exact_match:
        metric_module["compute_metrics"] = ComputeExactMatch(tokenizer=tokenizer)
        metric_module["preprocess_logits_for_metrics"] = eval_logit_processor
    elif finetuning_args.compute_accuracy:
        metric_module["compute_metrics"] = ComputeAccuracy()
        metric_module["preprocess_logits_for_metrics"] = eval_logit_processor


    # Keyword arguments for `model.generate`
    gen_kwargs = generating_args.to_dict(obey_generation_config=True)

    # Compatible with Transformers v4 and Transformers v5
    if is_transformers_version_greater_than("4.58.0"):
        extra_ids = getattr(tokenizer, "additional_special_tokens_ids", None)
        if not isinstance(extra_ids, list):
            extra_special_tokens = getattr(tokenizer, "_extra_special_tokens", [])
            string_tokens = [str(t) for t in extra_special_tokens]
            extra_ids = tokenizer.convert_tokens_to_ids(string_tokens)
        add_think_id = tokenizer.convert_tokens_to_ids("<add_think>")
        extra_ids = [i for i in extra_ids if i != -1 and i != add_think_id]
        all_eos_ids = [tokenizer.eos_token_id] + [i for i in extra_ids if i != -1]
        unique_eos_ids = list(dict.fromkeys(all_eos_ids))
        gen_kwargs["eos_token_id"] = unique_eos_ids
    else:
        add_think_id = tokenizer.convert_tokens_to_ids("<add_think>")
        eos_ids = [tokenizer.eos_token_id] + tokenizer.additional_special_tokens_ids
        gen_kwargs["eos_token_id"] = [i for i in eos_ids if i != add_think_id]
    gen_kwargs["pad_token_id"] = tokenizer.pad_token_id

    # Initialize our Trainer
    if model_args.use_kt:
        from ktransformers.sft.lora import KTrainer  # type: ignore
        from ktransformers.util.globals import GLOBAL_CONFIG  # type: ignore

        GLOBAL_CONFIG._config["mod"] = "sft"

        trainer = KTrainer(
            model=model,
            args=training_args,
            tokenizer=tokenizer_module,
            data_collator=data_collator,
            callbacks=callbacks,
            **dataset_module,
            **metric_module,
        )
        trainer.model_accepts_loss_kwargs = False
        model.config.use_cache = False

    else:
        trainer = CustomSeq2SeqTrainer(
            model=model,
            args=training_args,
            finetuning_args=finetuning_args,
            data_collator=data_collator,
            callbacks=callbacks,
            gen_kwargs=gen_kwargs,
            data_args=data_args,
            **dataset_module,
            **tokenizer_module,
            **metric_module,
        )

    # Training
    if training_args.do_train:
        train_result = trainer.train(resume_from_checkpoint=training_args.resume_from_checkpoint)
        trainer.save_model()
        if finetuning_args.include_effective_tokens_per_second:
            train_result.metrics["effective_tokens_per_sec"] = calculate_tps(
                dataset_module["train_dataset"], train_result.metrics, stage="sft"
            )

        trainer.log_metrics("train", train_result.metrics)
        trainer.save_metrics("train", train_result.metrics)
        trainer.save_state()
        if trainer.is_world_process_zero() and finetuning_args.plot_loss:
            keys = ["loss"]
            if isinstance(dataset_module.get("eval_dataset"), dict):
                keys += sum(
                    [[f"eval_{key}_loss", f"eval_{key}_accuracy"] for key in dataset_module["eval_dataset"].keys()], []
                )
            else:
                keys += ["eval_loss", "eval_accuracy"]

            plot_loss(training_args.output_dir, keys=keys)

    if training_args.predict_with_generate:
        tokenizer.padding_side = "left"  # use left-padding in generation

    # Evaluation
    if training_args.do_eval:
        metrics = trainer.evaluate(metric_key_prefix="eval", **gen_kwargs)
        trainer.log_metrics("eval", metrics)
        trainer.save_metrics("eval", metrics)

    # Predict
    if training_args.do_predict:
        logger.warning_rank0_once("Batch generation can be very slow. Consider using `scripts/vllm_infer.py` instead.")
        predict_results = trainer.predict(dataset_module["eval_dataset"], metric_key_prefix="predict", **gen_kwargs)
        trainer.log_metrics("predict", predict_results.metrics)
        trainer.save_metrics("predict", predict_results.metrics)
        trainer.save_predictions(dataset_module["eval_dataset"], predict_results, generating_args.skip_special_tokens)

    # Entropy-strategy analysis
    if finetuning_args.do_entropy_analysis:
        eval_dataset = dataset_module.get("eval_dataset")
        if eval_dataset is None:
            logger.warning_rank0("No eval_dataset found. Skipping entropy analysis.")
        else:
            # If eval_dataset is a dict (multi-dataset), analyse each split
            if isinstance(eval_dataset, dict):
                for split_name, split_ds in eval_dataset.items():
                    logger.info_rank0(f"Running entropy analysis on split: {split_name}")
                    trainer.analyze_entropy_strategies(
                        dataset=split_ds,
                        top_k_entropy_pct=finetuning_args.entropy_top_k_positions,
                        top_k_tokens=finetuning_args.entropy_top_k_tokens,
                        blend_alpha_sweep=finetuning_args.entropy_blend_alpha_sweep,
                        blend_alpha_steps=finetuning_args.entropy_blend_alpha_steps,
                        max_new_tokens=finetuning_args.entropy_max_new_tokens,
                        logit_weight_threshold=finetuning_args.entropy_logit_weight_threshold,
                        blend_alpha=finetuning_args.entropy_blend_alpha,
                        use_answer_tokens=finetuning_args.entropy_use_answer_tokens,
                        hidden_drop_last_kv=finetuning_args.entropy_hidden_drop_last_kv,
                        fit_hidden_to_topk=finetuning_args.entropy_fit_hidden_to_topk,
                        analyze_at_add_think_positions=finetuning_args.entropy_analyze_at_add_think_positions,
                    )
            else:
                trainer.analyze_entropy_strategies(
                    dataset=eval_dataset,
                    top_k_entropy_pct=finetuning_args.entropy_top_k_positions,
                    top_k_tokens=finetuning_args.entropy_top_k_tokens,
                    blend_alpha_sweep=finetuning_args.entropy_blend_alpha_sweep,
                    blend_alpha_steps=finetuning_args.entropy_blend_alpha_steps,
                    max_new_tokens=finetuning_args.entropy_max_new_tokens,
                    logit_weight_threshold=finetuning_args.entropy_logit_weight_threshold,
                    blend_alpha=finetuning_args.entropy_blend_alpha,
                    use_answer_tokens=finetuning_args.entropy_use_answer_tokens,
                    hidden_drop_last_kv=finetuning_args.entropy_hidden_drop_last_kv,
                    fit_hidden_to_topk=finetuning_args.entropy_fit_hidden_to_topk,
                    analyze_at_add_think_positions=finetuning_args.entropy_analyze_at_add_think_positions,
                )

    # Mark low-confidence positions with <add_think>
    if finetuning_args.do_mark_low_confidence:
        eval_dataset = dataset_module.get("eval_dataset")
        if eval_dataset is None:
            logger.warning_rank0("No eval_dataset found. Skipping low-confidence marking.")
        else:
            if isinstance(eval_dataset, dict):
                for split_name, split_ds in eval_dataset.items():
                    logger.info_rank0(f"Running low-confidence marking on split: {split_name}")
                    trainer.mark_low_confidence_positions(
                        dataset=split_ds,
                        prob_threshold=finetuning_args.low_confidence_prob_threshold,
                        entropy_threshold=finetuning_args.low_confidence_entropy_threshold,
                        sim_threshold=finetuning_args.low_confidence_sim_threshold,
                        insert_position=finetuning_args.low_confidence_insert_position,
                    )
            else:
                trainer.mark_low_confidence_positions(
                    dataset=eval_dataset,
                    prob_threshold=finetuning_args.low_confidence_prob_threshold,
                    entropy_threshold=finetuning_args.low_confidence_entropy_threshold,
                    sim_threshold=finetuning_args.low_confidence_sim_threshold,
                    insert_position=finetuning_args.low_confidence_insert_position,
                )

    # Create model card
    create_modelcard_and_push(trainer, model_args, data_args, training_args, finetuning_args)
