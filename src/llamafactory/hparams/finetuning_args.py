# Copyright 2025 the LlamaFactory team.
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

from dataclasses import asdict, dataclass, field
from typing import Any, Literal


@dataclass
class FreezeArguments:
    r"""Arguments pertaining to the freeze (partial-parameter) training."""

    freeze_trainable_layers: int = field(
        default=2,
        metadata={
            "help": (
                "The number of trainable layers for freeze (partial-parameter) fine-tuning. "
                "Positive numbers mean the last n layers are set as trainable, "
                "negative numbers mean the first n layers are set as trainable."
            )
        },
    )
    freeze_trainable_modules: str = field(
        default="all",
        metadata={
            "help": (
                "Name(s) of trainable modules for freeze (partial-parameter) fine-tuning. "
                "Use commas to separate multiple modules. "
                "Use `all` to specify all the available modules."
            )
        },
    )
    freeze_extra_modules: str | None = field(
        default=None,
        metadata={
            "help": (
                "Name(s) of modules apart from hidden layers to be set as trainable "
                "for freeze (partial-parameter) fine-tuning. "
                "Use commas to separate multiple modules."
            )
        },
    )


@dataclass
class LoraArguments:
    r"""Arguments pertaining to the LoRA training."""

    additional_target: str | None = field(
        default=None,
        metadata={
            "help": (
                "Name(s) of modules apart from LoRA layers to be set as trainable "
                "and saved in the final checkpoint. "
                "Use commas to separate multiple modules."
            )
        },
    )
    lora_alpha: int | None = field(
        default=None,
        metadata={"help": "The scale factor for LoRA fine-tuning (default: lora_rank * 2)."},
    )
    lora_dropout: float = field(
        default=0.0,
        metadata={"help": "Dropout rate for the LoRA fine-tuning."},
    )
    lora_rank: int = field(
        default=8,
        metadata={"help": "The intrinsic dimension for LoRA fine-tuning."},
    )
    lora_target: str = field(
        default="all",
        metadata={
            "help": (
                "Name(s) of target modules to apply LoRA. "
                "Use commas to separate multiple modules. "
                "Use `all` to specify all the linear modules."
            )
        },
    )
    loraplus_lr_ratio: float | None = field(
        default=None,
        metadata={"help": "LoRA plus learning rate ratio (lr_B / lr_A)."},
    )
    loraplus_lr_embedding: float = field(
        default=1e-6,
        metadata={"help": "LoRA plus learning rate for lora embedding layers."},
    )
    use_rslora: bool = field(
        default=False,
        metadata={"help": "Whether or not to use the rank stabilization scaling factor for LoRA layer."},
    )
    use_dora: bool = field(
        default=False,
        metadata={"help": "Whether or not to use the weight-decomposed lora method (DoRA)."},
    )
    pissa_init: bool = field(
        default=False,
        metadata={"help": "Whether or not to initialize a PiSSA adapter."},
    )
    pissa_iter: int = field(
        default=16,
        metadata={"help": "The number of iteration steps performed by FSVD in PiSSA. Use -1 to disable it."},
    )
    pissa_convert: bool = field(
        default=False,
        metadata={"help": "Whether or not to convert the PiSSA adapter to a normal LoRA adapter."},
    )
    create_new_adapter: bool = field(
        default=False,
        metadata={"help": "Whether or not to create a new adapter with randomly initialized weight."},
    )


@dataclass
class OFTArguments:
    r"""Arguments pertaining to the OFT training."""

    additional_target: str | None = field(
        default=None,
        metadata={
            "help": (
                "Name(s) of modules apart from LoRA layers to be set as trainable "
                "and saved in the final checkpoint. "
                "Use commas to separate multiple modules."
            )
        },
    )
    module_dropout: float = field(
        default=0.0,
        metadata={"help": "Dropout rate for the OFT fine-tuning."},
    )
    oft_rank: int = field(
        default=0,
        metadata={"help": "The intrinsic dimension for OFT fine-tuning."},
    )
    oft_block_size: int = field(
        default=32,
        metadata={"help": "The intrinsic dimension for OFT fine-tuning."},
    )
    oft_target: str = field(
        default="all",
        metadata={
            "help": (
                "Name(s) of target modules to apply OFT. "
                "Use commas to separate multiple modules. "
                "Use `all` to specify all the linear modules."
            )
        },
    )
    create_new_adapter: bool = field(
        default=False,
        metadata={"help": "Whether or not to create a new adapter with randomly initialized weight."},
    )


@dataclass
class RLHFArguments:
    r"""Arguments pertaining to the PPO, DPO and KTO training."""

    pref_beta: float = field(
        default=0.1,
        metadata={"help": "The beta parameter in the preference loss."},
    )
    pref_ftx: float = field(
        default=0.0,
        metadata={"help": "The supervised fine-tuning loss coefficient in DPO training."},
    )
    pref_bco_weight: float = field(
        default=0.0,
        metadata={"help": "The Binary Classifier Optimization coefficient in DPO training."},
    )
    pref_loss: Literal["sigmoid", "hinge", "ipo", "kto_pair", "orpo", "simpo"] = field(
        default="sigmoid",
        metadata={"help": "The type of DPO loss to use."},
    )
    dpo_label_smoothing: float = field(
        default=0.0,
        metadata={"help": "The robust DPO label smoothing parameter in cDPO that should be between 0 and 0.5."},
    )
    kto_chosen_weight: float = field(
        default=1.0,
        metadata={"help": "The weight factor of the desirable losses in KTO training."},
    )
    kto_rejected_weight: float = field(
        default=1.0,
        metadata={"help": "The weight factor of the undesirable losses in KTO training."},
    )
    simpo_gamma: float = field(
        default=0.5,
        metadata={"help": "The target reward margin term in SimPO loss."},
    )
    ppo_buffer_size: int = field(
        default=1,
        metadata={"help": "The number of mini-batches to make experience buffer in a PPO optimization step."},
    )
    ppo_epochs: int = field(
        default=4,
        metadata={"help": "The number of epochs to perform in a PPO optimization step."},
    )
    ppo_score_norm: bool = field(
        default=False,
        metadata={"help": "Use score normalization in PPO training."},
    )
    ppo_target: float = field(
        default=6.0,
        metadata={"help": "Target KL value for adaptive KL control in PPO training."},
    )
    ppo_whiten_rewards: bool = field(
        default=False,
        metadata={"help": "Whiten the rewards before compute advantages in PPO training."},
    )
    ref_model: str | None = field(
        default=None,
        metadata={"help": "Path to the reference model used for the PPO or DPO training."},
    )
    ref_model_adapters: str | None = field(
        default=None,
        metadata={"help": "Path to the adapters of the reference model."},
    )
    ref_model_quantization_bit: int | None = field(
        default=None,
        metadata={"help": "The number of bits to quantize the reference model."},
    )
    reward_model: str | None = field(
        default=None,
        metadata={"help": "Path to the reward model used for the PPO training."},
    )
    reward_model_adapters: str | None = field(
        default=None,
        metadata={"help": "Path to the adapters of the reward model."},
    )
    reward_model_quantization_bit: int | None = field(
        default=None,
        metadata={"help": "The number of bits to quantize the reward model."},
    )
    reward_model_type: Literal["lora", "full", "api"] = field(
        default="lora",
        metadata={"help": "The type of the reward model in PPO training. Lora model only supports lora training."},
    )
    ld_alpha: float | None = field(
        default=None,
        metadata={
            "help": (
                "Alpha parameter from the LD-DPO paper, which controls the weighting of"
                " the verbose token log-probabilities in responses."
            )
        },
    )


@dataclass
class GaloreArguments:
    r"""Arguments pertaining to the GaLore algorithm."""

    use_galore: bool = field(
        default=False,
        metadata={"help": "Whether or not to use the gradient low-Rank projection (GaLore)."},
    )
    galore_target: str = field(
        default="all",
        metadata={
            "help": (
                "Name(s) of modules to apply GaLore. Use commas to separate multiple modules. "
                "Use `all` to specify all the linear modules."
            )
        },
    )
    galore_rank: int = field(
        default=16,
        metadata={"help": "The rank of GaLore gradients."},
    )
    galore_update_interval: int = field(
        default=200,
        metadata={"help": "Number of steps to update the GaLore projection."},
    )
    galore_scale: float = field(
        default=2.0,
        metadata={"help": "GaLore scaling coefficient."},
    )
    galore_proj_type: Literal["std", "reverse_std", "right", "left", "full"] = field(
        default="std",
        metadata={"help": "Type of GaLore projection."},
    )
    galore_layerwise: bool = field(
        default=False,
        metadata={"help": "Whether or not to enable layer-wise update to further save memory."},
    )


@dataclass
class ApolloArguments:
    r"""Arguments pertaining to the APOLLO algorithm."""

    use_apollo: bool = field(
        default=False,
        metadata={"help": "Whether or not to use the APOLLO optimizer."},
    )
    apollo_target: str = field(
        default="all",
        metadata={
            "help": (
                "Name(s) of modules to apply APOLLO. Use commas to separate multiple modules. "
                "Use `all` to specify all the linear modules."
            )
        },
    )
    apollo_rank: int = field(
        default=16,
        metadata={"help": "The rank of APOLLO gradients."},
    )
    apollo_update_interval: int = field(
        default=200,
        metadata={"help": "Number of steps to update the APOLLO projection."},
    )
    apollo_scale: float = field(
        default=32.0,
        metadata={"help": "APOLLO scaling coefficient."},
    )
    apollo_proj: Literal["svd", "random"] = field(
        default="random",
        metadata={"help": "Type of APOLLO low-rank projection algorithm (svd or random)."},
    )
    apollo_proj_type: Literal["std", "right", "left"] = field(
        default="std",
        metadata={"help": "Type of APOLLO projection."},
    )
    apollo_scale_type: Literal["channel", "tensor"] = field(
        default="channel",
        metadata={"help": "Type of APOLLO scaling (channel or tensor)."},
    )
    apollo_layerwise: bool = field(
        default=False,
        metadata={"help": "Whether or not to enable layer-wise update to further save memory."},
    )
    apollo_scale_front: bool = field(
        default=False,
        metadata={"help": "Whether or not to use the norm-growth limiter in front of gradient scaling."},
    )


@dataclass
class BAdamArgument:
    r"""Arguments pertaining to the BAdam optimizer."""

    use_badam: bool = field(
        default=False,
        metadata={"help": "Whether or not to use the BAdam optimizer."},
    )
    badam_mode: Literal["layer", "ratio"] = field(
        default="layer",
        metadata={"help": "Whether to use layer-wise or ratio-wise BAdam optimizer."},
    )
    badam_start_block: int | None = field(
        default=None,
        metadata={"help": "The starting block index for layer-wise BAdam."},
    )
    badam_switch_mode: Literal["ascending", "descending", "random", "fixed"] | None = field(
        default="ascending",
        metadata={"help": "the strategy of picking block to update for layer-wise BAdam."},
    )
    badam_switch_interval: int | None = field(
        default=50,
        metadata={
            "help": "Number of steps to update the block for layer-wise BAdam. Use -1 to disable the block update."
        },
    )
    badam_update_ratio: float = field(
        default=0.05,
        metadata={"help": "The ratio of the update for ratio-wise BAdam."},
    )
    badam_mask_mode: Literal["adjacent", "scatter"] = field(
        default="adjacent",
        metadata={
            "help": (
                "The mode of the mask for BAdam optimizer. "
                "`adjacent` means that the trainable parameters are adjacent to each other, "
                "`scatter` means that trainable parameters are randomly choosed from the weight."
            )
        },
    )
    badam_verbose: int = field(
        default=0,
        metadata={
            "help": (
                "The verbosity level of BAdam optimizer. "
                "0 for no print, 1 for print the block prefix, 2 for print trainable parameters."
            )
        },
    )


@dataclass
class SwanLabArguments:
    use_swanlab: bool = field(
        default=False,
        metadata={"help": "Whether or not to use the SwanLab (an experiment tracking and visualization tool)."},
    )
    swanlab_project: str | None = field(
        default="llamafactory",
        metadata={"help": "The project name in SwanLab."},
    )
    swanlab_workspace: str | None = field(
        default=None,
        metadata={"help": "The workspace name in SwanLab."},
    )
    swanlab_run_name: str | None = field(
        default=None,
        metadata={"help": "The experiment name in SwanLab."},
    )
    swanlab_mode: Literal["cloud", "local"] = field(
        default="cloud",
        metadata={"help": "The mode of SwanLab."},
    )
    swanlab_api_key: str | None = field(
        default=None,
        metadata={"help": "The API key for SwanLab."},
    )
    swanlab_logdir: str | None = field(
        default=None,
        metadata={"help": "The log directory for SwanLab."},
    )
    swanlab_lark_webhook_url: str | None = field(
        default=None,
        metadata={"help": "The Lark(飞书) webhook URL for SwanLab."},
    )
    swanlab_lark_secret: str | None = field(
        default=None,
        metadata={"help": "The Lark(飞书) secret for SwanLab."},
    )


@dataclass
class FinetuningArguments(
    SwanLabArguments,
    BAdamArgument,
    ApolloArguments,
    GaloreArguments,
    RLHFArguments,
    LoraArguments,
    OFTArguments,
    FreezeArguments,
):
    r"""Arguments pertaining to which techniques we are going to fine-tuning with."""

    pure_bf16: bool = field(
        default=False,
        metadata={"help": "Whether or not to train model in purely bf16 precision (without AMP)."},
    )
    stage: Literal["pt", "sft", "rm", "ppo", "dpo", "kto", "distill"] = field(
        default="sft",
        metadata={"help": "Which stage will be performed in training."},
    )
    # Distillation (stage="distill")
    teacher_models: list[str] | None = field(
        default=None,
        metadata={"help": "Paths to teacher model(s) for distillation. Comma-separated for multiple teachers."},
    )
    teacher_parallel: Literal["no", "pp"] = field(
        default="no",
        metadata={"help": "How to load teacher model: 'no' for single GPU, 'pp' for pipeline parallel."},
    )
    distill_loss_type: str = field(
        default="logits-kld",
        metadata={
            "help": (
                "Distillation loss type: logits-kld, logits-kld-v2, logits-reversed_kld, "
                "hidden-states-layerwise-kld, hidden-states-layerwise-mse, logits-and-hidden-states-kld."
            )
        },
    )
    distill_temperature: float = field(
        default=2.0,
        metadata={"help": "Temperature for distillation (logits-based losses)."},
    )
    hidden_states_alpha: float = field(
        default=0.9,
        metadata={"help": "Weight for hidden-states distillation loss when combined with student loss."},
    )
    distill_alpha: float = field(
        default=0.1,
        metadata={"help": "Weight for student CE loss: total = distill_alpha * student_loss + (1 - distill_alpha) * distill_loss."},
    )
    finetuning_type: Literal["lora", "oft", "freeze", "full"] = field(
        default="lora",
        metadata={"help": "Which fine-tuning method to use."},
    )
    use_llama_pro: bool = field(
        default=False,
        metadata={"help": "Whether or not to make only the parameters in the expanded blocks trainable."},
    )
    use_adam_mini: bool = field(
        default=False,
        metadata={"help": "Whether or not to use the Adam-mini optimizer."},
    )
    use_mca: bool = field(
        default=False,
        metadata={
            "help": (
                "Whether or not to use MCA (Megatron Core Adapter) training. "
                "Controlled by USE_MCA environment variable."
            )
        },
    )
    use_muon: bool = field(
        default=False,
        metadata={"help": "Whether or not to use the Muon optimizer."},
    )
    use_dft_loss: bool = field(
        default=False,
        metadata={"help": "Whether to use the DFT loss."},
    )
    use_eaft_loss: bool = field(
        default=False,
        metadata={"help": "Whether to use the EAFT loss."},
    )
    eaft_alpha: float = field(
        default=1.0,
        metadata={"help": "The alpha parameter for EAFT loss to control the power of adaptive weight."},
    )
    reasoning_loss_weight: float = field(
        default=1.0,
        metadata={"help": "Coefficient for reasoning loss. total_loss = loss_sft + reasoning_loss_weight * loss_reasoning."},
    )
    use_align_loss: bool = field(
        default=False,
        metadata={
            "help": (
                "Whether to enable <add_think> alignment loss in standard SFT branch. "
                "When enabled, trainer computes an auxiliary cosine alignment loss "
                "between <add_think> embedding and pre-<add_think> hidden-state soft projection."
            )
        },
    )
    align_loss_weight: float = field(
        default=0.1,
        metadata={"help": "Weight for alignment loss. total_loss += align_loss_weight * loss_align."},
    )
    use_ortho_loss: bool = field(
        default=False,
        metadata={
            "help": (
                "Whether to enable orthogonal loss for <add_think> embedding. "
                "When enabled, trainer minimizes the absolute cosine similarity "
                "between <add_think> and sampled normal token embeddings."
            )
        },
    )
    ortho_loss_weight: float = field(
        default=0.1,
        metadata={"help": "Weight for orthogonal loss. total_loss += ortho_loss_weight * loss_ortho."},
    )
    use_rgha: bool = field(
        default=False,
        metadata={
            "help": (
                "Whether to enable Risk-Gated Hidden Adapter (RGHA) in standard SFT. "
                "RGHA refines hidden states at high-risk positions before auxiliary CE."
            )
        },
    )
    rgha_weight: float = field(
        default=0.05,
        metadata={"help": "Weight for RGHA auxiliary CE loss. total_loss += rgha_weight * loss_rgha."},
    )
    rgha_entropy_alpha: float = field(
        default=0.5,
        metadata={"help": "Alpha coefficient for normalized entropy in RGHA risk score."},
    )
    rgha_sim_beta: float = field(
        default=0.5,
        metadata={"help": "Beta coefficient for (1 - max_cosine_sim) in RGHA risk score."},
    )
    rgha_threshold: float = field(
        default=0.55,
        metadata={"help": "Risk threshold to trigger RGHA positions."},
    )
    rgha_hidden_size: int = field(
        default=256,
        metadata={"help": "Hidden size of RGHA MLP adapter."},
    )
    rgha_warmup_steps: int = field(
        default=200,
        metadata={"help": "Warmup steps before RGHA masking is enabled."},
    )
    freeze_vision_tower: bool = field(
        default=True,
        metadata={"help": "Whether ot not to freeze the vision tower in MLLM training."},
    )
    freeze_multi_modal_projector: bool = field(
        default=True,
        metadata={"help": "Whether or not to freeze the multi modal projector in MLLM training."},
    )
    freeze_language_model: bool = field(
        default=False,
        metadata={"help": "Whether or not to freeze the language model in MLLM training."},
    )
    compute_accuracy: bool = field(
        default=False,
        metadata={"help": "Whether or not to compute the token-level accuracy at evaluation."},
    )
    compute_exact_match: bool = field(
        default=False,
        metadata={"help": "Whether or not to compute exact match at evaluation. Checks if generated response matches label."},
    )
    do_entropy_analysis: bool = field(
        default=False,
        metadata={
            "help": (
                "Whether to run entropy-strategy analysis after prediction. "
                "Performs autoregressive generation, identifies the top-20 highest-entropy "
                "positions per sample, then probes with three input strategies "
                "(hidden_norm, logit_weighted_embed, standard_embed) and saves "
                "distribution comparisons to CSV."
            )
        },
    )
    entropy_top_k_positions: int = field(
        default=20,
        metadata={"help": "Percentage (0-100) of highest-entropy positions to probe per sample. E.g. 20 means top 20%%."},
    )
    entropy_top_k_tokens: int = field(
        default=10,
        metadata={"help": "Number of top tokens (with probabilities) to record per strategy during entropy analysis."},
    )
    entropy_max_new_tokens: int = field(
        default=512,
        metadata={"help": "Maximum number of new tokens to generate per sample during entropy analysis."},
    )
    entropy_logit_weight_threshold: float = field(
        default=0.01,
        metadata={"help": "Probability threshold for Strategy B (logit_weighted_embed). Only tokens with prob > threshold are used."},
    )
    entropy_blend_alpha: float = field(
        default=0.6,
        metadata={"help": "Weight of Strategy A in the blended Strategy D: D = alpha * A_embed + (1-alpha) * B_embed. Default 0.6."},
    )
    entropy_use_answer_tokens: bool = field(
        default=False,
        metadata={
            "help": (
                "If True, entropy analysis uses ground-truth answer tokens via teacher-forcing "
                "instead of autoregressive generation. entropy_max_new_tokens is ignored in this mode."
            )
        },
    )
    entropy_hidden_drop_last_kv: bool = field(
        default=False,
        metadata={
            "help": (
                "If True, Strategy A (hidden_norm) uses a KV cache that is one token shorter "
                "than the other strategies: gen[pos-1] is excluded from the attention context. "
                "Has no effect when pos==0. B / C / D are unaffected."
            )
        },
    )
    entropy_fit_hidden_to_topk: bool = field(
        default=False,
        metadata={
            "help": (
                "If True, during entropy-strategy analysis fit normed_h with top-k word embeddings "
                "at each high-entropy position and save results to sample_<idx>_hidden_topk_fit.csv."
            )
        },
    )
    entropy_blend_alpha_sweep: bool = field(
        default=False,
        metadata={
            "help": (
                "If True, during entropy-strategy analysis sweep alpha in the blended Strategy D "
                "between 0 and 1 and record KL divergences and top-k overlaps between the blended "
                "distribution and Strategies A/B. Results are written to sample_<idx>_blend_alpha_sweep.csv."
            )
        },
    )
    entropy_blend_alpha_steps: int = field(
        default=21,
        metadata={
            "help": (
                "Number of alpha points in [0,1] (inclusive) for the blended Strategy D sweep when "
                "entropy_blend_alpha_sweep is True. Must be >=2."
            )
        },
    )
    entropy_analyze_at_add_think_positions: bool = field(
        default=False,
        metadata={
            "help": (
                "If True and use_answer_tokens (answer-mode entropy analysis) is enabled, "
                "probe at <add_think> token positions instead of top-k%% highest-entropy positions. "
                "Samples with no <add_think> are skipped."
            )
        },
    )
    logits_analysis_in_sft: bool = field(
        default=False,
        metadata={
            "help": (
                "If True, during standard SFT record per-token entropy, max cosine (LN hidden vs lm_head), "
                "and max logit before/after each optimizer step and append grouped stats to jsonl "
                "(requires CustomSeq2SeqTrainer standard branch)."
            )
        },
    )
    logits_analysis_threshold_entropy: float = field(
        default=1.0,
        metadata={"help": "High-entropy threshold H > this value counts as high-entropy positions."},
    )
    logits_analysis_threshold_cosine: float = field(
        default=0.5,
        metadata={"help": "Low max-cosine threshold: max_k cos(h, w_k) < this counts as low-cosine."},
    )
    logits_analysis_log_every_n_steps: int = field(
        default=100,
        metadata={"help": "Write one jsonl record every N completed optimizer steps."},
    )
    logits_analysis_output_path: str = field(
        default="./analysis_logs/",
        metadata={"help": "Directory for logits analysis jsonl files."},
    )
    logits_analysis_cosine_seq_chunk: int = field(
        default=512,
        metadata={"help": "Sequence chunk size for max-cosine matmul to limit peak memory."},
    )
    logits_analysis_max_positions: int = field(
        default=256,
        metadata={
            "help": (
                "Max number of supervised (masked) token positions per step for which to track "
                "focused logits (top-k or prob threshold). 0 means all masked positions."
            )
        },
    )
    logits_analysis_focus_mode: Literal["topk", "prob_threshold"] = field(
        default="topk",
        metadata={
            "help": (
                "How to pick vocabulary ids per position before optimizer.step: "
                "'topk' = top logits_analysis_focus_topk probs; "
                "'prob_threshold' = all tokens with prob >= logits_analysis_focus_prob_threshold "
                "(capped by logits_analysis_focus_max_tokens)."
            )
        },
    )
    logits_analysis_focus_topk: int = field(
        default=10,
        metadata={"help": "When focus_mode=topk, number of highest-probability tokens per position."},
    )
    logits_analysis_focus_prob_threshold: float = field(
        default=0.01,
        metadata={"help": "When focus_mode=prob_threshold, minimum softmax probability to include."},
    )
    logits_analysis_focus_max_tokens: int = field(
        default=32,
        metadata={
            "help": (
                "When focus_mode=prob_threshold, max tokens per position if many exceed the threshold."
            )
        },
    )
    logits_analysis_save_focus_per_position: bool = field(
        default=False,
        metadata={
            "help": (
                "If True, include per-position focused logits (top-k or prob-threshold) in each jsonl record. "
                "Large; default False — clustering stats (groups A–D) are always saved."
            )
        },
    )
    add_think_token: bool = field(
        default=False,
        metadata={
            "help": (
                "If True, register `<add_think>` as a special token and resize embeddings before "
                "SFT, without enabling recurrent hidden-as-input training. Use this for normal "
                "SFT where `<add_think>` is treated like any other token (standard embeddings + "
                "LM head). Mutually complements `recurrent_add_think_training` (which also adds "
                "the token if missing)."
            )
        },
    )
    recurrent_add_think_training: bool = field(
        default=False,
        metadata={
            "help": (
                "If True, use recurrent forward for <add_think>: at <add_think> and the next "
                "token, input is the previous position's hidden state (through a learnable "
                "LayerNorm) instead of token embeddings. Loss ignores <add_think> targets; "
                "whole batch uses this branch when enabled."
            )
        },
    )
    recurrent_add_think_backprop_hidden: bool = field(
        default=False,
        metadata={
            "help": (
                "If True, in recurrent_add_think_training allow gradients to flow through "
                "the hidden-as-input path (i.e. loss backpropagates through the <add_think> "
                "input embedding that is derived from previous hidden states). This increases "
                "memory usage. When False (default), the recurrent path is computed without "
                "backprop through the hidden replacements."
            )
        },
    )
    do_mark_low_confidence: bool = field(
        default=False,
        metadata={
            "help": (
                "Whether to run low-confidence position marking. "
                "Performs teacher-forcing forward on the eval dataset, identifies "
                "positions where the model's probability for the correct next token "
                "is below `low_confidence_prob_threshold`, inserts <add_think> tokens "
                "after those positions, and saves the augmented dataset as JSONL."
            )
        },
    )
    low_confidence_prob_threshold: float = field(
        default=0.3,
        metadata={
            "help": (
                "Probability threshold for marking low-confidence positions. "
                "Positions where the model assigns less than this probability to the "
                "correct next token (within the response portion) are marked with <add_think>."
            )
        },
    )
    low_confidence_entropy_threshold: float = field(
        default=2.0,
        metadata={
            "help": (
                "Entropy threshold for B_lowconf marking. "
                "Only response positions with entropy >= this value are considered."
            )
        },
    )
    low_confidence_sim_threshold: float = field(
        default=0.3,
        metadata={
            "help": (
                "Cosine-similarity threshold for B_lowconf marking. "
                "A position is marked only when max cosine similarity between "
                "last hidden state and token embeddings is below this value."
            )
        },
    )
    low_confidence_insert_position: str = field(
        default="before",
        metadata={
            "help": (
                "Where to insert <add_think> around marked B_lowconf tokens. "
                "Choices: `before` or `after`."
            )
        },
    )
    disable_shuffling: bool = field(
        default=False,
        metadata={"help": "Whether or not to disable the shuffling of the training set."},
    )
    early_stopping_steps: int | None = field(
        default=None,
        metadata={"help": "Number of steps to stop training if the `metric_for_best_model` does not improve."},
    )
    plot_loss: bool = field(
        default=False,
        metadata={"help": "Whether or not to save the training loss curves."},
    )
    include_effective_tokens_per_second: bool = field(
        default=False,
        metadata={"help": "Whether or not to compute effective tokens per second."},
    )

    def __post_init__(self):
        def split_arg(arg):
            if isinstance(arg, str):
                return [item.strip() for item in arg.split(",")]
            return arg

        self.freeze_trainable_modules: list[str] = split_arg(self.freeze_trainable_modules)
        self.freeze_extra_modules: list[str] | None = split_arg(self.freeze_extra_modules)
        self.lora_alpha: int = self.lora_alpha or self.lora_rank * 2
        self.lora_target: list[str] = split_arg(self.lora_target)
        self.oft_target: list[str] = split_arg(self.oft_target)
        self.additional_target: list[str] | None = split_arg(self.additional_target)
        self.galore_target: list[str] = split_arg(self.galore_target)
        self.apollo_target: list[str] = split_arg(self.apollo_target)
        if self.teacher_models is not None and isinstance(self.teacher_models, str):
            self.teacher_models = [p.strip() for p in self.teacher_models.split(",") if p.strip()]
        self.use_ref_model = self.stage == "dpo" and self.pref_loss not in ["orpo", "simpo"]

        assert self.finetuning_type in ["lora", "oft", "freeze", "full"], "Invalid fine-tuning method."
        assert self.ref_model_quantization_bit in [None, 8, 4], "We only accept 4-bit or 8-bit quantization."
        assert self.reward_model_quantization_bit in [None, 8, 4], "We only accept 4-bit or 8-bit quantization."

        if self.stage == "distill" and (not self.teacher_models or len(self.teacher_models) == 0):
            raise ValueError("`teacher_models` is necessary for distillation training.")

        if self.low_confidence_insert_position not in {"before", "after"}:
            raise ValueError("`low_confidence_insert_position` must be either `before` or `after`.")

        if self.align_loss_weight < 0:
            raise ValueError("`align_loss_weight` must be non-negative.")
        if self.ortho_loss_weight < 0:
            raise ValueError("`ortho_loss_weight` must be non-negative.")
        if self.rgha_weight < 0:
            raise ValueError("`rgha_weight` must be non-negative.")
        if self.rgha_entropy_alpha < 0 or self.rgha_sim_beta < 0:
            raise ValueError("`rgha_entropy_alpha` and `rgha_sim_beta` must be non-negative.")
        if self.rgha_hidden_size <= 0:
            raise ValueError("`rgha_hidden_size` must be positive.")
        if self.rgha_warmup_steps < 0:
            raise ValueError("`rgha_warmup_steps` must be non-negative.")

        if self.stage == "ppo" and self.reward_model is None:
            raise ValueError("`reward_model` is necessary for PPO training.")

        if self.stage == "ppo" and self.reward_model_type == "lora" and self.finetuning_type != "lora":
            raise ValueError("`reward_model_type` cannot be lora for Freeze/Full PPO training.")

        if self.stage == "ppo" and self.reward_model_type == "oft" and self.finetuning_type != "oft":
            raise ValueError("`reward_model_type` cannot be oft for Freeze/Full PPO training.")

        if self.stage == "dpo" and self.pref_loss != "sigmoid" and self.dpo_label_smoothing > 1e-6:
            raise ValueError("`dpo_label_smoothing` is only valid for sigmoid loss function.")

        if self.use_llama_pro and self.finetuning_type == "full":
            raise ValueError("`use_llama_pro` is only valid for Freeze or LoRA training.")

        if self.finetuning_type == "lora" and (self.use_galore or self.use_apollo or self.use_badam):
            raise ValueError("Cannot use LoRA with GaLore, APOLLO or BAdam together.")

        if int(self.use_galore) + int(self.use_apollo) + (self.use_badam) > 1:
            raise ValueError("Cannot use GaLore, APOLLO or BAdam together.")

        if self.pissa_init and (self.stage in ["ppo", "kto"] or self.use_ref_model):
            raise ValueError("Cannot use PiSSA for current training stage.")

        if self.finetuning_type != "lora":
            if self.loraplus_lr_ratio is not None:
                raise ValueError("`loraplus_lr_ratio` is only valid for LoRA training.")

            if self.use_rslora:
                raise ValueError("`use_rslora` is only valid for LoRA training.")

            if self.use_dora:
                raise ValueError("`use_dora` is only valid for LoRA training.")

            if self.pissa_init:
                raise ValueError("`pissa_init` is only valid for LoRA training.")

    def to_dict(self) -> dict[str, Any]:
        args = asdict(self)
        args = {k: f"<{k.upper()}>" if k.endswith("api_key") else v for k, v in args.items()}
        return args
