python experiment/analyze_entropy_for_logits.py \
  --data /mnt/tidal-alsh01/dataset/zeus/hecunjie/train_outputs/llama_3b_base_sft_3_epoch/vllm_infer/last_ckp_infer_4_gsm_nl.jsonl \
  --model /mnt/tidal-alsh01/dataset/zeus/hecunjie/train_outputs/llama_3b_base_sft_3_epoch/checkpoint-1404 \
  --max_samples 8000 \
  --batch_size 4 \
  --entropy_threshold 0.01 \
  --high_entropy_topk 20 \
  --sim_threshold 0.18 \
  --only_wrong \
  --export_low_lse_positions \
  --export_low_lse_bottom_quantile 0.05 \
  --output_plot /mnt/tidal-alsh01/dataset/zeus/hecunjie/train_outputs/llama_3b_base_sft_3_epoch/vllm_infer/gsm_nl/fork_points/last_ckp_entropy_analysis.png \
  --output_jsonl /mnt/tidal-alsh01/dataset/zeus/hecunjie/train_outputs/llama_3b_base_sft_3_epoch/vllm_infer/gsm_nl/fork_point/last_ckp_entropy_results.jsonl
  # --lse_threshold_report \
  # --log_sum_exp_threshold 22.0 \
  # --lse_layer_probe \
  # --lse_layer_probe_bottom_quantile 0.2 \
  # --export_low_entropy_lse \
  # --export_low_joint_quantile 0.3 \

python experiment/analyze_partial_cancellation.py \
  --data /mnt/tidal-alsh01/dataset/zeus/hecunjie/train_outputs/llama_3b_base_sft_3_epoch/vllm_infer/last_ckp_infer_4_gsm_nl.jsonl \
  --model /mnt/tidal-alsh01/dataset/zeus/hecunjie/train_outputs/llama_3b_base_sft_3_epoch/checkpoint-1404 \
  --max_samples 8000 \
  --batch_size 1 \
  --use_multi_process \
  --gpu_ids "0,1,2,3,4,5,6,7" \
  --num_processes 8 \
  --output_dir /mnt/tidal-alsh01/dataset/zeus/hecunjie/train_outputs/llama_3b_base_sft_3_epoch/vllm_infer/gsm_nl/partial_cancellation

python experiment/analyze_entropy_for_logits.py \
  --data data \
  --dataset gsm8k_sft_test \
  --template llama3 \
  --model /mnt/tidal-alsh01/dataset/zeus/hecunjie/train_outputs/llama_3b_base_sft_2_epoch_baseline/checkpoint-936 \
  --max_samples 8000 \
  --batch_size 4 \
  --high_entropy_topk 20 \
  --sim_threshold 0.18 \
  --only_correct \
  --output_plot /mnt/tidal-alsh01/dataset/zeus/hecunjie/train_outputs/llama_3b_base_sft_2_epoch_baseline/vllm_infer/gsm_sft_test/wrong_log_sum_exp/last_ckp_entropy_analysis.png \
  --output_jsonl /mnt/tidal-alsh01/dataset/zeus/hecunjie/train_outputs/llama_3b_base_sft_2_epoch_baseline/vllm_infer/gsm_sft_test/wrong_log_sum_exp/last_ckp_entropy_results.jsonl

##只看正确
python experiment/analyze_entropy_for_logits.py \
  --data /mnt/tidal-alsh01/dataset/zeus/hecunjie/train_outputs/llama_3b_base_sft_2_epoch_baseline/vllm_infer/gsm_nl/last_ckp_on_gsm_nl.jsonl \
  --model /mnt/tidal-alsh01/dataset/zeus/hecunjie/train_outputs/llama_3b_base_sft_2_epoch_baseline/checkpoint-936 \
  --max_samples 8000 \
  --batch_size 4 \
  --high_entropy_topk 20 \
  --sim_threshold 0.18 \
  --only_correct \
  --lse_threshold_report \
  --log_sum_exp_threshold 22.0 \
  --lse_layer_probe \
  --lse_layer_probe_bottom_quantile 0.2 \
  --output_plot /mnt/tidal-alsh01/dataset/zeus/hecunjie/train_outputs/llama_3b_base_sft_2_epoch_baseline/vllm_infer/gsm_nl/correct_log_sum_exp/last_ckp_entropy_analysis.png \
  --output_jsonl /mnt/tidal-alsh01/dataset/zeus/hecunjie/train_outputs/llama_3b_base_sft_2_epoch_baseline/vllm_infer/gsm_nl/correct_log_sum_exp/last_ckp_entropy_results.jsonl

python scripts/stat_add_think_logits_entropy_dist.py \
  --model /mnt/tidal-alsh01/dataset/zeus/hecunjie/train_outputs/llama_3b_base_sft_2_epoch_baseline/checkpoint-936 \
  --dataset gsm8k_add_think \
  --dataset_dir data \
  --fallback_marker_token  "<|reserved_special_token_1|>" \
  --template llama3 \
  --max_samples 8000 \
  --drop_think_position \
  --output_dir /mnt/tidal-alsh01/dataset/zeus/hecunjie/train_outputs/llama_3b_base_sft_2_epoch_baseline/vllm_infer/train_data/add_think_logits_entropy_dists


python scripts/stat_add_think_entropy_shift.py \
  --data /mnt/tidal-alsh01/dataset/zeus/hecunjie/train_outputs/llama_3b_base_2_aligh_0.0_add_think_on_v2_data/vllm_infer/last_ckp_svamp.jsonl \
  --model /mnt/tidal-alsh01/dataset/zeus/hecunjie/train_outputs/llama_3b_base_2_aligh_0.0_add_think_on_v2_data/checkpoint-936 \
  --output_dir /mnt/tidal-alsh01/dataset/zeus/hecunjie/train_outputs/llama_3b_base_2_aligh_0.0_add_think_on_v2_data/vllm_infer/svamp/add_think_entropy_shift \
  --hist_bins 60

python3 scripts/stat_add_think_top5_by_case.py /mnt/tidal-alsh01/dataset/zeus/hecunjie/train_outputs/llama_3b_base_2_aligh_0.1_add_think_on_v2_data/vllm_infer/last_ckp_gsm_nl_entropy_results.jsonl \
  --output-json /mnt/tidal-alsh01/dataset/zeus/hecunjie/train_outputs/llama_3b_base_2_aligh_0.1_add_think_on_v2_data/vllm_infer/gsm_nl/add_think_top5_case_summary.json \
  --output-csv /mnt/tidal-alsh01/dataset/zeus/hecunjie/train_outputs/llama_3b_base_2_aligh_0.1_add_think_on_v2_data/vllm_infer/gsm_nl/add_think_top5_case_correctness.csv \

## 推理
python scripts/rgha_infer.py \
  --model_name_or_path /mnt/tidal-alsh01/dataset/zeus/hecunjie/train_outputs/llama_3b_base_rgha_maxcos_thre/checkpoint-936 \
  --dataset gsm_nl_test \
  --dataset_dir data \
  --template llama3 \
  --output /mnt/tidal-alsh01/dataset/zeus/hecunjie/train_outputs/llama_3b_base_rgha_maxcos_thre/vllm_infer/gsm_nl/last_ckp_no_rgha_preds.jsonl \
  --do_sample \
  --temperature 1 \
  --top_p 0.95 \
  --num_generations 4 \
  --batch_size 32 \
  --data_parallel_gpus 8
# --use_rgha \

python -m train_sae.train \
  --data_path /mnt/ali-sh-1/dataset/zeus/hecunjie/rl_data/gsm8k/zeroshot_16_samples_qwen2.5-3b_instruct_on_math_train_full.jsonl \
  --model_name /mnt/tidal-alsh01/dataset/zeus/hecunjie/models/Qwen/Qwen2.5-3B-Instruct \
  --device cuda

python experiment/analyze_entropy.py \
  --data_path /mnt/ali-sh-1/dataset/zeus/hecunjie/rl_data/gsm8k/zeroshot_16_samples_qwen2.5-3b_instruct_on_math_train_full.jsonl \
  --model_name /mnt/tidal-alsh01/dataset/zeus/hecunjie/models/Qwen/Qwen2.5-3B-Instruct \
  --output_dir ./entropy_analysis \
  --max_samples 2000 \
  --max_predicts 16 \


python /mnt/ali-sh-1/dataset/zeus/hecunjie/gitlab-source/LlamaFactory/scripts/convert_prompt_response_jsonl_to_sft.py -i /mnt/tidal-alsh01/dataset/zeus/hecunjie/train_outputs/add_think_token/low_confidence_marked/marked_dataset.jsonl -o /mnt/ali-sh-1/dataset/zeus/hecunjie/rl_data/gsm8k/add_think_train.jsonl

python  /mnt/ali-sh-1/dataset/zeus/hecunjie/projects/CMT2/xlsx2html.py /mnt/tidal-alsh01/dataset/zeus/hecunjie/train_outputs/blend_alpha_sweep_AB_stage1_on_gsm8k_add_think_test_data/entropy_analysis/sample_0/strategy_distributions.csv /mnt/ali-sh-1/dataset/zeus/hecunjie/workspace  /mnt/ali-sh-1/dataset/zeus/hecunjie/workspace


python scripts/plot_blend_alpha_sweep.py \
      --analysis_dir /mnt/tidal-alsh01/dataset/zeus/hecunjie/train_outputs/blend_alpha_sweep_test_AC/entropy_analysis \
      --output /mnt/tidal-alsh01/dataset/zeus/hecunjie/train_outputs/blend_alpha_sweep_test_AC/blend_analysis.png



python experiment/run_experiment.py \
  --model /mnt/tidal-alsh01/dataset/zeus/hecunjie/models/Qwen/Qwen2.5-3B-Instruct \
  --train_sampled_jsonl /mnt/ali-sh-1/dataset/zeus/hecunjie/rl_data/gsm8k/zeroshot_16_samples_qwen2.5-3b_instruct_on_alpaca_train_2k.jsonl \
  --test_sampled_jsonl /mnt/ali-sh-1/dataset/zeus/hecunjie/rl_data/gsm8k/zeroshot_16_samples_qwen2.5-3b_instruct_on_alpaca_test_100.jsonl \
  --skip_collect

python scripts/plot_blend_alpha_sweep.py       --analysis_dir /mnt/tidal-alsh01/dataset/zeus/hecunjie/train_outputs/blend_alpha_sweep_AB_stage1_on_gsm8k_add_think_test_data/entropy_analysis --output /mnt/tidal-alsh01/dataset/zeus/hecunjie/train_outputs/blend_alpha_sweep_AB_stage1_on_gsm8k_add_think_test_data/blend_alpha_sweep.png --max_sample 100


python /mnt/ali-sh-1/dataset/zeus/hecunjie/gitlab-source/LlamaFactory/scripts/word_embedding_pca.py --model_name_or_path /mnt/tidal-alsh01/dataset/zeus/hecunjie/models/Qwen/Qwen2.5-3B --output_dir /mnt/tidal-alsh01/dataset/zeus/hecunjie/train_outputs/pca_word_emebdding


python -m train_sae.analyze_features \
  --checkpoint ./sae_checkpoints/epoch_019.pt \
  --cache_path ./sae_cache/hidden_states.pt \
  --output_dir ./feature_analysis \
  --top_n 20 \
  --context_examples 100 \
  --device cuda