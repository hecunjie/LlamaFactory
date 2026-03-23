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