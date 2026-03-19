cd /mnt/ali-sh-1/dataset/zeus/hecunjie/gitlab-source/LlamaFactory
# model_name_or_path: /mnt/tidal-alsh01/dataset/zeus/hecunjie/train_outputs/gsm_128_bs_1_epoch/checkpoint-117

python scripts/vllm_infer.py \
  --model_name_or_path /mnt/tidal-alsh01/dataset/zeus/hecunjie/models/Qwen/Qwen2.5-3B \
  --template qwen \
  --dataset gsm8k_analysis \
  --batch_size 32 \
  --max_samples 200 \
  --max_new_tokens 8192 \
  --num_generations 16 \
  --temperature 0.8 \
  --top_p 0.95