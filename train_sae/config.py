"""SAE 训练与数据提取的超参数（见 cursor_prompt/train_sqe.md）。"""

from __future__ import annotations

# LLM 配置
LLM_NAME = "Qwen/Qwen2.5-3B-Instruct"
HIDDEN_DIM = 2048  # Qwen2.5-3B 的 hidden size
HIDDEN_LAYER_INDEX = -2  # 取倒数第二层 hidden states

# 高熵位置筛选（百分位：>= 该分位的 token 视为高熵，即约 top (100-p)%）
ENTROPY_PERCENTILE = 80

# SAE 配置
SAE_EXPANSION = 8  # 字典大小 = HIDDEN_DIM * SAE_EXPANSION
TOP_K = 64  # 每次激活最多 K 个特征

# 损失权重
LAMBDA_QUALITY = 0.1  # 推理质量分类损失权重

# 训练配置
BATCH_SIZE = 512
LEARNING_RATE = 1e-4
NUM_EPOCHS = 20
WARMUP_STEPS = 200
GRAD_CLIP = 1.0
WEIGHT_DECAY = 0.01

# 数据配置
DATA_PATH = "./data/samples.jsonl"
MAX_PREDICTS_PER_SAMPLE = 10
CACHE_DIR = "./sae_cache"  # hidden states 缓存目录

# 保存配置
SAVE_DIR = "./sae_checkpoints"
SAVE_EVERY = 2
LOG_EVERY = 50

DEVICE = "cuda"
DTYPE = "bfloat16"
SEED = 42

# 提取进度：每处理多少道题打印一次
EXTRACT_PROGRESS_EVERY = 20
