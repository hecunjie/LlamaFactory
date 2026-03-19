"""Centralized experiment configuration."""

MODEL_NAME = "Qwen/Qwen2.5-3B-Instruct"
DATASET_NAME = "gsm8k"
DATASET_CONFIG = "main"

N_TRAJECTORIES = 20
TEMPERATURE = 0.8
TOP_P = 0.95
TOP_K_ESOFT = 20
MIN_TRAJS_PER_GROUP = 3
GAMMA_VALUES = [0.0, 0.25, 0.5, 0.75, 1.0]

MAX_NEW_TOKENS_COLLECT = 300
MAX_NEW_TOKENS_INTERVENE = 80

HIDDEN_LAYER_INDEX = -1
DELTA_SOURCE = "train"  # "train" or "test"

DEVICE = "cuda"
DTYPE = "bfloat16"
SAVE_DIR = "./results"
