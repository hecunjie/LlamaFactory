#!/usr/bin/env bash
# 多卡训练 DIT 或 DIT-P（torchrun + train.py 内 DDP / 评估 all_reduce）
#
# 用法（在 LlamaFactory 仓库根目录执行）:
#   bash dit-p/train_multigpu.sh dit
#   bash dit-p/train_multigpu.sh ditp
#   NPROC_PER_NODE=4 bash dit-p/train_multigpu.sh ditp
#
# 常用环境变量覆盖:
#   NPROC_PER_NODE  默认 8；设为 1 则等价于 python 单进程
#   MODEL_NAME  SAVE_PATH  M_DIT  EPOCHS  LR  BATCH_SIZE  TEMPLATE_STYLE
#   DATASET_INFO_PATH  DATA_DIR  TRAIN_DATASET_NAME  TEST_DATASET_NAME
#   MAX_LENGTH  MAX_NEW_TOKENS  SAVE_STEPS

set -euo pipefail

MODE="${1:-}"
if [[ "${MODE}" != "dit" && "${MODE}" != "ditp" ]]; then
  echo "Usage: bash dit-p/train_multigpu.sh <dit|ditp>"
  exit 1
fi

SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
# 假设本脚本在 .../LlamaFactory/dit-p/，数据在 .../LlamaFactory/data/
REPO_ROOT="$(cd "${SCRIPT_DIR}/.." && pwd)"

NPROC_PER_NODE="${NPROC_PER_NODE:-8}"

MODEL_NAME="${MODEL_NAME:-/mnt/tidal-alsh01/dataset/zeus/hecunjie/models/Llama/Llama-3.2-3B}"
M_DIT="${M_DIT:-5}"
EPOCHS="${EPOCHS:-3}"
LR="${LR:-1e-5}"
BATCH_SIZE="${BATCH_SIZE:-8}"
TEMPLATE_STYLE="${TEMPLATE_STYLE:-llama3}"
MAX_LENGTH="${MAX_LENGTH:-1024}"
MAX_NEW_TOKENS="${MAX_NEW_TOKENS:-256}"
SAVE_STEPS="${SAVE_STEPS:-0}"

DATASET_INFO_PATH="${DATASET_INFO_PATH:-${REPO_ROOT}/data/dataset_info.json}"
DATA_DIR="${DATA_DIR:-${REPO_ROOT}/data}"
TRAIN_DATASET_NAME="${TRAIN_DATASET_NAME:-gsm8k_sft_train}"
TEST_DATASET_NAME="${TEST_DATASET_NAME:-gsm8k_sft_test}"

if [[ -z "${SAVE_PATH:-}" ]]; then
  if [[ "${MODE}" == "ditp" ]]; then
    SAVE_PATH="/mnt/tidal-alsh01/dataset/zeus/hecunjie/train_outputs/llama_3b_base_ditp_5_3_epoch_lr1e-5/"
  else
    SAVE_PATH="/mnt/tidal-alsh01/dataset/zeus/hecunjie/train_outputs/llama_3b_base_dit_5_3_epoch_lr1e-5/"
  fi
fi

mkdir -p "${SAVE_PATH}"

echo "[INFO] MODE=${MODE}"
echo "[INFO] NPROC_PER_NODE=${NPROC_PER_NODE}"
echo "[INFO] MODEL_NAME=${MODEL_NAME}"
echo "[INFO] SAVE_PATH=${SAVE_PATH}"
echo "[INFO] DATASET_INFO_PATH=${DATASET_INFO_PATH}"

TRAIN_PY="${SCRIPT_DIR}/train.py"
COMMON_ARGS=(
  --model_name "${MODEL_NAME}"
  --mode "${MODE}"
  --m_dit "${M_DIT}"
  --epochs "${EPOCHS}"
  --lr "${LR}"
  --batch_size "${BATCH_SIZE}"
  --save_path "${SAVE_PATH}"
  --template_style "${TEMPLATE_STYLE}"
  --max_length "${MAX_LENGTH}"
  --max_new_tokens "${MAX_NEW_TOKENS}"
  --save_steps "${SAVE_STEPS}"
  --dataset_info_path "${DATASET_INFO_PATH}"
  --data_dir "${DATA_DIR}"
  --train_dataset_name "${TRAIN_DATASET_NAME}"
  --test_dataset_name "${TEST_DATASET_NAME}"
)

if [[ "${NPROC_PER_NODE}" -gt 1 ]]; then
  exec torchrun --standalone --nproc_per_node "${NPROC_PER_NODE}" "${TRAIN_PY}" "${COMMON_ARGS[@]}"
else
  exec python "${TRAIN_PY}" "${COMMON_ARGS[@]}"
fi
