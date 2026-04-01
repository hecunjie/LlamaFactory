#!/usr/bin/env bash
set -euo pipefail

# Usage:
#   bash run.sh dit
#   bash run.sh ditp

MODE="${1:-ditp}"                 # dit | ditp
MODEL_NAME="${MODEL_NAME:-meta-llama/Llama-3.2-3B-Instruct}"
M_DIT="${M_DIT:-5}"
EPOCHS="${EPOCHS:-1}"
LR="${LR:-1e-5}"
BATCH_SIZE="${BATCH_SIZE:-1}"
SAVE_STEPS="${SAVE_STEPS:-0}"
MAX_LENGTH="${MAX_LENGTH:-1024}"
MAX_NEW_TOKENS="${MAX_NEW_TOKENS:-256}"
TEMPLATE_STYLE="${TEMPLATE_STYLE:-llama3}"
TRAIN_DATASET_NAME="${TRAIN_DATASET_NAME:-gsm8k_sft_train}"
TEST_DATASET_NAME="${TEST_DATASET_NAME:-gsm8k_sft_test}"

SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
DATASET_INFO_PATH="${DATASET_INFO_PATH:-${SCRIPT_DIR}/../data/dataset_info.json}"
DATA_DIR="${DATA_DIR:-${SCRIPT_DIR}/../data}"
SAVE_ROOT="${SAVE_ROOT:-${SCRIPT_DIR}/outputs}"
SAVE_PATH="${SAVE_PATH:-${SAVE_ROOT}/${MODE}_md${M_DIT}_ep${EPOCHS}_lr${LR}}"

if [[ "${MODE}" != "dit" && "${MODE}" != "ditp" ]]; then
  echo "[ERROR] MODE must be 'dit' or 'ditp', got: ${MODE}"
  exit 1
fi

if [[ ! -f "${DATASET_INFO_PATH}" ]]; then
  echo "[ERROR] dataset_info.json not found: ${DATASET_INFO_PATH}"
  exit 1
fi

# Quick checks to avoid "runs without input" confusion.
TRAIN_FILE="$(python - "${DATASET_INFO_PATH}" "${DATA_DIR}" "${TRAIN_DATASET_NAME}" <<'PY'
import json, os, sys
info_path, data_dir, ds = sys.argv[1], sys.argv[2], sys.argv[3]
info = json.load(open(info_path, "r", encoding="utf-8"))
meta = info.get(ds, {})
name = meta.get("file_name", "")
print(os.path.join(data_dir, name) if name else "")
PY
)"

TEST_FILE="$(python - "${DATASET_INFO_PATH}" "${DATA_DIR}" "${TEST_DATASET_NAME}" <<'PY'
import json, os, sys
info_path, data_dir, ds = sys.argv[1], sys.argv[2], sys.argv[3]
info = json.load(open(info_path, "r", encoding="utf-8"))
meta = info.get(ds, {})
name = meta.get("file_name", "")
print(os.path.join(data_dir, name) if name else "")
PY
)"

if [[ -z "${TRAIN_FILE}" || ! -f "${TRAIN_FILE}" ]]; then
  echo "[ERROR] train dataset file missing: ${TRAIN_FILE}"
  exit 1
fi
if [[ -z "${TEST_FILE}" || ! -f "${TEST_FILE}" ]]; then
  echo "[ERROR] test dataset file missing: ${TEST_FILE}"
  exit 1
fi

mkdir -p "${SAVE_PATH}"

echo "[INFO] MODE=${MODE}"
echo "[INFO] MODEL_NAME=${MODEL_NAME}"
echo "[INFO] TRAIN_FILE=${TRAIN_FILE}"
echo "[INFO] TEST_FILE=${TEST_FILE}"
echo "[INFO] SAVE_PATH=${SAVE_PATH}"
echo "[INFO] SAVE_STEPS=${SAVE_STEPS}"
echo "[INFO] TEMPLATE_STYLE=${TEMPLATE_STYLE}"

python "${SCRIPT_DIR}/train.py" \
  --model_name "${MODEL_NAME}" \
  --mode "${MODE}" \
  --m_dit "${M_DIT}" \
  --epochs "${EPOCHS}" \
  --lr "${LR}" \
  --batch_size "${BATCH_SIZE}" \
  --save_steps "${SAVE_STEPS}" \
  --max_length "${MAX_LENGTH}" \
  --max_new_tokens "${MAX_NEW_TOKENS}" \
  --template_style "${TEMPLATE_STYLE}" \
  --dataset_info_path "${DATASET_INFO_PATH}" \
  --data_dir "${DATA_DIR}" \
  --train_dataset_name "${TRAIN_DATASET_NAME}" \
  --test_dataset_name "${TEST_DATASET_NAME}" \
  --save_path "${SAVE_PATH}"
