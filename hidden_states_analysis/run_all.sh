#!/usr/bin/env bash
# 一键运行：抽取高熵 hidden states → UMAP+聚类 → 可视化 → 抽样解释 → 定量统计
#
# 用法（在 LlamaFactory 仓库根目录执行，或任意路径执行本脚本均可）：
#   chmod +x hidden_states_analysis/run_all.sh
#   MODEL_NAME_OR_PATH=meta-llama/Meta-Llama-3-8B-Instruct ./hidden_states_analysis/run_all.sh
#
# 或传参：第 1 个参数为模型，第 2 个为 dataset_info 中的数据集名（可选）
#   ./hidden_states_analysis/run_all.sh meta-llama/Meta-Llama-3-8B-Instruct gsm8k_sft_train
#
# 常用环境变量（可选）：
#   DATASET_DIR       默认 data（其下需有 dataset_info.json）
#   DATASET           默认 gsm8k_sft_train
#   TEMPLATE          默认 llama3
#   OUTPUT_DIR        默认 outputs/hidden_states_run
#   MAX_SAMPLES       默认 2000（调试可设 200）
#   TOP_K_PERCENT     默认 10
#   DTYPE             默认 bfloat16（可选 float16 float32）
#   SKIP_EXTRACT=1    仅跑聚类/图/统计（需已有 OUTPUT_DIR 下的 hidden_states.npy）
#
set -euo pipefail

ROOT="$(cd "$(dirname "${BASH_SOURCE[0]}")/.." && pwd)"
export PYTHONPATH="${ROOT}/src:${PYTHONPATH:-}"

MODEL="${MODEL_NAME_OR_PATH:-${MODEL:-${1:-}}}"
DATASET_NAME="${DATASET:-${2:-gsm8k_sft_train}}"
DATASET_DIR="${DATASET_DIR:-data}"
TEMPLATE="${TEMPLATE:-llama3}"
OUTPUT_DIR="${OUTPUT_DIR:-outputs/hidden_states_run}"
MAX_SAMPLES="${MAX_SAMPLES:-2000}"
TOP_K_PERCENT="${TOP_K_PERCENT:-10}"
DTYPE="${DTYPE:-bfloat16}"
SKIP_EXTRACT="${SKIP_EXTRACT:-0}"

ABS_OUT="${ROOT}/${OUTPUT_DIR}"
HIDDEN_NPY="${ABS_OUT}/hidden_states.npy"

if [[ "${SKIP_EXTRACT}" != "1" ]]; then
  if [[ -z "${MODEL}" ]]; then
    echo "错误：未指定模型。请设置 MODEL_NAME_OR_PATH 或第一个参数传入模型 ID/路径。" >&2
    echo "示例：MODEL_NAME_OR_PATH=meta-llama/Meta-Llama-3-8B-Instruct ${ROOT}/hidden_states_analysis/run_all.sh" >&2
    exit 1
  fi

  echo "==> [1/5] 抽取 hidden states"
  echo "    model=${MODEL} dataset=${DATASET_NAME} dataset_dir=${DATASET_DIR} out=${ABS_OUT}"
  python "${ROOT}/hidden_states_analysis/extract_hidden_states.py" \
    --model_name_or_path "${MODEL}" \
    --dataset_dir "${ROOT}/${DATASET_DIR}" \
    --dataset "${DATASET_NAME}" \
    --template "${TEMPLATE}" \
    --output_dir "${ABS_OUT}" \
    --max_samples "${MAX_SAMPLES}" \
    --top_k_percent "${TOP_K_PERCENT}" \
    --dtype "${DTYPE}"
else
  echo "==> 跳过抽取 (SKIP_EXTRACT=1)，使用已有：${HIDDEN_NPY}"
  if [[ ! -f "${HIDDEN_NPY}" ]]; then
    echo "错误：找不到 ${HIDDEN_NPY}" >&2
    exit 1
  fi
fi

echo "==> [2/5] UMAP + KMeans"
python "${ROOT}/hidden_states_analysis/cluster_analysis.py" --input_dir "${ABS_OUT}"

echo "==> [3/5] 可视化 UMAP"
python "${ROOT}/hidden_states_analysis/visualize.py" --input_dir "${ABS_OUT}"

echo "==> [4/5] 聚类上下文抽样（人工可读）"
python "${ROOT}/hidden_states_analysis/interpret_clusters.py" --input_dir "${ABS_OUT}"

echo "==> [5/5] 定量统计"
python "${ROOT}/hidden_states_analysis/quantitative_features.py" --input_dir "${ABS_OUT}"

echo "全部完成。输出目录：${ABS_OUT}"
ls -la "${ABS_OUT}"
