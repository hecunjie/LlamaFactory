#!/usr/bin/env bash
set -euo pipefail

# Example:
#   MODEL=meta-llama/Llama-3.1-8B-Instruct NPROC=4 bash intervention_experiment/sweep.sh

MODEL="${MODEL:-meta-llama/Llama-3.1-8B-Instruct}"
RESULT_DIR="${RESULT_DIR:-intervention_experiment/results}"
NPROC="${NPROC:-1}"
DTYPE="${DTYPE:-bfloat16}"
MAX_SAMPLES="${MAX_SAMPLES:-}"
LF_DATASET_NAME="${LF_DATASET_NAME:-}"
DATASET_INFO_PATH="${DATASET_INFO_PATH:-data/dataset_info.json}"
LF_SPLIT="${LF_SPLIT:-}"

mkdir -p "${RESULT_DIR}"

BASELINE_PATH="${RESULT_DIR}/baseline.jsonl"

COMMON_ARGS=(--model_name_or_path "${MODEL}" --dtype "${DTYPE}")
if [[ -n "${MAX_SAMPLES}" ]]; then
  COMMON_ARGS+=(--max_samples "${MAX_SAMPLES}")
fi
if [[ -n "${LF_DATASET_NAME}" ]]; then
  COMMON_ARGS+=(--lf_dataset_name "${LF_DATASET_NAME}" --dataset_info_path "${DATASET_INFO_PATH}")
  if [[ -n "${LF_SPLIT}" ]]; then
    COMMON_ARGS+=(--lf_split "${LF_SPLIT}")
  fi
fi

run_distributed() {
  local script="$1"
  shift
  if [[ "${NPROC}" -gt 1 ]]; then
    torchrun --nproc_per_node="${NPROC}" "${script}" "${COMMON_ARGS[@]}" "$@"
  else
    python "${script}" "${COMMON_ARGS[@]}" "$@"
  fi
}

echo "[1/4] baseline"
run_distributed intervention_experiment/run_baseline.py --output_path "${BASELINE_PATH}"
python intervention_experiment/evaluate.py \
  --pred_path "${BASELINE_PATH}" \
  --save_path "${RESULT_DIR}/baseline_eval.json"

echo "[2/4] local_temp sweep (15 configs)"
THRESHOLDS=(19.0 19.5 19.764647 20.0 20.5)
LOCAL_TEMPS=(0.3 0.5 0.7)
for th in "${THRESHOLDS[@]}"; do
  for lt in "${LOCAL_TEMPS[@]}"; do
    out="${RESULT_DIR}/local_temp_${th}_${lt}.jsonl"
    run_distributed intervention_experiment/run_local_temp.py \
      --threshold "${th}" \
      --local_temp "${lt}" \
      --output_path "${out}"
    python intervention_experiment/evaluate.py \
      --pred_path "${out}" \
      --baseline_path "${BASELINE_PATH}" \
      --save_path "${RESULT_DIR}/local_temp_${th}_${lt}_eval.json"
  done
done

echo "[3/4] soft_step sweep (3 configs)"
SOFT_THRESHOLDS=(19.0 19.764647 20.5)
for th in "${SOFT_THRESHOLDS[@]}"; do
  out="${RESULT_DIR}/soft_step_${th}.jsonl"
  run_distributed intervention_experiment/run_soft_step.py \
    --threshold "${th}" \
    --output_path "${out}"
  python intervention_experiment/evaluate.py \
    --pred_path "${out}" \
    --baseline_path "${BASELINE_PATH}" \
    --save_path "${RESULT_DIR}/soft_step_${th}_eval.json"
done

echo "[4/4] done: ${RESULT_DIR}"
