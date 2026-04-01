#!/usr/bin/env bash
# 仅执行聚类及后续步骤（需已运行过抽取并生成 hidden_states.npy）
# 等价于：SKIP_EXTRACT=1 ./hidden_states_analysis/run_all.sh
set -euo pipefail
SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
export SKIP_EXTRACT=1
exec "${SCRIPT_DIR}/run_all.sh" "$@"
