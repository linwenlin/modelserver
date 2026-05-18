#!/usr/bin/env bash
set -euo pipefail

source /data/miniconda3/etc/profile.d/conda.sh
conda activate sglang
export OPENAI_API_KEY="${OPENAI_API_KEY:-shuzuan2025-minimax}"

cd "$(dirname "$0")"

python run_sglang_app_benchmark.py \
  --mode "${MODE:-all}" \
  --host "${HOST:-127.0.0.1}" \
  --port "${PORT:-8000}" \
  --api-key "$OPENAI_API_KEY" \
  --results-dir "${RESULTS_DIR:-results/deepseekv4flash_$(date +%Y%m%d_%H%M%S)}" \
  --single-workloads "${SINGLE_WORKLOADS:-8192:1024,16384:1024,32768:1024,65536:1024,131072:1024,262144:1024,393216:1024,524288:1024,786432:1024,983040:512}" \
  --concurrent-workloads "${CONCURRENT_WORKLOADS:-8192:1024,65536:1024,131072:1024,262144:1024,524288:1024,983040:512}" \
  --concurrency-levels "${CONCURRENCY_LEVELS:-50,100,150,200}" \
  --refine-min-gap "${REFINE_MIN_GAP:-10}" \
  "$@"
