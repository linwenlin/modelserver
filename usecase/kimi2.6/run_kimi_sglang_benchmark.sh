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
  --results-dir "${RESULTS_DIR:-results/kimi_$(date +%Y%m%d_%H%M%S)}" \
  --concurrency-levels "${CONCURRENCY_LEVELS:-100,200,300,400,500}" \
  --refine-min-gap "${REFINE_MIN_GAP:-25}" \
  "$@"
