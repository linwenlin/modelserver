#!/usr/bin/env bash
set -euo pipefail

source /data/miniconda3/etc/profile.d/conda.sh
conda activate sglang
export OPENAI_API_KEY="${OPENAI_API_KEY:-shuzuan2025-minimax}"

cd "$(dirname "$0")"

python probe_sglang_real_concurrency.py \
  --host "${HOST:-127.0.0.1}" \
  --port "${PORT:-8000}" \
  --api-key "$OPENAI_API_KEY" \
  --server-log "${SERVER_LOG:-/data/lin/modelserver/logs/kimi_rank0.log}" \
  --results-dir "${RESULTS_DIR:-results/real_concurrency_$(date +%Y%m%d_%H%M%S)}" \
  --workloads "${WORKLOADS:-8192:1024,16384:1024,32768:1024,65536:1024,131072:1024,245760:512}" \
  --client-concurrency "${CLIENT_CONCURRENCY:-200}" \
  --first-client-concurrency-levels "${FIRST_CLIENT_CONCURRENCY_LEVELS:-50,100,150,200}" \
  --num-prompts "${NUM_PROMPTS:-200}" \
  --probe-seconds "${PROBE_SECONDS:-1800}" \
  --min-probe-seconds "${MIN_PROBE_SECONDS:-120}" \
  --stable-seconds "${STABLE_SECONDS:-60}" \
  --stable-samples "${STABLE_SAMPLES:-3}" \
  --target-token-usage "${TARGET_TOKEN_USAGE:-0.90}" \
  ${USE_FIRST_MAX_FOR_REST:+--use-first-max-for-rest} \
  "$@"

