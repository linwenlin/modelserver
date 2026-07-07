#!/usr/bin/env bash
set -euo pipefail

# Qwen3.6-35B-A3B-FP8 / 六卡4090 TP4 应用级压测 wrapper
source /home/shuzuan/miniconda3/etc/profile.d/conda.sh
conda activate sglang_env
export OPENAI_API_KEY="${OPENAI_API_KEY:-EMPTY}"
export HF_HUB_OFFLINE=1
export TRANSFORMERS_OFFLINE=1
export BENCH_TOKENIZER="${BENCH_TOKENIZER:-/home/shuzuan/Project/lin/model/Qwen3.6-35B-A3B-FP8}"

cd "$(dirname "$0")"

python run_sglang_app_benchmark.py \
  --mode "${MODE:-all}" \
  --host "${HOST:-127.0.0.1}" \
  --port "${PORT:-11450}" \
  --api-key "$OPENAI_API_KEY" \
  --results-dir "${RESULTS_DIR:-results/qwen3.6_fp8_$(date +%Y%m%d_%H%M%S)}" \
  --single-workloads "${SINGLE_WORKLOADS:-8192:1024,16384:1024,32768:1024,65536:1024,98304:1024,131072:1024,163840:1024,196608:1024,229376:1024,245760:512}" \
  --concurrent-workloads "${CONCURRENT_WORKLOADS:-8192:1024,16384:1024,32768:1024,65536:1024,131072:1024,245760:512}" \
  --concurrency-levels "${CONCURRENCY_LEVELS:-50,100,150,200}" \
  ${REAL_CONCURRENCY_SUMMARY:+--real-concurrency-summary "$REAL_CONCURRENCY_SUMMARY"} \
  ${STOP_ON_BAD:+--stop-on-bad} \
  --refine-min-gap "${REFINE_MIN_GAP:-25}" \
  "$@"
