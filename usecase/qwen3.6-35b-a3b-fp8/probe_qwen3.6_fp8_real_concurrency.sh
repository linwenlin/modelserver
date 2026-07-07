#!/usr/bin/env bash
set -euo pipefail

# Qwen3.6-35B-A3B-FP8 / 六卡4090 TP4 真实并发探测 wrapper
source /home/shuzuan/miniconda3/etc/profile.d/conda.sh
conda activate sglang_env
export OPENAI_API_KEY="${OPENAI_API_KEY:-EMPTY}"
export HF_HUB_OFFLINE=1
export TRANSFORMERS_OFFLINE=1
export BENCH_TOKENIZER="${BENCH_TOKENIZER:-/home/shuzuan/Project/lin/model/Qwen3.6-35B-A3B-FP8}"

cd "$(dirname "$0")"

python probe_sglang_real_concurrency.py \
  --host "${HOST:-127.0.0.1}" \
  --port "${PORT:-11450}" \
  --api-key "$OPENAI_API_KEY" \
  --server-log "${SERVER_LOG:-/home/shuzuan/Project/lin/modelserver/logs/qwen3.6-fp8-tp4.log}" \
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
