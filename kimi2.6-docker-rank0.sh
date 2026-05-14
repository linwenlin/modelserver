#!/usr/bin/env bash
set -euo pipefail

IMAGE="${IMAGE:-lmsysorg/sglang:deepseek-v4-hopper}"
MODEL_PATH="${MODEL_PATH:-/data/models/Kimi-K2.6/}"
API_KEY="${API_KEY:-shuzuan2025-minimax}"
MASTER_ADDR="${MASTER_ADDR:-192.168.100.48}"
MASTER_PORT="${MASTER_PORT:-20000}"
PORT="${PORT:-8000}"
NCCL_IFNAME="${NCCL_SOCKET_IFNAME:-bond1}"
GLOO_IFNAME="${GLOO_SOCKET_IFNAME:-bond1}"

docker run --rm -it \
  --gpus all \
  --network host \
  --ipc=host \
  --shm-size 32g \
  --device /dev/infiniband:/dev/infiniband \
  --cap-add IPC_LOCK \
  --ulimit memlock=-1 \
  -v /data/models:/data/models \
  -e NCCL_SOCKET_IFNAME="$NCCL_IFNAME" \
  -e GLOO_SOCKET_IFNAME="$GLOO_IFNAME" \
  -e NCCL_DEBUG=INFO \
  -e TORCH_DISTRIBUTED_DEBUG=DETAIL \
  -e TVM_FFI_CUDA_ARCH_LIST=9.0 \
  "$IMAGE" \
  bash -lc "sglang serve \
    --model-path '$MODEL_PATH' \
    --tp-size 16 \
    --nnodes 2 \
    --node-rank 0 \
    --dist-init-addr '${MASTER_ADDR}:${MASTER_PORT}' \
    --trust-remote-code \
    --api-key '$API_KEY' \
    --host 0.0.0.0 \
    --port '$PORT'"
