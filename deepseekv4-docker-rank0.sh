#!/usr/bin/env bash
set -euo pipefail

IMAGE="${IMAGE:-lmsysorg/sglang:deepseek-v4-hopper}"
MODEL_PATH="${MODEL_PATH:-/data/models/DeepSeek-V4-Pro}"
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
  -e NCCL_BLOCKING_WAIT=1 \
  -e NCCL_TIMEOUT=3600 \
  -e TORCH_NCCL_ASYNC_ERROR_HANDLING=1 \
  -e SGLANG_DSV4_FP4_EXPERTS=1 \
  -e NVSHMEM_DEBUG=INFO \
  -e NVSHMEM_DEBUG_SUBSYS=INIT,TRANSPORT,TOPO \
  -e TORCH_DISTRIBUTED_DEBUG=DETAIL \
  -e PYTORCH_ALLOC_CONF=expandable_segments:True \
  -e TVM_FFI_CUDA_ARCH_LIST=9.0 \
  "$IMAGE" \
  bash -lc "sglang serve \
    --model-path '$MODEL_PATH' \
    --tp-size 16 \
    --dp-size 1 \
    --enable-dp-attention \
    --moe-a2a-backend deepep \
    --mem-fraction-static 0.92 \
    --disable-cuda-graph \
    --disable-overlap-schedule \
    --max-running-requests 2 \
    --context-length 8192 \
    --chunked-prefill-size 2048 \
    --nnodes 2 \
    --node-rank 0 \
    --dist-init-addr '${MASTER_ADDR}:${MASTER_PORT}' \
    --trust-remote-code \
    --api-key '$API_KEY' \
    --host 0.0.0.0 \
    --port '$PORT'"
