#!/usr/bin/env bash
set -euo pipefail

IMAGE="${IMAGE:-lmsysorg/sglang:deepseek-v4-hopper}"

docker run --rm \
  --gpus all \
  --network host \
  --ipc=host \
  --shm-size 32g \
  --device /dev/infiniband:/dev/infiniband \
  --cap-add IPC_LOCK \
  --ulimit memlock=-1 \
  -e NCCL_SOCKET_IFNAME="${NCCL_SOCKET_IFNAME:-bond1}" \
  -e GLOO_SOCKET_IFNAME="${GLOO_SOCKET_IFNAME:-bond1}" \
  "$IMAGE" \
  bash -lc 'echo "=== nvidia-smi ===" && nvidia-smi -L && echo && echo "=== bond1 ===" && if [ -d /sys/class/net/bond1 ]; then ls -ld /sys/class/net/bond1 && cat /sys/class/net/bond1/operstate && cat /sys/class/net/bond1/address; else echo "bond1 missing"; fi && echo && echo "=== infiniband ===" && ls -l /dev/infiniband'
