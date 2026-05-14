# MiniMax-M2.7 启动脚本

## 服务信息

- 部署机器：**H800_1** `111.6.70.75:20010`
- 主机私网：`192.168.100.48`
- 模型路径：`/data/models/MiniMax-M2.7`
- conda 环境：`sglang`
- 服务端口：`8000`
- API Key：`shuzuan2025-minimax`
- 并行配置：`TP8 + EP8`

## 启动命令

```bash
source ~/miniconda3/etc/profile.d/conda.sh
conda activate sglang
cd /data/models/MiniMax-M2.7

export NCCL_SOCKET_IFNAME=bond1
export GLOO_SOCKET_IFNAME=bond1
export NCCL_DEBUG=INFO

python -m sglang.launch_server \
  --model-path /data/models/MiniMax-M2.7 \
  --tp-size 8 \
  --ep-size 8 \
  --trust-remote-code \
  --tool-call-parser minimax-m2 \
  --reasoning-parser minimax-append-think \
  --mem-fraction-static 0.85 \
  --api-key shuzuan2025-minimax \
  --host 0.0.0.0 \
  --port 8000
```

## 说明

- 服务监听 `0.0.0.0:8000`
- 使用 `bond1` 对应当前私网链路
- 4月13日启动，已稳定运行 9天+
