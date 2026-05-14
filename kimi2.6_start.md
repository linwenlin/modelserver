# Kimi-K2.6 启动说明

## 服务信息

- 主机：**H800_1** `111.6.70.75:20010`
- 从机：**H800_2** `111.6.70.85:20010`
- 主机私网：`192.168.100.48`
- 从机私网：`192.168.100.50`
- 模型路径：`/data/models/Kimi-K2.6/`
- conda 环境：`sglang`
- 接口类型：**OpenAI 兼容接口**
- 服务端口：`8000`
- API Key：`shuzuan2025-minimax`

> 说明：这不是 Anthropic 原生接口，使用的是 OpenAI 兼容协议（如 `/v1/models`、`/v1/chat/completions`）。

## 启动顺序

推荐顺序：

1. 先启动 **H800_1 主机**（rank 0）
2. 主机日志出现 `Init torch distributed begin.` 后
3. 再启动 **H800_2 从机**（rank 1）

## 启动命令

### H800_1 主机（rank 0）

```bash
source /home/shuzuan/miniconda3/etc/profile.d/conda.sh
conda activate sglang
cd /data/models/Kimi-K2.6/

export NCCL_SOCKET_IFNAME=bond1
export GLOO_SOCKET_IFNAME=bond1
export NCCL_DEBUG=INFO
export MASTER_ADDR=192.168.100.48
export MASTER_PORT=20000

python -m sglang.launch_server \
  --model-path /data/models/Kimi-K2.6/ \
  --tp-size 16 \
  --nnodes 2 \
  --node-rank 0 \
  --dist-init-addr 192.168.100.48:20000 \
  --trust-remote-code \
  --reasoning-parser kimi_k2 \
  --tool-call-parser kimi_k2 \
  --api-key shuzuan2025-minimax \
  --host 0.0.0.0 \
  --port 8000
```

### H800_2 从机（rank 1）

```bash
source /home/shuzuan/miniconda3/etc/profile.d/conda.sh
conda activate sglang
cd /data/models/Kimi-K2.6/

export NCCL_SOCKET_IFNAME=bond1
export GLOO_SOCKET_IFNAME=bond1
export NCCL_DEBUG=INFO
export MASTER_ADDR=192.168.100.48
export MASTER_PORT=20000

python -m sglang.launch_server \
  --model-path /data/models/Kimi-K2.6/ \
  --tp-size 16 \
  --nnodes 2 \
  --node-rank 1 \
  --dist-init-addr 192.168.100.48:20000 \
  --trust-remote-code \
  --reasoning-parser kimi_k2 \
  --tool-call-parser kimi_k2 \
  --api-key shuzuan2025-minimax \
  --host 0.0.0.0 \
  --port 8000
```

## 后台启动示例

### H800_1 主机

```bash
nohup bash -lc '
source /home/shuzuan/miniconda3/etc/profile.d/conda.sh
conda activate sglang
cd /data/models/Kimi-K2.6/
export NCCL_SOCKET_IFNAME=bond1
export GLOO_SOCKET_IFNAME=bond1
export NCCL_DEBUG=INFO
export MASTER_ADDR=192.168.100.48
export MASTER_PORT=20000
python -m sglang.launch_server \
  --model-path /data/models/Kimi-K2.6/ \
  --tp-size 16 \
  --nnodes 2 \
  --node-rank 0 \
  --dist-init-addr 192.168.100.48:20000 \
  --trust-remote-code \
  --reasoning-parser kimi_k2 \
  --tool-call-parser kimi_k2 \
  --api-key shuzuan2025-minimax \
  --host 0.0.0.0 \
  --port 8000
' > ~/sglang_kimi_k2_rank0.log 2>&1 &
```

### H800_2 从机

```bash
nohup bash -lc '
source /home/shuzuan/miniconda3/etc/profile.d/conda.sh
conda activate sglang
cd /data/models/Kimi-K2.6/
export NCCL_SOCKET_IFNAME=bond1
export GLOO_SOCKET_IFNAME=bond1
export NCCL_DEBUG=INFO
export MASTER_ADDR=192.168.100.48
export MASTER_PORT=20000
python -m sglang.launch_server \
  --model-path /data/models/Kimi-K2.6/ \
  --tp-size 16 \
  --nnodes 2 \
  --node-rank 1 \
  --dist-init-addr 192.168.100.48:20000 \
  --trust-remote-code \
  --reasoning-parser kimi_k2 \
  --tool-call-parser kimi_k2 \
  --api-key shuzuan2025-minimax \
  --host 0.0.0.0 \
  --port 8000
' > ~/sglang_kimi_k2_rank1.log 2>&1 &
```

## 启动日志

- 主机日志：`~/sglang_kimi_k2_rank0.log`
- 从机日志：`~/sglang_kimi_k2_rank1.log`

初始化过程中常见阶段：

1. `Init torch distributed begin.`
2. `Load weight begin/end`
3. `KV Cache is allocated`
4. `Capture cuda graph begin`
5. `Application startup complete`
6. `Uvicorn running on http://0.0.0.0:8000`

## 成功判断

### 1. 检查 8000 端口

```bash
ss -ltnp | grep :8000
```

### 2. 检查模型列表

```bash
curl http://127.0.0.1:8000/v1/models \
  -H "Authorization: Bearer shuzuan2025-minimax"
```

正常返回示例：

```json
{"object":"list","data":[{"id":"/data/models/Kimi-K2.6/","object":"model"}]}
```

### 3. 最小对话测试

```bash
curl http://127.0.0.1:8000/v1/chat/completions \
  -H "Content-Type: application/json" \
  -H "Authorization: Bearer shuzuan2025-minimax" \
  -d '{
    "model": "/data/models/Kimi-K2.6/",
    "messages": [{"role": "user", "content": "你好，回复ok，不要解释。"}],
    "max_tokens": 128,
    "temperature": 0
  }'
```

本次实测返回成功，`content` 为：

```json
"content":"ok"
```

未带鉴权头访问时会返回：

```json
{"error":"Unauthorized"}
```

## 接口说明

### 支持

- OpenAI 兼容客户端
- OpenAI SDK 风格调用
- 自定义 `base_url=http://<host>:8000/v1`

### 不直接支持

- Anthropic 原生 Messages API
- Claude SDK 直连 Anthropic 风格接口

如果上层系统只支持 Anthropic 协议，需要额外做一层协议转换。
