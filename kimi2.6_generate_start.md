# Kimi-K2.6 `/generate` 启动说明

## 结论

- 当前 Kimi-K2.6 已可通过 **SGLang native `/generate` 接口**访问
- 这次切换到 `/generate` 的核心改动是：**去掉 parser 参数**
- 保留原有双机分布式拓扑、API Key、端口不变

## 服务信息

- 主机：**H800_1** `111.6.70.75:20010`
- 从机：**H800_2** `111.6.70.85:20010`
- 主机私网：`192.168.100.48`
- 从机私网：`192.168.100.50`
- 模型路径：`/data/models/Kimi-K2.6/`
- conda 环境：`sglang`
- 服务端口：`8000`
- API Key：`shuzuan2025-minimax`
- 核心接口：`/generate`

## 与之前版本的区别

之前 OpenAI 兼容版本使用了：

```bash
--reasoning-parser kimi_k2
--tool-call-parser kimi_k2
```

如果要偏向 SGLang native `/generate` 用法，去掉这两个参数即可。

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
  --api-key shuzuan2025-minimax \
  --host 0.0.0.0 \
  --port 8000
```

## 测试命令

### 1. 模型列表测试

本地/远程都可以测：

```bash
curl http://111.6.70.75:8000/v1/models \
  -H "Authorization: Bearer shuzuan2025-minimax"
```

本次实测返回成功：

```json
{"object":"list","data":[{"id":"/data/models/Kimi-K2.6/","object":"model","created":1777365666,"owned_by":"sglang","root":"/data/models/Kimi-K2.6/","parent":null,"max_model_len":262144}]}
```

### 2. `/generate` 最小测试

```bash
curl http://111.6.70.75:8000/generate \
  -H "Content-Type: application/json" \
  -H "Authorization: Bearer shuzuan2025-minimax" \
  -d '{
    "text": "你好，回复ok，不要解释。",
    "sampling_params": {
      "max_new_tokens": 8,
      "temperature": 0
    }
  }'
```

本次实测返回成功，示例返回：

```json
{"text":"  \n 2、别人问一句，","output_ids":[4170,220,17,343,4365,1126,13664,378],"meta_info":{"id":"e18ffe3f57ae4b52b7a75bd65a2110ba","finish_reason":{"type":"length","length":8},"prompt_tokens":8,"weight_version":"default","total_retractions":0,"reasoning_tokens":0,"completion_tokens":8,"cached_tokens":0,"cached_tokens_details":null,"dp_rank":null,"e2e_latency":0.7847804430020915,"response_sent_to_client_ts":1777365669.6031263}}
```

说明：

- `/generate` 已经可用
- 返回字段是 SGLang native 风格，不是 OpenAI Chat Completions 风格
- 如果上层是 `sglang-proxy`，应让代理对接 `/generate`，并自己构造 `text`

### 3. 未带鉴权头测试

```bash
curl http://111.6.70.75:8000/v1/models
```

应返回未授权错误。

## 接口说明

### 当前可用

- `GET /v1/models`
- `POST /generate`
- 仍保留 OpenAI 兼容入口，但当前目标是使用 native `/generate`

### 适配建议

如果上层是 `sglang-proxy` 且核心链路是：

- 自己拼 prompt
- 请求 `/generate`

那么当前这版启动方式更适合接入。

## 资源说明

- 去掉 `--reasoning-parser kimi_k2` 和 `--tool-call-parser kimi_k2`，**不会显著增加资源消耗**
- 主要资源消耗仍来自：模型加载、KV Cache、16TP 分布式、CUDA graph capture
- 启动速度和显存占用不会因为去掉这两个 parser 参数发生本质变化
