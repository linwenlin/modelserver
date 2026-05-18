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

## KV Cache / 并发调参建议

当前默认启动下，SGLang 实测服务侧参数约为：

```text
mem_fraction_static: 0.732337744140625
max_total_num_tokens: 295115
max_req_input_len: 262138
```

压测 `64K input / 1K output / c200` 时观察到：

```text
显存使用：约 67.3GB / 81.6GB
token usage：约 0.89 - 0.90
真实 running request：4
queue request：196
```

说明当前瓶颈主要是 KV/token capacity，而不是客户端并发没压上来。可以通过提高 `--mem-fraction-static` 扩大 KV 池，但需要给 CUDA graph、通信 buffer、临时张量和显存碎片保留空间。

建议第一轮使用：

```bash
--mem-fraction-static 0.82
```

按当前实测值粗略线性估算：

| mem_fraction_static | 预估 max_total_num_tokens | 说明 |
|---:|---:|---|
| 0.76 | 306K | 保守 |
| 0.78 | 314K | 较保守 |
| 0.80 | 322K | 推荐保守起点 |
| 0.82 | 330K | 推荐本轮测试值 |
| 0.84 | 338K | 激进，若 0.82 稳定再尝试 |
| 0.86 | 346K | 较高风险，可能启动或运行期 OOM |

真实可同时运行请求数可按下面粗略估算：

```text
running_requests ≈ max_total_num_tokens / (input_tokens + output_tokens)
```

以 `--mem-fraction-static 0.82`、预估 `max_total_num_tokens ≈ 330K` 计算：

| 输入/输出 | 预估真实 running 并发 |
|---|---:|
| 8K input / 1K output | 约 35 |
| 64K input / 1K output | 约 4-5 |
| 128K input / 1K output | 约 2 |
| 240K input / 512 output | 约 1 |

判断 KV/显存是否跑满，优先看服务端日志中的：

```text
token usage
#running-req
#queue-req
gen throughput
```

判定标准：

- `token usage >= 0.90`：KV/token 池已接近满载。
- `#queue-req > 0` 且 `#running-req` 不再增加：真实 running 并发已到上限。
- GPU 利用率接近 `100%`：计算侧也在满负载工作。
- 显存不应该追求 100% 占满；建议稳定运行时保留至少 `3-6GB/卡` 空余。
- 如果启动阶段在 `Load weight`、`KV Cache is allocated`、`Capture cuda graph` 附近 OOM，说明 `mem_fraction_static` 过高，需要回退。

建议测试顺序：

1. 先用 `--mem-fraction-static 0.82` 启动。
2. 启动成功后记录 `max_total_num_tokens`。
3. 用 `c200` 压测不同上下文，只观察真实 `#running-req`：
   - `8K / 1K`
   - `64K / 1K`
   - `128K / 1K`
   - `240K / 512`
4. 如果 0.82 稳定且每卡仍有明显空余，再尝试 `0.84`。
5. 如果 0.84 启动或压测不稳定，回退到 `0.82` 或 `0.80`。

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
source /data/miniconda3/etc/profile.d/conda.sh
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
  --mem-fraction-static 0.82 \
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
source /data/miniconda3/etc/profile.d/conda.sh
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
  --mem-fraction-static 0.82 \
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
