# Qwen3.6-35B-A3B 单机启动与部署方案

> 文件名按任务保留为 `qwen3.5-a3b-start.md`，实际模型为 **Qwen3.6-35B-A3B**。

## 结论

- 模型已下载到 **H800_1**：`/data/models/Qwen3.6-35B-A3B`
- 推荐先做 **H800_1 单机 8 卡 SGLang 部署**，对外提供 OpenAI-compatible API。
- SGLang 原生支持 OpenAI 兼容接口：`/v1/models`、`/v1/chat/completions`、`/v1/completions`。
- Anthropic Messages API 不是 SGLang 原生接口；如果需要 `/v1/messages`，建议在同机单独加一层轻量协议转换网关。
- API Key 沿用 Kimi 文档：`shuzuan2025-minimax`。
- 当前 H800_1 / H800_2 原 Kimi-K2.6 服务已停止，H800_1 GPU 可用于单机启动。

## 模型信息

来自已下载模型 `README.md` 与 `config.json`：

| 项 | 值 |
|---|---|
| 模型 | Qwen3.6-35B-A3B |
| 架构 | `Qwen3_5MoeForConditionalGeneration` |
| 类型 | Causal Language Model with Vision Encoder |
| 参数量 | 35B total / 3B activated |
| MoE | 256 experts，8 routed + 1 shared activated |
| 层数 | 40 |
| Hidden size | 2048 |
| 原生上下文 | 262,144 tokens |
| 可扩展上下文 | 官方说明最高可扩展到 1,010,000 tokens |
| dtype | bfloat16 |
| License | Apache-2.0 |
| 推荐 SGLang | `sglang>=0.5.10` |

H800_1 当前环境已确认：

```text
sglang 0.5.11
transformers 5.8.0.dev0
flashinfer-python 0.6.8.post1
torchcodec 0.11.1
ffmpeg 8.0.1
```

## 服务信息

- 主机：**H800_1** `111.6.70.75:20010`
- 模型路径：`/data/models/Qwen3.6-35B-A3B`
- conda 环境：`sglang`
- 服务端口：`8000`
- API Key：`shuzuan2025-minimax`
- 对外 OpenAI base URL：`http://111.6.70.75:8000/v1`
- 内网/本机测试地址：`http://127.0.0.1:8000/v1`

## 启动前检查

```bash
ssh -p 20010 shuzuan@111.6.70.75

source /data/miniconda3/etc/profile.d/conda.sh
conda activate sglang

nvidia-smi
du -sh /data/models/Qwen3.6-35B-A3B
ls /data/models/Qwen3.6-35B-A3B/model-*.safetensors | wc -l
```

预期：

- 8 张 H800 显存基本空闲。
- 模型目录约 `67G`。
- 权重分片为 `26` 个。

## 图片/视频解析环境

Qwen3.6-35B-A3B 是带 vision encoder 的模型。如果启动日志出现下面的报错：

```text
Ignore import error when loading sglang.srt.multimodal.processors.mimo_v2: Could not load libtorchcodec
OSError: libavutil.so.60: cannot open shared object file
```

说明当前 Python 环境中 `torchcodec` 已安装，但找不到 FFmpeg 共享库。这个问题会影响视频解码能力；图片通常主要依赖 Pillow/torchvision，不一定受影响，但多模态部署应修复。

### 当前已修复状态

H800_1 的 `sglang` 环境已安装 conda-forge FFmpeg 8.0.1，并验证通过：

```text
torchcodec: OK 0.11.1+cpu
torchvision: OK 0.26.0+cu128
PIL: OK 12.1.1
VideoDecoder: OK
sglang.srt.multimodal.processors.mimo_v2: OK
```

### 修复命令

如以后重建环境或在 H800_2 上部署，需要执行：

```bash
source /data/miniconda3/etc/profile.d/conda.sh
conda activate sglang

conda install -n sglang -c conda-forge ffmpeg -y
```

安装后验证：

```bash
source /data/miniconda3/etc/profile.d/conda.sh
conda activate sglang

which ffmpeg
ffmpeg -version | head -3
ls -lh "$CONDA_PREFIX"/lib/libavutil.so* "$CONDA_PREFIX"/lib/libavcodec.so* "$CONDA_PREFIX"/lib/libavformat.so*

python - <<'PY'
import importlib
for m in ["torch", "torchcodec", "torchvision", "PIL"]:
    mod = importlib.import_module(m)
    print("{}: OK {}".format(m, getattr(mod, "__version__", "unknown")))
from torchcodec.decoders import VideoDecoder
print("VideoDecoder: OK")
importlib.import_module("sglang.srt.multimodal.processors.mimo_v2")
print("mimo_v2 processor: OK")
PY
```

### 启动前建议

修复后重新启动 SGLang。如果启动日志里不再出现 `Could not load libtorchcodec`，视频解析依赖已正常加载。

## 推荐启动命令：OpenAI 兼容接口

这是当前建议使用的生产候选启动方式。官方 README 推荐 Qwen3.6 在 SGLang 下使用 `--mem-fraction-static 0.8`，H800_1 实测 `0.83` 启动后每卡仍保留约 `5.6-6.7GiB` 空闲显存，符合图片/视频多模态场景建议保留 `6GB+` 的余量。

```text
--tp-size 8
--mem-fraction-static 0.83
--context-length 262144
--reasoning-parser qwen3
```

启动：

```bash
ssh -p 20010 shuzuan@111.6.70.75

source /data/miniconda3/etc/profile.d/conda.sh
conda activate sglang

mkdir -p /data/lin/modelserver/logs
cd /data/models/Qwen3.6-35B-A3B

export CUDA_VISIBLE_DEVICES=0,1,2,3,4,5,6,7
export NCCL_DEBUG=INFO

nohup python -m sglang.launch_server \
  --model-path /data/models/Qwen3.6-35B-A3B \
  --served-model-name qwen3.6-35b-a3b \
  --tp-size 8 \
  --mem-fraction-static 0.83 \
  --context-length 262144 \
  --trust-remote-code \
  --reasoning-parser qwen3 \
  --tool-call-parser qwen3_coder \
  --api-key shuzuan2025-minimax \
  --host 0.0.0.0 \
  --port 8000 \
  --enable-metrics \
  > /data/lin/modelserver/logs/qwen3.6-35b-a3b.log 2>&1 &

echo $! > /data/lin/modelserver/logs/qwen3.6-35b-a3b.pid
tail -f /data/lin/modelserver/logs/qwen3.6-35b-a3b.log
```

### 参数说明

| 参数 | 说明 |
|---|---|
| `--tp-size 8` | 单机 8 卡 tensor parallel |
| `--mem-fraction-static 0.83` | H800_1 当前实测值，每卡约保留 5.6-6.7GiB 空闲显存，兼顾 KV cache 与多模态余量 |
| `--context-length 262144` | 使用官方原生 256K 上下文 |
| `--reasoning-parser qwen3` | Qwen3 reasoning/thinking 解析 |
| `--tool-call-parser qwen3_coder` | 支持 OpenAI 工具调用格式 |
| `--served-model-name qwen3.6-35b-a3b` | 对外暴露模型名，避免返回本地路径 |
| `--api-key shuzuan2025-minimax` | 与 Kimi 文档一致 |
| `--enable-metrics` | 便于后续观测吞吐和服务状态 |

## MTP 启动候选

官方 README 对 Qwen3.6 推荐了 MTP 配置。SGLang 0.5.11 的参数名是 `--speculative-algorithm`，不是 README 示例里的 `--speculative-algo`。

如果标准启动稳定后要测试生成吞吐，可以再试：

```bash
nohup python -m sglang.launch_server \
  --model-path /data/models/Qwen3.6-35B-A3B \
  --served-model-name qwen3.6-35b-a3b \
  --tp-size 8 \
  --mem-fraction-static 0.83 \
  --context-length 262144 \
  --trust-remote-code \
  --reasoning-parser qwen3 \
  --tool-call-parser qwen3_coder \
  --speculative-algorithm NEXTN \
  --speculative-num-steps 3 \
  --speculative-eagle-topk 1 \
  --speculative-num-draft-tokens 4 \
  --api-key shuzuan2025-minimax \
  --host 0.0.0.0 \
  --port 8000 \
  --enable-metrics \
  > /data/lin/modelserver/logs/qwen3.6-35b-a3b-mtp.log 2>&1 &
```

建议先不要直接用 MTP 作为首轮压测基线；先得到标准启动的并发/吞吐，再对比 MTP。

## 停止服务

```bash
ssh -p 20010 shuzuan@111.6.70.75

PID=$(cat /data/lin/modelserver/logs/qwen3.6-35b-a3b.pid 2>/dev/null || true)
if [ -n "$PID" ] && ps -p "$PID" >/dev/null 2>&1; then
  PGID=$(ps -o pgid= -p "$PID" | tr -d ' ')
  kill -TERM -- -"$PGID"
  sleep 10
  ps -p "$PID" >/dev/null 2>&1 && kill -KILL -- -"$PGID"
fi

nvidia-smi
```

如果 pid 文件不存在，用下面命令查：

```bash
ps -eo pid,ppid,pgid,etime,cmd | grep -E 'sglang.launch_server|Qwen3.6-35B-A3B' | grep -v grep
```

## 接口测试

### 1. 模型列表

```bash
curl http://111.6.70.75:8000/v1/models \
  -H "Authorization: Bearer shuzuan2025-minimax"
```

预期返回模型 ID：`qwen3.6-35b-a3b`。

### 2. OpenAI Chat Completions

```bash
curl http://111.6.70.75:8000/v1/chat/completions \
  -H "Content-Type: application/json" \
  -H "Authorization: Bearer shuzuan2025-minimax" \
  -d '{
    "model": "qwen3.6-35b-a3b",
    "messages": [
      {"role": "user", "content": "你好，回复 ok，不要解释。"}
    ],
    "temperature": 0,
    "max_tokens": 16
  }'
```

### 3. 工具调用测试

```bash
curl http://111.6.70.75:8000/v1/chat/completions \
  -H "Content-Type: application/json" \
  -H "Authorization: Bearer shuzuan2025-minimax" \
  -d '{
    "model": "qwen3.6-35b-a3b",
    "messages": [
      {"role": "user", "content": "查询北京天气，调用工具。"}
    ],
    "tools": [{
      "type": "function",
      "function": {
        "name": "get_weather",
        "description": "查询天气",
        "parameters": {
          "type": "object",
          "properties": {"city": {"type": "string"}},
          "required": ["city"]
        }
      }
    }],
    "tool_choice": "auto",
    "temperature": 0,
    "max_tokens": 256
  }'
```

## OpenAI 接口部署方案

### 推荐方案

```text
client
  -> http://111.6.70.75:8000/v1
  -> SGLang OpenAI-compatible server
  -> Qwen3.6-35B-A3B on H800_1 8 GPUs
```

客户端配置：

```text
OPENAI_BASE_URL=http://111.6.70.75:8000/v1
OPENAI_API_KEY=shuzuan2025-minimax
model=qwen3.6-35b-a3b
```

适用接口：

- `GET /v1/models`
- `POST /v1/chat/completions`
- `POST /v1/completions`

如果已有上层代理，优先让代理直接转发 OpenAI 协议，不要再拼 SGLang native `/generate`，这样工具调用、reasoning parser 和客户端 SDK 兼容性更好。

## Anthropic 接口部署方案

SGLang 不原生提供 Anthropic Messages API。要同时提供 OpenAI 和 Anthropic 两套接口，建议同机部署一层协议转换网关。

### 推荐拓扑

```text
OpenAI clients
  -> :8000/v1/*
  -> SGLang

Anthropic clients
  -> :8001/v1/messages
  -> Anthropic-compatible adapter
  -> :8000/v1/chat/completions
  -> SGLang
```

建议：

- SGLang 继续监听 `8000`，只负责模型推理和 OpenAI-compatible API。
- Anthropic adapter 监听 `8001`，只做协议转换和鉴权。
- 两边 API Key 初期都使用 `shuzuan2025-minimax`，减少运维差异。
- 对外如果只暴露一个域名，可以后续用 Nginx 按路径转发：
  - `/v1/chat/completions` -> `127.0.0.1:8000`
  - `/v1/models` -> `127.0.0.1:8000`
  - `/v1/messages` -> `127.0.0.1:8001`

### Anthropic adapter 必须转换的字段

请求转换：

| Anthropic Messages | OpenAI Chat Completions |
|---|---|
| `model` | `model`，可强制替换为 `qwen3.6-35b-a3b` |
| `system` | `messages[0].role=system` |
| `messages[].role=user/assistant` | 同名 role |
| `messages[].content[].type=text` | OpenAI text content |
| `max_tokens` | `max_tokens` |
| `temperature` | `temperature` |
| `top_p` | `top_p` |
| `stream` | `stream` |
| `tools` | OpenAI `tools`，需要转换 schema |
| `tool_choice` | OpenAI `tool_choice` |

响应转换：

| OpenAI Chat Completions | Anthropic Messages |
|---|---|
| `choices[0].message.content` | `content[].type=text` |
| `choices[0].message.tool_calls` | `content[].type=tool_use` |
| `finish_reason=stop` | `stop_reason=end_turn` |
| `finish_reason=length` | `stop_reason=max_tokens` |
| `finish_reason=tool_calls` | `stop_reason=tool_use` |
| `usage.prompt_tokens` | `usage.input_tokens` |
| `usage.completion_tokens` | `usage.output_tokens` |

### Anthropic adapter 第一版边界

第一版建议只实现：

- `POST /v1/messages`
- 非流式 `stream=false`
- 文本输入/输出
- 基础 tool use 映射
- `x-api-key: shuzuan2025-minimax` 鉴权

第二版再补：

- streaming SSE
- image content 转换
- prompt caching 字段兼容
- 更完整的 stop sequence / thinking 字段映射

原因：Anthropic streaming 事件格式和 OpenAI chunk 格式差异较大，第一版直接做全量兼容会增加不必要复杂度。

## Benchmark 方案

参考 `usecase/kimi2.6` 中 Kimi 的测试方式，推荐分两阶段：

1. 真实 running 并发探测：从服务端日志读取 `#running-req`、`#queue-req`、`token usage`。
2. 应用级性能压测：基于真实 running 上限生成并发档位，输出 TTFT、TPOT、ITL、E2E latency 和 tokens/sec。

### 准备测试脚本

H800_1 上已有仓库路径：`/data/lin/modelserver`。

如果本地脚本更新后需要同步到 H800_1：

```bash
rsync -av -e 'ssh -p 20010' usecase/kimi2.6/ shuzuan@111.6.70.75:/data/lin/modelserver/usecase/qwen3.6/
```

也可以直接复用 Kimi 目录脚本，只改环境变量。

### 阶段一：服务健康与单请求基线

```bash
ssh -p 20010 shuzuan@111.6.70.75

source /data/miniconda3/etc/profile.d/conda.sh
conda activate sglang
export OPENAI_API_KEY=shuzuan2025-minimax

python -m sglang.bench_serving \
  --backend sglang \
  --host 127.0.0.1 \
  --port 8000 \
  --dataset-name random-ids \
  --tokenize-prompt \
  --random-input-len 8192 \
  --random-output-len 1024 \
  --random-range-ratio 1.0 \
  --num-prompts 3 \
  --max-concurrency 1 \
  --request-rate inf \
  --output-file /data/lin/modelserver/usecase/qwen3.6_8k_single.jsonl
```

### 阶段二：真实 running 并发探测

复制 Kimi 脚本后执行：

```bash
ssh -p 20010 shuzuan@111.6.70.75

source /data/miniconda3/etc/profile.d/conda.sh
conda activate sglang
cd /data/lin/modelserver/usecase/qwen3.6

export OPENAI_API_KEY=shuzuan2025-minimax
export SERVER_LOG=/data/lin/modelserver/logs/qwen3.6-35b-a3b.log
export RESULTS_DIR=results/qwen3.6_real_concurrency_$(date +%Y%m%d_%H%M%S)
export WORKLOADS=8192:1024,16384:1024,32768:1024,65536:1024,131072:1024,245760:512
export CLIENT_CONCURRENCY=200
export FIRST_CLIENT_CONCURRENCY_LEVELS=50,100,150,200
export NUM_PROMPTS=200
export PROBE_SECONDS=1800
export MIN_PROBE_SECONDS=120
export STABLE_SECONDS=60
export STABLE_SAMPLES=3
export TARGET_TOKEN_USAGE=0.90
export USE_FIRST_MAX_FOR_REST=1

bash probe_kimi_real_concurrency.sh
```

输出：

```text
results/qwen3.6_real_concurrency_*/summary.csv
results/qwen3.6_real_concurrency_*/summary.md
```

重点看：

- `max_running`
- `avg_decode_running`
- `max_queue`
- `max_token_usage`

真实 running 上限判定：

```text
token usage >= 0.90
#queue-req > 0
#running-req 不再增加
```

### 阶段三：基于真实并发上限做性能压测

```bash
ssh -p 20010 shuzuan@111.6.70.75

source /data/miniconda3/etc/profile.d/conda.sh
conda activate sglang
cd /data/lin/modelserver/usecase/qwen3.6

export OPENAI_API_KEY=shuzuan2025-minimax
REAL_SUMMARY=results/qwen3.6_real_concurrency_YYYYmmdd_HHMMSS/summary.csv

python run_sglang_app_benchmark.py \
  --mode all \
  --host 127.0.0.1 \
  --port 8000 \
  --api-key "$OPENAI_API_KEY" \
  --real-concurrency-summary "$REAL_SUMMARY" \
  --single-workloads 8192:1024,16384:1024,32768:1024,65536:1024,98304:1024,131072:1024,163840:1024,196608:1024,229376:1024,245760:512 \
  --concurrent-workloads 8192:1024,16384:1024,32768:1024,65536:1024,131072:1024,245760:512 \
  --results-dir results/qwen3.6_perf_from_real_concurrency_$(date +%Y%m%d_%H%M%S) \
  --stop-on-bad
```

输出：

```text
results/qwen3.6_perf_from_real_concurrency_*/summary.csv
results/qwen3.6_perf_from_real_concurrency_*/summary.md
```

### 报告指标优先级

报告性能时优先使用 token 吞吐：

- `output_throughput`：输出 tokens/sec
- `total_throughput`：输入 + 输出 tokens/sec
- `mean_tpot_ms` / `p99_tpot_ms`：每 token 生成耗时
- `mean_ttft_ms` / `p99_ttft_ms`：首 token 延迟
- `mean_e2e_latency_ms` / `p99_e2e_latency_ms`：端到端延迟

每个上下文长度最后汇总：

| input/output | real max_running | stable concurrency | knee concurrency | output tok/s | total tok/s | reason |
|---|---:|---:|---:|---:|---:|---|

### 第一轮推荐测试顺序

1. 启动标准版，不开 MTP。
2. 跑 8K/1K 单请求，确认接口和脚本正常。
3. 跑真实 running 并发探测。
4. 基于真实 running 上限跑完整性能压测。
5. 当前推荐固定使用 `mem_fraction_static=0.83` 做标准版基线；如果压测或视频请求出现 OOM，再回退到 `0.80-0.82`。
6. 如果标准版结果稳定，再单独启动 MTP 版本，对比相同 workload 的 `output_throughput` 和 TPOT。

## KV Cache / 并发调参建议

Qwen3.6 默认 256K 上下文，长上下文下真实并发主要受 KV/token capacity 限制。粗略估算：

```text
running_requests ≈ max_total_num_tokens / (input_tokens + output_tokens)
```

启动后先从日志记录：

```text
mem_fraction_static
max_total_num_tokens
max_req_input_len
```

再决定是否调整：

| 参数 | 建议 |
|---|---|
| `--mem-fraction-static 0.80` | 官方推荐起点，最保守 |
| `--mem-fraction-static 0.82` | 保守生产候选 |
| `--mem-fraction-static 0.83` | 当前 H800_1 推荐值，已观察到每卡约 5.6-6.7GiB 空闲 |
| `--mem-fraction-static 0.84` | 略激进，只建议压测验证后再考虑 |
| `--context-length 131072` | 如果 256K 启动 OOM 或实际业务不需要 256K，可降到 128K 换并发 |

判断 KV/显存是否跑满，优先看服务端日志：

```text
token usage
#running-req
#queue-req
gen throughput
```

判定标准：

- `token usage >= 0.90`：KV/token 池接近满载。
- `#queue-req > 0` 且 `#running-req` 不再增加：真实 running 并发到上限。
- GPU 利用率接近 `100%`：计算侧满载。
- 稳定运行不追求显存 100%，建议每卡保留至少 `3-6GB`。
- 如果启动在 load weight、KV cache allocation、CUDA graph capture 附近 OOM，回退 `mem_fraction_static` 或降低 `context-length`。

## 单机部署建议

### 第一阶段：单服务直接暴露

```text
H800_1:8000
  SGLang OpenAI-compatible API
  model: qwen3.6-35b-a3b
  auth: Bearer shuzuan2025-minimax
```

优点：链路短、性能损耗最小、最容易压测。

### 第二阶段：增加 Anthropic adapter

```text
H800_1:8000  SGLang OpenAI API
H800_1:8001  Anthropic Messages adapter
```

优点：不影响 OpenAI 客户端，Anthropic 兼容问题集中在 adapter 内。

### 第三阶段：统一入口

如需统一端口或域名，再加 Nginx：

```text
:80 or :443
  /v1/models           -> 127.0.0.1:8000
  /v1/chat/completions -> 127.0.0.1:8000
  /v1/completions      -> 127.0.0.1:8000
  /v1/messages         -> 127.0.0.1:8001
```

如果只是内网调用，第一阶段和第二阶段足够，不需要 Nginx。

## 注意事项

- Qwen3.6 模型带 vision encoder；当前方案按文本/工具调用优先验证。
- 官方建议遇到 OOM 可降低 context window，但最好保持至少 128K，以保留复杂任务能力。
- 不建议一开始做 1M context；先把 256K 的启动、并发和性能基线跑稳定。
- `/generate` 可作为 SGLang native 调试接口，但对外服务优先使用 OpenAI-compatible API。
- Anthropic API 兼容层要明确“兼容 Messages API”，不是 Claude 原生能力；模型输出质量仍取决于 Qwen3.6 和提示词模板。
