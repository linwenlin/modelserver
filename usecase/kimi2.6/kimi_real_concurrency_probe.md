# Kimi / SGLang 真实并发数探测说明

## 目标

这个测试用于回答：客户端一次压入 `200` 个请求时，SGLang 服务端**真实同时运行**了多少个请求。

客户端并发不是模型真实并发。推荐策略是：先用 8K workload 按 `50,100,150,200` 探测全局 running 上限；后续更长上下文默认只用第一档探测出的 `max_running` 作为客户端并发，避免无效排队。真实并发需要看服务端日志：

```text
#running-req
#queue-req
token usage
```

例如：

```text
Decode batch, #running-req: 4, #token: 262168, token usage: 0.89, cuda graph: True, gen throughput (token/s): 9.5, #queue-req: 196
```

含义是：客户端可能压了 200 个请求，但服务端当前只有 4 个请求在 running，另有 196 个请求排队。

## 脚本

```text
usecase/probe_sglang_real_concurrency.py
usecase/probe_kimi_real_concurrency.sh
```

脚本会：

1. 用 `sglang.bench_serving` 发起固定 workload。
2. 同时读取 rank0 服务端日志。
3. 解析 `Prefill batch` / `Decode batch` 中的：
   - `#running-req`
   - `#queue-req`
   - `token usage`
4. 每个 workload 探测固定时长后主动停止 benchmark。
5. 输出：
   - `summary.csv`
   - `summary.md`
   - 每档 `.samples.jsonl`
   - 每档 benchmark stdout 日志

## 前置条件

服务需要已经启动并可访问：

```bash
curl http://127.0.0.1:8000/v1/models \
  -H "Authorization: Bearer shuzuan2025-minimax"
```

需要传入 rank0 的服务端日志路径，例如：

```text
/data/lin/modelserver/logs/kimi_bg_20260513_182712_rank0.log
```

如果你按新的 `kimi2.6_generate_start.md` 启动，建议把 rank0 日志固定写成：

```text
/data/lin/modelserver/logs/kimi_rank0.log
```

## 推荐命令

在 H800_1 上运行：

```bash
cd /data/lin/modelserver/usecase

SERVER_LOG=/data/lin/modelserver/logs/kimi_rank0.log \
FIRST_CLIENT_CONCURRENCY_LEVELS=50,100,150,200 \
USE_FIRST_MAX_FOR_REST=1 \
NUM_PROMPTS=200 \
PROBE_SECONDS=1800 \
bash probe_kimi_real_concurrency.sh
```

如果当前日志是某次后台启动的具体文件，例如：

```text
/data/lin/modelserver/logs/kimi_bg_20260513_182712_rank0.log
```

则运行：

```bash
cd /data/lin/modelserver/usecase

SERVER_LOG=/data/lin/modelserver/logs/kimi_bg_20260513_182712_rank0.log \
FIRST_CLIENT_CONCURRENCY_LEVELS=50,100,150,200 \
USE_FIRST_MAX_FOR_REST=1 \
NUM_PROMPTS=200 \
PROBE_SECONDS=1800 \
bash probe_kimi_real_concurrency.sh
```

## 默认测试档位

```text
8K input / 1K output
16K input / 1K output
32K input / 1K output
64K input / 1K output
128K input / 1K output
240K input / 512 output
```

对应参数：

```text
8192:1024,16384:1024,32768:1024,65536:1024,131072:1024,245760:512
```

可自定义：

```bash
WORKLOADS=8192:1024,65536:1024,131072:1024 \
SERVER_LOG=/data/lin/modelserver/logs/kimi_rank0.log \
bash probe_kimi_real_concurrency.sh
```

## 输出解释

默认 wrapper 行为：

```text
第一档 workload：使用 FIRST_CLIENT_CONCURRENCY_LEVELS=50,100,150,200 探测上限
后续 workload：使用第一档得到的 max_running 作为客户端并发
```

如果要强制所有 workload 都用同一个客户端并发，可以直接调用 Python 脚本，不传 `--use-first-max-for-rest`。

`summary.csv` 关键字段：

| 字段 | 含义 |
|---|---|
| `client_concurrency` | 客户端最大并发，默认 200 |
| `num_prompts` | 总请求数，默认 200 |
| `max_running` | 探测期间观察到的最大真实 running 请求数 |
| `avg_decode_running` | decode 阶段平均真实 running 请求数 |
| `max_queue` | 最大排队请求数 |
| `max_token_usage` | 最大 token/KV 池使用率 |
| `decode_sample_count` | 采集到的 decode 日志样本数量 |

判断标准：

```text
token usage >= 0.90
且 #queue-req > 0
且 #running-req 不再增加
```

说明当前上下文长度下的真实 running 并发已经到上限。

## 和 benchmark throughput 的区别

这个脚本不是为了得到完整 E2E latency 或最终吞吐，而是为了快速观察真实并发上限。

因此它会在 `PROBE_SECONDS` 后主动停止当前 benchmark，避免像完整压测一样跑几个小时。

如果要做完整性能统计，仍然使用：

```text
usecase/run_sglang_app_benchmark.py
```

## 预期估算

真实 running 并发可粗略估算为：

```text
running_requests ≈ max_total_num_tokens / (input_tokens + output_tokens)
```

当前默认服务约：

```text
max_total_num_tokens ≈ 295K
```

如果使用 `--mem-fraction-static 0.82`，预计约：

```text
max_total_num_tokens ≈ 330K
```

因此 `0.82` 下大致预期：

| 输入/输出 | 预期真实 running 并发 |
|---|---:|
| 8K / 1K | 约 35 |
| 16K / 1K | 约 19 |
| 32K / 1K | 约 9 |
| 64K / 1K | 约 4-5 |
| 128K / 1K | 约 2 |
| 240K / 512 | 约 1 |

