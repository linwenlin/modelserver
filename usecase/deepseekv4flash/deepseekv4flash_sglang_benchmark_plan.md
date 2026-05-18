# DeepSeek-V4-Flash / SGLang 应用级 Benchmark 方案

## 结论

Kimi-K2.6 的两阶段测试方法对 DeepSeek-V4-Flash **方法论通用**，但不能直接复用同一套档位和预期值。

通用部分：

1. 先测模型上下文长度限制和单请求长上下文性能曲线。
2. 再测不同上下文长度下的真实 running 并发数。
3. 根据上下文长度和真实 running 并发，分段测试 prefill 时间、decode 后 token/s、整体输出吞吐和端到端延迟。

模型相关部分：

- 上下文档位不同：Kimi-K2.6 主要按 256K 上限设计；DeepSeek-V4-Flash 标称 1M context，需要增加 256K、384K、512K、768K、960K 档位。
- 并发预期不同：DeepSeek-V4-Flash 是 284B MoE、13B 激活参数，权重和 KV cache 占用与 Kimi 不同，真实 running 上限必须重新从服务端日志探测。
- 启动形态不同：DeepSeek-V4-Flash 可以单机 TP8/EP8，也可以双机 TP16/EP16；测试报告必须注明启动形态，否则结果不能横向比较。

## 测试对象

当前计划对应 `deepseekv4flash_generate_start.md` 中的 SGLang native `/generate` 服务。

建议优先测试两种部署形态：

| 形态 | 用途 | 推荐说明 |
|---|---|---|
| 单机 H800 8 卡，TP8 + EP8 + DeepEP | 首轮验证、短/中上下文性能基线 | 优先在 H800_2 跑，避免影响 H800_1 现有服务 |
| 双机 H800 16 卡，TP16 + EP16 + DeepEP | 长上下文、更大 KV cache、更高并发 | 用于对比单机结果和验证 1M context 承载能力 |

每次测试记录：

```text
模型路径、启动机器、tp-size、ep-size、nnodes、mem-fraction-static、cuda-graph-max-bs、max-running-requests、max-total-num-tokens、服务端日志路径
```

## 已知模型信息

`/data/models/DeepSeek-V4-Flash`：

- 权重目录大小：约 `149G`
- 总参数：`284B`
- 激活参数：`13B`
- 上下文长度：`1M`
- 精度：`FP4 + FP8 Mixed`
- `model_type=deepseek_v4`
- `moe_intermediate_size=2048`
- `n_routed_experts=256`
- `num_experts_per_tok=6`
- `max_position_embeddings=1048576`

## 工具和目录

本目录包含：

```text
usecase/deepseekv4flash/probe_sglang_real_concurrency.py
usecase/deepseekv4flash/probe_deepseekv4flash_real_concurrency.sh
usecase/deepseekv4flash/run_sglang_app_benchmark.py
usecase/deepseekv4flash/run_deepseekv4flash_sglang_benchmark.sh
```

工具基于 `python -m sglang.bench_serving`：

- `--backend sglang` 请求 `/generate`
- `--dataset-name random-ids` 使用 synthetic token id，不依赖 HuggingFace 数据集
- `--tokenize-prompt` 固定输入 token 数
- `--random-range-ratio 1.0` 固定输入/输出长度
- `--max-concurrency` 控制客户端并发
- 真实 running 并发从 SGLang rank0 日志解析 `#running-req`、`#queue-req`、`token usage`

## 阶段一：上下文大小限制和单请求性能曲线

### 目的

先在 `max-concurrency=1` 下确认：

1. 服务实际可接受的最大 `input_tokens + output_tokens`。
2. 不同上下文长度下的 prefill 时间。
3. 扣除 prefill 后的 decode token/s。
4. 长上下文性能从哪个档位开始明显下降。

### 推荐档位

DeepSeek-V4-Flash 标称 1M context，建议第一轮档位：

| 输入 tokens | 输出 tokens | 说明 |
|---:|---:|---|
| 8192 | 1024 | 8K 基线 |
| 16384 | 1024 | 16K |
| 32768 | 1024 | 32K |
| 65536 | 1024 | 64K |
| 131072 | 1024 | 128K |
| 262144 | 1024 | 256K |
| 393216 | 1024 | 384K，Think Max 推荐下限附近 |
| 524288 | 1024 | 512K |
| 786432 | 1024 | 768K |
| 983040 | 512 | 960K，接近 1M 上限时给输出留空间 |

如果 960K 失败，可继续二分：

```text
917504:512
884736:512
851968:512
```

如果 512K 以上单请求耗时过长，可先只跑：

```text
8K, 64K, 128K, 256K, 512K, 960K
```

### 推荐命令

在模型服务所在机器本机执行：

```bash
cd /data/lin/modelserver/usecase/deepseekv4flash

MODE=single \
HOST=127.0.0.1 \
PORT=8000 \
RESULTS_DIR=results/deepseekv4flash_single_$(date +%Y%m%d_%H%M%S) \
bash run_deepseekv4flash_sglang_benchmark.sh
```

只跑关键档位：

```bash
cd /data/lin/modelserver/usecase/deepseekv4flash

MODE=single \
SINGLE_WORKLOADS=8192:1024,65536:1024,131072:1024,262144:1024,524288:1024,983040:512 \
RESULTS_DIR=results/deepseekv4flash_single_key_$(date +%Y%m%d_%H%M%S) \
bash run_deepseekv4flash_sglang_benchmark.sh
```

### 记录指标

重点记录：

- `Successful requests`
- `Mean / Median / P90 / P99 E2E Latency`
- `Mean / Median / P99 TTFT`
- `Mean / Median / P99 TPOT`
- `Mean / P95 / P99 ITL`
- `Output token throughput`
- `Total token throughput`

派生指标：

```text
prefill近似时间(s) = mean_ttft_ms / 1000
扣除prefill生成速度(token/s) = 1000 / mean_tpot_ms
整体输出吞吐(token/s) = output_throughput
e2e_ratio = 当前 E2E / 8K E2E
ttft_ratio = 当前 TTFT / 8K TTFT
tpot_ratio = 当前 TPOT / 8K TPOT
output_tps_ratio = 当前 output token throughput / 8K output token throughput
```

### 单请求上下文限制判定

如果某档失败，先确认失败类型：

| 现象 | 解释 |
|---|---|
| HTTP 400 / max model len 报错 | 请求 token 数超过服务配置或模型上限 |
| OOM / 服务退出 | KV cache 或运行时显存不足 |
| 长时间无返回但服务存活 | prefill 过慢或排队/调度异常 |
| TTFT 非线性跳升 | 长上下文 prefill 性能拐点 |

单请求可用不代表并发可用；长上下文后续还必须进入阶段二探测真实 running 并发。

## 阶段二：真实 running 并发数探测

### 目的

客户端并发不是模型真实并发。DeepSeek-V4-Flash 必须从服务端日志读取：

```text
#running-req
#queue-req
token usage
```

只有当服务端日志显示请求真实进入 running，才算模型实际并发。

### 推荐命令

单机启动时，建议 rank0 日志固定为：

```text
/data/lin/modelserver/logs/deepseekv4flash_rank0.log
```

运行：

```bash
cd /data/lin/modelserver/usecase/deepseekv4flash

SERVER_LOG=/data/lin/modelserver/logs/deepseekv4flash_rank0.log \
FIRST_CLIENT_CONCURRENCY_LEVELS=50,100,150,200 \
USE_FIRST_MAX_FOR_REST=1 \
NUM_PROMPTS=200 \
PROBE_SECONDS=1800 \
bash probe_deepseekv4flash_real_concurrency.sh
```

双机启动时，也读取 H800_1 rank0 日志：

```bash
cd /data/lin/modelserver/usecase/deepseekv4flash

SERVER_LOG=/data/lin/modelserver/logs/deepseekv4flash_rank0.log \
FIRST_CLIENT_CONCURRENCY_LEVELS=50,100,150,200 \
USE_FIRST_MAX_FOR_REST=1 \
NUM_PROMPTS=200 \
PROBE_SECONDS=1800 \
bash probe_deepseekv4flash_real_concurrency.sh
```

如果短上下文 8K 真实 running 超过 200，可把第一档客户端并发提高：

```bash
FIRST_CLIENT_CONCURRENCY_LEVELS=100,200,300,400 \
CLIENT_CONCURRENCY=400 \
bash probe_deepseekv4flash_real_concurrency.sh
```

### 默认探测档位

```text
8192:1024
16384:1024
32768:1024
65536:1024
131072:1024
262144:1024
393216:1024
524288:1024
786432:1024
983040:512
```

如果长上下文探测耗时过长，先跑关键档位：

```bash
WORKLOADS=8192:1024,65536:1024,131072:1024,262144:1024,524288:1024,983040:512 \
SERVER_LOG=/data/lin/modelserver/logs/deepseekv4flash_rank0.log \
bash probe_deepseekv4flash_real_concurrency.sh
```

### 真实 running 上限判定

满足下面条件时，认为当前 workload 已达到真实 running 容量上限：

```text
token usage >= 0.90
#queue-req > 0
#running-req 不再增加
```

粗略估算公式：

```text
running_requests ≈ max_total_num_tokens / (input_tokens + output_tokens)
```

其中 `max_total_num_tokens` 必须以实际启动日志为准。不要复用 Kimi 的实测值。

## 阶段三：基于真实 running 上限做分段性能测试

### 目的

阶段三才正式回答每个上下文长度下的：

- prefill 时间
- 扣除 prefill 后生成速度 token/s
- 整体输出吞吐 token/s
- 稳定并发上限
- 性能拐点
- 不可用点

### 并发档位生成原则

每个 workload 先从阶段二结果拿到：

```text
max_running
```

再自动生成：

```text
1
1/3 * max_running
1/2 * max_running
2/3 * max_running
max_running
```

如果 `max_running` 较大，再加入不超过 `max_running` 的整十档：

```text
10, 20, 30, 40, ...
```

### 推荐命令

假设阶段二输出为：

```text
results/deepseekv4flash_real_concurrency_xxx/summary.csv
```

运行：

```bash
cd /data/lin/modelserver/usecase/deepseekv4flash

MODE=concurrency \
RESULTS_DIR=results/deepseekv4flash_perf_from_real_$(date +%Y%m%d_%H%M%S) \
bash run_deepseekv4flash_sglang_benchmark.sh \
  --real-concurrency-summary results/deepseekv4flash_real_concurrency_xxx/summary.csv
```

只测关键上下文：

```bash
cd /data/lin/modelserver/usecase/deepseekv4flash

MODE=concurrency \
CONCURRENT_WORKLOADS=8192:1024,65536:1024,131072:1024,262144:1024,524288:1024,983040:512 \
RESULTS_DIR=results/deepseekv4flash_perf_key_$(date +%Y%m%d_%H%M%S) \
bash run_deepseekv4flash_sglang_benchmark.sh \
  --real-concurrency-summary results/deepseekv4flash_real_concurrency_xxx/summary.csv
```

如果没有真实并发 summary，脚本会使用固定客户端并发：

```text
50,100,150,200
```

但这只能作为粗测，正式报告应优先使用阶段二结果。

## 停止规则

### 硬失败停止

出现任一情况，停止继续增加该上下文长度的并发：

```text
失败率 >= 5%
服务端返回 5xx
请求大量超时
服务端 OOM
服务进程退出
GPU 进入异常状态
```

### 明显卡顿停止

出现任一情况，停止继续增加：

```text
P99 TPOT >= 1000 ms/token
Mean TPOT >= 500 ms/token
P99 ITL >= 1000 ms/token
Mean E2E latency 比上一档增加 2x 以上，且吞吐没有同步增加
P99 TTFT 比上一档增加 2x 以上
```

### 相对基线退化停止

以同一上下文长度 `concurrency=1` 为基线：

```text
output token throughput 连续 2 档持平或下降
Mean TPOT 达到 concurrency=1 的 3x 以上
P99 E2E latency 达到 concurrency=1 的 5x 以上
```

## 报告口径

每个部署形态单独出报告，不要混合单机和双机结果。

### 单请求上下文阶梯表

```text
input_tokens | output_tokens | success | prefill_s | decode_tok/s | output_tok/s | total_tok/s | E2E_p50 | E2E_p99 | TTFT_p50 | TTFT_p99 | TPOT_mean | ITL_p99 | conclusion
```

### 真实 running 并发表

```text
input_tokens | output_tokens | client_concurrency | max_running | avg_decode_running | max_queue | max_token_usage | conclusion
```

### 并发性能表

```text
input_tokens | output_tokens | concurrency | num_prompts | success_rate | prefill_s | decode_tok/s | output_tok/s | total_tok/s | E2E_p50 | E2E_p99 | TTFT_p50 | TTFT_p99 | TPOT_mean | TPOT_p99 | ITL_p99 | conclusion
```

### 拐点汇总表

```text
input_tokens | output_tokens | real_running_limit | stable_concurrency | knee_concurrency | failure_concurrency | reason
```

## 建议执行顺序

1. 启动 DeepSeek-V4-Flash 服务，并固定 rank0 日志路径。
2. 用 `/v1/models` 和 `/generate` 做最小可用性测试。
3. 跑阶段一单请求上下文阶梯，确认服务实际上下文上限和 prefill 曲线。
4. 跑阶段二真实 running 并发探测，得到每个上下文长度的 `max_running`。
5. 基于阶段二 summary 跑阶段三并发性能测试。
6. 按单机 TP8/EP8 和双机 TP16/EP16 分别汇总，不混合比较。

## 最小可用性测试

```bash
curl http://127.0.0.1:8000/v1/models \
  -H "Authorization: Bearer shuzuan2025-minimax"
```

```bash
curl http://127.0.0.1:8000/generate \
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
