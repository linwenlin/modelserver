# Kimi / SGLang 应用级 Benchmark 方案

## 目标

本方案用于在模型服务器本机测试 SGLang `/generate` 服务的应用级访问能力，重点回答：

1. 单个请求在不同上下文长度下的响应时间和生成速度如何变化。
2. 以 8K 单请求作为最快基线，长上下文相对基线从哪里开始明显变慢。
3. 在关键上下文长度下，最大可承受并发是多少。
4. 并发递增时，在哪个点开始出现明显卡顿、超时、失败、吞吐下降或 token 生成时间超过阈值。

当前 Kimi-K2.6 服务参考 `kimi2.6_generate_start.md`，使用 SGLang native `/generate` 接口。

## 已验证工具

H800_1 当前 `sglang` 环境已验证可用：

```bash
python -m sglang.bench_serving
```

已确认：

- `--backend sglang` 会请求 `/generate`。
- 支持 `--host 127.0.0.1 --port 8000` 做服务器本机访问测试。
- 支持 `--dataset-name random-ids` 做离线 synthetic prompt 测试，不需要访问 HuggingFace。
- 支持 `--tokenize-prompt`，可直接传 token ids，适合控制长上下文 token 数。
- 支持 `--max-concurrency` 控制并发上限。
- 支持输出 TTFT、TPOT、ITL、E2E latency、吞吐等指标。

注意：不要用 `--dataset-name random`，它会尝试下载 HuggingFace ShareGPT 数据集。当前服务器可能无法访问 HuggingFace。

## 基础命令模板

```bash
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
  --ready-check-timeout-sec 60 \
  --output-file usecase/results/kimi_manual/kimi_bench_8k_1k_c1.jsonl
```

关键参数说明：

- `--random-range-ratio 1.0`：固定输入/输出 token 长度。必须设置，否则默认会在 `1..目标长度` 之间随机采样。
- `--tokenize-prompt`：使用 token ids 请求，减少文本 decode 后 token 数不准的问题。
- `--request-rate inf`：一次性发出请求，由 `--max-concurrency` 控制最大同时运行请求数。
- `--num-prompts`：总请求数。并发测试时建议至少等于并发数，通常设置为并发数的 `2x` 或 `3x`。

## 阶段一：单访问上下文阶梯测试

### 目的

先在 `max-concurrency=1` 下建立不同上下文长度的基线曲线，避免把并发排队、KV cache 压力和长上下文 prefill 压力混在一起。

8K 单请求作为最快基线，其它上下文长度都按 8K 归一化比较。

### Kimi-K2.6 256K 上下文建议档位

```text
8K
16K
32K
64K
96K
128K
160K
192K
224K
240K
```

输出长度建议：

```text
8K    - 224K: output 1024 tokens
240K:        output 512 tokens
```

原因：上下文上限通常是 `input_tokens + output_tokens <= max_model_len`，接近 256K 上限时需要给输出留空间。

### 1M 上下文模型建议档位

如果测试对象是 1M 上下文模型，建议扩展为：

```text
8K
16K
32K
64K
128K
256K
384K
512K
768K
960K
```

输出长度建议：

```text
8K - 768K: output 1024 tokens
960K:      output 512 tokens
```

### 单访问不是只跑一次

这里的“1个访问”定义为：

```text
max-concurrency = 1
```

但每个档位建议跑 3 次或 5 次，取中位数，降低偶发抖动影响：

```text
num-prompts = 3 或 5
max-concurrency = 1
```

### 单访问记录指标

重点记录：

- `Successful requests`
- `Mean / Median / P90 / P99 E2E Latency`
- `Mean / Median / P99 TTFT`
- `Mean / Median / P99 TPOT`
- `Mean / P95 / P99 ITL`
- `Output token throughput`
- `Total token throughput`

额外派生指标：

```text
e2e_ratio = 当前 E2E / 8K E2E
ttft_ratio = 当前 TTFT / 8K TTFT
tpot_ratio = 当前 TPOT / 8K TPOT
output_tps_ratio = 当前 output token throughput / 8K output token throughput
avg_e2e_ms_per_output_token = E2E(ms) / output_tokens
```

### 单访问拐点判断

以 8K 单请求为 baseline：

```text
正常：      output_tps_ratio >= 0.70，且 TPOT 没有明显跳升
轻微下降：  output_tps_ratio < 0.70
明显下降：  output_tps_ratio < 0.50，或 TTFT/E2E 出现非线性跳升
严重下降：  output_tps_ratio < 0.30，或 TPOT 达到 baseline 的 3x 以上
不可用：    超时、失败、OOM、服务端 5xx、E2E 超过业务可接受上限
```

如果 TTFT 随上下文长度基本线性增长，属于预期；如果某个长度后 TTFT 或 E2E 突然跳升，说明该长度附近可能是长上下文性能拐点。

## 阶段二：真实 running 并发探测

### 目的

并发测试不再先猜测 `100 / 200 / 500` 这类客户端并发，而是先测每个上下文长度下服务端真实能同时运行多少请求。

客户端并发只是压入请求数；真正决定模型是否同时执行的是 SGLang 日志中的：

```text
#running-req
#queue-req
token usage
```

例如：

```text
Decode batch, #running-req: 44, #token: 362252, token usage: 0.91, cuda graph: True, gen throughput (token/s): 71.0, #queue-req: 156
```

表示客户端虽然压入了 200 个请求，但服务端真实 running 是 44 个，另外 156 个排队。

### 探测方式

对每个 workload 先运行真实并发探测脚本。推荐使用智能策略：

1. 第一个短上下文 workload，例如 `8K / 1K`，用客户端并发 `50,100,150,200` 递增探测全局 running 上限。
2. 后续更长上下文 workload 不再继续使用 `c200`，而是使用第一档探测出的 `max_running` 作为客户端并发，避免大量请求无效排队。

```bash
cd /data/lin/modelserver/usecase

SERVER_LOG=/data/lin/modelserver/logs/kimi_rank0.log \
FIRST_CLIENT_CONCURRENCY_LEVELS=50,100,150,200 \
USE_FIRST_MAX_FOR_REST=1 \
NUM_PROMPTS=200 \
PROBE_SECONDS=1800 \
bash probe_kimi_real_concurrency.sh
```

默认探测档位：

```text
8K input / 1K output
16K input / 1K output
32K input / 1K output
64K input / 1K output
128K input / 1K output
240K input / 512 output
```

输出结果：

```text
usecase/results/real_concurrency_*/summary.csv
usecase/results/real_concurrency_*/summary.md
```

重点字段：

| 字段 | 含义 |
|---|---|
| `max_running` | 探测期间观察到的最大真实 running 请求数 |
| `avg_decode_running` | decode 阶段平均真实 running 请求数 |
| `max_queue` | 最大排队请求数 |
| `max_token_usage` | 最大 KV/token 池使用率 |

### 真实 running 上限判定

如果满足：

```text
token usage >= 0.90
#queue-req > 0
#running-req 不再增加
```

则认为当前 workload 已达到真实 running 容量上限。

真实 running 上限也可以按下面公式粗略估算：

```text
running_requests ≈ max_total_num_tokens / (input_tokens + output_tokens)
```

例如 `mem_fraction_static=0.82` 时，服务端实测：

```text
max_total_num_tokens = 399304
```

则估算：

| 输入/输出 | 估算真实 running 并发 |
|---|---:|
| 8K / 1K | 约 43 |
| 16K / 1K | 约 22 |
| 32K / 1K | 约 11 |
| 64K / 1K | 约 6 |
| 128K / 1K | 约 3 |
| 240K / 512 | 约 1 |

## 阶段三：基于真实 running 上限生成性能测试档位

### 核心原则

性能测试并发档位应基于真实 `max_running` 生成，而不是手工猜测。

对于每个 workload，先从真实并发探测结果拿到：

```text
max_running
```

然后生成测试档位：

```text
1
1/3 * max_running
1/2 * max_running
2/3 * max_running
max_running
```

如果 `max_running` 较大，再加入所有不超过 `max_running` 的整十档：

```text
10, 20, 30, 40, ...
```

最终去重、排序。如果 `max_running = 1`，则只测试：

```text
1
```

### 示例

如果 `8K / 1K` 探测到：

```text
max_running = 44
```

自动生成：

```text
1, 10, 15, 20, 29, 30, 40, 44
```

如果 `64K / 1K` 探测到：

```text
max_running = 6
```

自动生成：

```text
1, 2, 3, 4, 6
```

如果 `128K / 1K` 探测到：

```text
max_running = 3
```

自动生成：

```text
1, 2, 3
```

如果 `240K / 512` 探测到：

```text
max_running = 1
```

自动生成：

```text
1
```

### 脚本用法

`run_sglang_app_benchmark.py` 支持读取真实并发探测 summary，并自动生成并发档位：

```bash
cd /data/lin/modelserver/usecase

python run_sglang_app_benchmark.py   --mode concurrency   --host 127.0.0.1   --port 8000   --api-key shuzuan2025-minimax   --real-concurrency-summary results/real_concurrency_xxx/summary.csv   --concurrent-workloads 8192:1024,16384:1024,32768:1024,65536:1024,131072:1024,245760:512   --results-dir results/kimi_perf_from_real_concurrency_xxx
```

如果没有传 `--real-concurrency-summary`，脚本仍会使用 `--concurrency-levels` 指定的固定档位。

### 并发测试请求数

并发测试时 `num-prompts` 不应小于并发数，否则无法充分打满该并发档位。

默认规则：

```text
num_prompts = max(concurrency * 3, 3)
```

长上下文高并发如果耗时过长，可以显式指定：

```bash
--num-prompts <concurrency>
```

但报告中必须注明。

### 并发测试输出长度

第一轮建议固定：

```text
8K / 16K / 32K / 64K / 128K: output 1024 tokens
240K:                         output 512 tokens
```

如果测试目标是纯并发承载，也可以增加短输出版本：

```text
output 256 tokens
```

但报告中必须分开标注，不能和 1024 输出的结果混合比较。

## 并发停止规则

每个上下文长度从低并发开始递增。出现以下任一情况时，停止继续增加该上下文长度的并发档位，并记录该档位为瓶颈点或不可用点。

### 硬失败停止

立即停止继续增加：

```text
失败率 >= 5%
服务端返回 5xx
请求大量超时
服务端 OOM
服务进程退出
GPU 进入异常状态
```

### 明显卡顿停止

如果满足任一条件，停止继续增加：

```text
P99 TPOT >= 1000 ms/token
Mean TPOT >= 500 ms/token
P99 ITL >= 1000 ms/token
Mean E2E latency 比上一档增加 2x 以上，且吞吐没有同步增加
P99 TTFT 比上一档增加 2x 以上
```

### 相对基线退化停止

以同一上下文长度 `concurrency=1` 为并发基线：

```text
output token throughput 不再随并发增加而提升，连续 2 档持平或下降
Mean TPOT 达到 concurrency=1 的 3x 以上
P99 E2E latency 达到 concurrency=1 的 5x 以上
```

### 业务阈值停止

如果有业务可接受阈值，优先使用业务阈值。例如：

```text
Mean TPOT > 200 ms/token
P99 TPOT > 500 ms/token
P99 TTFT > 120s
P99 E2E > 300s
```

当前没有明确业务阈值时，先使用“明显卡顿停止”规则。

## 结果判定

每个上下文长度最后给出四个点：

```text
真实 running 上限：探测得到的 max_running
稳定并发上限：最后一个满足成功率和延迟阈值的并发档位
性能拐点：吞吐不再增长或 TPOT/TTFT/E2E 明显跳升的第一个档位
不可用点：失败率、超时、OOM 或严重卡顿首次出现的档位
```

示例：

```text
8K / 1K output:
- 真实 running 上限：44
- 性能测试档位：1,10,15,20,29,30,40,44
- 稳定并发上限：待性能测试确定
- 性能拐点：待性能测试确定
```

## 推荐报告表

### 单访问上下文阶梯表

```text
input_tokens | output_tokens | success | E2E_p50 | E2E_p99 | TTFT_p50 | TTFT_p99 | TPOT_mean | ITL_p99 | output_tok/s | E2E_ratio | TTFT_ratio | TPOT_ratio | conclusion
```

### 并发递增表

```text
input_tokens | output_tokens | concurrency | num_prompts | success_rate | E2E_p50 | E2E_p99 | TTFT_p50 | TTFT_p99 | TPOT_mean | TPOT_p99 | ITL_p99 | output_tok/s | total_tok/s | conclusion
```

### 拐点汇总表

```text
input_tokens | output_tokens | stable_concurrency | knee_concurrency | failure_concurrency | reason
```

## 建议执行顺序

1. 先跑 8K 单访问，得到最快基线。
2. 跑完整单访问上下文阶梯，找长上下文性能拐点。
3. 选择 8K、64K、128K、上限附近做并发递增。
4. 每个上下文长度从低并发向高并发递增。
5. 出现停止规则后，不继续增加该上下文长度的并发，转到下一个上下文长度。
6. 最后汇总稳定并发上限、性能拐点、不可用点。

## 单访问示例命令

8K 输入、1K 输出、单访问：

```bash
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
  --ready-check-timeout-sec 60 \
  --output-file usecase/results/kimi_manual/kimi_single_8k_1k_c1.jsonl
```

240K 输入、512 输出、单访问：

```bash
python -m sglang.bench_serving \
  --backend sglang \
  --host 127.0.0.1 \
  --port 8000 \
  --dataset-name random-ids \
  --tokenize-prompt \
  --random-input-len 245760 \
  --random-output-len 512 \
  --random-range-ratio 1.0 \
  --num-prompts 3 \
  --max-concurrency 1 \
  --request-rate inf \
  --ready-check-timeout-sec 60 \
  --output-file usecase/results/kimi_manual/kimi_single_240k_512_c1.jsonl
```

## 并发示例命令

8K 输入、1K 输出、500 并发：

```bash
python -m sglang.bench_serving \
  --backend sglang \
  --host 127.0.0.1 \
  --port 8000 \
  --dataset-name random-ids \
  --tokenize-prompt \
  --random-input-len 8192 \
  --random-output-len 1024 \
  --random-range-ratio 1.0 \
  --num-prompts 1000 \
  --max-concurrency 500 \
  --request-rate inf \
  --ready-check-timeout-sec 60 \
  --output-file usecase/results/kimi_manual/kimi_concurrent_8k_1k_c500.jsonl
```

128K 输入、1K 输出、64 并发：

```bash
python -m sglang.bench_serving \
  --backend sglang \
  --host 127.0.0.1 \
  --port 8000 \
  --dataset-name random-ids \
  --tokenize-prompt \
  --random-input-len 131072 \
  --random-output-len 1024 \
  --random-range-ratio 1.0 \
  --num-prompts 128 \
  --max-concurrency 64 \
  --request-rate inf \
  --ready-check-timeout-sec 60 \
  --output-file usecase/results/kimi_manual/kimi_concurrent_128k_1k_c64.jsonl
```

## H800_1 实测结果解析口径：prefill 与扣除 prefill 后生成速度

结果来源：

```text
/data/lin/modelserver/usecase/results/real_concurrency_082_reprobe_20260515_095453/summary.csv
/data/lin/modelserver/usecase/results/perf_from_real_082_full_clean_20260515_113907/summary.csv
```

关注口径：

- `prefill近似时间(s)`：使用 `mean_ttft_ms / 1000`。低并发，尤其 `c1` 时，更接近冷请求 prefill 时间；高并发时会混入排队和调度时间。
- `扣除prefill生成速度(token/s)`：使用 `1000 / mean_tpot_ms`，表示进入 decode 后，单请求平均每秒生成 token 数。
- `整体输出吞吐(token/s)`：使用 `output_throughput`，包含 prefill、排队、decode 的端到端批量输出吞吐。
- `最大真实并发`：来自真实 running 探测的 `max_running`，不是客户端压入并发。

### 最大真实并发

| 上下文 | 输出 | 最大真实并发 |
|---|---:|---:|
| 8K | 1K | 44 |
| 16K | 1K | 23 |
| 32K | 1K | 11 |
| 64K | 1K | 6 |
| 128K | 1K | 2 |
| 240K | 512 | 2 |

### 不同上下文、不同并发结果

| 上下文/输出 | 最大真实并发 | 客户端并发 | prefill近似(s) | 扣除prefill生成速度(token/s) | 整体输出吞吐(token/s) | bad | 原因 |
|---|---:|---:|---:|---:|---:|---:|---|
| 8K/1K | 44 | 1 | 0.13 | 39.84 | 39.68 | 0 |  |
| 8K/1K | 44 | 10 | 0.40 | 6.49 | 64.81 | 1 | p99_ttft jumped 3.37x |
| 8K/1K | 44 | 15 | 9.05 | 4.00 | 57.94 | 1 | p99_ttft jumped 269.82x |
| 8K/1K | 44 | 20 | 7.59 | 3.11 | 60.76 | 1 | p99_ttft jumped 271.39x |
| 8K/1K | 44 | 29 | 14.01 | 2.24 | 63.10 | 1 | p99_ttft jumped 500.76x |
| 8K/1K | 44 | 30 | 42.56 | 2.11 | 58.28 | 1 | p99_ttft jumped 828.81x |
| 8K/1K | 44 | 40 | 55.90 | 1.58 | 58.16 | 1 | mean_tpot 633.5ms >= 500.0ms; p99_ttft jumped 1113.89x |
| 8K/1K | 44 | 44 | 89.98 | 1.39 | 54.34 | 1 | mean_tpot 718.0ms >= 500.0ms; p99_ttft jumped 1228.50x |
| 16K/1K | 23 | 1 | 4.08 | 26.49 | 23.98 | 0 |  |
| 16K/1K | 23 | 8 | 0.56 | 4.92 | 35.18 | 0 |  |
| 16K/1K | 23 | 10 | 0.75 | 3.54 | 35.31 | 0 |  |
| 16K/1K | 23 | 12 | 3.48 | 2.94 | 34.94 | 1 | p99_ttft jumped 16.19x |
| 16K/1K | 23 | 15 | 32.46 | 2.23 | 31.24 | 1 | p99_ttft jumped 55.39x |
| 16K/1K | 23 | 20 | 44.12 | 1.67 | 31.11 | 1 | mean_tpot 600.0ms >= 500.0ms; mean_e2e jumped 2.27x without enough throughput growth; p99_ttft jumped 75.52x |
| 16K/1K | 23 | 23 | 50.24 | 1.46 | 31.21 | 1 | mean_tpot 684.9ms >= 500.0ms; mean_e2e jumped 2.59x without enough throughput growth; p99_ttft jumped 86.21x |
| 32K/1K | 11 | 1 | 13.75 | 15.87 | 13.09 | 0 |  |
| 32K/1K | 11 | 4 | 35.13 | 4.21 | 14.73 | 1 | p99_ttft jumped 3.99x |
| 32K/1K | 11 | 6 | 45.99 | 2.99 | 14.74 | 1 | p99_ttft jumped 5.98x |
| 32K/1K | 11 | 7 | 53.23 | 2.51 | 14.88 | 1 | p99_ttft jumped 6.80x |
| 32K/1K | 11 | 10 | 74.13 | 1.68 | 14.99 | 1 | mean_tpot 595.2ms >= 500.0ms; p99_ttft jumped 9.79x |
| 32K/1K | 11 | 11 | 81.87 | 1.53 | 15.03 | 1 | mean_tpot 652.2ms >= 500.0ms; p99_ttft jumped 10.77x |
| 64K/1K | 6 | 1 | 49.93 | 8.82 | 6.17 | 0 |  |
| 64K/1K | 6 | 2 | 73.93 | 4.16 | 6.40 | 0 |  |
| 64K/1K | 6 | 3 | 95.29 | 2.81 | 6.45 | 0 |  |
| 64K/1K | 6 | 4 | 121.41 | 2.01 | 6.51 | 0 |  |
| 64K/1K | 6 | 6 | 131.04 | 1.34 | 6.50 | 1 | mean_tpot 747.4ms >= 500.0ms |
| 128K/1K | 2 | 1 | 191.01 | 4.67 | 2.50 | 0 |  |
| 128K/1K | 2 | 2 | 291.42 | 1.98 | 2.53 | 1 | mean_tpot 505.0ms >= 500.0ms |
| 240K/512 | 2 | 1 | 670.55 | 2.56 | 0.59 | 0 |  |

截至记录时完成 `29/30` 个档位；最后 `240K/512 c2` 仍在运行，尚未写入 summary。

### 结果解读清单

- 冷请求 prefill 随上下文长度显著增长：`c1` 下从 8K 的约 `0.13s`，增长到 64K 的约 `49.93s`、128K 的约 `191.01s`、240K 的约 `670.55s`。
- 扣除 prefill 后的单请求 decode 速度也随上下文变长下降：`c1` 下 8K 约 `39.84 token/s`，16K 约 `26.49 token/s`，32K 约 `15.87 token/s`，64K 约 `8.82 token/s`，128K 约 `4.67 token/s`，240K 约 `2.56 token/s`。
- 真实最大并发随上下文长度快速下降：8K 为 `44`，16K 为 `23`，32K 为 `11`，64K 为 `6`，128K/240K 为 `2`。
- 高并发下 `prefill近似时间` 会显著变大，主要混入排队、调度和 batch 竞争，不应直接当纯 prefill 时间。
- 长上下文场景下，整体输出吞吐更多受 prefill 和排队影响；连续对话如果 prefix cache 命中，体感会更接近“只 prefill 新增 token + decode”的情况。
