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

## 阶段二：并发递增测试

### 目的

在阶段一确认单请求基线后，选择关键上下文长度做并发递增测试，找实际服务上限和出问题的点。

并发测试不是只测低并发，需要主动向上压到 100、200、500，必要时继续增加，直到出现明显退化或失败。

### 关键上下文长度

Kimi-K2.6 建议选择：

```text
8K       # 最快基线，观察短上下文最大并发
64K      # 中等长上下文
128K     # 长上下文
240K     # 接近 256K 上限
```

1M 上下文模型建议选择：

```text
8K
64K
128K
512K
960K
```

### 并发递增档位

建议用递增阶梯，而不是固定只测几个小档位。

短上下文，例如 8K：

```text
1, 2, 4, 8, 16, 32, 64, 100, 200, 500
```

中等上下文，例如 64K：

```text
1, 2, 4, 8, 16, 32, 64, 100, 200
```

长上下文，例如 128K：

```text
1, 2, 4, 8, 16, 32, 64, 100
```

接近上限，例如 240K 或 960K：

```text
1, 2, 4, 8, 16, 32
```

如果某个档位仍然稳定，且没有达到停止条件，可以继续增加：

```text
1000, 1500, 2000
```

是否继续增加以停止规则为准。

### 并发测试请求数

并发测试时 `num-prompts` 不应小于并发数，否则无法充分打满该并发档位。

建议：

```text
num-prompts = max(concurrency * 2, 20)
```

例如：

```text
concurrency=1:   num-prompts=20
concurrency=32:  num-prompts=64
concurrency=100: num-prompts=200
concurrency=500: num-prompts=1000
```

长上下文高并发可能耗时很长，可以根据实际情况降为：

```text
num-prompts = concurrency
```

但要在结果中注明。

### 并发测试输出长度

第一轮建议固定：

```text
8K / 64K / 128K: output 1024 tokens
240K / 960K:     output 512 tokens
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

每个上下文长度最后给出三个点：

```text
稳定并发上限：最后一个满足成功率和延迟阈值的并发档位
性能拐点：吞吐不再增长或 TPOT/TTFT/E2E 明显跳升的第一个档位
不可用点：失败率、超时、OOM 或严重卡顿首次出现的档位
```

示例：

```text
8K / 1K output:
- 稳定并发上限：200
- 性能拐点：500
- 不可用点：未测到或 1000

128K / 1K output:
- 稳定并发上限：16
- 性能拐点：32
- 不可用点：64
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
