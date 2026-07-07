# Qwen3.6-35B-A3B-FP8 / SGLang 并发与效率测试方案

> 六卡4090 · TP4（GPU 2,3,4,5）· port 11450 · 纯文本压测
> 实测结果见同目录 `qwen3.6_fp8_benchmark_results.md`
> 启动/停止见仓库根目录 `qwen3.5-a3b-f8-start.md`

## 目标

在服务器本机测试 SGLang 服务的应用级访问能力，回答：

1. 不同上下文长度下单请求的 **prefill 存入时间** 与 **纯 decode 速度**。
2. 8K 单请求作为最快基线，长上下文从哪里开始明显变慢。
3. 每个上下文长度下服务端 **真实 running 并发上限**（不是客户端并发）。
4. 并发递增时的稳定并发上限、性能拐点、不可用点。
5. 性能优先以 tokens/sec 汇报，并发表以 **prefill存入时间(TTFT) + 纯decode(1/TPOT)** 两个每请求指标为主。

## 服务假设

- 主机：六卡4090 `shuzuan@58.211.6.130 -p 102`
- conda：`source ~/miniconda3/etc/profile.d/conda.sh && conda activate sglang_env`（sglang 0.5.9）
- 模型：`~/Project/lin/model/Qwen3.6-35B-A3B-FP8`（`qwen3_5_moe`，Mamba+Attention 混合，256K 上下文）
- 端口：`11450`（启动无 `--api-key`，压测无需鉴权）
- KV token 池 `max_total_num_tokens=602416`；每 token KV ≈ 10KB

## 已验证工具

```bash
python -m sglang.bench_serving --backend sglang --host 127.0.0.1 --port 11450
```

- `--dataset-name random-ids --tokenize-prompt`：离线 synthetic prompt，不依赖 HuggingFace。
- `--random-range-ratio 1.0`：固定输入/输出 token 长度。
- `--max-concurrency`：控制客户端最大并发。
- **必须加** `--tokenizer <本地模型路径>` + `HF_HUB_OFFLINE=1 TRANSFORMERS_OFFLINE=1`，否则 bench 会联网找 tokenizer 超时失败（本机不通外网）。已在 wrapper 内置。
- 本环境 **不支持** `--ready-check-timeout-sec`（脚本已去除）。

脚本（本目录）：

```text
probe_sglang_real_concurrency.py         # 真实并发探测（压测 + 读日志解析 running）
probe_qwen3.6_fp8_real_concurrency.sh    # 探测 wrapper（fp8 默认值）
run_sglang_app_benchmark.py              # 完整压测驱动（读 probe summary 自动生成并发档位）
run_qwen3.6_fp8_sglang_benchmark.sh      # 压测 wrapper（fp8 默认值）
```

---

## 阶段一：单请求上下文阶梯

在 `max-concurrency=1` 下建立每个上下文长度的 prefill/decode 基线，8K 为最快基线。

档位：`8192,16384,32768,65536,98304,131072,163840,196608,229376` 输出 1024；`245760` 输出 512（接近 256K 上限给输出留空间）。

```bash
cd ~/Project/lin/modelserver/usecase/qwen3.6-35b-a3b-fp8
MODE=single RESULTS_DIR=results/single_$(date +%Y%m%d_%H%M%S) \
bash run_qwen3.6_fp8_sglang_benchmark.sh
```

记录：`prefill存入时间(mean_ttft)`、`纯decode tok/s(1000/mean_tpot)`、整体输出吞吐。

## 阶段二：真实 running 并发探测

客户端并发 ≠ 模型真实并发。真实并发看服务端日志 `#running-req` / `token usage` / `#queue-req`。
探测脚本一边压测一边读 rank0 日志，取 `avg_decode_running`（稳态 decode 并发）为真实并发。

```bash
cd ~/Project/lin/modelserver/usecase/qwen3.6-35b-a3b-fp8
SERVER_LOG=~/Project/lin/modelserver/logs/qwen3.6-fp8-tp4.log \
RESULTS_DIR=results/real_concurrency_$(date +%Y%m%d_%H%M%S) \
WORKLOADS=8192:1024,16384:1024,32768:1024,65536:1024,131072:1024,245760:512 \
FIRST_CLIENT_CONCURRENCY_LEVELS=100,150,200 \
USE_FIRST_MAX_FOR_REST=1 NUM_PROMPTS=200 \
PROBE_SECONDS=400 MIN_PROBE_SECONDS=75 STABLE_SECONDS=35 STABLE_SAMPLES=3 \
bash probe_qwen3.6_fp8_real_concurrency.sh
```

> 提示：第一档客户端并发直接从 **高于上限**（100）起，会立即排队触发早停（~90s/档），避免 sub-cap 档跑满 200 请求的漫长等待。
> `max_running` 列含正在 prefill 的请求会虚高；用 `avg_decode_running` 看稳态真实并发。

真实并发判定：`token usage ≥ 0.90` 且 `#queue-req > 0` 且 `#running-req 不再增加`。
理论估算：`真实并发 ≈ max_total_num_tokens / (输入+输出)`。

## 阶段三：基于真实并发上限做性能压测

读阶段二的 `summary.csv`，脚本按 `max_running` 自动生成并发档位（`1/⅓/½/⅔/max` + 整十档）。

```bash
cd ~/Project/lin/modelserver/usecase/qwen3.6-35b-a3b-fp8
MODE=concurrency \
REAL_CONCURRENCY_SUMMARY=results/real_concurrency_YYYYmmdd_HHMMSS/summary.csv \
CONCURRENT_WORKLOADS=8192:1024,16384:1024,32768:1024,65536:1024,131072:1024,245760:512 \
RESULTS_DIR=results/perf_$(date +%Y%m%d_%H%M%S) \
bash run_qwen3.6_fp8_sglang_benchmark.sh --prompts-multiplier 2 --no-refine
```

> `--no-refine` 不做二分细化；不加 `STOP_ON_BAD` 则跑完所有档（得到完整曲线，只标 bad 不早停）。
> 若要第一个 bad 档即停，加 `STOP_ON_BAD=1`。

并发表汇报两个每请求指标：**prefill存入时间(TTFT)** + **纯decode tok/s(1/TPOT)**。

## 停止 / 判定规则

单档判 bad（`run_sglang_app_benchmark.py` 默认阈值）：

```text
成功率 < 95%
mean_tpot ≥ 500ms  或  p99_tpot ≥ 1000ms  或  p99_itl ≥ 1000ms
mean_e2e 比上一档跳 2x 且吞吐没同步增长
p99_ttft 比上一档跳 2x
```

每个上下文最终给四个点：真实 running 上限 / 稳定并发上限 / 性能拐点（吞吐不再增长）/ 不可用点（失败或崩溃）。

## 稳定性注意：radix cache

默认（radix 开启）在 KV 池被高并发打满时会触发 sglang 0.5.9 分配器 bug（`token_to_kv_pool_allocator memory leak detected`）并 `SIGQUIT` 崩溃（本机在 8K/1K c60~c67）。
**纯合成随机压测**（零前缀命中）用 `--disable-radix-cache` 绕开且略快；**生产有共享前缀时应保持开启**。详见 `qwen3.5-a3b-f8-start.md`。

## 本次实测摘要（2026-07-06，详见 results 文档）

| 上下文/输出 | 单请求prefill | 单请求纯decode | 最大真实并发 | 峰值聚合输出tok/s |
|---|---:|---:|---:|---:|
| 8K/1K | 0.29s | 123 tok/s | 64 | 584 |
| 32K/1K | 1.65s | 117 tok/s | 17 | 178 |
| 64K/1K | 3.45s | 110 tok/s | 12 | 93 |
| 128K/1K | 4.31s | 100 tok/s | 4 | 45 |
| 240K/512 | 20.4s | 88 tok/s | 3 | 12 |

- 满载总吞吐（prefill+decode）恒定 ~5-6K tok/s；输出吞吐随上下文反比下降。
- 并发增益随上下文衰减：8K 4.4× → 240K 1.1×（长上下文单请求即 prefill-bound）。
