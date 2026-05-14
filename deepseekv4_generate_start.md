# DeepSeek-V4-Pro `/generate` 启动说明

## 结论

- 当前 DeepSeek-V4-Pro 已完成依赖兼容，可按 **SGLang native `/generate` 接口**方式启动
- 这版启动命令改为 **MoE 并行版本**：`TP16 + EP16 + DeepEP`
- 为降低单卡启动峰值显存，这版默认加入更保守的显存参数：`--mem-fraction-static 0.80`、`--cuda-graph-max-bs 8`、`--max-running-requests 32`
- 保留当前双机分布式拓扑、API Key、端口不变
- **注意**：仅升级到最新 `sglang` 还不够，还需要把 `transformers` 升到支持 `deepseek_v4` 的版本
- **注意**：本地原始模型路径直接使用 **plain `TP16`** 会命中 FP8 block quant 对齐错误，不能再按旧命令启动

## 服务信息

- 主机：**H800_1** `111.6.70.75:20010`
- 从机：**H800_2** `111.6.70.85:20010`
- 主机私网：`192.168.100.48`
- 从机私网：`192.168.100.50`
- 模型路径：`/data/models/DeepSeek-V4-Pro`
- conda 环境：`sglang`
- 服务端口：`8000`
- API Key：`shuzuan2025-minimax`
- 核心接口：`/generate`

## 环境前置

当前双机已验证通过的环境组合：

- `sglang==0.5.11`
- `transformers==5.8.0.dev0`

原因：

- DeepSeek-V4-Pro 的 `config.json` 中 `model_type` 为 `deepseek_v4`
- `sglang 0.5.11` 自带依赖会把 `transformers` 固定到 `5.6.0`
- 仅升级 `sglang` 时，启动仍会报 `Transformers does not recognize this architecture`

建议先在 **H800_1** 和 **H800_2** 都执行：

```bash
source /data/miniconda3/etc/profile.d/conda.sh
conda activate sglang
pip install -U sglang
pip install -U git+https://github.com/huggingface/transformers.git
```

如果某台机器无法直连 GitHub，可先在能联网的机器上构建 wheel，再拷贝后安装。

## 启动顺序

推荐顺序：

1. 先启动 **H800_1 主机**（rank 0）
2. 主机日志出现 `Init torch distributed begin.` 后
3. 再启动 **H800_2 从机**（rank 1）

## 为什么不能继续用旧版 `TP16`

当前本地 `/data/models/DeepSeek-V4-Pro` 在 SGLang 的 FP8 MoE 路径下，如果直接使用：

```bash
--tp-size 16
```

会在启动阶段报错：

```text
ValueError: The output_size of gate's and up's weight = 192 is not divisible by weight quantization block_n = 128.
```

根因是：

- 该模型的 MoE 权重在 **plain TP16** 分片后，某些 gate/up 权重维度变成 `192`
- 当前 SGLang FP8 block quant 路径要求该维度可被 `128` 整除
- 因此不能继续只靠 `TP16` 硬切，需要改用 **MoE-aware 并行方式**

## 启动命令

## OOM 调参说明

这次启动报错不是总显存不够，而是**单张卡**在启动阶段已经接近打满，因此需要优先降低单卡峰值显存。

当前文档默认加入这 3 个参数：

- `--mem-fraction-static 0.80`
  - 降低静态显存占用比例，给权重加载后的临时张量、通信 buffer、CUDA graph capture 预留更多空间
- `--cuda-graph-max-bs 8`
  - 降低 CUDA graph 的最大 batch size，减少 graph capture 带来的额外显存消耗
- `--max-running-requests 32`
  - 降低运行期最大并发请求数，减少 KV cache 和调度侧预留显存

如果后续仍然 OOM，建议按这个顺序继续收紧：

1. 先把 `--mem-fraction-static` 从 `0.80` 继续降到 `0.78` 或 `0.76`
2. 再把 `--cuda-graph-max-bs` 从 `8` 降到 `4`
3. 最后把 `--max-running-requests` 从 `32` 降到 `16`

### H800_1 主机（rank 0）

```bash
source /data/miniconda3/etc/profile.d/conda.sh
conda activate sglang
cd /data/models/DeepSeek-V4-Pro

export NCCL_SOCKET_IFNAME=bond1
export GLOO_SOCKET_IFNAME=bond1
export NCCL_DEBUG=INFO
export MASTER_ADDR=192.168.100.48
export MASTER_PORT=20000

python -m sglang.launch_server \
  --model-path /data/models/DeepSeek-V4-Pro \
  --tp-size 16 \
  --ep-size 16 \
  --moe-a2a-backend deepep \
  --mem-fraction-static 0.80 \
  --cuda-graph-max-bs 8 \
  --max-running-requests 32 \
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
cd /data/models/DeepSeek-V4-Pro

export NCCL_SOCKET_IFNAME=bond1
export GLOO_SOCKET_IFNAME=bond1
export NCCL_DEBUG=INFO
export MASTER_ADDR=192.168.100.48
export MASTER_PORT=20000

python -m sglang.launch_server \
  --model-path /data/models/DeepSeek-V4-Pro \
  --tp-size 16 \
  --ep-size 16 \
  --moe-a2a-backend deepep \
  --mem-fraction-static 0.80 \
  --cuda-graph-max-bs 8 \
  --max-running-requests 32 \
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

正常返回时，`id` / `root` 应为：

```json
"/data/models/DeepSeek-V4-Pro"
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

说明：

- `/generate` 返回的是 **SGLang native 风格**
- 不是 OpenAI Chat Completions 风格
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

## 补充说明

- DeepSeek-V4-Pro 模型目录中包含官方 `inference` 示例，但这里统一采用 **SGLang 服务化启动方式**，便于和当前 Kimi / MiniMax 的部署方式保持一致
- 这版命令不再使用 plain `16TP`，而是优先验证 **`TP16 + EP16 + DeepEP`** 能否稳定拉起
- 如果后续要进一步追求低延迟或更强吞吐，再考虑按 SGLang 的 DeepSeek-V4 官方文档增加更激进的并行/后端参数
- 当前文档沿用端口 `8000`；如果 H800_1 上仍在运行 MiniMax 服务，也会占用 `8000`，需要先停掉原服务或改成新端口并同步修改测试命令
