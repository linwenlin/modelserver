# DeepSeek-V4-Flash `/generate` 启动说明

## 重要前提：必须使用 sgl-project 重排过的 FP8 权重

**不要使用 DeepSeek 官方原始权重 `deepseek-ai/DeepSeek-V4-Flash` 直接喂给 sglang。**

原始权重（`/data/models/deepseek-ai/DeepSeek-V4-Flash`，约 149G）的 expert 权重命名为
`layers.N.ffn.experts.M.w1/w2/w3`，是 DeepSeek 原生 layout。sglang 的
`fused_moe_triton` loader 期望的是 sgl-project 重新打包过的 FP8 layout，两者不兼容。

用原始权重启动时，三次尝试分别死在：

1. `--moe-a2a-backend deepep` + `SGLANG_DSV4_FP4_EXPERTS=1`：权重侥幸加载，但
   cuda graph capture 时 TP rank 集合通信序列号对不上
   （`Rank 0 SequenceNumber=30 vs Rank 1 SequenceNumber=18`，BARRIER/BROADCAST mismatch）
2. `--moe-runner-backend marlin` + `SGLANG_DSV4_FP4_EXPERTS=0`：
   `expert_data.copy_(loaded_weight): tensor a (4096) must match tensor b (2048) at dim 1`
3. `--moe-a2a-backend deepep` + `SGLANG_DSV4_FP4_EXPERTS=0` + `--disable-cuda-graph`：
   `_load_w2: start (0) + length (2048) exceeds dimension size (1024)`

结论：这不是显存/模型大小问题，也不是量化方式问题（两份都是 FP8 `weight_block_size=[128,128]`），
而是 **权重 layout 与 sglang loader 不匹配**。必须下载 sgl-project 重排版本。

- 正确权重：`sgl-project/DeepSeek-V4-Flash-FP8`，ModelScope 地址
  `https://www.modelscope.cn/models/sgl-project/DeepSeek-V4-Flash-FP8`
- 实际大小约 **294G**（46 个 safetensors 分片），约为原始 Flash 的 2 倍
- 本地路径：`/data/models/DeepSeek-V4-Flash-FP8`

下载命令（H800 conda `sglang` 环境内）：

```bash
source /data/miniconda3/etc/profile.d/conda.sh
conda activate sglang
modelscope download \
  --model sgl-project/DeepSeek-V4-Flash-FP8 \
  --local_dir /data/models/DeepSeek-V4-Flash-FP8
```

## 结论

- DeepSeek-V4-Flash-FP8 已在 H800_1 验证通过，可按 **SGLang native `/generate` 接口**方式启动
- 启动统一使用官方 Hopper Docker 镜像：`lmsysorg/sglang:deepseek-v4-hopper`
- **不要带 `--moe-a2a-backend deepep`，不要带 `--moe-runner-backend marlin`，不要带 `--ep-size`**
  让 sglang 自动选 backend（FP8 走 DeepGEMM），这是当前唯一验证通过的组合
- `SGLANG_DSV4_FP4_EXPERTS=0`（FP8 权重不要强制走 FP4 experts 路径）
- 单机 **8 卡 H800：TP8**，`--mem-fraction-static 0.92`
- 双机 **16 卡 H800：TP16**，`--mem-fraction-static 0.88`，为跨机 NCCL 通信保留更大显存空间
- 模型规模 **284B 总参数 / 13B 激活参数**，上下文 **1M（1048576）**
- 单卡显存约 81.6GB，单机总显存约 652GB；TP8 `0.88` 启动后每卡占用约 72GB，`0.92` 会分配更多 KV cache

## 已验证启动结果

H800_1 上验证通过的关键指标：

- 8 卡均吃到约 72GB 显存
- cuda graph capture 完成（约 266s）
- `max_total_num_tokens=2138880`，`context_len=1048576`
- `/v1/models`、`/generate` 正常返回，未鉴权请求返回 401
- 首次启动需 DeepGEMM JIT 预编译，整体到 ready 约 **14 分钟**
  （可用 `python3 -m sglang.compile_deep_gemm` 预编译降低后续启动开销）

## 服务信息

### H800_1

- 主机：**H800_1** `111.6.70.75:20010`
- 私网：`192.168.100.48`
- 模型路径：`/data/models/DeepSeek-V4-Flash-FP8`
- 工作目录：`/data/lin/modelserver`

### H800_2

- 主机：**H800_2** `111.6.70.85:20010`
- 私网：`192.168.100.50`
- 模型路径：`/data/models/DeepSeek-V4-Flash-FP8`（需在 H800_2 同样下载）
- 工作目录：`/data/lin/modelserver`

### 通用配置

- Docker 镜像：`lmsysorg/sglang:deepseek-v4-hopper`
- 服务端口：`8000`
- API Key：`shuzuan2025-minimax`
- 核心接口：`/generate`
- NCCL / GLOO 网卡：`bond1`
- 如需 sudo：用户名 `shuzuan`，密码 `Free2024`

## 已检查环境

H800_1 已确认：

- 单卡显存约 `81559 MiB`，每台机器 8 张 H800
- `/data/models/DeepSeek-V4-Flash-FP8` 已下载完整（46 个分片）
- Docker / NVIDIA Container Toolkit 路线可用
- 官方 Hopper 镜像 `lmsysorg/sglang:deepseek-v4-hopper` 已在本地

## 启动方案：Docker 单机 H800 8 卡（已验证）

```bash
cd /data/lin/modelserver

sudo docker run --rm --name dsv4flash \
  --gpus all \
  --network host \
  --ipc=host \
  --shm-size 32g \
  --device /dev/infiniband:/dev/infiniband \
  --cap-add IPC_LOCK \
  --ulimit memlock=-1 \
  -v /data/models:/data/models \
  -v /data/lin/modelserver:/data/lin/modelserver \
  -e NCCL_SOCKET_IFNAME=bond1 \
  -e GLOO_SOCKET_IFNAME=bond1 \
  -e SGLANG_DSV4_FP4_EXPERTS=0 \
  -e PYTORCH_ALLOC_CONF=expandable_segments:True \
  -e TVM_FFI_CUDA_ARCH_LIST=9.0 \
  lmsysorg/sglang:deepseek-v4-hopper \
  bash -lc "cd /data/lin/modelserver && sglang serve \
    --model-path /data/models/DeepSeek-V4-Flash-FP8 \
    --tp-size 8 \
    --mem-fraction-static 0.92 \
    --cuda-graph-max-bs 8 \
    --trust-remote-code \
    --api-key shuzuan2025-minimax \
    --host 0.0.0.0 \
    --port 8000"
```

后台启动并落日志（推荐，便于排查首次 JIT 编译过程）：

```bash
LOG=/data/lin/modelserver/logs/dsv4flash_fp8_$(date +%Y%m%d_%H%M%S).log
sudo docker rm -f dsv4flash 2>/dev/null
nohup sudo docker run --rm --name dsv4flash \
  --gpus all --network host --ipc=host --shm-size 32g \
  --device /dev/infiniband:/dev/infiniband --cap-add IPC_LOCK --ulimit memlock=-1 \
  -v /data/models:/data/models -v /data/lin/modelserver:/data/lin/modelserver \
  -e NCCL_SOCKET_IFNAME=bond1 -e GLOO_SOCKET_IFNAME=bond1 \
  -e SGLANG_DSV4_FP4_EXPERTS=0 -e PYTORCH_ALLOC_CONF=expandable_segments:True \
  -e TVM_FFI_CUDA_ARCH_LIST=9.0 \
  lmsysorg/sglang:deepseek-v4-hopper \
  bash -lc "cd /data/lin/modelserver && sglang serve \
    --model-path /data/models/DeepSeek-V4-Flash-FP8 \
    --tp-size 8 --mem-fraction-static 0.92 --cuda-graph-max-bs 8 \
    --trust-remote-code \
    --api-key shuzuan2025-minimax --host 0.0.0.0 --port 8000" >"$LOG" 2>&1 &
```

判断启动完成：日志出现 `The server is fired up and ready to roll!`，
且 `ss -ltnp | grep :8000` 有 listener。


## 启动方案：Docker 双机 H800 16 卡 TP16（前台打印，需带 --disable-cuda-graph）

> **已知问题：当前 sglang 镜像版本，DeepSeek-V4-Flash 在 TP16 下 CUDA graph capture 必崩。**
>
> 报错形如 `Capture cuda graph failed: shape '[N, 0, -1]' is invalid for input of size M`，
> 关键是中间维度被切成 `0`：调 `--cuda-graph-max-bs`（8→4）只让 `N`/`M` 跟着变
> （`[8,0,-1]`/16384 → `[4,0,-1]`/8192），`0` 始终在，错误不消失。
>
> 根因：DeepSeek-V2/V3/MLA 在 sglang 里 `num_local_heads = num_heads // attn_tp_size`，
> 没有整除校验。Flash 是 13B 激活的小 MoE，TP16 下某 attention 维度被整除成 0，
> capture 时 reshape `[bs, 0, -1]` 必然失败。**与跨机 NCCL、显存无关**，
> 报错里那 4 条「solutions」对这个 layout bug 无效。社区现状见
> [sglang #23743](https://github.com/sgl-project/sglang/issues/23743)
> （DeepSeek-V4-Flash bring-up 跟踪 issue，官方 workaround 即 `--disable-cuda-graph`）。
>
> 因此当前阶段双机命令**必须带 `--disable-cuda-graph`**，先验证服务能起来、能响应。
> 代价：decode 阶段性能显著下降（无 CUDA graph）。若后续要保住性能，需改并行策略
> （如 `TP8 + DP attention` 让 `attn_tp_size` 仍为 8，避免维度被切到 0），属于另一轮验证。

双机方案使用 H800_1 作为 rank 0 / master，H800_2 作为 rank 1。两台机器都用 Docker 前台启动，日志会直接打印在当前终端；先在 H800_1 执行命令，看到 `Init torch distributed begin.` 后，再尽快在 H800_2 执行命令。双机使用 `--mem-fraction-static 0.88`，为跨机 NCCL 通信和运行时 workspace 保留更大显存空间。当前版本 TP16 必须带 `--disable-cuda-graph` 才能跳过会崩溃的 capture 阶段。

### H800_1 主机（rank 0）

```bash
cd /data/lin/modelserver
sudo docker rm -f dsv4flash 2>/dev/null
sudo docker run --rm --name dsv4flash \
  --gpus all \
  --network host \
  --ipc=host \
  --shm-size 32g \
  --device /dev/infiniband:/dev/infiniband \
  --cap-add IPC_LOCK \
  --ulimit memlock=-1 \
  -v /data/models:/data/models \
  -v /data/lin/modelserver:/data/lin/modelserver \
  -e NCCL_SOCKET_IFNAME=bond1 \
  -e GLOO_SOCKET_IFNAME=bond1 \
  -e NCCL_DEBUG=INFO \
  -e MASTER_ADDR=192.168.100.48 \
  -e MASTER_PORT=20000 \
  -e SGLANG_DSV4_FP4_EXPERTS=0 \
  -e PYTORCH_ALLOC_CONF=expandable_segments:True \
  -e TVM_FFI_CUDA_ARCH_LIST=9.0 \
  lmsysorg/sglang:deepseek-v4-hopper \
  bash -lc "cd /data/lin/modelserver && python3 -m sglang.launch_server \
    --model-path /data/models/DeepSeek-V4-Flash-FP8 \
    --tp-size 16 \
    --mem-fraction-static 0.88 \
    --nnodes 2 \
    --node-rank 0 \
    --dist-init-addr 192.168.100.48:20000 \
    --disable-cuda-graph \
    --trust-remote-code \
    --api-key shuzuan2025-minimax \
    --host 0.0.0.0 \
    --port 8000"
```

### H800_2 从机（rank 1）

```bash
cd /data/lin/modelserver
sudo docker rm -f dsv4flash 2>/dev/null
sudo docker run --rm --name dsv4flash \
  --gpus all \
  --network host \
  --ipc=host \
  --shm-size 32g \
  --device /dev/infiniband:/dev/infiniband \
  --cap-add IPC_LOCK \
  --ulimit memlock=-1 \
  -v /data/models:/data/models \
  -v /data/lin/modelserver:/data/lin/modelserver \
  -e NCCL_SOCKET_IFNAME=bond1 \
  -e GLOO_SOCKET_IFNAME=bond1 \
  -e NCCL_DEBUG=INFO \
  -e MASTER_ADDR=192.168.100.48 \
  -e MASTER_PORT=20000 \
  -e SGLANG_DSV4_FP4_EXPERTS=0 \
  -e PYTORCH_ALLOC_CONF=expandable_segments:True \
  -e TVM_FFI_CUDA_ARCH_LIST=9.0 \
  lmsysorg/sglang:deepseek-v4-hopper \
  bash -lc "cd /data/lin/modelserver && python3 -m sglang.launch_server \
    --model-path /data/models/DeepSeek-V4-Flash-FP8 \
    --tp-size 16 \
    --mem-fraction-static 0.88 \
    --nnodes 2 \
    --node-rank 1 \
    --dist-init-addr 192.168.100.48:20000 \
    --disable-cuda-graph \
    --trust-remote-code \
    --api-key shuzuan2025-minimax \
    --host 0.0.0.0 \
    --port 8000"
```

双机启动后，对外访问 H800_1 的 `8000` 端口。判断 ready 同样看 H800_1 终端输出是否出现 `The server is fired up and ready to roll!`。

### OOM 调参

如果启动阶段 OOM，按下面顺序收紧：

1. 单机把 `--mem-fraction-static` 从 `0.92` 降到 `0.90`、`0.88`、`0.85` 或 `0.80`；双机从 `0.88` 降到 `0.85` 或 `0.80`
2. 单机把 `--cuda-graph-max-bs` 从 `8` 降到 `4`（双机已带 `--disable-cuda-graph`，不涉及此项）
3. 如果需要人为限制活跃并发，再加 `--max-running-requests 32` 或 `16`
4. 单机仍 OOM 再临时加 `--disable-cuda-graph` 验证是否是 capture 峰值（双机已默认带）

### 不要做的事（已被验证会失败）

- 不要用原始 `deepseek-ai/DeepSeek-V4-Flash` 权重（layout 不兼容）
- 不要加 `--moe-a2a-backend deepep`（cuda graph capture TP rank 路径分叉）
- 不要加 `--moe-runner-backend marlin`（FP8 MoE 路径报
  `'Fp8MoEMethod' object has no attribute 'runner'`）
- 不要设 `SGLANG_DSV4_FP4_EXPERTS=1`（FP8 权重强制走 FP4 路径会引发不一致）
- 不要带 `--ep-size`（让 sglang 自动）

## 测试命令

下面测试以服务部署在 H800_1 为例。如果部署在 H800_2，把地址改成 `111.6.70.85`。

### 1. 模型列表测试

```bash
curl http://111.6.70.75:8000/v1/models \
  -H "Authorization: Bearer shuzuan2025-minimax"
```

正常返回时，`id` / `root` 应为：

```json
"/data/models/DeepSeek-V4-Flash-FP8"
```

`max_model_len` 应为 `1048576`。

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

- `/generate` 返回的是 **SGLang native 风格**，不是 OpenAI Chat Completions 风格
- 当前镜像是 base 续写行为；如需对话格式应由上层 `sglang-proxy` 自己拼 prompt
- 如果上层是 `sglang-proxy`，应让代理对接 `/generate`，并自己构造 `text`

### 3. 未带鉴权头测试

```bash
curl http://111.6.70.75:8000/v1/models
```

应返回 401 未授权错误。

## 接口说明

### 当前可用

- `GET /v1/models`
- `POST /generate`
- 仍保留 OpenAI 兼容入口，但当前目标是使用 native `/generate`

### 适配建议

如果上层是 `sglang-proxy` 且核心链路是「自己拼 prompt + 请求 `/generate`」，
那么当前启动方式可以直接接入。

## 补充说明

- 首次启动有 DeepGEMM JIT 预编译，到 ready 约 14 分钟；可提前用
  `python3 -m sglang.compile_deep_gemm` 同参数预编译降低开销
- 单机方案默认使用端口 `8000`；如果目标机器已有服务占用，需先停掉或改端口并同步测试命令
- Flash 支持 1M context，但生产使用长上下文前需单独压测 KV cache、并发和 tokens/sec
- H800_2 若要部署，需先在 H800_2 同样下载 `sgl-project/DeepSeek-V4-Flash-FP8`
- 双机 16 卡 TP16：当前 sglang 镜像版本 CUDA graph capture 必崩（`shape '[N,0,-1]' invalid`，
  attention 维度被 TP16 整除成 0），**必须带 `--disable-cuda-graph` 才能起来**，性能有损；
  保性能需改并行策略（如 TP8+DP attention），属另一轮验证
