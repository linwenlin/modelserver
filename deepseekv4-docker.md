# DeepSeek-V4-Pro Docker 部署计划

## 结论

- 当前两台 **H800（每台 8 卡，80GB）** 可以视为 **Hopper 系**，官方 DeepSeek-V4 对应路线应优先对齐 **Hopper 镜像**。
- 对于 **DeepSeek-V4-Pro FP8 checkpoint**，SGLang 官方 DeepSeek-V4 页面给出了 **H200/Hopper 2 节点、16 GPU** 的可用路线，说明这类拓扑 **方向上是可行的**。
- 但文档明确写的是 **H200**，不是 H800；而且当前 H800 为 **80GB**，显存余量通常会比文档默认参考硬件更紧，因此结论应定为：
  - **具备较强可行性，值得实现验证**
  - **但不能在动手前承诺一定拉起成功**
- 当前本地 `conda + pip sglang==0.5.11` 路线已经多次落到 **Transformers fallback**，因此下一步应切到 **官方 Docker 镜像路线** 验证。

## 为什么要改走 Docker

- 当前通用 `sglang==0.5.11` 虽然已是最新，但 DeepSeek-V4-Pro 实际日志仍反复出现：
  - `Using Transformers backend.`
- 这说明当前环境没有稳定命中官方预期的 DeepSeek-V4 原生 Hopper 路线。
- SGLang 官方 DeepSeek-V4 文档给了按硬件拆分的专用镜像，其中 Hopper 对应：
  - `lmsysorg/sglang:deepseek-v4-hopper`
- 因此更合理的验证路径是：
  1. 使用 Hopper 专用镜像
  2. 对齐官方生成器给出的 `sglang serve` 路线
  3. 再验证双机 16 卡、IB、DeepEP 是否可稳定启动

## 可行性调研结果

### 1. 两台 H800 是否属于可尝试的目标硬件

是。

- NVIDIA 官方 Hopper 页面明确把 **H800** 列在 Hopper 架构下。
- H800 显存规格包含：
  - `H800 PCIe 80GB`
  - `H800 SXM5 80GB`
  - `H800 NVL 94GB`
- 当前你的机器为 **H800 × 8 / 80GB 每卡**，因此应按 **Hopper 系** 处理。

### 2. 两节点 16 卡跑 DeepSeek-V4-Pro 是否有官方依据

有，而且方向匹配。

- SGLang DeepSeek-V4 文档对 **H200/Hopper** 提供了两类路径：
  1. **Original FP4 checkpoints**：只支持 TP，偏单机 TP 路线
  2. **Converted FP8 checkpoints**：支持更广的并行/特性组合
- 文档中的 H200/Hopper Pro 路线里，存在 **2 节点 / 16 GPU / `tp=16`** 的生成器方案。
- 当前你的模型日志已明确：
  - `Detected fp8 checkpoint.`
- 因此从 checkpoint 形态上，当前更应对齐 **Hopper + FP8 + 多节点** 路线，而不是继续沿用旧的通用 pip 环境命令。

### 3. 有哪些前提条件必须满足

必须至少满足以下条件：

- 使用 Hopper 对应镜像：
  - `lmsysorg/sglang:deepseek-v4-hopper`
- 容器需要完整拿到 GPU：
  - `--gpus all`
- 容器建议使用宿主机网络与 IPC：
  - `--network host`
  - `--ipc=host`
  - `--shm-size 32g`
- 如需 IB / RDMA / DeepEP 跨机通信，容器要暴露 IB 设备与相关权限：
  - `--device /dev/infiniband:/dev/infiniband`
  - `--cap-add IPC_LOCK`
  - `--ulimit memlock=-1`
  - 如仍不足，可考虑 `--privileged`
- 容器内继续绑定当前私网 NIC：
  - `NCCL_SOCKET_IFNAME=bond1`
  - `GLOO_SOCKET_IFNAME=bond1`

### 4. 当前阶段的主要风险

- 文档是按 **H200/Hopper** 给方案，不是按 H800 单独验证；H800 虽同属 Hopper，但显存/频率/平台细节仍可能带来差异。
- SGLang 安装文档写明：默认 Docker 镜像是 **CUDA 13** 环境；如果宿主驱动/运行时更适合 CUDA 12，需要进一步确认是否存在对应可用 tag 或是否需额外对齐。
- 之前在本地 conda 环境里，DeepSeek-V4-Pro 在权重创建阶段已有明显显存压力；即使换 Docker，若官方 Hopper 路线本身也非常贴边，仍可能需要按照官方 recipe 进一步调整。

## 实施计划

### 阶段 1：确认镜像与运行方式

目标：确认两台 H800 都具备 Docker + NVIDIA Container Toolkit 运行条件，并准备 Hopper 镜像。

步骤：

1. 确认两台机器都可执行 `docker run --gpus all ... nvidia-smi`
2. 确认容器内可见 8 张 H800
3. 确认容器内可见 `/dev/infiniband`
4. 拉取 Hopper 镜像：
   - `lmsysorg/sglang:deepseek-v4-hopper`
5. 如默认 tag 与当前驱动/CUDA 不匹配，再评估 CUDA 12 变体或镜像替代方案

### 阶段 2：容器级联通验证

目标：先确认 Docker 不会破坏当前多机通信条件。

步骤：

1. 用 `--network host` 启动测试容器
2. 容器内检查：
   - `ip addr`
   - `env | grep -E 'NCCL|GLOO'`
   - `ls /dev/infiniband`
3. 确认容器内可绑定 `bond1`
4. 视需要补充：
   - `--device /dev/infiniband:/dev/infiniband`
   - `--cap-add IPC_LOCK`
   - `--ulimit memlock=-1`
   - `--privileged`

### 阶段 3：按 Hopper 路线做最小启动验证

目标：先验证是否进入正确 backend，而不是继续落到 Transformers fallback。

步骤：

1. 用容器内 `sglang serve` 替代旧的 `python -m sglang.launch_server`
2. 首先验证日志中不再出现：
   - `Using Transformers backend.`
3. 如果仍 fallback，则优先检查：
   - 镜像是否正确
   - checkpoint 类型是否匹配 Hopper recipe
   - 启动参数是否与官方生成器一致

### 阶段 4：双机 16 卡正式拉起

目标：在正确镜像和正确 backend 下，验证 H800_1 + H800_2 的完整服务启动。

步骤：

1. 先启动 H800_1（rank 0）
2. 出现 distributed init 相关日志后启动 H800_2（rank 1）
3. 观察是否完成：
   - distributed init
   - load weight
   - server ready
4. 测试：
   - `GET /v1/models`
   - `POST /generate`

### 阶段 5：若仍失败的分叉策略

如果仍失败，按以下顺序判断：

1. **仍然 Transformers fallback**
   - 说明镜像/recipe/checkpoint 仍未对齐
2. **native backend 但 OOM**
   - 说明方向对了，但 H800 80GB 余量仍紧，需要按 Hopper recipe 继续收紧或改 recipe
3. **IB / NCCL / DeepEP 问题**
   - 优先检查 Docker 权限、`/dev/infiniband`、`bond1`、`memlock`
4. **官方 Hopper 路线仍无法在 H800 成功**
   - 结论改为：H200 官方支持不能直接等价推出 H800 一定可用

## 本次实现范围

本轮先实现以下内容：

1. 新建本文档
2. 明确可行性结论与风险边界
3. 给出 Docker 部署验证计划
4. 补一版双机 Docker 启动命令模板，作为下一步落地基线

## 当前已验证进展

截至目前，以下前置条件已经验证通过：

- 两台 H800 已安装：
  - Docker
  - NVIDIA Container Toolkit
- 两台机器都已成功拉取 Hopper 镜像，并可使用本地 tag：
  - `lmsysorg/sglang:deepseek-v4-hopper`
- 容器内已验证：
  - 8 张 H800 GPU 可见
  - `bond1` 可见且状态为 `up`
  - `/dev/infiniband` 已成功透传

这说明当前 Docker / GPU / IB / bond1 这条基础链路已经打通。

## 实际执行方式

推荐优先直接执行服务器上的脚本，而不是手工敲长命令。

### H800_1 先跑 rank 0

```bash
cd /data/lin/modelserver
sudo ./deepseekv4-docker-rank0.sh
```

### H800_2 再跑 rank 1

等 H800_1 日志进入 distributed 初始化后，再在 H800_2 执行：

```bash
cd /data/lin/modelserver
sudo ./deepseekv4-docker-rank1.sh
```

说明：

- 当前脚本默认已加入：`--mem-fraction-static 0.86`
- 这是根据当前实测错误从默认 `0.805` 上调后的下一步尝试值
- 如果后续还需要微调，优先继续围绕 `mem-fraction-static` 调整，而不是先改 TP/EP

## 双机 Docker 启动命令模板

以下命令先作为 **H800 双机 Hopper 镜像验证模板**，后续再根据实际容器测试结果微调。

### 共同约定

- 模型目录挂载：`/data/models:/data/models`
- 使用宿主机网络：`--network host`
- 使用宿主机 IPC：`--ipc=host`
- 共享内存：`--shm-size 32g`
- 暴露全部 GPU：`--gpus all`
- 暴露 IB 设备：`--device /dev/infiniband:/dev/infiniband`
- 锁页内存权限：`--cap-add IPC_LOCK --ulimit memlock=-1`
- 镜像：`lmsysorg/sglang:deepseek-v4-hopper`

### H800_1（rank 0）

```bash
docker run --rm -it \
  --gpus all \
  --network host \
  --ipc=host \
  --shm-size 32g \
  --device /dev/infiniband:/dev/infiniband \
  --cap-add IPC_LOCK \
  --ulimit memlock=-1 \
  -v /data/models:/data/models \
  -e NCCL_SOCKET_IFNAME=bond1 \
  -e GLOO_SOCKET_IFNAME=bond1 \
  -e NCCL_DEBUG=INFO \
  lmsysorg/sglang:deepseek-v4-hopper \
  bash -lc '
    sglang serve \
      --model-path /data/models/DeepSeek-V4-Pro \
      --tp-size 16 \
      --ep-size 16 \
      --moe-a2a-backend deepep \
      --mem-fraction-static 0.86 \
      --nnodes 2 \
      --node-rank 0 \
      --dist-init-addr 192.168.100.48:20000 \
      --trust-remote-code \
      --api-key shuzuan2025-minimax \
      --host 0.0.0.0 \
      --port 8000
  '
```

### H800_2（rank 1）

```bash
docker run --rm -it \
  --gpus all \
  --network host \
  --ipc=host \
  --shm-size 32g \
  --device /dev/infiniband:/dev/infiniband \
  --cap-add IPC_LOCK \
  --ulimit memlock=-1 \
  -v /data/models:/data/models \
  -e NCCL_SOCKET_IFNAME=bond1 \
  -e GLOO_SOCKET_IFNAME=bond1 \
  -e NCCL_DEBUG=INFO \
  lmsysorg/sglang:deepseek-v4-hopper \
  bash -lc '
    sglang serve \
      --model-path /data/models/DeepSeek-V4-Pro \
      --tp-size 16 \
      --ep-size 16 \
      --moe-a2a-backend deepep \
      --mem-fraction-static 0.86 \
      --nnodes 2 \
      --node-rank 1 \
      --dist-init-addr 192.168.100.48:20000 \
      --trust-remote-code \
      --api-key shuzuan2025-minimax \
      --host 0.0.0.0 \
      --port 8000
  '
```

## 首轮验证目标

这两条命令的第一目标不是直接追求一次成功，而是优先验证：

1. 容器内是否能正常看到 8 卡
2. 容器内是否能使用 `bond1`
3. 容器内是否能访问 `/dev/infiniband`
4. DeepSeek-V4-Pro 是否仍然落到 `Using Transformers backend.`
5. 如果不再 fallback，再继续观察显存与启动稳定性

## 当前已知报错与判断

当前 Docker Hopper 路线下，模型已经明显比之前走得更远：

- 没有再先撞到之前那类 early-stage 权重初始化问题
- 现在报错点已经推进到 scheduler / memory profiling 阶段

当前实测关键日志为：

```text
RuntimeError: Not enough memory. Please try to increase --mem-fraction-static.
Current value: self.server_args.mem_fraction_static=0.805
```

同时伴随：

```text
Memory profiling: available_gpu_memory=11.66 GB, total_gpu_memory=77.80 GB,
mem_fraction_static=0.81, rest_memory=-3.51 GB
```

这说明：

- 当前问题已不是 Docker / IB / GPU 可见性问题
- 也不是之前那种更早阶段的 fallback/启动链路问题
- 而是进入 DeepSeek-V4 这条路径后，scheduler 计算出的静态显存比例仍然不够

基于这次日志，当前第一优先级应改为：

1. 先把 `--mem-fraction-static` 从默认 `0.805` 提到 `0.86`
2. 如果 `0.85` 仍不足，再继续尝试 `0.86` / `0.87`
3. 只有当提高 `mem-fraction-static` 后又重新撞回更早阶段 OOM，才重新评估别的参数

所以当前 Docker 脚本和模板命令都已同步更新为：

```bash
--mem-fraction-static 0.86
```

> 补充：`deepseekv4-docker-check.sh` 为了便于通过 ssh 非交互执行，实际脚本中使用的是非交互 `docker run --rm`；上面的 rank 启动模板仍保留 `-it` 便于手工前台观察日志。

## 已落地文件

- `deepseekv4-docker-rank0.sh`
  - H800_1 上的 rank 0 Docker 启动脚本模板
- `deepseekv4-docker-rank1.sh`
  - H800_2 上的 rank 1 Docker 启动脚本模板
- `deepseekv4-docker-check.sh`
  - 容器内 GPU / `bond1` / `/dev/infiniband` 可见性检查脚本

## 推荐执行顺序

1. 先在两台机器上执行 `deepseekv4-docker-check.sh`
2. 确认容器内能看到：
   - 8 张 H800
   - `bond1`
   - `/dev/infiniband`
3. 在 H800_1 上执行 `deepseekv4-docker-rank0.sh`
4. 待 rank 0 进入 distributed init 后，在 H800_2 上执行 `deepseekv4-docker-rank1.sh`
5. 重点看日志是否仍出现：
   - `Using Transformers backend.`
6. 如果 backend 正确，再看是否有新的 OOM / NCCL / IB 问题

