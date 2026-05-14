# DeepSeek-V4 / Docker / Conda 现状排查记录（summary1）

## 目的

保存当前阶段的关键检查结果，便于后续继续定位：

- 为什么 DeepSeek-V4-Pro 在当前 H800 双机环境下跑不起来
- 为什么 Docker 路线和原始 conda 路线都在相近阶段卡住
- 最近对 NVIDIA / CUDA / NCCL / OFED / conda 环境的修改是否引入回归

## 当前结论（截至本文件更新时）

当前最可疑的问题已经不只是 DeepSeek-V4 的启动参数，而是：

- **宿主机 GPU / 通信运行时栈发生过修改**
- 并且 **conda `sglang` 环境出现了明显的 CUDA 12 / CUDA 13 运行时混装**
- 这类混装非常可能导致：
  - NCCL 初始化可以完成
  - 但 scheduler / worker ready 阶段挂起
  - Docker 路线和 conda 路线都在相近位置卡住

也就是说，当前更像是：

- **系统与 Python 运行时栈回归**
- 而不只是 DeepSeek-V4 某一组 `tp/ep/dp` 参数写错

## 最早问题背景

在非 Docker / conda + SGLang 路线下：

- Kimi-K2.6 之前是可以跑起来的
- DeepSeek-V4-Pro 出现的关键日志包括：
  - `Load weight begin. avail mem=77.75 GB`
  - `Using Transformers backend.`

这说明当时 DeepSeek-V4-Pro 没有稳定命中预期的原生 Hopper 路线，而是 fallback 到 Transformers backend。

## Docker 路线的主要发现

### 已确认打通的部分

Docker Hopper 镜像路线下，以下内容已经验证过：

- 镜像：`lmsysorg/sglang:deepseek-v4-hopper`
- 两台机器容器内能看到：
  - 8 张 H800 GPU
  - `bond1`
  - `/dev/infiniband`
- 宿主机 `nvidia_peermem` 已存在并已加载
- 分布式初始化时，16 个 rank 的 NCCL `Init COMPLETE` 能正常出现

### 但当前仍然失败的部分

尽管 NCCL 初始化完成，服务仍然会卡住：

- 卡在 `ncclCommInitRank ... Init COMPLETE` 之后
- 主进程 `Ctrl+C` 栈显示卡在：
  - `sglang/srt/entrypoints/engine.py:_wait_for_scheduler_ready`
- 从机在主机退出后会打印：
  - `TCPStore.cpp ... Connection reset by peer`
  - `ProcessGroupNCCL::HeartbeatMonitor::runLoop()`

这说明：

- **不是 NCCL 初始化本身卡住**
- 而是 NCCL 初始化完成后，某个 scheduler / worker 没有 ready
- 于是 rank0 / master 一直在等子进程就绪

## 试过的 DeepSeek 参数方向

### 1. 旧尝试：`tp-size 16 + ep-size 16 + deepep`

曾经多次使用：

```bash
--tp-size 16
--ep-size 16
--moe-a2a-backend deepep
```

现象：

- 能跑到比最早阶段更后面
- 也曾遇到：
  - `Using Transformers backend.`
  - `Not enough memory. Please try to increase --mem-fraction-static.`
  - `nvidia_peermem` / NVSHMEM 相关错误
- 在修复/加载 `nvidia_peermem` 后，NCCL 初始化可以完成
- 但后续卡在 scheduler ready 之前/附近

### 2. 对照尝试：去掉 `deepep`

结论：

- 去掉 `--moe-a2a-backend deepep` 后，仍然会卡住
- 说明挂起**不只是** `deepep` 一项导致

### 3. 对照尝试：`tp-size 8 + ep-size 2`

结论：

- 这组参数会让 H800_1 只实际启用 4 张卡
- `docker top` 可见仅有：
  - `sglang::scheduler_TP0_EP0`
  - `sglang::scheduler_TP1_EP0`
  - `sglang::scheduler_TP2_EP0`
  - `sglang::scheduler_TP3_EP0`
- GPU 0-3 有使用，4-7 基本空闲

因此：

- 这组参数不满足当前“双机 16 卡完整验证”的目标
- 且它会引入新的变量，不能作为最终方向

### 4. 按官方文档改成 `tp + dp + enable-dp-attention + deepep`

根据官方 DeepSeek-V4 文档，Hopper / H200 / Pro / FP8 / 2 节点 16 GPU 路线更接近：

```bash
--tp-size 16
--dp-size 16
--enable-dp-attention
--moe-a2a-backend deepep
--mem-fraction-static 0.88
--cuda-graph-max-bs 8
--max-running-requests 32
```

并加环境变量：

```bash
SGLANG_DSV4_FP4_EXPERTS=0
```

当前脚本已经按这条方向同步过，但实际运行后依然会在 NCCL 初始化完成后挂住。

这进一步说明：

- 启动参数可能还有优化空间
- 但**仅靠调参已经无法解释 conda / Docker 都卡在相似位置**

## Kimi Docker 对照实验结果

尝试用 DeepSeek Docker 镜像直接跑 Kimi2.6，结果失败在更早阶段：

```text
RuntimeError: SGLANG_APPLY_CONFIG_BACKUP=auto could not read a numeric num_hidden_layers from /data/models/Kimi-K2.6/config.json (got None).
```

这说明：

- `lmsysorg/sglang:deepseek-v4-hopper` 是偏 DeepSeek-V4 定制镜像
- 它不适合作为“通用 SGLang Docker 基线”直接验证 Kimi
- 因此 Kimi Docker 失败**不能**用来证明 Docker 通信本身有问题

## 当前宿主机系统栈检查结果

两台机器（H800_1 / H800_2）当前检查结果一致。

### 宿主机 GPU / 驱动 / OFED

- Kernel: `5.15.0-94-generic`
- NVIDIA Driver: `570.133.20`
- `nvidia-smi` CUDA Version: `12.8`
- `nvidia` 内核模块版本：`570.133.20`
- `nvidia_peermem` 模块版本：`570.133.20`
- `nvidia_peermem` 已加载
- OFED: `MLNX_OFED_LINUX-5.8-4.1.5.0`
- IB 设备可见，包含：
  - `mlx5_0`
  - `mlx5_bond_0`
  - 以及多个 `mlx5_*`

### DKMS 状态

两台机器均显示已安装：

- `mlnx-ofed-kernel/5.8`
- `nvidia/570.133.20`
- `iser/5.8`
- `isert/5.8`
- `srp/5.8`
- `knem/...`

### 最近模块文件时间戳

`/lib/modules/5.15.0-94-generic/updates/dkms` 下：

- `nvidia.ko`
- `nvidia-peermem.ko`
- `nvidia-uvm.ko`
- `nvidia-modeset.ko`

均显示时间为：

- `May 30 2025`

## 当前 conda `sglang` 环境检查结果

两台机器一致：

- `torch 2.11.0+cu128`
- `cuda 12.8`
- `nccl (2, 28, 9)`（Python 侧读取）
- `sglang 0.5.11`
- `transformers 5.8.0.dev0`

### 关键异常：Python 依赖出现明显混装

在 `conda list` 中同时出现了：

#### CUDA 12 相关包

- `nvidia-cuda-cupti-cu12 12.8.90`
- `nvidia-cuda-nvrtc-cu12 12.8.93`
- `nvidia-cuda-runtime-cu12 12.8.90`
- `nvidia-cudnn-cu12 9.19.0.56`
- `nvidia-nccl-cu12 2.30.4`

#### CUDA 13 相关包

- `nvidia-cuda-cupti 13.0.85`
- `nvidia-cuda-nvrtc 13.0.88`
- `nvidia-cuda-runtime 13.0.96`
- `nvidia-cudnn-cu13 9.19.0.56`
- `nvidia-nccl-cu13 2.30.4`
- `cuda-python 13.2.0`

并且同时还有：

- `cuda-toolkit 12.8.1`
- `flashinfer-python 0.6.8.post1`
- `triton 3.6.0`

### 对该环境的判断

这是当前最可疑的问题之一。

原因：

- `torch` 本身是 **cu128** 版本
- 但环境里同时混入了 **CUDA 13** 的 runtime / nvrtc / cuDNN / NCCL 包
- 还保留了 **CUDA 12.8** 的一套包

这种混装非常容易造成：

- 低层初始化还能部分通过
- 但高层 distributed / scheduler / worker 初始化出现静默挂起

## Shell 历史中的高价值信息

两台机器 shell 历史都能看到最近做过：

```bash
sudo apt-get -y install cuda-toolkit-12-8
```

并且能看到大量 DeepSeek-V4 的启动尝试，包括：

- `--tp-size 16`
- `--ep-size 16`
- `--moe-a2a-backend deepep`
- `--mem-fraction-static` 多种值
- `--cuda-graph-max-bs`
- `--max-running-requests`
- `--disable-cuda-graph`
- `--model-impl sglang`

以及：

```bash
sudo modprobe nvidia_peermem
```

这和当前判断一致：

- 最近确实发生过 CUDA / NVIDIA / peermem 相关改动

## 当前最强结论

当前最值得优先怀疑的是：

1. **最近对宿主机 CUDA / NVIDIA 栈的修改引入了运行时回归**
2. **`sglang` conda 环境出现了明显的 cu12 / cu13 混装**
3. 这类问题会同时影响：
   - 原始 conda 路线
   - Docker 路线（尤其当容器与宿主机驱动/IB链路耦合时）

因此，现在不应继续只围绕 DeepSeek 参数打转，而应该优先：

- 导出当前环境快照
- 新建一套**干净且版本一致**的 `sglang` 环境
- 再用新环境做最小复现

## 建议的下一步

### 1. 保存当前环境快照

建议保存以下信息到文件：

- `conda list`
- `pip freeze`
- `python -c 'import torch; ...'`
- `nvidia-smi`
- `dkms status`
- `dpkg -l | grep -E 'nvidia|cuda|nccl|mlnx|ofed'`

### 2. 新建干净环境

目标：

- 不再混装 CUDA 12 / CUDA 13 Python runtime 包
- 保持一套一致的：
  - `torch 2.11.0+cu128`
  - 与之兼容的 `sglang`
  - `transformers`（保留 `deepseek_v4` 支持）

### 3. 新环境先做最小复现

优先顺序建议：

1. 用新环境先验证 Kimi / 或最小 NCCL 冒烟
2. 再用新环境跑 DeepSeek-V4-Pro
3. 若新环境恢复原先更靠后的行为，则基本坐实旧环境已被污染/回归

## 新建隔离环境进展（`sglang_clean_20260512`）

### 已完成

两台机器都已新建独立 conda 环境：

- `sglang_clean_20260512`

并且已确认：

- Python: `3.12.13`
- 当前 `torch` 已被强制钉回：
  - `2.11.0+cu128`
- `torch.version.cuda` 当前为：
  - `12.8`
- `torch.cuda.nccl.version()` 当前为：
  - `(2, 28, 9)`

### 重要新发现

虽然这是一个新建的干净环境，但安装 `sglang==0.5.11` 后，`sglang` 自己的依赖声明会继续拉入一批 CUDA 13 相关 Python 包。

从 `sglang` 的 `Requires-Dist` 可直接看到：

- `cuda-python>=13.0`
- `flashinfer_python==0.6.8.post1`
- `flashinfer_cubin==0.6.8.post1`
- `transformers==5.6.0`

在实际安装结果中，两台机器的新环境里都再次出现：

- `cuda-python==13.2.0`
- `nvidia-cuda-runtime==13.0.96`
- `nvidia-cuda-nvrtc==13.0.88`
- `nvidia-cudnn-cu13==9.19.0.56`
- `nvidia-nccl-cu13==2.28.9`

同时仍然保留了 `torch cu128` 对应的 CUDA 12.8 运行时包。

这意味着：

- **当前看到的 cu12 / cu13 混合，并不完全是旧环境脏了**
- **至少有一部分是 `sglang==0.5.11` 当前打包依赖本身就会引入的**

这是一条非常关键的新证据。

### 当前阻塞点

为让新环境支持 `deepseek_v4`，原计划继续把 `transformers` 升到 GitHub 最新版。

但两台机器都在拉取 GitHub 时失败：

- `curl 16 Error in the HTTP2 framing layer`
- `GnuTLS recv error (-110)`

因此目前新环境的 `transformers` 仍停留在：

- `transformers==5.6.0`

还没有完成 DeepSeek-V4 所需的升级。

### 当前状态总结

新环境隔离目标已经部分达成：

- 没有修改旧 `sglang` 环境
- 新环境已独立创建
- `torch` 已恢复为 `2.11.0+cu128`

但同时也暴露出更深层的问题：

- `sglang 0.5.11` 本身会带来一批 CUDA 13 Python 依赖
- 所以“完全不混”的 Python 层环境，未必能仅靠 `pip install sglang==0.5.11` 实现

这说明后续判断要分两层：

1. 旧环境是否因长期叠加操作进一步恶化
2. `sglang 0.5.11` 当前官方依赖组合本身是否就包含 cu12/cu13 混合设计

