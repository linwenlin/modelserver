# Qwen3.6-35B-A3B-FP8 六卡4090启动命令

> 文件名按任务保留为 `qwen3.5-a3b-f8-start.md`，实际模型为 **Qwen3.6-35B-A3B-FP8**。

## 服务信息

- 主机：**六卡4090** `shuzuan@58.211.6.130 -p 102`
- conda 环境：`sglang_env`
- 模型路径：`~/Project/lin/model/Qwen3.6-35B-A3B-FP8`
- 使用 GPU：`2,3,4,5`
- TP：`4`
- 服务端口：`11450`
- 对外 OpenAI base URL：`http://58.211.6.130:11450/v1`

## 启动前检查

```bash
ssh -p 102 shuzuan@58.211.6.130

source ~/miniconda3/etc/profile.d/conda.sh
conda activate sglang_env

nvidia-smi
du -sh ~/Project/lin/model/Qwen3.6-35B-A3B-FP8
ls ~/Project/lin/model/Qwen3.6-35B-A3B-FP8/*.safetensors | wc -l
```

## 启动命令

SGLang 当前环境帮助中确认的 tensor parallel 参数名是 `--tp-size`。

```bash
ssh -p 102 shuzuan@58.211.6.130

source ~/miniconda3/etc/profile.d/conda.sh
conda activate sglang_env

mkdir -p ~/Project/lin/modelserver/logs

export CUDA_VISIBLE_DEVICES=2,3,4,5

nohup python3 -m sglang.launch_server \
  --model-path ~/Project/lin/model/Qwen3.6-35B-A3B-FP8 \
  --served-model-name qwen3.6-35b-a3b-fp8 \
  --tp-size 4 \
  --host 0.0.0.0 \
  --port 11450 \
  --mem-fraction-static 0.85 \
  > ~/Project/lin/modelserver/logs/qwen3.6-35b-a3b-fp8.log 2>&1 &

echo $! > ~/Project/lin/modelserver/logs/qwen3.6-35b-a3b-fp8.pid
tail -f ~/Project/lin/modelserver/logs/qwen3.6-35b-a3b-fp8.log
```

等价的前台启动命令：

```bash
CUDA_VISIBLE_DEVICES=2,3,4,5 python3 -m sglang.launch_server \
  --model-path ~/Project/lin/model/Qwen3.6-35B-A3B-FP8 \
  --served-model-name qwen3.6-35b-a3b-fp8 \
  --tp-size 4 \
  --host 0.0.0.0 \
  --port 11450 \
  --mem-fraction-static 0.85
```

## 关于 radix cache（前缀缓存）与高并发稳定性

> 2026-07-06 压测实测记录，务必了解。

**radix cache 是什么**：sglang 默认开启的前缀缓存（RadixAttention）。它把已算过的 prompt 的 KV cache 按 token 序列存进基数树，新请求若与已缓存请求**共享开头前缀**，这段 KV 直接复用、跳过重复 prefill。对**相同 system prompt / 多轮对话 / RAG 重复上下文**是巨大加速，**生产环境应保持开启（默认）**。

**实测发现的崩溃 bug**（sglang 0.5.9）：用合成随机 prompt 高并发压测、KV 池（`max_total_num_tokens=602416`）被打满时（本机约在 `8K/1K` 并发 `c60~c67`），会触发假阳性断言并把全部 TP worker `SIGQUIT` 杀掉：

```
ValueError: token_to_kv_pool_allocator memory leak detected!
full_available_size=1393, full_evictable_size=601023, size=602416
```

其实没真泄漏——`1393 + 601023 = 602416`，601023 都在 radix 树里可回收，只是分配器只看即时空闲量而误判。

**两种用法**：

- **生产/日常**：保持默认（不加 `--disable-radix-cache`），享受前缀复用收益。已知代价是极端满载时可能撞上述 bug；如遇到可考虑升级 sglang 或降低并发/`--mem-fraction-static`。
- **纯合成压测**（随机 prompt、零前缀命中）：加 `--disable-radix-cache`。此时 radix 一次都不命中却仍付出维护树 + 惰性淘汰的记账开销，禁用后 KV 用完即刻归还，**略快且不会撞上面的崩溃**。

```bash
# 压测专用启动（在上面启动命令基础上加一行）
  --mem-fraction-static 0.85 \
  --disable-radix-cache
```

## 查看日志

```bash
tail -f ~/Project/lin/modelserver/logs/qwen3.6-35b-a3b-fp8.log
```

## 停止服务

```bash
PID=$(cat ~/Project/lin/modelserver/logs/qwen3.6-35b-a3b-fp8.pid 2>/dev/null || true)
if [ -n "$PID" ] && ps -p "$PID" >/dev/null 2>&1; then
  PGID=$(ps -o pgid= -p "$PID" | tr -d ' ')
  kill -TERM -- -"$PGID"
  sleep 10
  ps -p "$PID" >/dev/null 2>&1 && kill -KILL -- -"$PGID"
fi

nvidia-smi
```

如果 pid 文件不存在：

```bash
ps -eo pid,ppid,pgid,etime,cmd | grep -E 'sglang.launch_server|Qwen3.6-35B-A3B-FP8' | grep -v grep
```

## 接口测试

```bash
curl http://127.0.0.1:11450/v1/models
```

```bash
curl http://127.0.0.1:11450/v1/chat/completions \
  -H "Content-Type: application/json" \
  -d '{
    "model": "qwen3.6-35b-a3b-fp8",
    "messages": [
      {"role": "user", "content": "你好，回复 ok，不要解释。"}
    ],
    "temperature": 0,
    "max_tokens": 16
  }'
```
