# Unlimited-OCR 六卡4090启动命令

## 服务信息

- 主机：**六卡4090** `shuzuan@58.211.6.130 -p 102`
- conda 环境：`sglang_ocr`
- 模型来源：ModelScope `PaddlePaddle/Unlimited-OCR`
- 模型路径：`~/Project/lin/model/Unlimited-OCR`
- 使用 GPU：`1`
- 服务端口：`11451`
- 对外 OpenAI base URL：`http://58.211.6.130:11451/v1`
- 本机测试 base URL：`http://127.0.0.1:11451/v1`

> 说明：原计划端口 `8001` 在六卡4090上已被 SurrealDB 占用，因此改用 `11451`。该端口当前不对本机开放，测试在服务器本机执行。

## 已验证的环境状态

本次没有修改原 `sglang_env`，而是克隆出独立环境 `sglang_ocr` 运行 Unlimited-OCR：

- `sglang`: `0.0.0.dev11416+g92e8bb79e`，来自模型目录自带 wheel：`~/Project/lin/model/Unlimited-OCR/wheel/sglang-0.0.0.dev11416+g92e8bb79e-py3-none-any.whl`
- `sgl-kernel`: `0.3.21`
- `matplotlib`: `3.10.8`
- `easydict`: `1.13`

模型自带 SGLang wheel 要求 `sglang-kernel>=0.4.1`，但 PyPI 版 `sglang-kernel 0.4.4` 在这台 4090/CUDA 12.8 环境中导入失败；当前使用原环境可用的 `sgl-kernel 0.3.21`，并通过 `SGLANG_SKIP_SGL_KERNEL_VERSION_CHECK=1` 跳过版本检查。

## 下载模型

```bash
ssh -p 102 shuzuan@58.211.6.130

source ~/miniconda3/etc/profile.d/conda.sh
conda activate sglang_env

mkdir -p ~/Project/lin/model
modelscope download \
  --model PaddlePaddle/Unlimited-OCR \
  --local_dir ~/Project/lin/model/Unlimited-OCR \
  --max-workers 8
```

当前已下载完成：

```bash
$ du -sh ~/Project/lin/model/Unlimited-OCR
6.4G    /home/shuzuan/Project/lin/model/Unlimited-OCR

$ ls -lh ~/Project/lin/model/Unlimited-OCR/model-00001-of-000001.safetensors
-rw-rw-r-- 1 shuzuan shuzuan 6.3G ... model-00001-of-000001.safetensors
```

## 初始化隔离环境

如环境已存在可跳过本节。

```bash
ssh -p 102 shuzuan@58.211.6.130

source ~/miniconda3/etc/profile.d/conda.sh
conda create -y --name sglang_ocr --clone sglang_env
conda activate sglang_ocr

python3 -m pip install --force-reinstall --no-deps \
  ~/Project/lin/model/Unlimited-OCR/wheel/sglang-0.0.0.dev11416+g92e8bb79e-py3-none-any.whl

python3 -m pip install --no-deps matplotlib==3.10.8 easydict==1.13
python3 -m pip install --no-deps contourpy cycler fonttools kiwisolver pyparsing python-dateutil
python3 -m pip install --force-reinstall --no-deps -i https://pypi.tuna.tsinghua.edu.cn/simple sgl-kernel==0.3.21
```

检查环境：

```bash
source ~/miniconda3/etc/profile.d/conda.sh
conda activate sglang_ocr

python3 - <<'PY'
import importlib.metadata as m
import sglang
print('sglang', getattr(sglang, '__version__', 'unknown'))
for name in ['sgl-kernel', 'matplotlib', 'easydict']:
    print(name, m.version(name))
PY
```

## 启动前检查

```bash
ssh -p 102 shuzuan@58.211.6.130

source ~/miniconda3/etc/profile.d/conda.sh
conda activate sglang_ocr

nvidia-smi
ss -ltnp | grep ':11451' || true
du -sh ~/Project/lin/model/Unlimited-OCR
ls -lh ~/Project/lin/model/Unlimited-OCR/model-00001-of-000001.safetensors
```

## 启动命令

```bash
ssh -p 102 shuzuan@58.211.6.130

source ~/miniconda3/etc/profile.d/conda.sh
conda activate sglang_ocr

mkdir -p ~/Project/lin/modelserver/logs

export CUDA_VISIBLE_DEVICES=1
export SGLANG_ALLOW_OVERWRITE_LONGER_CONTEXT_LEN=1
export SGLANG_SKIP_SGL_KERNEL_VERSION_CHECK=1

nohup python3 -m sglang.launch_server \
  --model-path ~/Project/lin/model/Unlimited-OCR \
  --served-model-name Unlimited-OCR \
  --trust-remote-code \
  --attention-backend fa3 \
  --page-size 1 \
  --mem-fraction-static 0.8 \
  --context-length 32768 \
  --enable-custom-logit-processor \
  --disable-overlap-schedule \
  --skip-server-warmup \
  --host 0.0.0.0 \
  --port 11451 \
  > ~/Project/lin/modelserver/logs/unlimited-ocr-sglang.log 2>&1 &

echo $! > ~/Project/lin/modelserver/logs/unlimited-ocr-sglang.pid
tail -f ~/Project/lin/modelserver/logs/unlimited-ocr-sglang.log
```

等价的前台启动命令：

```bash
CUDA_VISIBLE_DEVICES=1 \
SGLANG_ALLOW_OVERWRITE_LONGER_CONTEXT_LEN=1 \
SGLANG_SKIP_SGL_KERNEL_VERSION_CHECK=1 \
python3 -m sglang.launch_server \
  --model-path ~/Project/lin/model/Unlimited-OCR \
  --served-model-name Unlimited-OCR \
  --trust-remote-code \
  --attention-backend fa3 \
  --page-size 1 \
  --mem-fraction-static 0.8 \
  --context-length 32768 \
  --enable-custom-logit-processor \
  --disable-overlap-schedule \
  --skip-server-warmup \
  --host 0.0.0.0 \
  --port 11451
```

## 查看日志

```bash
tail -f ~/Project/lin/modelserver/logs/unlimited-ocr-sglang.log
```

## 停止服务

```bash
PID=$(cat ~/Project/lin/modelserver/logs/unlimited-ocr-sglang.pid 2>/dev/null || true)
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
ps -eo pid,ppid,pgid,etime,cmd | grep -E 'sglang.launch_server|Unlimited-OCR|11451' | grep -v grep
```

## 接口检查

```bash
curl -i http://127.0.0.1:11451/health
curl http://127.0.0.1:11451/v1/models
```

当前服务检查结果：

```json
{"object":"list","data":[{"id":"Unlimited-OCR","object":"model","created":1782267439,"owned_by":"sglang","root":"Unlimited-OCR","parent":null,"max_model_len":32768}]}
```

GPU 1 当前显存占用约 `21537 MiB / 24564 MiB`。

## 服务器本机测试

测试文件已放到：

```bash
~/Project/lin/modelserver/Unlimited-OCR/test/
```

官方 README 明确支持单张文档图片、多图/多页文档、PDF 转图片后的多页解析。视频未在官方说明中明确支持，本次不做视频测试。

### 图片接口连通性测试

图片测试脚本已放到：

```bash
~/Project/lin/modelserver/test_unlimited_ocr_image.py
```

如需重建脚本：

```bash
cat > ~/Project/lin/modelserver/test_unlimited_ocr_image.py <<'PY'
import base64
import json
import os
import sys
import time

import requests
from sglang.srt.sampling.custom_logit_processor import DeepseekOCRNoRepeatNGramLogitProcessor

server_url = "http://127.0.0.1:11451"
image_path = sys.argv[1]
output_path = sys.argv[2] if len(sys.argv) > 2 else None

ext = os.path.splitext(image_path)[1].lower()
mime = "image/jpeg" if ext in (".jpg", ".jpeg") else "image/" + ext.lstrip(".")
with open(image_path, "rb") as f:
    data = base64.b64encode(f.read()).decode("utf-8")

payload = {
    "model": "Unlimited-OCR",
    "messages": [
        {
            "role": "user",
            "content": [
                {"type": "text", "text": "document parsing."},
                {"type": "image_url", "image_url": {"url": f"data:{mime};base64,{data}"}},
            ],
        }
    ],
    "temperature": 0,
    "skip_special_tokens": False,
    "images_config": {"image_mode": "gundam"},
    "custom_logit_processor": DeepseekOCRNoRepeatNGramLogitProcessor.to_str(),
    "custom_params": {"ngram_size": 35, "window_size": 128},
    "stream": True,
}

start = time.time()
resp = requests.post(
    f"{server_url}/v1/chat/completions",
    headers={"Content-Type": "application/json"},
    data=json.dumps(payload),
    timeout=1200,
    stream=True,
)
resp.raise_for_status()

chunks = []
for line in resp.iter_lines(decode_unicode=True):
    if not line or not line.startswith("data:"):
        continue
    item = line[len("data:"):].strip()
    if item == "[DONE]":
        break
    event = json.loads(item)
    delta = event["choices"][0].get("delta", {}).get("content", "")
    if delta:
        print(delta, end="", flush=True)
        chunks.append(delta)
print()
text = "".join(chunks)
print(f"\n---RESULT_STATS--- chars={len(text)} seconds={time.time() - start:.2f}")
if output_path:
    with open(output_path, "w", encoding="utf-8") as f:
        f.write(text)
PY
```

执行图片测试：

```bash
source ~/miniconda3/etc/profile.d/conda.sh
conda activate sglang_ocr

mkdir -p ~/Project/lin/modelserver/outputs/unlimited-ocr

python3 ~/Project/lin/modelserver/test_unlimited_ocr_image.py \
  ~/Project/lin/modelserver/Unlimited-OCR/test/image_首饰佩戴1.png \
  ~/Project/lin/modelserver/outputs/unlimited-ocr/image_首饰佩戴1.md \
  > ~/Project/lin/modelserver/outputs/unlimited-ocr/image_首饰佩戴1.run.log 2>&1

python3 ~/Project/lin/modelserver/test_unlimited_ocr_image.py \
  ~/Project/lin/modelserver/Unlimited-OCR/test/image_首饰佩戴2.png \
  ~/Project/lin/modelserver/outputs/unlimited-ocr/image_首饰佩戴2.md \
  > ~/Project/lin/modelserver/outputs/unlimited-ocr/image_首饰佩戴2.run.log 2>&1
```

查看测试输出：

```bash
ls -lh ~/Project/lin/modelserver/outputs/unlimited-ocr/
grep -- '---RESULT_STATS---' ~/Project/lin/modelserver/outputs/unlimited-ocr/*.run.log
sed -n '1,80p' ~/Project/lin/modelserver/outputs/unlimited-ocr/image_首饰佩戴1.md
```

### PDF 解析测试

PDF 测试脚本已放到：

```bash
~/Project/lin/modelserver/test_unlimited_ocr_pdf.py
```

PDF 测试依赖 PyMuPDF，已安装在隔离环境 `sglang_ocr` 中：

```bash
source ~/miniconda3/etc/profile.d/conda.sh
conda activate sglang_ocr
python3 -m pip install -i https://pypi.tuna.tsinghua.edu.cn/simple pymupdf
```

执行 PDF 前 3 页解析：

```bash
source ~/miniconda3/etc/profile.d/conda.sh
conda activate sglang_ocr

mkdir -p ~/Project/lin/modelserver/outputs/unlimited-ocr/pdf

python3 ~/Project/lin/modelserver/test_unlimited_ocr_pdf.py \
  ~/Project/lin/modelserver/Unlimited-OCR/test/商业模式解构与设计.pdf \
  ~/Project/lin/modelserver/outputs/unlimited-ocr/pdf/商业模式解构与设计_pages1-3.md \
  3 200 \
  > ~/Project/lin/modelserver/outputs/unlimited-ocr/pdf/商业模式解构与设计_pages1-3.run.log 2>&1

python3 ~/Project/lin/modelserver/test_unlimited_ocr_pdf.py \
  ~/Project/lin/modelserver/Unlimited-OCR/test/商业模式学原理（第二版）0728魏炜修订2.pdf \
  ~/Project/lin/modelserver/outputs/unlimited-ocr/pdf/商业模式学原理_pages1-3.md \
  3 200 \
  > ~/Project/lin/modelserver/outputs/unlimited-ocr/pdf/商业模式学原理_pages1-3.run.log 2>&1
```

查看 PDF 输出：

```bash
ls -lh ~/Project/lin/modelserver/outputs/unlimited-ocr/pdf/
grep -- '---RESULT_STATS---' ~/Project/lin/modelserver/outputs/unlimited-ocr/pdf/*.run.log
sed -n '1,120p' ~/Project/lin/modelserver/outputs/unlimited-ocr/pdf/商业模式解构与设计_pages1-3.md
```

图片不是本模型的主要验证场景；普通图片可能只返回检测区域而不是内容描述。当前两张普通图片输出均为 `<|det|>image [0, 0, 999, 999]<|/det|>`，只说明接口能处理图片请求，不代表 OCR 质量。

## 本次测试记录

- 服务已启动成功：`http://127.0.0.1:11451/v1/models` 返回 `Unlimited-OCR`，`max_model_len=32768`。
- 测试文件已复制到：`~/Project/lin/modelserver/Unlimited-OCR/test/`。
- PDF `商业模式解构与设计.pdf`：共 84 页，本次解析前 3 页，输出 2259 字符，用时 `4.06s`，结果保存到 `~/Project/lin/modelserver/outputs/unlimited-ocr/pdf/商业模式解构与设计_pages1-3.md`。
- PDF `商业模式学原理（第二版）0728魏炜修订2.pdf`：共 647 页，本次解析前 3 页，输出 1728 字符，用时 `3.67s`，结果保存到 `~/Project/lin/modelserver/outputs/unlimited-ocr/pdf/商业模式学原理_pages1-3.md`。
- 输出 md 中包含 `<|det|>title/text/...` 版面标签和识别出的中文内容，PDF 场景验证通过。
- 视频未测试：官方说明未明确支持原生视频输入。
