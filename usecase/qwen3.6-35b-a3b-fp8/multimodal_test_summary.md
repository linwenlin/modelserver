# Qwen3.6-35B-A3B-FP8 多模态测试结果

## 服务

- 服务器：六卡4090
- 外部接口：`http://58.211.6.130:10283/v1`
- 内部接口：`http://127.0.0.1:11450/v1`
- 模型名：`qwen3.6-35b-a3b-fp8`
- 启动 GPU：`CUDA_VISIBLE_DEVICES=2,3,4,5`

## 测试命令

```bash
python3 usecase/qwen3.6-35b-a3b-fp8/call_qwen3_6_fp8_multimodal.py \
  --results-dir usecase/qwen3.6-35b-a3b-fp8/results
```

视频默认使用抽帧模式：`--video-mode frames --video-frames 8`。本次视频实测使用 6 帧、最大边长 768：

```bash
python3 usecase/qwen3.6-35b-a3b-fp8/call_qwen3_6_fp8_multimodal.py \
  usecase/qwen3.6/data/20250624_172606.mp4 \
  --video-mode frames \
  --video-frames 6 \
  --max-image-size 768 \
  --results-dir usecase/qwen3.6-35b-a3b-fp8/results
```

## 结果

| 文件 | 类型 | 耗时 | prompt tokens | completion tokens | total tokens | 结果文件 |
|---|---:|---:|---:|---:|---:|---|
| `wiki只是图谱接口说明.png` | 图片/OCR | 2.58s | 842 | 245 | 1087 | `results/wiki只是图谱接口说明.png.txt` |
| `image_首饰佩戴1.png` | 图片/实拍 | 9.59s | 3248 | 714 | 3962 | `results/image_首饰佩戴1.png.txt` |
| `image_首饰佩戴2.png` | 图片/实拍 | 7.00s | 3248 | 420 | 3668 | `results/image_首饰佩戴2.png.txt` |
| `20250624_172606.mp4` | 视频/抽帧 | 2.98s | 2132 | 138 | 2270 | `results/20250624_172606.mp4.txt` |

## 结论

- 图片多模态可用：OCR 图已按原文直接还原，两张实拍图均正常返回解析结果。
- 视频理解可用，但建议当前 4090 FP8 服务先使用抽帧方式调用。
- 原生 `video_url` 调用触发过 CUDA OOM：`GPU 0 ... 214.56 MiB is free`。这是显存余量问题，不是图片能力不可用。
- 远端 `sglang_env` 中 `PIL`、`torchvision`、`decord`、`cv2`、`imageio` 可用；`torchcodec` 当前不可用，且环境里没有 `ffmpeg` 命令。

## 原生视频依赖检查

远端检查结果：

```text
PIL: OK 11.3.0
torchvision: OK 0.24.1+cu128
torchcodec: FAIL RuntimeError: Could not load libtorchcodec
decord: OK 3.0.0
cv2: OK 4.10.0
imageio: OK 2.36.0
sglang.srt.multimodal.processors.mimo_v2: FAIL ModuleNotFoundError
ffmpeg: not found
```

如果后续需要直接传 `video_url`，建议先补齐 `ffmpeg`/`torchcodec` 兼容依赖，并考虑降低 `--mem-fraction-static` 或减少上下文/KV cache 占用，给视频解码和视觉编码留更多显存。

