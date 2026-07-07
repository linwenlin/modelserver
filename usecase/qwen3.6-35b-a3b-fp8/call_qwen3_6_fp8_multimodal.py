#!/usr/bin/env python3
import argparse
import base64
import io
import json
import mimetypes
import os
import sys
import time
from pathlib import Path

import requests


DEFAULT_ENDPOINT = os.environ.get("QWEN3_6_FP8_ENDPOINT", "http://58.211.6.130:10283/v1/chat/completions")
DEFAULT_MODEL = os.environ.get("QWEN3_6_FP8_MODEL", "qwen3.6-35b-a3b-fp8")
DEFAULT_DATA_DIR = Path(__file__).resolve().parents[1] / "qwen3.6" / "data"

IMAGE_PROMPT = """请先判断图片类型：是文字文档/图表，还是风景/实拍。

如果是文字文档或图表：
- 只输出图片中可见的文字内容，不要分析、不要解释、不要总结
- 按 Markdown 尽量还原原始排版、层级、列表、代码块和表格
- 注意：图片第一行文字可能被截断显示不完整；如果确实截断，请从第一个完整的行开始识别
- 不要添加“识别结果”“分析”“总结”等图片中不存在的标题

如果是风景/实拍：
- 识别图中出现的关键物品及其空间位置
- 分析图片主体、细节、可见文字（如有）
- 分析图片的艺术风格（如光影处理、色彩基调、摄影流派）

请根据图片类型直接输出对应结果。"""

VIDEO_PROMPT = """请识别这个视频并生成一段中文视频描述。只输出一段话，不要标题、不要分点。必须小于500字，建议200字以内。概括场景、主要界面/物体、动作变化和可见文字线索即可；不要逐字抄写超长文件名，长文件名请概括为“长文件名”。"""

IMAGE_SUFFIXES = {".png", ".jpg", ".jpeg", ".webp", ".bmp", ".gif", ".tiff", ".tif"}
VIDEO_SUFFIXES = {".mp4", ".mov", ".mkv", ".webm", ".avi", ".m4v"}
PDF_SUFFIXES = {".pdf"}


def detect_kind(path: Path) -> str:
    suffix = path.suffix.lower()
    if suffix in IMAGE_SUFFIXES:
        return "image"
    if suffix in VIDEO_SUFFIXES:
        return "video"
    if suffix in PDF_SUFFIXES:
        return "pdf"
    mime_type, _ = mimetypes.guess_type(path.name)
    if mime_type and mime_type.startswith("image/"):
        return "image"
    if mime_type and mime_type.startswith("video/"):
        return "video"
    if mime_type == "application/pdf":
        return "pdf"
    raise ValueError(f"无法判断文件类型: {path}")


def image_to_data_uri(path: Path, max_image_size: int) -> str:
    try:
        from PIL import Image
    except ImportError:
        mime_type = mimetypes.guess_type(path.name)[0] or "image/png"
        return f"data:{mime_type};base64,{base64.b64encode(path.read_bytes()).decode('ascii')}"

    image = Image.open(path)
    if max_image_size > 0 and max(image.size) > max_image_size:
        ratio = max_image_size / max(image.size)
        image = image.resize((int(image.size[0] * ratio), int(image.size[1] * ratio)), Image.Resampling.LANCZOS)

    buffer = io.BytesIO()
    if image.mode in {"RGBA", "LA"} and path.suffix.lower() == ".png":
        image.save(buffer, format="PNG", optimize=True)
        mime_type = "image/png"
    else:
        if image.mode != "RGB":
            image = image.convert("RGB")
        image.save(buffer, format="JPEG", quality=90, optimize=True)
        mime_type = "image/jpeg"
    return f"data:{mime_type};base64,{base64.b64encode(buffer.getvalue()).decode('ascii')}"


def image_bytes_to_data_uri(image_bytes: bytes, max_image_size: int) -> str:
    try:
        from PIL import Image
    except ImportError:
        return f"data:image/jpeg;base64,{base64.b64encode(image_bytes).decode('ascii')}"

    image = Image.open(io.BytesIO(image_bytes))
    if max_image_size > 0 and max(image.size) > max_image_size:
        ratio = max_image_size / max(image.size)
        image = image.resize((int(image.size[0] * ratio), int(image.size[1] * ratio)), Image.Resampling.LANCZOS)
    if image.mode != "RGB":
        image = image.convert("RGB")
    buffer = io.BytesIO()
    image.save(buffer, format="JPEG", quality=85, optimize=True)
    return f"data:image/jpeg;base64,{base64.b64encode(buffer.getvalue()).decode('ascii')}"


def pdf_pages_to_data_uris(path: Path, max_image_size: int, pdf_pages: int, pdf_zoom: float) -> list[str]:
    try:
        import fitz
    except ImportError as exc:
        raise RuntimeError("PDF 输入需要安装 PyMuPDF: pip install pymupdf") from exc

    data_uris = []
    with fitz.open(path) as document:
        page_count = len(document) if pdf_pages <= 0 else min(pdf_pages, len(document))
        for index in range(page_count):
            page = document[index]
            pixmap = page.get_pixmap(matrix=fitz.Matrix(pdf_zoom, pdf_zoom), alpha=False)
            data_uris.append(image_bytes_to_data_uri(pixmap.tobytes("png"), max_image_size))
    if not data_uris:
        raise RuntimeError(f"PDF 没有可渲染页面: {path}")
    return data_uris


def sample_video_frames(path: Path, frame_count: int) -> list[bytes]:
    import cv2

    capture = cv2.VideoCapture(str(path))
    if not capture.isOpened():
        raise RuntimeError(f"无法打开视频: {path}")
    total_frames = int(capture.get(cv2.CAP_PROP_FRAME_COUNT) or 0)
    if total_frames <= 0:
        raise RuntimeError(f"无法读取视频帧数: {path}")

    count = min(frame_count, total_frames)
    indexes = [round((i + 0.5) * total_frames / count) for i in range(count)]
    frames = []
    for index in indexes:
        capture.set(cv2.CAP_PROP_POS_FRAMES, min(index, total_frames - 1))
        ok, frame = capture.read()
        if not ok:
            continue
        ok, encoded = cv2.imencode(".jpg", frame, [int(cv2.IMWRITE_JPEG_QUALITY), 85])
        if ok:
            frames.append(encoded.tobytes())
    capture.release()
    if not frames:
        raise RuntimeError(f"无法抽取视频帧: {path}")
    return frames


def file_to_data_uri(path: Path, media_type: str) -> str:
    return f"data:{media_type};base64,{base64.b64encode(path.read_bytes()).decode('ascii')}"


def build_content(path: Path, kind: str, prompt: str, args: argparse.Namespace) -> list[dict]:
    if kind == "image":
        return [
            {"type": "text", "text": prompt},
            {"type": "image_url", "image_url": {"url": image_to_data_uri(path, args.max_image_size)}},
        ]
    if kind == "pdf":
        data_uris = pdf_pages_to_data_uris(path, args.max_image_size, args.pdf_pages, args.pdf_zoom)
        content = [{"type": "text", "text": prompt + f"\n\n下面是这个 PDF 按页渲染得到的 {len(data_uris)} 张图片，请根据这些页面图片回答。"}]
        content.extend({"type": "image_url", "image_url": {"url": data_uri}} for data_uri in data_uris)
        return content
    if args.video_mode == "frames":
        content = [{"type": "text", "text": prompt + f"\n\n下面是按时间顺序抽取的 {args.video_frames} 张视频帧，请根据这些帧概括视频内容。"}]
        content.extend(
            {"type": "image_url", "image_url": {"url": image_bytes_to_data_uri(frame, args.max_image_size)}}
            for frame in sample_video_frames(path, args.video_frames)
        )
        return content
    media_type = mimetypes.guess_type(path.name)[0] or "video/mp4"
    return [
        {"type": "text", "text": prompt},
        {"type": "video_url", "video_url": {"url": file_to_data_uri(path, media_type)}},
    ]


def extract_text(response: dict) -> str:
    choices = response.get("choices") or []
    if not choices:
        return ""
    message = choices[0].get("message") or {}
    content = message.get("content")
    if isinstance(content, str):
        return content.strip()
    if isinstance(content, list):
        return "".join(block.get("text", "") for block in content if isinstance(block, dict)).strip()
    return ""


def call_model(args: argparse.Namespace, path: Path, kind: str, prompt: str) -> tuple[str, dict, float]:
    payload = {
        "model": args.model,
        "messages": [{"role": "user", "content": build_content(path, kind, prompt, args)}],
        "max_tokens": args.max_tokens,
        "temperature": args.temperature,
        "chat_template_kwargs": {"enable_thinking": False},
    }
    headers = {"Content-Type": "application/json"}
    if args.api_key:
        headers["Authorization"] = f"Bearer {args.api_key}"

    start = time.time()
    response = requests.post(args.endpoint, headers=headers, json=payload, timeout=args.timeout)
    elapsed = time.time() - start
    if response.status_code != 200:
        raise RuntimeError(f"HTTP {response.status_code}: {response.text[:2000]}")
    data = response.json()
    return extract_text(data), data, elapsed


def iter_default_files(data_dir: Path) -> list[Path]:
    return sorted(
        path for path in data_dir.iterdir()
        if path.is_file() and path.suffix.lower() in IMAGE_SUFFIXES | VIDEO_SUFFIXES | PDF_SUFFIXES
    )


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="调用六卡4090 qwen3.6-35b-a3b-fp8 多模态模型识别图片、PDF 或视频")
    parser.add_argument("files", nargs="*", help="图片、PDF 或视频路径；不传则默认测试 usecase/qwen3.6/data 下全部图片、PDF 和视频")
    parser.add_argument("--data-dir", default=str(DEFAULT_DATA_DIR))
    parser.add_argument("--prompt", help="自定义提示词；默认图片用 OCR/视觉分析提示词，视频用小于500字描述提示词")
    parser.add_argument("--endpoint", default=DEFAULT_ENDPOINT)
    parser.add_argument("--api-key", default=os.environ.get("QWEN3_6_FP8_API_KEY", ""))
    parser.add_argument("--model", default=DEFAULT_MODEL)
    parser.add_argument("--max-tokens", type=int, default=4096)
    parser.add_argument("--temperature", type=float, default=0)
    parser.add_argument("--timeout", type=int, default=600)
    parser.add_argument("--max-image-size", type=int, default=2048, help="图片最大边长；设为0表示不压缩")
    parser.add_argument("--video-mode", choices=["frames", "native"], default="frames", help="视频默认抽帧为多张图片，避免 video_url 在 4090 上触发 OOM")
    parser.add_argument("--video-frames", type=int, default=8)
    parser.add_argument("--pdf-pages", type=int, default=0, help="PDF 渲染页数；0 表示全部页面")
    parser.add_argument("--pdf-zoom", type=float, default=2.0, help="PDF 渲染缩放倍率")
    parser.add_argument("--results-dir", default=str(Path(__file__).resolve().parent / "results"))
    return parser.parse_args()


def main() -> int:
    args = parse_args()
    data_dir = Path(args.data_dir).expanduser().resolve()
    paths = [Path(file).expanduser().resolve() for file in args.files] if args.files else iter_default_files(data_dir)
    if not paths:
        print(f"没有找到测试文件: {data_dir}", file=sys.stderr)
        return 1

    results_dir = Path(args.results_dir).expanduser().resolve()
    results_dir.mkdir(parents=True, exist_ok=True)
    summary = []

    for path in paths:
        if not path.exists():
            print(f"文件不存在: {path}", file=sys.stderr)
            return 1
        try:
            kind = detect_kind(path)
            prompt = args.prompt or (VIDEO_PROMPT if kind == "video" else IMAGE_PROMPT)
            text, raw_response, elapsed = call_model(args, path, kind, prompt)
        except Exception as exc:
            print(f"调用失败: {path}: {exc}", file=sys.stderr)
            return 1

        output_base = results_dir / path.name
        text_path = output_base.with_suffix(output_base.suffix + ".txt")
        json_path = output_base.with_suffix(output_base.suffix + ".json")
        text_path.write_text(text, encoding="utf-8")
        json_path.write_text(json.dumps({
            "file": str(path),
            "kind": kind,
            "endpoint": args.endpoint,
            "model": args.model,
            "elapsed": elapsed,
            "prompt": prompt,
            "text": text,
            "raw_response": raw_response,
        }, ensure_ascii=False, indent=2), encoding="utf-8")

        usage = raw_response.get("usage") or {}
        summary.append({"file": path.name, "kind": kind, "elapsed": elapsed, "usage": usage, "text": text})
        print(f"\n===== {path.name} ({kind}, {elapsed:.2f}s) =====")
        print(text)
        if usage:
            print(f"tokens: {json.dumps(usage, ensure_ascii=False)}")

    (results_dir / "summary.json").write_text(json.dumps(summary, ensure_ascii=False, indent=2), encoding="utf-8")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
