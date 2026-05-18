#!/usr/bin/env python3
import argparse
import csv
import json
import os
import re
import signal
import subprocess
import sys
import time
from dataclasses import dataclass
from pathlib import Path


KIND_RE = re.compile(r"(?P<kind>Prefill|Decode) batch")
RUNNING_RE = re.compile(r"#running-req: (?P<value>\d+)")
QUEUE_RE = re.compile(r"#queue-req: (?P<value>\d+)")
USAGE_RE = re.compile(r"token usage: (?P<value>[0-9.]+)")


def parse_scheduler_line(line: str) -> dict | None:
    kind_match = KIND_RE.search(line)
    running_match = RUNNING_RE.search(line)
    queue_match = QUEUE_RE.search(line)
    usage_match = USAGE_RE.search(line)
    if not (kind_match and running_match and queue_match and usage_match):
        return None
    return {
        "kind": kind_match.group("kind"),
        "running": int(running_match.group("value")),
        "queue": int(queue_match.group("value")),
        "usage": float(usage_match.group("value")),
    }


@dataclass(frozen=True)
class Workload:
    input_tokens: int
    output_tokens: int


def parse_int_list(value: str) -> list[int]:
    return [int(x.strip()) for x in value.split(",") if x.strip()]


def parse_workloads(value: str) -> list[Workload]:
    workloads: list[Workload] = []
    for item in value.split(","):
        item = item.strip()
        if not item:
            continue
        inp, out = item.split(":", 1)
        workloads.append(Workload(int(inp), int(out)))
    return workloads


def short_tokens(n: int) -> str:
    if n % 1024 == 0:
        return f"{n // 1024}k"
    return str(n)


def read_new_log_lines(log_path: Path, offset: int) -> tuple[list[str], int]:
    if not log_path.exists():
        return [], offset
    with log_path.open("r", encoding="utf-8", errors="ignore") as f:
        f.seek(offset)
        lines = f.readlines()
        return lines, f.tell()


def max_or_zero(samples: list[dict], field: str) -> int | float:
    return max((s[field] for s in samples), default=0)


def average_or_empty(samples: list[dict], field: str, ndigits: int) -> float | str:
    if not samples:
        return ""
    return round(sum(s[field] for s in samples) / len(samples), ndigits)


def summarize_samples(samples: list[dict], target_token_usage: float) -> dict:
    prefill = [s for s in samples if s["kind"] == "Prefill"]
    decode = [s for s in samples if s["kind"] == "Decode"]
    capacity = [s for s in samples if s["usage"] >= target_token_usage and s["queue"] > 0]
    source = capacity or samples
    if not samples:
        return {
            "sample_count": 0,
            "prefill_sample_count": 0,
            "decode_sample_count": 0,
            "capacity_sample_count": 0,
            "max_running": 0,
            "max_all_running": 0,
            "max_prefill_running": 0,
            "max_decode_running": 0,
            "max_capacity_running": 0,
            "max_queue": 0,
            "max_token_usage": 0.0,
            "avg_prefill_running": "",
            "avg_decode_running": "",
            "avg_capacity_running": "",
            "avg_decode_queue": "",
            "avg_decode_token_usage": "",
        }
    return {
        "sample_count": len(samples),
        "prefill_sample_count": len(prefill),
        "decode_sample_count": len(decode),
        "capacity_sample_count": len(capacity),
        "max_running": max(s["running"] for s in source),
        "max_all_running": max_or_zero(samples, "running"),
        "max_prefill_running": max_or_zero(prefill, "running"),
        "max_decode_running": max_or_zero(decode, "running"),
        "max_capacity_running": max_or_zero(capacity, "running"),
        "max_queue": max_or_zero(samples, "queue"),
        "max_token_usage": max_or_zero(samples, "usage"),
        "avg_prefill_running": average_or_empty(prefill, "running", 3),
        "avg_decode_running": average_or_empty(decode, "running", 3),
        "avg_capacity_running": average_or_empty(capacity, "running", 3),
        "avg_decode_queue": average_or_empty(decode, "queue", 3),
        "avg_decode_token_usage": average_or_empty(decode, "usage", 4),
    }


def append_csv(path: Path, row: dict) -> None:
    fields = [
        "timestamp",
        "input_tokens",
        "output_tokens",
        "client_concurrency",
        "num_prompts",
        "probe_seconds",
        "sample_count",
        "prefill_sample_count",
        "decode_sample_count",
        "capacity_sample_count",
        "max_running",
        "max_all_running",
        "max_prefill_running",
        "max_decode_running",
        "max_capacity_running",
        "avg_prefill_running",
        "avg_decode_running",
        "avg_capacity_running",
        "max_queue",
        "avg_decode_queue",
        "max_token_usage",
        "avg_decode_token_usage",
        "bench_pid",
        "bench_returncode",
        "bench_stdout",
        "stop_reason",
        "elapsed_s",
    ]
    exists = path.exists()
    with path.open("a", newline="", encoding="utf-8") as f:
        writer = csv.DictWriter(f, fieldnames=fields)
        if not exists:
            writer.writeheader()
        writer.writerow({k: row.get(k, "") for k in fields})


def write_markdown(results_dir: Path, csv_path: Path) -> None:
    if not csv_path.exists():
        return
    with csv_path.open("r", encoding="utf-8") as f:
        rows = list(csv.DictReader(f))
    headers = [
        "input_tokens",
        "output_tokens",
        "client_concurrency",
        "num_prompts",
        "max_running",
        "max_all_running",
        "max_prefill_running",
        "max_decode_running",
        "max_capacity_running",
        "capacity_sample_count",
        "avg_decode_running",
        "max_queue",
        "max_token_usage",
        "decode_sample_count",
        "stop_reason",
    ]
    with (results_dir / "summary.md").open("w", encoding="utf-8") as f:
        f.write("# SGLang Real Concurrency Probe Summary\n\n")
        f.write(f"Generated: {time.strftime('%F %T')}\n\n")
        f.write("| " + " | ".join(headers) + " |\n")
        f.write("| " + " | ".join(["---"] * len(headers)) + " |\n")
        for r in rows:
            f.write("| " + " | ".join(str(r.get(h, "")) for h in headers) + " |\n")


def build_bench_cmd(args: argparse.Namespace, workload: Workload, output_file: Path, tag: str, client_concurrency: int, num_prompts: int) -> list[str]:
    return [
        sys.executable,
        "-m",
        "sglang.bench_serving",
        "--backend",
        "sglang",
        "--host",
        args.host,
        "--port",
        str(args.port),
        "--dataset-name",
        "random-ids",
        "--tokenize-prompt",
        "--random-input-len",
        str(workload.input_tokens),
        "--random-output-len",
        str(workload.output_tokens),
        "--random-range-ratio",
        "1.0",
        "--num-prompts",
        str(num_prompts),
        "--max-concurrency",
        str(client_concurrency),
        "--request-rate",
        "inf",
        "--ready-check-timeout-sec",
        str(args.ready_timeout),
        "--output-file",
        str(output_file),
        "--tag",
        tag,
        "--disable-tqdm",
    ]


def terminate_process(proc: subprocess.Popen, grace_seconds: int) -> int | None:
    if proc.poll() is not None:
        return proc.returncode
    proc.send_signal(signal.SIGTERM)
    try:
        return proc.wait(timeout=grace_seconds)
    except subprocess.TimeoutExpired:
        proc.kill()
        return proc.wait(timeout=30)


def run_probe(args: argparse.Namespace, workload: Workload, csv_path: Path, client_concurrency: int) -> dict:
    num_prompts = max(args.num_prompts, client_concurrency)
    tag = f"probe_{short_tokens(workload.input_tokens)}_{short_tokens(workload.output_tokens)}_c{client_concurrency}"
    bench_stdout = args.results_dir / f"{tag}.bench.log"
    bench_jsonl = args.results_dir / f"{tag}.bench.jsonl"
    samples_jsonl = args.results_dir / f"{tag}.samples.jsonl"
    cmd_file = args.results_dir / f"{tag}.cmd"
    cmd = build_bench_cmd(args, workload, bench_jsonl, tag, client_concurrency, num_prompts)
    cmd_file.write_text(" ".join(cmd) + "\n", encoding="utf-8")

    env = os.environ.copy()
    env["OPENAI_API_KEY"] = args.api_key or env.get("OPENAI_API_KEY", "")

    offset = args.server_log.stat().st_size if args.server_log.exists() else 0
    print(f"[{time.strftime('%F %T')}] start {tag}", flush=True)
    with bench_stdout.open("w", encoding="utf-8") as out:
        proc = subprocess.Popen(cmd, stdout=out, stderr=subprocess.STDOUT, env=env, text=True)

    samples: list[dict] = []
    start = time.time()
    stop_reason = "timeout"
    stable_hits = 0
    last_max_running = 0
    last_growth_time = start
    with samples_jsonl.open("w", encoding="utf-8") as sf:
        while True:
            elapsed = time.time() - start
            lines, offset = read_new_log_lines(args.server_log, offset)
            for line in lines:
                parsed = parse_scheduler_line(line)
                if not parsed:
                    continue
                sample = {
                    "ts": time.strftime("%F %T"),
                    **parsed,
                    "line": line.strip(),
                }
                samples.append(sample)
                sf.write(json.dumps(sample, ensure_ascii=False) + "\n")
                sf.flush()

                if sample["running"] > last_max_running:
                    last_max_running = sample["running"]
                    last_growth_time = time.time()

                capacity_hit = sample["usage"] >= args.target_token_usage and sample["queue"] > 0
                decode_hit = sample["kind"] == "Decode" and sample["queue"] > 0
                no_growth = time.time() - last_growth_time >= args.stable_seconds
                if elapsed >= args.min_probe_seconds and no_growth and (capacity_hit or decode_hit):
                    stable_hits += 1
                else:
                    stable_hits = 0

            if stable_hits >= args.stable_samples:
                stop_reason = "stable_capacity"
                break
            if proc.poll() is not None and elapsed >= args.min_probe_seconds:
                stop_reason = f"bench_exit_{proc.returncode}"
                break
            if elapsed >= args.probe_seconds:
                stop_reason = "timeout"
                break
            time.sleep(args.poll_interval)

    returncode = terminate_process(proc, args.terminate_grace_seconds)
    summary = summarize_samples(samples, args.target_token_usage)
    row = {
        "timestamp": time.strftime("%F %T"),
        "input_tokens": workload.input_tokens,
        "output_tokens": workload.output_tokens,
        "client_concurrency": client_concurrency,
        "num_prompts": num_prompts,
        "probe_seconds": args.probe_seconds,
        "bench_pid": proc.pid,
        "bench_returncode": returncode,
        "bench_stdout": str(bench_stdout),
        "stop_reason": stop_reason,
        "elapsed_s": round(time.time() - start, 3),
        **summary,
    }
    append_csv(csv_path, row)
    write_markdown(args.results_dir, csv_path)
    print(
        f"[{time.strftime('%F %T')}] done {tag}: max_running={summary['max_running']} "
        f"max_all_running={summary['max_all_running']} max_capacity_running={summary['max_capacity_running']} "
        f"max_queue={summary['max_queue']} max_token_usage={summary['max_token_usage']} stop_reason={stop_reason}",
        flush=True,
    )
    return row


def main() -> int:
    parser = argparse.ArgumentParser(description="Probe real SGLang running concurrency from server logs.")
    parser.add_argument("--host", default="127.0.0.1")
    parser.add_argument("--port", type=int, default=8000)
    parser.add_argument("--api-key", default=os.environ.get("OPENAI_API_KEY", "shuzuan2025-minimax"))
    parser.add_argument("--server-log", type=Path, required=True)
    parser.add_argument("--results-dir", type=Path, default=Path("results") / time.strftime("real_concurrency_%Y%m%d_%H%M%S"))
    parser.add_argument("--workloads", default="8192:1024,16384:1024,32768:1024,65536:1024,131072:1024,245760:512")
    parser.add_argument("--client-concurrency", type=int, default=200)
    parser.add_argument("--first-client-concurrency-levels", default="50,100,150,200", help="Client concurrency levels for the first workload to discover a global cap")
    parser.add_argument("--use-first-max-for-rest", action="store_true", default=False, help="Use the first workload's max_running as client concurrency for remaining workloads")
    parser.add_argument("--num-prompts", type=int, default=200)
    parser.add_argument("--probe-seconds", type=int, default=1800)
    parser.add_argument("--min-probe-seconds", type=int, default=120)
    parser.add_argument("--stable-seconds", type=int, default=60)
    parser.add_argument("--stable-samples", type=int, default=3)
    parser.add_argument("--target-token-usage", type=float, default=0.90)
    parser.add_argument("--poll-interval", type=float, default=2.0)
    parser.add_argument("--ready-timeout", type=int, default=60)
    parser.add_argument("--terminate-grace-seconds", type=int, default=20)
    args = parser.parse_args()

    args.results_dir.mkdir(parents=True, exist_ok=True)
    csv_path = args.results_dir / "summary.csv"
    workloads = parse_workloads(args.workloads)
    first_max_running = 0
    for index, workload in enumerate(workloads):
        if index == 0 and args.use_first_max_for_rest:
            rows = []
            for level in parse_int_list(args.first_client_concurrency_levels):
                row = run_probe(args, workload, csv_path, level)
                rows.append(row)
                first_max_running = max(first_max_running, int(row.get("max_running", 0) or 0))
                if first_max_running > 0 and first_max_running < level:
                    break
            continue

        client_concurrency = first_max_running if args.use_first_max_for_rest and first_max_running > 0 else args.client_concurrency
        run_probe(args, workload, csv_path, client_concurrency)
    print(f"results: {args.results_dir}", flush=True)
    return 0


if __name__ == "__main__":
    raise SystemExit(main())

