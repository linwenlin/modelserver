#!/usr/bin/env python3
import argparse
import csv
import json
import math
import os
import shlex
import subprocess
import sys
import time
from dataclasses import dataclass
from pathlib import Path


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
        if ":" not in item:
            raise ValueError(f"workload must be INPUT:OUTPUT, got {item!r}")
        inp, out = item.split(":", 1)
        workloads.append(Workload(int(inp), int(out)))
    return workloads


def load_real_concurrency(path: Path) -> dict[tuple[int, int], int]:
    if not path:
        return {}
    if not path.exists():
        raise FileNotFoundError(f"real concurrency summary not found: {path}")
    result: dict[tuple[int, int], int] = {}
    with path.open("r", encoding="utf-8") as f:
        for row in csv.DictReader(f):
            try:
                key = (int(row["input_tokens"]), int(row["output_tokens"]))
                value = int(float(row["max_running"]))
            except (KeyError, TypeError, ValueError):
                continue
            if value > 0:
                result[key] = value
    return result


def generate_concurrency_levels(max_running: int, max_levels: int = 8) -> list[int]:
    if max_running <= 1:
        return [1]

    levels = {
        1,
        max(1, round(max_running / 3)),
        max(1, round(max_running / 2)),
        max(1, round(max_running * 2 / 3)),
        max_running,
    }
    levels.update(range(10, max_running + 1, 10))
    ordered = sorted(x for x in levels if 1 <= x <= max_running)

    if len(ordered) <= max_levels:
        return ordered

    required = [1, max_running]
    middle = [x for x in ordered if x not in required]
    slots = max_levels - len(required)
    if slots <= 0:
        return sorted(required)
    if slots >= len(middle):
        return ordered

    selected = []
    for i in range(slots):
        idx = round(i * (len(middle) - 1) / max(slots - 1, 1))
        selected.append(middle[idx])
    return sorted(set(required + selected))


def short_tokens(n: int) -> str:
    if n % 1024 == 0:
        return f"{n // 1024}k"
    return str(n)


def load_summary(path: Path) -> dict:
    with path.open("r", encoding="utf-8") as f:
        for line in f:
            line = line.strip()
            if line:
                return json.loads(line)
    raise RuntimeError(f"empty benchmark output: {path}")


def metric(summary: dict, key: str, default: float = math.nan) -> float:
    value = summary.get(key, default)
    if value is None:
        return default
    return float(value)


def build_command(args: argparse.Namespace, workload: Workload, concurrency: int, num_prompts: int, output_file: Path, tag: str) -> list[str]:
    return [
        sys.executable,
        "-m",
        "sglang.bench_serving",
        "--backend",
        args.backend,
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
        str(concurrency),
        "--request-rate",
        "inf",
        "--ready-check-timeout-sec",
        str(args.ready_timeout),
        "--output-file",
        str(output_file),
        "--tag",
        tag,
    ] + (["--disable-tqdm"] if args.disable_tqdm else [])


def append_csv(csv_path: Path, row: dict) -> None:
    fieldnames = [
        "timestamp",
        "phase",
        "input_tokens",
        "output_tokens",
        "concurrency",
        "num_prompts",
        "completed",
        "success_rate",
        "bad",
        "bad_reasons",
        "duration_s",
        "request_throughput",
        "input_throughput",
        "output_throughput",
        "total_throughput",
        "mean_e2e_latency_ms",
        "p99_e2e_latency_ms",
        "mean_ttft_ms",
        "p99_ttft_ms",
        "mean_tpot_ms",
        "p99_tpot_ms",
        "mean_itl_ms",
        "p99_itl_ms",
        "output_file",
        "stdout_file",
    ]
    exists = csv_path.exists()
    with csv_path.open("a", newline="", encoding="utf-8") as f:
        writer = csv.DictWriter(f, fieldnames=fieldnames)
        if not exists:
            writer.writeheader()
        writer.writerow({k: row.get(k, "") for k in fieldnames})


def classify(summary: dict, num_prompts: int, prev: dict | None, args: argparse.Namespace) -> tuple[bool, list[str]]:
    reasons: list[str] = []
    completed = int(summary.get("completed", 0) or 0)
    success_rate = completed / max(num_prompts, 1)
    mean_tpot = metric(summary, "mean_tpot_ms")
    p99_tpot = metric(summary, "p99_tpot_ms")
    p99_itl = metric(summary, "p99_itl_ms")
    mean_e2e = metric(summary, "mean_e2e_latency_ms")
    p99_ttft = metric(summary, "p99_ttft_ms")
    output_tps = metric(summary, "output_throughput")

    if success_rate < args.min_success_rate:
        reasons.append(f"success_rate {success_rate:.3f} < {args.min_success_rate:.3f}")
    if mean_tpot >= args.mean_tpot_stop_ms:
        reasons.append(f"mean_tpot {mean_tpot:.1f}ms >= {args.mean_tpot_stop_ms:.1f}ms")
    if p99_tpot >= args.p99_tpot_stop_ms:
        reasons.append(f"p99_tpot {p99_tpot:.1f}ms >= {args.p99_tpot_stop_ms:.1f}ms")
    if p99_itl >= args.p99_itl_stop_ms:
        reasons.append(f"p99_itl {p99_itl:.1f}ms >= {args.p99_itl_stop_ms:.1f}ms")

    if prev:
        prev_mean_e2e = metric(prev, "mean_e2e_latency_ms")
        prev_p99_ttft = metric(prev, "p99_ttft_ms")
        prev_output_tps = metric(prev, "output_throughput")
        if mean_e2e >= prev_mean_e2e * args.e2e_jump_ratio and output_tps <= prev_output_tps * args.throughput_growth_floor:
            reasons.append(
                f"mean_e2e jumped {mean_e2e / max(prev_mean_e2e, 1e-9):.2f}x without enough throughput growth"
            )
        if p99_ttft >= prev_p99_ttft * args.ttft_jump_ratio:
            reasons.append(f"p99_ttft jumped {p99_ttft / max(prev_p99_ttft, 1e-9):.2f}x")

    return bool(reasons), reasons


def run_one(args: argparse.Namespace, workload: Workload, concurrency: int, phase: str, prev_summary: dict | None) -> tuple[dict | None, bool]:
    num_prompts = args.single_num_prompts if phase == "single" else max(args.min_num_prompts, concurrency * args.prompts_multiplier)
    if args.num_prompts:
        num_prompts = args.num_prompts

    tag = f"{phase}_{short_tokens(workload.input_tokens)}_{short_tokens(workload.output_tokens)}_c{concurrency}"
    output_file = args.results_dir / f"{tag}.jsonl"
    stdout_file = args.results_dir / f"{tag}.log"
    command_file = args.results_dir / f"{tag}.cmd"
    cmd = build_command(args, workload, concurrency, num_prompts, output_file, tag)
    command_file.write_text(" ".join(shlex.quote(x) for x in cmd) + "\n", encoding="utf-8")

    env = os.environ.copy()
    if args.api_key:
        env["OPENAI_API_KEY"] = args.api_key

    started = time.time()
    print(f"[{time.strftime('%F %T')}] run {tag}: num_prompts={num_prompts}", flush=True)
    with stdout_file.open("w", encoding="utf-8") as out:
        proc = subprocess.run(cmd, stdout=out, stderr=subprocess.STDOUT, env=env, text=True, timeout=args.command_timeout)

    if proc.returncode != 0:
        row = {
            "timestamp": time.strftime("%F %T"),
            "phase": phase,
            "input_tokens": workload.input_tokens,
            "output_tokens": workload.output_tokens,
            "concurrency": concurrency,
            "num_prompts": num_prompts,
            "completed": 0,
            "success_rate": 0,
            "bad": 1,
            "bad_reasons": f"bench_exit_{proc.returncode}",
            "duration_s": round(time.time() - started, 3),
            "output_file": str(output_file),
            "stdout_file": str(stdout_file),
        }
        append_csv(args.csv_file, row)
        print(f"[{time.strftime('%F %T')}] bad {tag}: bench_exit_{proc.returncode}", flush=True)
        return None, True

    summary = load_summary(output_file)
    bad, reasons = classify(summary, num_prompts, prev_summary, args)
    completed = int(summary.get("completed", 0) or 0)
    row = {
        "timestamp": time.strftime("%F %T"),
        "phase": phase,
        "input_tokens": workload.input_tokens,
        "output_tokens": workload.output_tokens,
        "concurrency": concurrency,
        "num_prompts": num_prompts,
        "completed": completed,
        "success_rate": round(completed / max(num_prompts, 1), 6),
        "bad": int(bad),
        "bad_reasons": "; ".join(reasons),
        "duration_s": summary.get("duration", ""),
        "request_throughput": summary.get("request_throughput", ""),
        "input_throughput": summary.get("input_throughput", ""),
        "output_throughput": summary.get("output_throughput", ""),
        "total_throughput": summary.get("total_throughput", ""),
        "mean_e2e_latency_ms": summary.get("mean_e2e_latency_ms", ""),
        "p99_e2e_latency_ms": summary.get("p99_e2e_latency_ms", ""),
        "mean_ttft_ms": summary.get("mean_ttft_ms", ""),
        "p99_ttft_ms": summary.get("p99_ttft_ms", ""),
        "mean_tpot_ms": summary.get("mean_tpot_ms", ""),
        "p99_tpot_ms": summary.get("p99_tpot_ms", ""),
        "mean_itl_ms": summary.get("mean_itl_ms", ""),
        "p99_itl_ms": summary.get("p99_itl_ms", ""),
        "output_file": str(output_file),
        "stdout_file": str(stdout_file),
    }
    append_csv(args.csv_file, row)
    print(
        f"[{time.strftime('%F %T')}] done {tag}: bad={bad} completed={completed}/{num_prompts} "
        f"out_tps={summary.get('output_throughput', math.nan):.2f} mean_tpot={summary.get('mean_tpot_ms', math.nan):.1f}ms",
        flush=True,
    )
    if reasons:
        print(f"[{time.strftime('%F %T')}] reasons {tag}: {'; '.join(reasons)}", flush=True)
    return summary, bad


def read_csv_rows(csv_file: Path) -> list[dict]:
    if not csv_file.exists():
        return []
    with csv_file.open("r", encoding="utf-8") as f:
        return list(csv.DictReader(f))


def write_markdown(results_dir: Path, csv_file: Path) -> None:
    rows = read_csv_rows(csv_file)
    md = results_dir / "summary.md"
    headers = [
        "phase",
        "input_tokens",
        "output_tokens",
        "concurrency",
        "success_rate",
        "bad",
        "output_throughput",
        "mean_tpot_ms",
        "p99_tpot_ms",
        "mean_ttft_ms",
        "p99_e2e_latency_ms",
        "bad_reasons",
    ]
    with md.open("w", encoding="utf-8") as f:
        f.write("# SGLang Benchmark Summary\n\n")
        f.write(f"Generated: {time.strftime('%F %T')}\n\n")
        f.write("| " + " | ".join(headers) + " |\n")
        f.write("| " + " | ".join(["---"] * len(headers)) + " |\n")
        for row in rows:
            f.write("| " + " | ".join(str(row.get(h, "")) for h in headers) + " |\n")


def refine_between(args: argparse.Namespace, workload: Workload, low_c: int, low_summary: dict | None, high_c: int) -> None:
    prev_summary = low_summary
    lo = low_c
    hi = high_c
    while hi - lo > args.refine_min_gap:
        mid = (lo + hi) // 2
        if mid in args.tested:
            break
        args.tested.add(mid)
        summary, bad = run_one(args, workload, mid, "refine", prev_summary)
        write_markdown(args.results_dir, args.csv_file)
        if bad:
            hi = mid
        else:
            lo = mid
            prev_summary = summary


def run_single_ladder(args: argparse.Namespace) -> None:
    for workload in args.single_workloads:
        run_one(args, workload, 1, "single", None)
        write_markdown(args.results_dir, args.csv_file)


def concurrency_levels_for_workload(args: argparse.Namespace, workload: Workload) -> list[int]:
    max_running = args.real_concurrency_by_workload.get((workload.input_tokens, workload.output_tokens))
    if max_running:
        levels = generate_concurrency_levels(max_running, args.auto_levels_max_count)
        print(
            f"[{time.strftime('%F %T')}] workload input={workload.input_tokens} output={workload.output_tokens} "
            f"real max_running={max_running}; generated concurrency levels={levels}",
            flush=True,
        )
        return levels
    return args.concurrency_levels


def run_concurrency_search(args: argparse.Namespace) -> None:
    for workload in args.concurrent_workloads:
        print(f"[{time.strftime('%F %T')}] workload input={workload.input_tokens} output={workload.output_tokens}", flush=True)
        prev_c = 0
        prev_summary = None
        args.tested = set()
        levels = concurrency_levels_for_workload(args, workload)
        for concurrency in levels:
            args.tested.add(concurrency)
            summary, bad = run_one(args, workload, concurrency, "concurrency", prev_summary)
            write_markdown(args.results_dir, args.csv_file)
            if bad:
                if args.stop_on_bad and args.refine and prev_c > 0:
                    refine_between(args, workload, prev_c, prev_summary, concurrency)
                if args.stop_on_bad:
                    break
                continue
            prev_c = concurrency
            prev_summary = summary


def main() -> int:
    parser = argparse.ArgumentParser(description="Run application-level SGLang serving benchmarks.")
    parser.add_argument("--mode", choices=["single", "concurrency", "all"], default="all")
    parser.add_argument("--host", default="127.0.0.1")
    parser.add_argument("--port", type=int, default=8000)
    parser.add_argument("--backend", default="sglang")
    parser.add_argument("--api-key", default=os.environ.get("OPENAI_API_KEY", ""))
    parser.add_argument("--results-dir", type=Path, default=Path("results") / time.strftime("kimi_%Y%m%d_%H%M%S"))
    parser.add_argument("--single-workloads", default="8192:1024,16384:1024,32768:1024,65536:1024,98304:1024,131072:1024,163840:1024,196608:1024,229376:1024,245760:512")
    parser.add_argument("--concurrent-workloads", default="8192:1024,65536:1024,131072:1024,245760:512")
    parser.add_argument("--concurrency-levels", default="100,200,300,400,500")
    parser.add_argument("--real-concurrency-summary", type=Path, default=None, help="CSV from probe_sglang_real_concurrency.py; when provided, concurrency levels are generated from max_running per workload")
    parser.add_argument("--auto-levels-max-count", type=int, default=8, help="Maximum generated concurrency levels per workload when using --real-concurrency-summary")
    parser.add_argument("--single-num-prompts", type=int, default=3)
    parser.add_argument("--num-prompts", type=int, default=0)
    parser.add_argument("--min-num-prompts", type=int, default=3)
    parser.add_argument("--prompts-multiplier", type=int, default=3)
    parser.add_argument("--ready-timeout", type=int, default=60)
    parser.add_argument("--command-timeout", type=int, default=0)
    parser.add_argument("--disable-tqdm", action="store_true", default=True)
    parser.add_argument("--refine", action="store_true", default=True)
    parser.add_argument("--no-refine", action="store_false", dest="refine")
    parser.add_argument("--refine-min-gap", type=int, default=25)
    parser.add_argument("--stop-on-bad", action="store_true", default=False, help="Stop each workload after the first bad concurrency level")
    parser.add_argument("--min-success-rate", type=float, default=0.95)
    parser.add_argument("--mean-tpot-stop-ms", type=float, default=500.0)
    parser.add_argument("--p99-tpot-stop-ms", type=float, default=1000.0)
    parser.add_argument("--p99-itl-stop-ms", type=float, default=1000.0)
    parser.add_argument("--e2e-jump-ratio", type=float, default=2.0)
    parser.add_argument("--ttft-jump-ratio", type=float, default=2.0)
    parser.add_argument("--throughput-growth-floor", type=float, default=1.10)
    args = parser.parse_args()

    if args.command_timeout == 0:
        args.command_timeout = None
    args.single_workloads = parse_workloads(args.single_workloads)
    args.concurrent_workloads = parse_workloads(args.concurrent_workloads)
    args.concurrency_levels = parse_int_list(args.concurrency_levels)
    args.real_concurrency_by_workload = load_real_concurrency(args.real_concurrency_summary) if args.real_concurrency_summary else {}
    args.results_dir.mkdir(parents=True, exist_ok=True)
    args.csv_file = args.results_dir / "summary.csv"

    if args.mode in ("single", "all"):
        run_single_ladder(args)
    if args.mode in ("concurrency", "all"):
        run_concurrency_search(args)
    write_markdown(args.results_dir, args.csv_file)
    print(f"results: {args.results_dir}", flush=True)
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
