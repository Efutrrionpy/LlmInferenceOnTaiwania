#!/usr/bin/env python3

import argparse
import concurrent.futures
import json
import math
import os
import statistics
import sys
import time
import urllib.error
import urllib.request
from pathlib import Path
from typing import Any, Dict, List, Optional, Tuple

from transformers import AutoTokenizer


def percentile(values: List[float], pct: float) -> float:
    if not values:
        return float("nan")
    ordered = sorted(values)
    index = max(0, min(len(ordered) - 1, math.ceil((pct / 100.0) * len(ordered)) - 1))
    return ordered[index]


def mean(values: List[float]) -> float:
    return statistics.fmean(values) if values else float("nan")


def build_prompt(tokenizer: Any, target_tokens: int) -> Tuple[str, int]:
    seed = (
        "We are benchmarking large language model inference on Taiwania 2 "
        "with NVIDIA V100 GPUs. Please continue with a concise technical "
        "explanation of distributed inference performance. "
    )
    seed_ids = tokenizer.encode(seed, add_special_tokens=False)
    if not seed_ids:
        raise RuntimeError("Tokenizer produced no tokens for the seed prompt.")

    repeated = (seed_ids * ((target_tokens // len(seed_ids)) + 2))[:target_tokens]
    prompt = tokenizer.decode(repeated, skip_special_tokens=True)
    actual_tokens = len(tokenizer.encode(prompt, add_special_tokens=False))
    return prompt, actual_tokens


def wait_for_server(base_url: str, timeout_s: float) -> None:
    deadline = time.monotonic() + timeout_s
    models_url = f"{base_url.rstrip('/')}/models"
    last_error = None  # type: Optional[Exception]

    while time.monotonic() < deadline:
        try:
            with urllib.request.urlopen(models_url, timeout=5) as response:
                if response.status == 200:
                    return
        except Exception as exc:  # noqa: BLE001
            last_error = exc
        time.sleep(5)

    raise TimeoutError(f"Server did not become ready at {models_url}: {last_error}")


def stream_completion(
    base_url: str,
    model: str,
    prompt: str,
    output_tokens: int,
    timeout_s: float,
) -> Dict[str, Any]:
    payload = {
        "model": model,
        "prompt": prompt,
        "max_tokens": output_tokens,
        "min_tokens": output_tokens,
        "temperature": 0,
        "stream": True,
        "ignore_eos": True,
    }
    request = urllib.request.Request(
        f"{base_url.rstrip('/')}/completions",
        data=json.dumps(payload).encode("utf-8"),
        headers={"Content-Type": "application/json"},
        method="POST",
    )

    started = time.perf_counter()
    first_token_at = None  # type: Optional[float]
    chunks = []  # type: List[str]
    stream_events = 0

    try:
        with urllib.request.urlopen(request, timeout=timeout_s) as response:
            while True:
                line = response.readline()
                if not line:
                    break
                line = line.strip()
                if not line or not line.startswith(b"data:"):
                    continue

                raw = line[len(b"data:") :].strip()
                if raw == b"[DONE]":
                    break

                event = json.loads(raw.decode("utf-8"))
                stream_events += 1
                text = event.get("choices", [{}])[0].get("text", "")
                if text and first_token_at is None:
                    first_token_at = time.perf_counter()
                chunks.append(text)
    except urllib.error.HTTPError as exc:
        body = exc.read().decode("utf-8", errors="replace")
        raise RuntimeError(f"HTTP {exc.code} from vLLM: {body}") from exc

    finished = time.perf_counter()
    if first_token_at is None:
        first_token_at = finished

    return {
        "text": "".join(chunks),
        "latency_s": finished - started,
        "ttft_s": first_token_at - started,
        "stream_events": stream_events,
    }


def run_one(
    tokenizer: Any,
    base_url: str,
    model: str,
    prompt: str,
    input_tokens: int,
    output_tokens: int,
    timeout_s: float,
    phase: str,
    index: int,
) -> Dict[str, Any]:
    result = stream_completion(base_url, model, prompt, output_tokens, timeout_s)
    output_text_tokens = len(tokenizer.encode(result["text"], add_special_tokens=False))
    output_token_count = max(output_text_tokens, int(result.get("stream_events", 0)))
    latency_s = result["latency_s"]
    ttft_s = result["ttft_s"]
    decode_s = max(0.0, latency_s - ttft_s)

    measured = {
        "phase": phase,
        "index": index,
        "input_tokens": input_tokens,
        "output_tokens": output_token_count,
        "output_text_tokens": output_text_tokens,
        "stream_events": result.get("stream_events", 0),
        "latency_s": latency_s,
        "ttft_s": ttft_s,
        "decode_s": decode_s,
        "output_tokens_per_s": output_token_count / latency_s if latency_s > 0 else float("nan"),
        "decode_tokens_per_s": (
            max(output_token_count - 1, 0) / decode_s if decode_s > 0 else float("nan")
        ),
    }
    print(json.dumps(measured, sort_keys=True), flush=True)
    return measured


def summarize(rows: List[Dict[str, Any]], metadata: Dict[str, Any]) -> Dict[str, Any]:
    latencies = [row["latency_s"] for row in rows]
    ttfts = [row["ttft_s"] for row in rows]
    output_tps = [row["output_tokens_per_s"] for row in rows]
    decode_tps = [row["decode_tokens_per_s"] for row in rows if math.isfinite(row["decode_tokens_per_s"])]

    return {
        "metadata": metadata,
        "requests": len(rows),
        "latency_s": {
            "mean": mean(latencies),
            "p50": percentile(latencies, 50),
            "p95": percentile(latencies, 95),
        },
        "ttft_s": {
            "mean": mean(ttfts),
            "p50": percentile(ttfts, 50),
            "p95": percentile(ttfts, 95),
        },
        "output_tokens_per_s": {
            "mean": mean(output_tps),
            "p50": percentile(output_tps, 50),
            "p95": percentile(output_tps, 95),
        },
        "decode_tokens_per_s": {
            "mean": mean(decode_tps),
            "p50": percentile(decode_tps, 50),
            "p95": percentile(decode_tps, 95),
        },
    }


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Benchmark vLLM OpenAI completions.")
    parser.add_argument("--base-url", default="http://127.0.0.1:8000/v1")
    parser.add_argument("--model", required=True)
    parser.add_argument("--tokenizer", required=True)
    parser.add_argument("--input-tokens", type=int, default=512)
    parser.add_argument("--output-tokens", type=int, default=128)
    parser.add_argument("--warmup", type=int, default=3)
    parser.add_argument("--runs", type=int, default=20)
    parser.add_argument("--concurrency", type=int, default=1)
    parser.add_argument("--results-dir", default="runs/local")
    parser.add_argument("--server-timeout", type=float, default=1800)
    parser.add_argument("--request-timeout", type=float, default=600)
    return parser.parse_args()


def main() -> int:
    args = parse_args()
    results_dir = Path(args.results_dir)
    results_dir.mkdir(parents=True, exist_ok=True)

    tokenizer = AutoTokenizer.from_pretrained(args.tokenizer, trust_remote_code=True, use_fast=True)
    prompt, actual_input_tokens = build_prompt(tokenizer, args.input_tokens)

    metadata = {
        "base_url": args.base_url,
        "model": args.model,
        "tokenizer": args.tokenizer,
        "target_input_tokens": args.input_tokens,
        "actual_input_tokens": actual_input_tokens,
        "target_output_tokens": args.output_tokens,
        "warmup": args.warmup,
        "runs": args.runs,
        "concurrency": args.concurrency,
        "pid": os.getpid(),
    }
    (results_dir / "metadata.json").write_text(json.dumps(metadata, indent=2) + "\n")
    (results_dir / "prompt.txt").write_text(prompt)

    wait_for_server(args.base_url, args.server_timeout)

    for index in range(args.warmup):
        run_one(
            tokenizer,
            args.base_url,
            args.model,
            prompt,
            actual_input_tokens,
            args.output_tokens,
            args.request_timeout,
            "warmup",
            index,
        )

    rows = []
    measured_started = time.perf_counter()
    with (results_dir / "requests.jsonl").open("w", encoding="utf-8") as handle:
        if args.concurrency <= 1:
            for index in range(args.runs):
                row = run_one(
                    tokenizer,
                    args.base_url,
                    args.model,
                    prompt,
                    actual_input_tokens,
                    args.output_tokens,
                    args.request_timeout,
                    "measured",
                    index,
                )
                rows.append(row)
                handle.write(json.dumps(row, sort_keys=True) + "\n")
                handle.flush()
        else:
            with concurrent.futures.ThreadPoolExecutor(max_workers=args.concurrency) as executor:
                futures = [
                    executor.submit(
                        run_one,
                        tokenizer,
                        args.base_url,
                        args.model,
                        prompt,
                        actual_input_tokens,
                        args.output_tokens,
                        args.request_timeout,
                        "measured",
                        index,
                    )
                    for index in range(args.runs)
                ]
                for future in concurrent.futures.as_completed(futures):
                    row = future.result()
                    rows.append(row)
                    handle.write(json.dumps(row, sort_keys=True) + "\n")
                    handle.flush()
    measured_finished = time.perf_counter()

    summary = summarize(rows, metadata)
    measured_wall_s = measured_finished - measured_started
    total_output_tokens = sum(row["output_tokens"] for row in rows)
    summary["measured_wall_s"] = measured_wall_s
    summary["total_output_tokens"] = total_output_tokens
    summary["aggregate_output_tokens_per_s"] = (
        total_output_tokens / measured_wall_s if measured_wall_s > 0 else float("nan")
    )
    (results_dir / "summary.json").write_text(json.dumps(summary, indent=2) + "\n")
    print(json.dumps(summary, indent=2), flush=True)
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
