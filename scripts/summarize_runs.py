#!/usr/bin/env python3

import argparse
import json
from pathlib import Path
from typing import Dict, Iterable, List


def parse_env(path: Path) -> Dict[str, str]:
    values: Dict[str, str] = {}
    if not path.exists():
        return values
    for line in path.read_text(encoding="utf-8").splitlines():
        if not line or line.startswith("#") or "=" not in line:
            continue
        key, value = line.split("=", 1)
        values[key] = value
    return values


def detect_backend(log_path: Path) -> str:
    if not log_path.exists():
        return "unknown"
    backend = "unknown"
    for line in log_path.read_text(encoding="utf-8", errors="replace").splitlines():
        if "Using XFormers backend" in line:
            backend = "XFormers"
        elif "FLASH_ATTN_V100" in line:
            backend = "FlashAttention V100"
        elif "Using FlashAttention" in line:
            backend = "FlashAttention"
        elif "Cannot use FlashAttention" in line and backend == "unknown":
            backend = "FlashAttention rejected"
    return backend


def detect_failure(run_dir: Path) -> str:
    for log_name in ("vllm-server.log", "benchmark.log"):
        log_path = run_dir / log_name
        if not log_path.exists():
            continue
        text = log_path.read_text(encoding="utf-8", errors="replace")
        if "input size is not aligned with the quantized weight shape" in text:
            return "failed: GPTQ TP alignment"
        if "Engine process failed to start" in text:
            return "failed: engine startup"
    return "missing summary"


def fmt(value: object, digits: int = 2) -> str:
    if isinstance(value, (int, float)):
        return f"{value:.{digits}f}"
    return str(value)


def rows_for_runs(runs_dir: Path, job_ids: Iterable[str]) -> List[List[str]]:
    rows: List[List[str]] = []
    for job_id in job_ids:
        run_dir = runs_dir / job_id
        env = parse_env(run_dir / "experiment.env")
        summary_path = run_dir / "summary.json"
        if not summary_path.exists():
            rows.append([job_id, env.get("EXPERIMENT_NAME", ""), detect_failure(run_dir)])
            continue

        summary = json.loads(summary_path.read_text(encoding="utf-8"))
        metadata = summary.get("metadata", {})
        rows.append(
            [
                job_id,
                env.get("EXPERIMENT_NAME", metadata.get("model", "")),
                env.get("TP_SIZE", ""),
                env.get("PP_SIZE", ""),
                env.get("CONCURRENCY", str(metadata.get("concurrency", ""))),
                env.get("MAX_NUM_SEQS", ""),
                env.get("MAX_NUM_BATCHED_TOKENS", ""),
                detect_backend(run_dir / "vllm-server.log"),
                fmt(summary.get("aggregate_output_tokens_per_s", ""), 2),
                fmt(summary.get("decode_tokens_per_s", {}).get("mean", ""), 2),
                fmt(summary.get("latency_s", {}).get("mean", ""), 3),
                fmt(summary.get("ttft_s", {}).get("mean", ""), 3),
            ]
        )
    return rows


def print_markdown(rows: List[List[str]]) -> None:
    header = [
        "job",
        "experiment",
        "TP",
        "PP",
        "conc",
        "max seqs",
        "batched toks",
        "backend",
        "agg out tok/s",
        "decode tok/s",
        "lat mean s",
        "TTFT mean s",
    ]
    print("| " + " | ".join(header) + " |")
    print("| " + " | ".join("---" for _ in header) + " |")
    for row in rows:
        print("| " + " | ".join(row) + " |")


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Summarize vLLM experiment runs.")
    parser.add_argument("job_ids", nargs="+")
    parser.add_argument("--runs-dir", default="runs")
    return parser.parse_args()


def main() -> int:
    args = parse_args()
    print_markdown(rows_for_runs(Path(args.runs_dir), args.job_ids))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
