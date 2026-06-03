#!/usr/bin/env python3
"""Summarize lightweight nvidia-smi CSV profiles."""

from __future__ import annotations

import argparse
import csv
import json
from datetime import datetime
from pathlib import Path
from statistics import mean


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser()
    parser.add_argument("csv_files", nargs="+", type=Path)
    parser.add_argument("--window-start", type=Path)
    parser.add_argument("--window-end", type=Path)
    parser.add_argument("--output-json", type=Path)
    parser.add_argument("--output-md", type=Path)
    return parser.parse_args()


def parse_time(value: str) -> datetime | None:
    value = value.strip()
    for fmt in ("%Y/%m/%d %H:%M:%S.%f", "%Y/%m/%d %H:%M:%S"):
        try:
            return datetime.strptime(value, fmt)
        except ValueError:
            pass
    return None


def read_time_file(path: Path | None) -> datetime | None:
    if not path or not path.exists():
        return None
    return parse_time(path.read_text(encoding="utf-8").strip().splitlines()[0])


def get_value(row: dict[str, str], prefix: str) -> str:
    prefix = prefix.lower()
    for key, value in row.items():
        if key.strip().lower().startswith(prefix):
            return value.strip()
    return ""


def parse_float(value: str) -> float | None:
    if not value:
        return None
    try:
        return float(value.replace("N/A", "").strip())
    except ValueError:
        return None


def source_name(path: Path) -> str:
    name = path.stem
    return name.removeprefix("gpu-profile-")


def load_rows(path: Path, start: datetime | None, end: datetime | None) -> list[dict[str, float | str]]:
    if not path.exists():
        return []

    rows: list[dict[str, float | str]] = []
    with path.open(newline="", encoding="utf-8") as handle:
        reader = csv.DictReader(handle)
        for raw in reader:
            timestamp = parse_time(get_value(raw, "timestamp"))
            if timestamp is None:
                continue
            if start and timestamp < start:
                continue
            if end and timestamp > end:
                continue

            gpu_util = parse_float(get_value(raw, "utilization.gpu"))
            mem_util = parse_float(get_value(raw, "utilization.memory"))
            mem_used = parse_float(get_value(raw, "memory.used"))
            mem_total = parse_float(get_value(raw, "memory.total"))
            power = parse_float(get_value(raw, "power.draw"))
            temp = parse_float(get_value(raw, "temperature.gpu"))

            if gpu_util is None or mem_used is None or mem_total in (None, 0):
                continue

            rows.append(
                {
                    "source": source_name(path),
                    "gpu": get_value(raw, "index"),
                    "gpu_util_pct": gpu_util,
                    "memory_util_pct": mem_util if mem_util is not None else 0.0,
                    "memory_used_mib": mem_used,
                    "memory_total_mib": mem_total,
                    "memory_used_pct": 100.0 * mem_used / mem_total,
                    "power_w": power if power is not None else 0.0,
                    "temperature_c": temp if temp is not None else 0.0,
                }
            )
    return rows


def summarize(rows: list[dict[str, float | str]], label: str) -> dict[str, float | int | str]:
    if not rows:
        return {
            "source": label,
            "samples": 0,
            "avg_gpu_util_pct": 0.0,
            "max_gpu_util_pct": 0.0,
            "avg_memory_used_gib": 0.0,
            "max_memory_used_gib": 0.0,
            "avg_memory_used_pct": 0.0,
            "max_memory_used_pct": 0.0,
            "avg_power_w": 0.0,
            "avg_temperature_c": 0.0,
        }

    return {
        "source": label,
        "samples": len(rows),
        "avg_gpu_util_pct": mean(float(row["gpu_util_pct"]) for row in rows),
        "max_gpu_util_pct": max(float(row["gpu_util_pct"]) for row in rows),
        "avg_memory_used_gib": mean(float(row["memory_used_mib"]) for row in rows) / 1024.0,
        "max_memory_used_gib": max(float(row["memory_used_mib"]) for row in rows) / 1024.0,
        "avg_memory_used_pct": mean(float(row["memory_used_pct"]) for row in rows),
        "max_memory_used_pct": max(float(row["memory_used_pct"]) for row in rows),
        "avg_power_w": mean(float(row["power_w"]) for row in rows),
        "avg_temperature_c": mean(float(row["temperature_c"]) for row in rows),
    }


def to_markdown(summaries: list[dict[str, float | int | str]]) -> str:
    lines = [
        "| Source | Samples | Avg GPU util % | Max GPU util % | Avg memory GiB | Max memory GiB | Avg memory % | Avg power W | Avg temp C |",
        "| --- | ---: | ---: | ---: | ---: | ---: | ---: | ---: | ---: |",
    ]
    for row in summaries:
        lines.append(
            "| {source} | {samples} | {avg_gpu_util_pct:.1f} | {max_gpu_util_pct:.1f} | "
            "{avg_memory_used_gib:.2f} | {max_memory_used_gib:.2f} | "
            "{avg_memory_used_pct:.1f} | {avg_power_w:.1f} | {avg_temperature_c:.1f} |".format(
                **row
            )
        )
    return "\n".join(lines) + "\n"


def main() -> None:
    args = parse_args()
    start = read_time_file(args.window_start)
    end = read_time_file(args.window_end)

    by_source: dict[str, list[dict[str, float | str]]] = {}
    all_rows: list[dict[str, float | str]] = []
    for path in args.csv_files:
        rows = load_rows(path, start, end)
        if not rows:
            continue
        by_source.setdefault(source_name(path), []).extend(rows)
        all_rows.extend(rows)

    summaries = [summarize(rows, source) for source, rows in sorted(by_source.items())]
    summaries.append(summarize(all_rows, "all"))

    payload = {
        "window_start": start.isoformat() if start else None,
        "window_end": end.isoformat() if end else None,
        "summaries": summaries,
    }
    markdown = to_markdown(summaries)

    if args.output_json:
        args.output_json.write_text(json.dumps(payload, indent=2) + "\n", encoding="utf-8")
    if args.output_md:
        args.output_md.write_text(markdown, encoding="utf-8")

    print(markdown, end="")


if __name__ == "__main__":
    main()
