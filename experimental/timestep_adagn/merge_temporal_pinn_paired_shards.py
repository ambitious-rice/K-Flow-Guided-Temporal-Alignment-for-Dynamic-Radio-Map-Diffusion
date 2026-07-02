#!/usr/bin/env python3
"""Merge shard outputs from sample_temporal_pinn_paired.py."""

from __future__ import annotations

import _paths  # noqa: F401

import argparse
import csv
import json
from pathlib import Path

import numpy as np


BASE_FIELDS = ["global_clip_index", "source_shard", "clip_index", "start"]


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser()
    parser.add_argument("--output_dir", required=True)
    parser.add_argument("--shard_dirs", nargs="+", required=True)
    return parser.parse_args()


def summarize(rows: list[dict[str, float]], metrics: list[str]) -> dict[str, dict[str, float]]:
    out = {}
    for key in metrics:
        arr = np.asarray([float(row[key]) for row in rows], dtype=np.float64)
        out[key] = {
            "mean": float(arr.mean()) if arr.size else float("nan"),
            "std": float(arr.std()) if arr.size else float("nan"),
            "count": int(arr.size),
        }
    return out


def main() -> None:
    args = parse_args()
    out_dir = Path(args.output_dir)
    out_dir.mkdir(parents=True, exist_ok=True)
    merged_rows: list[dict[str, float]] = []
    shard_summaries = []
    metrics: list[str] | None = None
    for shard_arg in args.shard_dirs:
        shard_dir = Path(shard_arg)
        csv_path = shard_dir / "per_clip_paired.csv"
        summary_path = shard_dir / "summary_paired.json"
        if not csv_path.exists():
            raise FileNotFoundError(csv_path)
        summary = json.loads(summary_path.read_text(encoding="utf-8")) if summary_path.exists() else {}
        offset = int(summary.get("start_index", 0))
        with csv_path.open("r", newline="", encoding="utf-8") as handle:
            reader = csv.DictReader(handle)
            if reader.fieldnames is None:
                raise ValueError(f"No CSV header found in {csv_path}")
            current_metrics = [name for name in reader.fieldnames if name not in {"clip_index", "start"}]
            if metrics is None:
                metrics = current_metrics
            elif metrics != current_metrics:
                raise ValueError(f"Metric columns differ in {csv_path}: {current_metrics} vs {metrics}")
            rows = list(reader)
        for row in rows:
            clip_index = int(row["clip_index"])
            global_clip_index = clip_index if clip_index >= offset else offset + clip_index
            parsed = {
                "global_clip_index": global_clip_index,
                "source_shard": str(shard_dir),
                "clip_index": clip_index,
                "start": int(row["start"]),
            }
            for key in metrics:
                parsed[key] = float(row[key])
            merged_rows.append(parsed)
        shard_summaries.append({"path": str(shard_dir), "rows": len(rows), "summary": summary})

    metrics = metrics or []
    fields = BASE_FIELDS + metrics
    merged_rows.sort(key=lambda row: int(row["global_clip_index"]))
    csv_out = out_dir / "per_clip_paired_merged.csv"
    with csv_out.open("w", newline="", encoding="utf-8") as handle:
        writer = csv.DictWriter(handle, fieldnames=fields)
        writer.writeheader()
        writer.writerows(merged_rows)

    summary = {
        "num_clips": len(merged_rows),
        "metrics": summarize(merged_rows, metrics),
        "shards": shard_summaries,
        "merged_csv": str(csv_out),
    }
    summary_out = out_dir / "summary_paired_merged.json"
    summary_out.write_text(json.dumps(summary, indent=2), encoding="utf-8")
    print(json.dumps(summary, indent=2), flush=True)


if __name__ == "__main__":
    main()
