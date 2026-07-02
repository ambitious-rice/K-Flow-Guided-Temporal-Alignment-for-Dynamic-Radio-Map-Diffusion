#!/usr/bin/env python3
"""Merge per-shard full evaluation CSV/JSON outputs."""

import argparse
import csv
import json
from pathlib import Path

import numpy as np


def parse_args():
    parser = argparse.ArgumentParser(description="Merge full-eval shard outputs.")
    parser.add_argument("--eval_dir", required=True)
    parser.add_argument("--split", default="test")
    parser.add_argument("--num_shards", type=int, default=2)
    return parser.parse_args()


def summarize(values):
    arr = np.asarray([v for v in values if np.isfinite(v)], dtype=np.float64)
    if arr.size == 0:
        return {"mean": float("nan"), "std": float("nan"), "min": float("nan"), "max": float("nan"), "count": 0}
    return {
        "mean": float(arr.mean()),
        "std": float(arr.std()),
        "min": float(arr.min()),
        "max": float(arr.max()),
        "count": int(arr.size),
    }


def main():
    args = parse_args()
    root = Path(args.eval_dir)
    rows = []
    metric_names = ["MSE", "MAE", "RMSE", "NMSE", "PSNR", "SSIM"]
    metrics = {name: [] for name in metric_names}

    for shard_rank in range(args.num_shards):
        path = root / f"details_{args.split}_shard{shard_rank}of{args.num_shards}.csv"
        if not path.exists():
            raise FileNotFoundError(path)
        with path.open() as f:
            reader = csv.DictReader(f)
            for row in reader:
                rows.append(row)
                for name in metric_names:
                    metrics[name].append(float(row[name]))

    rows.sort(key=lambda row: int(row["global_index"]))
    merged_csv = root / f"details_{args.split}_merged.csv"
    with merged_csv.open("w", newline="") as f:
        writer = csv.DictWriter(f, fieldnames=list(rows[0].keys()))
        writer.writeheader()
        writer.writerows(rows)

    summaries = []
    for shard_rank in range(args.num_shards):
        path = root / f"summary_{args.split}_shard{shard_rank}of{args.num_shards}.json"
        if path.exists():
            summaries.append(json.loads(path.read_text()))

    merged = {
        "split": args.split,
        "num_shards": args.num_shards,
        "total_samples": len(rows),
        "metrics": {name: summarize(values) for name, values in metrics.items()},
        "shard_summaries": summaries,
        "merged_details_csv": str(merged_csv),
    }
    merged_json = root / f"summary_{args.split}_merged.json"
    merged_json.write_text(json.dumps(merged, indent=2))
    print(json.dumps(merged, indent=2))


if __name__ == "__main__":
    main()
