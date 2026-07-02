#!/usr/bin/env python3
"""Merge per-shard outputs from sample_temporal_pinn.py."""

from __future__ import annotations

import argparse
import csv
import json
from pathlib import Path

import numpy as np


FIELDS = [
    "clip_index",
    "start",
    "mae",
    "mse",
    "psnr",
    "cal_mae",
    "cal_mse",
    "cal_pinn",
    "cal_kflow",
    "pred_kflow_epe",
    "pred_kflow_l1",
]
METRIC_FIELDS = FIELDS[2:]


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser()
    parser.add_argument("--output_dir", required=True)
    parser.add_argument("--shard_dirs", nargs="+", required=True)
    return parser.parse_args()


def summarize(rows: list[dict[str, float]]) -> dict[str, dict[str, float]]:
    out = {}
    for key in METRIC_FIELDS:
        arr = np.asarray([float(row[key]) for row in rows], dtype=np.float64)
        out[key] = {
            "mean": float(arr.mean()) if arr.size else float("nan"),
            "std": float(arr.std()) if arr.size else float("nan"),
            "count": int(arr.size),
        }
    return out


def read_shard(path: Path) -> tuple[list[dict[str, float]], dict]:
    csv_path = path / "per_clip.csv"
    if not csv_path.exists():
        raise FileNotFoundError(csv_path)
    rows: list[dict[str, float]] = []
    with csv_path.open("r", newline="", encoding="utf-8") as handle:
        reader = csv.DictReader(handle)
        for row in reader:
            parsed: dict[str, float] = {}
            for key in FIELDS:
                if key in ("clip_index", "start"):
                    parsed[key] = int(row[key])
                else:
                    parsed[key] = float(row[key])
            rows.append(parsed)
    summary_path = path / "summary.json"
    summary = json.loads(summary_path.read_text(encoding="utf-8")) if summary_path.exists() else {}
    return rows, summary


def main() -> None:
    args = parse_args()
    out_dir = Path(args.output_dir)
    out_dir.mkdir(parents=True, exist_ok=True)

    merged_rows: list[dict[str, float]] = []
    shard_summaries = []
    for shard_dir_arg in args.shard_dirs:
        shard_dir = Path(shard_dir_arg)
        rows, summary = read_shard(shard_dir)
        offset = int(summary.get("start_index", 0))
        for row in rows:
            row = dict(row)
            row["source_shard"] = str(shard_dir)
            row["global_clip_index"] = offset + int(row["clip_index"])
            merged_rows.append(row)
        shard_summaries.append({"path": str(shard_dir), "summary": summary, "rows": len(rows)})

    merged_rows.sort(key=lambda row: int(row["global_clip_index"]))
    fieldnames = ["global_clip_index", "source_shard"] + FIELDS
    csv_path = out_dir / "per_clip_merged.csv"
    with csv_path.open("w", newline="", encoding="utf-8") as handle:
        writer = csv.DictWriter(handle, fieldnames=fieldnames)
        writer.writeheader()
        writer.writerows(merged_rows)

    summary = {
        "num_clips": len(merged_rows),
        "metrics": summarize(merged_rows),
        "shards": shard_summaries,
        "merged_csv": str(csv_path),
    }
    summary_path = out_dir / "summary_merged.json"
    summary_path.write_text(json.dumps(summary, indent=2), encoding="utf-8")
    print(json.dumps(summary, indent=2), flush=True)


if __name__ == "__main__":
    main()
