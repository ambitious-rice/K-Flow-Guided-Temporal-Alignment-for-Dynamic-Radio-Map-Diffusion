#!/usr/bin/env python3
"""Compare two sample_temporal_pinn.py summary files."""

from __future__ import annotations

import argparse
import json
from pathlib import Path


PRIMARY = ("mae", "mse", "psnr")
SECONDARY = ("cal_mae", "cal_mse", "cal_pinn", "cal_kflow", "pred_kflow_epe", "pred_kflow_l1")


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser()
    parser.add_argument("--baseline", required=True)
    parser.add_argument("--candidate", required=True)
    parser.add_argument("--output", required=True)
    return parser.parse_args()


def metric(summary: dict, name: str) -> float:
    return float(summary["metrics"][name]["mean"])


def main() -> None:
    args = parse_args()
    baseline = json.loads(Path(args.baseline).read_text(encoding="utf-8"))
    candidate = json.loads(Path(args.candidate).read_text(encoding="utf-8"))
    comparison = {
        "baseline": args.baseline,
        "candidate": args.candidate,
        "baseline_num_clips": int(baseline.get("num_clips", -1)),
        "candidate_num_clips": int(candidate.get("num_clips", -1)),
        "primary_metrics": {},
        "secondary_metrics": {},
    }
    for group, names in (("primary_metrics", PRIMARY), ("secondary_metrics", SECONDARY)):
        for name in names:
            b_value = metric(baseline, name)
            c_value = metric(candidate, name)
            comparison[group][name] = {
                "baseline": b_value,
                "candidate": c_value,
                "delta": c_value - b_value,
            }
    output = Path(args.output)
    output.parent.mkdir(parents=True, exist_ok=True)
    output.write_text(json.dumps(comparison, indent=2), encoding="utf-8")
    print(json.dumps(comparison, indent=2), flush=True)


if __name__ == "__main__":
    main()
