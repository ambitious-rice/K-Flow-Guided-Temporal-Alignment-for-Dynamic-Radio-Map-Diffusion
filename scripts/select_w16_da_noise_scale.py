#!/usr/bin/env python3
"""Select one estimated-noise DA scale from a calibration split."""

from __future__ import annotations

import argparse
import json
from collections import defaultdict
from pathlib import Path

from rmdm.evaluation.metrics import MetricAccumulator
from rmdm_hvdit_v4_joint.training.engine import write_json_atomic


def _load(root: Path) -> list[dict]:
    values = []
    for path in sorted(root.glob("rank_*/*.json")):
        with path.open(encoding="utf-8") as handle:
            values.append(json.load(handle))
    if not values:
        raise ValueError("no DA calibration units found")
    return values


def _aggregate(values: list[dict], method: str) -> dict:
    accumulator = MetricAccumulator()
    for value in values:
        accumulator.add_raw(value["methods"][method]["raw"])
    return accumulator.compute()


def _scale_from_method(name: str) -> float | None:
    if name == "estimated_noise_da":
        return 1.0
    prefix = "estimated_noise_da_scale"
    if name.startswith(prefix):
        return float(name.removeprefix(prefix).replace("p", "."))
    return None


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--units-dir", required=True)
    parser.add_argument("--output", required=True)
    parser.add_argument("--rates", required=True)
    parser.add_argument("--true-sigma", type=float, required=True)
    args = parser.parse_args()
    selected_rates = {float(item) for item in args.rates.split(",") if item}
    units = [
        unit for unit in _load(Path(args.units_dir))
        if float(unit["rate"]) in selected_rates and float(unit["true_sigma"]) == args.true_sigma
    ]
    if not units:
        raise ValueError("no units match the requested rates and true sigma")
    by_rate: dict[float, list[dict]] = defaultdict(list)
    for unit in units:
        by_rate[float(unit["rate"])].append(unit)
    if set(by_rate) != selected_rates:
        raise ValueError("every selected rate must have calibration units")

    methods = [name for name in units[0]["methods"] if _scale_from_method(name) is not None]
    if not methods:
        raise ValueError("units do not include estimated-noise DA methods")
    scores = {}
    for method in methods:
        per_rate = {
            f"p{rate:g}": _aggregate(values, method)
            for rate, values in sorted(by_rate.items())
        }
        scores[method] = {
            "scale": _scale_from_method(method),
            "macro_full_image_nmse": sum(
                value["full_image"]["nmse"] for value in per_rate.values()
            ) / len(per_rate),
            "macro_full_image_psnr": sum(
                value["full_image"]["psnr"] for value in per_rate.values()
            ) / len(per_rate),
            "per_rate": per_rate,
        }
    selected_method = min(scores, key=lambda name: scores[name]["macro_full_image_nmse"])
    write_json_atomic(
        args.output,
        {
            "schema": "w16_da_noise_scale_selection_v1",
            "selection_split": "stage_a",
            "true_sigma": args.true_sigma,
            "rates": sorted(selected_rates),
            "unit_count": len(units),
            "candidates": scores,
            "selected_method": selected_method,
            "selected_scale": scores[selected_method]["scale"],
        },
    )


if __name__ == "__main__":
    main()
