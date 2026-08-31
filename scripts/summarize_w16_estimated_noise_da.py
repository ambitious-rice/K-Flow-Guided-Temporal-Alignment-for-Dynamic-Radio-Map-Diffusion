#!/usr/bin/env python3
"""Aggregate estimated-noise DA metrics with paired-video bootstrap intervals."""

from __future__ import annotations

import argparse
import json
from collections import defaultdict
from pathlib import Path

import numpy as np

from rmdm.evaluation.metrics import MetricAccumulator
from rmdm_hvdit_v4_joint.training.engine import write_json_atomic


DOMAINS = ("full_image", "unobserved_free_space")


def _load(root: Path) -> list[dict]:
    units = []
    for path in sorted(root.glob("rank_*/*.json")):
        with path.open("r", encoding="utf-8") as handle:
            units.append(json.load(handle))
    if not units:
        raise ValueError("no estimated-noise DA units found")
    return units


def _aggregate(values: list[dict], method: str) -> dict:
    accumulator = MetricAccumulator()
    for item in values:
        accumulator.add_raw(item["methods"][method]["raw"])
    return {"metrics": accumulator.compute(), "raw": accumulator.raw()}


def _bootstrap(values: list[dict], method: str, domain: str, seed: int, draws: int) -> list[float]:
    scenes: dict[str, dict[str, list[dict]]] = defaultdict(lambda: defaultdict(list))
    for item in values:
        scenes[item["scene_id"]][item["video_id"]].append(item)
    rng = np.random.default_rng(seed)
    deltas = []
    for _ in range(draws):
        sample = []
        for videos in scenes.values():
            names = sorted(videos)
            for name in rng.choice(names, size=len(names), replace=True):
                sample.extend(videos[str(name)])
        baseline = _aggregate(sample, "no_da")["metrics"][domain]["nmse"]
        candidate = _aggregate(sample, method)["metrics"][domain]["nmse"]
        deltas.append(candidate - baseline)
    return [float(value) for value in np.quantile(deltas, [0.025, 0.975])]


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--units-dir", required=True)
    parser.add_argument("--output", required=True)
    parser.add_argument("--bootstrap-draws", type=int, default=2000)
    parser.add_argument("--bootstrap-seed", type=int, default=20260805)
    args = parser.parse_args()
    units = _load(Path(args.units_dir))
    method_set = set(units[0]["methods"])
    if "no_da" not in method_set:
        raise ValueError("units must include no_da")
    if any(set(unit["methods"]) != method_set for unit in units):
        raise ValueError("all units must have the same method set")
    methods = ("no_da", *sorted(method_set - {"no_da"}))
    grouped: dict[tuple[float, float], list[dict]] = defaultdict(list)
    for unit in units:
        grouped[(float(unit["rate"]), float(unit["true_sigma"]))].append(unit)

    groups = {}
    for group_index, ((rate, sigma), values) in enumerate(sorted(grouped.items())):
        aggregates = {method: _aggregate(values, method) for method in methods}
        comparisons = {}
        for method in methods[1:]:
            comparisons[method] = {}
            for domain_index, domain in enumerate(DOMAINS):
                baseline = aggregates["no_da"]["metrics"][domain]["nmse"]
                candidate = aggregates[method]["metrics"][domain]["nmse"]
                comparisons[method][domain] = {
                    "delta_nmse": candidate - baseline,
                    "delta_nmse_95ci": _bootstrap(
                        values, method, domain,
                        args.bootstrap_seed + group_index * 10 + domain_index,
                        args.bootstrap_draws,
                    ),
                }
        groups[f"p{rate:g}_sigma{sigma:g}"] = {
            "rate": rate,
            "true_sigma": sigma,
            "windows": len(values),
            "methods": aggregates,
            "comparisons_vs_no_da": comparisons,
            "mean_estimated_sigma": float(np.mean([item["estimated_sigma"] for item in values])),
        }
    write_json_atomic(
        args.output,
        {
            "schema": "w16_estimated_noise_da_summary_v1",
            "unit_count": len(units),
            "methods": list(methods),
            "groups": groups,
        },
    )


if __name__ == "__main__":
    main()
