#!/usr/bin/env python3
"""Summarize blind single-window noise estimates and apply fixed gates."""

from __future__ import annotations

import argparse
import json
from collections import defaultdict
from pathlib import Path

import numpy as np

from rmdm_hvdit_v4_joint.training.engine import write_json_atomic


PRIMARY_SIGMAS = (0.0, 0.005, 0.01, 0.02, 0.03, 0.05)
PRIMARY_METHOD = "calibrated_ensemble_mle"
METHODS = (
    "naive",
    "raw_ensemble_mle",
    "constant_variance_mle",
    "calibrated_ensemble_mle",
    "finite_sample_oracle",
)


def _load_units(root: Path) -> list[dict]:
    units = []
    for path in sorted(root.glob("rank_*/*.json")):
        with path.open("r", encoding="utf-8") as handle:
            units.append(json.load(handle))
    if not units:
        raise ValueError("no estimation units found")
    return units


def _stratified_bootstrap(values: list[dict], seed: int, draws: int) -> dict[str, list[float]]:
    rng = np.random.default_rng(seed)
    scenes: dict[str, dict[str, list[dict]]] = defaultdict(lambda: defaultdict(list))
    for item in values:
        scenes[item["scene_id"]][item["video_id"]].append(item)
    medians, maes = [], []
    for _ in range(draws):
        sample = []
        for videos in scenes.values():
            names = sorted(videos)
            chosen = rng.choice(names, size=len(names), replace=True)
            for name in chosen:
                sample.extend(videos[str(name)])
        estimates = np.asarray([item["methods"][PRIMARY_METHOD]["sigma"] for item in sample])
        sigma = float(sample[0]["true_sigma"])
        medians.append(float(np.median(estimates)))
        maes.append(float(np.abs(estimates - sigma).mean()))
    return {
        "median_estimate_95ci": [float(value) for value in np.quantile(medians, [0.025, 0.975])],
        "mae_95ci": [float(value) for value in np.quantile(maes, [0.025, 0.975])],
    }


def _group_metrics(units: list[dict], bootstrap_seed: int, bootstrap_draws: int) -> dict[str, dict]:
    grouped: dict[tuple[float, float], list[dict]] = defaultdict(list)
    for unit in units:
        grouped[(float(unit["rate"]), float(unit["true_sigma"]))].append(unit)
    result = {}
    for (rate, sigma), values in sorted(grouped.items()):
        estimates = np.asarray(
            [item["methods"][PRIMARY_METHOD]["sigma"] for item in values]
        )
        oracle = np.asarray(
            [item["methods"]["finite_sample_oracle"]["sigma"] for item in values]
        )
        method_metrics = {}
        for method in METHODS:
            method_estimates = np.asarray([item["methods"][method]["sigma"] for item in values])
            method_metrics[method] = {
                "mean_estimate": float(method_estimates.mean()),
                "median_estimate": float(np.median(method_estimates)),
                "bias": float((method_estimates - sigma).mean()),
                "mae": float(np.abs(method_estimates - sigma).mean()),
                "rmse": float(np.sqrt(((method_estimates - sigma) ** 2).mean())),
            }
        result[f"p{rate:g}_sigma{sigma:g}"] = {
            "rate": rate,
            "true_sigma": sigma,
            "windows": len(values),
            "mean_estimate": float(estimates.mean()),
            "median_estimate": float(np.median(estimates)),
            "mae": float(np.abs(estimates - sigma).mean()),
            "rmse": float(np.sqrt(((estimates - sigma) ** 2).mean())),
            "mean_oracle": float(oracle.mean()),
            "methods": method_metrics,
            "paired_video_bootstrap": _stratified_bootstrap(
                values,
                bootstrap_seed + int(rate * 1000) + int(sigma * 1_000_000),
                bootstrap_draws,
            ),
        }
    return result


def _weighted_mean(units: list[dict], field: str) -> float:
    counts = np.asarray([item["observation_count"] for item in units], dtype=np.float64)
    values = np.asarray([item["prior_diagnostics"][field] for item in units])
    return float(np.average(values, weights=counts))


def _gates(units: list[dict], groups: dict[str, dict]) -> dict:
    clean = [item for item in units if float(item["true_sigma"]) == 0.0]
    nonzero = [item for item in units if float(item["true_sigma"]) in PRIMARY_SIGMAS[1:]]
    clean_estimates = np.asarray([item["methods"][PRIMARY_METHOD]["sigma"] for item in clean])
    nonzero_errors = np.asarray(
        [abs(item["methods"][PRIMARY_METHOD]["sigma"] - float(item["true_sigma"])) for item in nonzero]
    )
    rates = sorted({float(item["rate"]) for item in units})
    monotonic = {}
    per_rate_mae = {}
    for rate in rates:
        medians = []
        rate_nonzero = []
        for sigma in PRIMARY_SIGMAS:
            values = [
                item["methods"][PRIMARY_METHOD]["sigma"]
                for item in units
                if float(item["rate"]) == rate and float(item["true_sigma"]) == sigma
            ]
            medians.append(float(np.median(values)))
        monotonic[f"{rate:g}"] = {
            "medians": medians,
            "passed": all(left < right for left, right in zip(medians, medians[1:])),
        }
        for item in nonzero:
            if float(item["rate"]) == rate:
                rate_nonzero.append(
                    abs(item["methods"][PRIMARY_METHOD]["sigma"] - float(item["true_sigma"]))
                )
        per_rate_mae[f"{rate:g}"] = float(np.mean(rate_nonzero))

    pooled_medians = []
    for sigma in PRIMARY_SIGMAS:
        values = [
            item["methods"][PRIMARY_METHOD]["sigma"]
            for item in units
            if float(item["true_sigma"]) == sigma
        ]
        pooled_medians.append(float(np.median(values)))
    x = np.asarray(PRIMARY_SIGMAS)
    y = np.asarray(pooled_medians)
    slope, intercept = np.polyfit(x, y, 1)
    fitted = slope * x + intercept
    r2 = 1.0 - float(((y - fitted) ** 2).sum() / max(((y - y.mean()) ** 2).sum(), 1.0e-12))

    scene_bias = {}
    for scene in sorted({item["scene_id"] for item in clean}):
        scene_bias[scene] = _weighted_mean(
            [item for item in clean if item["scene_id"] == scene], "mean_bias"
        )
    standardized_mean = _weighted_mean(clean, "standardized_mean")
    checks = {
        "clean_floor": float(np.median(clean_estimates)) <= 0.01,
        "strict_monotonic_all_rates": all(item["passed"] for item in monotonic.values()),
        "regression": abs(intercept) <= 0.005 and 0.8 <= slope <= 1.2 and r2 >= 0.95,
        "window_mae": float(nonzero_errors.mean()) <= 0.01,
        "rate_robustness": all(value <= 0.015 for value in per_rate_mae.values()),
        "mean_bias": abs(standardized_mean) <= 0.2
        and all(abs(value) <= 0.01 for value in scene_bias.values()),
    }
    return {
        "checks": checks,
        "passed_primary": all(checks[name] for name in (
            "clean_floor", "strict_monotonic_all_rates", "regression", "window_mae", "mean_bias"
        )),
        "passed_all_rates": all(checks.values()),
        "clean_median_sigma": float(np.median(clean_estimates)),
        "window_mae_nonzero_primary": float(nonzero_errors.mean()),
        "regression": {"slope": float(slope), "intercept": float(intercept), "r2": r2},
        "monotonic": monotonic,
        "per_rate_mae": per_rate_mae,
        "mean_bias": {"standardized_mean": standardized_mean, "by_scene": scene_bias},
    }


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--units-dir", required=True)
    parser.add_argument("--output", required=True)
    parser.add_argument("--bootstrap-draws", type=int, default=2000)
    parser.add_argument("--bootstrap-seed", type=int, default=20260805)
    args = parser.parse_args()
    units = _load_units(Path(args.units_dir))
    groups = _group_metrics(units, args.bootstrap_seed, args.bootstrap_draws)
    em_gaps = np.asarray(
        [item["methods"][PRIMARY_METHOD]["em_mle_gap"] for item in units], dtype=np.float64
    )
    write_json_atomic(
        args.output,
        {
            "schema": "w16_noise_estimation_summary_v1",
            "unit_count": len(units),
            "primary_sigmas": list(PRIMARY_SIGMAS),
            "primary_method": PRIMARY_METHOD,
            "groups": groups,
            "gates": _gates(units, groups),
            "numerical_audit": {
                "maximum_em_mle_variance_gap": float(em_gaps.max()),
                "median_em_mle_variance_gap": float(np.median(em_gaps)),
            },
        },
    )


if __name__ == "__main__":
    main()
