#!/usr/bin/env python3
"""Fit the clean-only W16 ensemble-variance calibration."""

from __future__ import annotations

import argparse
from collections import defaultdict
from pathlib import Path

import numpy as np
import torch

from rmdm_hvdit_v4_joint.training.engine import write_json_atomic
from rmdm_noise_estimation.calibration import fit_variance_calibration


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--units-dir", required=True)
    parser.add_argument("--output", required=True)
    args = parser.parse_args()

    raw_by_rate: dict[str, list[np.ndarray]] = defaultdict(list)
    squared_by_rate: dict[str, list[np.ndarray]] = defaultdict(list)
    unit_count = 0
    for path in sorted(Path(args.units_dir).glob("rank_*/*.pt")):
        unit = torch.load(path, map_location="cpu", weights_only=False)
        if float(unit["true_sigma"]) != 0.0:
            raise ValueError(f"calibration unit is not clean: {path}")
        rate = f"{float(unit['rate']):g}"
        raw_by_rate[rate].append(unit["raw_variance"].numpy())
        residual = unit["target"].numpy() - unit["prior_mean"].numpy()
        squared_by_rate[rate].append(residual**2)
        unit_count += 1
    if unit_count == 0:
        raise ValueError("no calibration units found")

    raw = {rate: np.concatenate(values) for rate, values in raw_by_rate.items()}
    squared = {rate: np.concatenate(squared_by_rate[rate]) for rate in raw}
    calibration = fit_variance_calibration(raw, squared)
    calibrated_residuals = []
    standardized = []
    rate_diagnostics = {}
    for rate in sorted(raw, key=float):
        variance = calibration.apply(raw[rate])
        residual = squared[rate] ** 0.5
        standardized_rate = residual / variance**0.5
        calibrated_residuals.append(float(np.mean(squared[rate])))
        standardized.append(standardized_rate)
        rate_diagnostics[rate] = {
            "points": int(raw[rate].size),
            "mean_raw_variance": float(raw[rate].mean()),
            "mean_squared_residual": float(squared[rate].mean()),
            "standardized_second_moment": float((standardized_rate**2).mean()),
        }
    all_standardized = np.concatenate(standardized)
    write_json_atomic(
        args.output,
        {
            "schema": "w16_noise_variance_calibration_v1",
            "unit_count": unit_count,
            "variance_calibration": calibration.to_dict(),
            "constant_variance": float(np.mean(calibrated_residuals)),
            "equal_rate_weighting": True,
            "clean_only": True,
            "rates": rate_diagnostics,
            "diagnostics": {
                "standardized_second_moment": float((all_standardized**2).mean()),
                "coverage_68": float((all_standardized <= 1.0).mean()),
                "coverage_95": float((all_standardized <= 1.96).mean()),
            },
        },
    )


if __name__ == "__main__":
    main()
