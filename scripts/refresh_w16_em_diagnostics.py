#!/usr/bin/env python3
"""Refresh EM consistency diagnostics from saved cross-fit vectors."""

from __future__ import annotations

import argparse
import json
from pathlib import Path

import numpy as np
import torch

from rmdm_hvdit_v4_joint.training.engine import write_json_atomic
from rmdm_noise_estimation.statistics import em_variance


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--evaluation-dir", required=True)
    parser.add_argument("--calibration", required=True)
    args = parser.parse_args()

    root = Path(args.evaluation_dir)
    calibration = json.loads(Path(args.calibration).read_text(encoding="utf-8"))[
        "variance_calibration"
    ]
    refreshed = 0
    for unit_path in sorted((root / "units").glob("rank_*/*.json")):
        vector_path = root / "vectors" / unit_path.parent.name / f"{unit_path.stem}.pt"
        unit = json.loads(unit_path.read_text(encoding="utf-8"))
        vectors = torch.load(vector_path, map_location="cpu", weights_only=False)
        observed = vectors["observed"].double().numpy()
        prior_mean = vectors["prior_mean"].double().numpy()
        raw_variance = vectors["raw_variance"].double().numpy()
        prior_variance = np.maximum(
            float(calibration["scale"]) * raw_variance + float(calibration["offset"]),
            float(calibration.get("floor", 1.0e-8)),
        )
        em_value, iterations = em_variance((observed - prior_mean) ** 2, prior_variance)
        estimate = unit["methods"]["calibrated_ensemble_mle"]
        estimate["em_variance"] = em_value
        estimate["em_sigma"] = em_value**0.5
        estimate["em_mle_gap"] = abs(em_value - float(estimate["variance"]))
        estimate["iterations"] = iterations
        write_json_atomic(unit_path, unit)
        refreshed += 1
    print(f"refreshed {refreshed} EM diagnostics")


if __name__ == "__main__":
    main()
