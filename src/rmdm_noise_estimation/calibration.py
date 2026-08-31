"""Clean-only affine calibration for DDIM ensemble variance."""

from __future__ import annotations

from dataclasses import asdict, dataclass

import numpy as np
from scipy.optimize import minimize


@dataclass(frozen=True)
class VarianceCalibration:
    scale: float
    offset: float
    floor: float = 1.0e-8

    def apply(self, raw_variance: np.ndarray) -> np.ndarray:
        return np.maximum(self.scale * np.asarray(raw_variance) + self.offset, self.floor)

    def to_dict(self) -> dict[str, float]:
        return asdict(self)


def fit_variance_calibration(
    raw_variance_by_rate: dict[str, np.ndarray],
    squared_residual_by_rate: dict[str, np.ndarray],
    *,
    floor: float = 1.0e-8,
) -> VarianceCalibration:
    """Fit one affine calibration with equal weight for every sampling rate."""

    rates = sorted(raw_variance_by_rate, key=float)
    if not rates or set(rates) != set(squared_residual_by_rate):
        raise ValueError("raw variance and residual dictionaries must share non-empty rates")
    pairs = []
    for rate in rates:
        raw = np.asarray(raw_variance_by_rate[rate], dtype=np.float64).reshape(-1)
        squared = np.asarray(squared_residual_by_rate[rate], dtype=np.float64).reshape(-1)
        if raw.size == 0 or raw.shape != squared.shape:
            raise ValueError(f"invalid calibration arrays for rate {rate}")
        pairs.append((raw, squared))

    initial_offset = max(float(np.mean(np.concatenate([item[1] for item in pairs]))), floor)
    raw_reference = max(float(np.mean(np.concatenate([item[0] for item in pairs]))), floor)

    def objective(parameters: np.ndarray) -> float:
        scaled_coefficient, offset = parameters
        values = []
        for raw, squared in pairs:
            variance = np.maximum(scaled_coefficient * (raw / raw_reference) + offset, floor)
            values.append(float(np.mean(np.log(variance) + squared / variance)))
        return float(np.mean(values))

    result = minimize(
        objective,
        x0=np.asarray([raw_reference, initial_offset], dtype=np.float64),
        method="L-BFGS-B",
        bounds=((0.0, None), (0.0, None)),
        options={"ftol": 1.0e-15, "gtol": 1.0e-10, "maxiter": 1000},
    )
    if not result.success:
        raise RuntimeError(f"variance calibration failed: {result.message}")
    return VarianceCalibration(
        float(result.x[0] / raw_reference), float(result.x[1]), float(floor)
    )


__all__ = ["VarianceCalibration", "fit_variance_calibration"]
