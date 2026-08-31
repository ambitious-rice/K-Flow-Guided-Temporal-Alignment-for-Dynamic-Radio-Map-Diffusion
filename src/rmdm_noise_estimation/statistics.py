"""One-window Gaussian marginal MLE and EM diagnostics."""

from __future__ import annotations

import math
from dataclasses import dataclass

import numpy as np


@dataclass(frozen=True)
class NoiseEstimate:
    sigma: float
    variance: float
    em_sigma: float
    em_variance: float
    em_mle_gap: float
    nll: float
    iterations: int


def marginal_nll(variance: float, squared_residual: np.ndarray, prior_variance: np.ndarray) -> float:
    total = np.maximum(prior_variance + float(variance), 1.0e-12)
    return float(0.5 * np.mean(np.log(total) + squared_residual / total))


def _golden_minimize(function, lower: float, upper: float, iterations: int = 96) -> tuple[float, float]:
    ratio = (math.sqrt(5.0) - 1.0) / 2.0
    left, right = float(lower), float(upper)
    x1 = right - ratio * (right - left)
    x2 = left + ratio * (right - left)
    f1, f2 = function(x1), function(x2)
    for _ in range(iterations):
        if f1 <= f2:
            right, x2, f2 = x2, x1, f1
            x1 = right - ratio * (right - left)
            f1 = function(x1)
        else:
            left, x1, f1 = x1, x2, f2
            x2 = left + ratio * (right - left)
            f2 = function(x2)
    point = 0.5 * (left + right)
    return point, function(point)


def marginal_mle(
    squared_residual: np.ndarray,
    prior_variance: np.ndarray,
    *,
    maximum_variance: float = 0.25,
) -> tuple[float, float]:
    squared = np.asarray(squared_residual, dtype=np.float64).reshape(-1)
    prior = np.asarray(prior_variance, dtype=np.float64).reshape(-1)
    if squared.size == 0 or squared.shape != prior.shape:
        raise ValueError("residual and variance arrays must be non-empty and aligned")
    def objective(value: float) -> float:
        return marginal_nll(value, squared, prior)
    interior, interior_value = _golden_minimize(objective, 0.0, maximum_variance)
    candidates = [(0.0, objective(0.0)), (interior, interior_value), (maximum_variance, objective(maximum_variance))]
    return min(candidates, key=lambda item: item[1])


def em_variance(
    squared_residual: np.ndarray,
    prior_variance: np.ndarray,
    *,
    tolerance: float = 1.0e-10,
    maximum_iterations: int = 2000,
) -> tuple[float, int]:
    squared = np.asarray(squared_residual, dtype=np.float64).reshape(-1)
    prior = np.maximum(np.asarray(prior_variance, dtype=np.float64).reshape(-1), 1.0e-12)
    value = max(float(np.mean(squared - prior)), 1.0e-6)
    for iteration in range(1, maximum_iterations + 1):
        denominator = prior + value
        updated = float(np.mean((value * value / denominator**2) * squared + prior * value / denominator))
        if abs(updated - value) <= tolerance * max(1.0, value):
            return max(updated, 0.0), iteration
        value = max(updated, 1.0e-12)
    return value, maximum_iterations


def estimate_window_noise(
    observed: np.ndarray,
    prior_mean: np.ndarray,
    prior_variance: np.ndarray,
    *,
    maximum_variance: float = 0.25,
) -> NoiseEstimate:
    residual = np.asarray(observed, dtype=np.float64).reshape(-1) - np.asarray(
        prior_mean, dtype=np.float64
    ).reshape(-1)
    squared = residual**2
    variance, nll = marginal_mle(squared, prior_variance, maximum_variance=maximum_variance)
    em_value, iterations = em_variance(squared, prior_variance)
    return NoiseEstimate(
        sigma=math.sqrt(variance),
        variance=variance,
        em_sigma=math.sqrt(em_value),
        em_variance=em_value,
        em_mle_gap=abs(em_value - variance),
        nll=nll,
        iterations=iterations,
    )


__all__ = [
    "NoiseEstimate",
    "em_variance",
    "estimate_window_noise",
    "marginal_mle",
    "marginal_nll",
]
