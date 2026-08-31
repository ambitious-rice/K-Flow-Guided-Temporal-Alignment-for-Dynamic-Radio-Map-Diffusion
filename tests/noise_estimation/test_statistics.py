from __future__ import annotations

import numpy as np

from rmdm_noise_estimation.calibration import fit_variance_calibration
from rmdm_noise_estimation.statistics import em_variance, estimate_window_noise, marginal_mle


def test_marginal_mle_recovers_known_noise_and_matches_em() -> None:
    rng = np.random.default_rng(5)
    count = 200_000
    prior_variance = rng.uniform(0.0002, 0.002, size=count)
    sensor_variance = 0.03**2
    residual = rng.normal(size=count) * np.sqrt(prior_variance + sensor_variance)

    result = estimate_window_noise(
        observed=residual,
        prior_mean=np.zeros_like(residual),
        prior_variance=prior_variance,
    )

    assert abs(result.sigma - 0.03) < 5.0e-4
    assert abs(result.em_sigma - result.sigma) < 5.0e-4


def test_em_reaches_slow_near_boundary_mle() -> None:
    rng = np.random.default_rng(0)
    prior_variance = np.exp(rng.uniform(np.log(1.0e-4), np.log(2.0e-2), size=100))
    squared_residual = rng.normal(size=100) ** 2 * (prior_variance + 1.25e-4)

    mle_variance, _ = marginal_mle(squared_residual, prior_variance)
    em_estimate, iterations = em_variance(squared_residual, prior_variance)

    assert iterations > 500
    assert abs(em_estimate - mle_variance) < 2.0e-8


def test_affine_variance_calibration_uses_equal_rate_objective() -> None:
    rng = np.random.default_rng(8)
    raw = {
        "1": rng.uniform(0.0001, 0.001, size=20_000),
        "5": rng.uniform(0.0001, 0.001, size=2_000),
    }
    squared = {}
    for rate, values in raw.items():
        true_variance = 2.5 * values + 0.0003
        squared[rate] = rng.normal(size=values.size) ** 2 * true_variance

    calibration = fit_variance_calibration(raw, squared)

    assert abs(calibration.scale - 2.5) < 0.25
    assert abs(calibration.offset - 0.0003) < 8.0e-5


def test_affine_variance_calibration_handles_tiny_raw_variances() -> None:
    rng = np.random.default_rng(12)
    raw = {
        "1": rng.uniform(0.2e-6, 8.0e-6, size=30_000),
        "10": rng.uniform(0.2e-6, 8.0e-6, size=30_000),
    }
    squared = {}
    for rate, values in raw.items():
        true_variance = 400.0 * values + 0.0001
        squared[rate] = rng.normal(size=values.size) ** 2 * true_variance

    calibration = fit_variance_calibration(raw, squared)

    assert abs(calibration.scale - 400.0) < 20.0
    assert abs(calibration.offset - 0.0001) < 3.0e-5
