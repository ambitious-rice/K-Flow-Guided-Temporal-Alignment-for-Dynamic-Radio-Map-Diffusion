"""Rebuild corrected sparse W16 conditions from saved cross-fit vectors."""

from __future__ import annotations

from typing import Any

import torch

from .posterior import posterior_clean_observations


def corrected_sparse_batch(
    sparse_batch: dict[str, Any],
    vectors: dict[str, Any],
    variance_calibration: dict[str, float],
    *,
    noise_variance: float,
    prior_variance_mode: str = "pointwise",
    constant_prior_variance: float | None = None,
    correction_alpha: float = 1.0,
) -> tuple[dict[str, Any], dict[str, float]]:
    """Build a posterior-corrected condition from cross-fitted moments.

    ``prior_variance_mode='constant'`` is the no-per-observation-variance
    ablation. ``correction_alpha`` linearly blends the posterior mean back to
    the original measurement, so zero leaves the condition unchanged.
    """

    device = sparse_batch["observed_rss"].device
    indices = vectors["flat_indices"].to(device=device, dtype=torch.long)
    observed = vectors["observed"].to(device=device, dtype=torch.float32)
    prior_mean = vectors["prior_mean"].to(device=device, dtype=torch.float32)
    raw_variance = vectors["raw_variance"].to(device=device, dtype=torch.float32)
    pointwise_variance = (
        float(variance_calibration["scale"]) * raw_variance
        + float(variance_calibration["offset"])
    ).clamp_min(float(variance_calibration.get("floor", 1.0e-8)))
    if prior_variance_mode == "pointwise":
        prior_variance = pointwise_variance
    elif prior_variance_mode == "constant":
        if constant_prior_variance is None or constant_prior_variance <= 0.0:
            raise ValueError("constant_prior_variance must be positive for constant mode")
        prior_variance = torch.full_like(observed, float(constant_prior_variance))
    else:
        raise ValueError(f"unknown prior_variance_mode: {prior_variance_mode}")
    if not 0.0 <= correction_alpha <= 1.0:
        raise ValueError("correction_alpha must be in [0, 1]")
    posterior_mean, posterior_variance, weight = posterior_clean_observations(
        observed,
        prior_mean,
        prior_variance,
        noise_variance,
    )
    corrected = dict(sparse_batch)
    corrected_values = sparse_batch["observed_rss"].clone()
    conditioned_mean = observed + correction_alpha * (posterior_mean - observed)
    corrected_values.reshape(-1)[indices] = conditioned_mean
    corrected["observed_rss"] = corrected_values
    return corrected, {
        "mean_measurement_weight": float(weight.mean()),
        "mean_posterior_variance": float(posterior_variance.mean()),
        "mean_absolute_correction": float((conditioned_mean - observed).abs().mean()),
        "correction_alpha": float(correction_alpha),
        "prior_variance_mode": prior_variance_mode,
    }


__all__ = ["corrected_sparse_batch"]
