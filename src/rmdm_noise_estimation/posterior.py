"""Gaussian observation correction from a calibrated W16 prior."""

from __future__ import annotations

import torch


def posterior_clean_observations(
    observed: torch.Tensor,
    prior_mean: torch.Tensor,
    prior_variance: torch.Tensor,
    noise_variance: float,
) -> tuple[torch.Tensor, torch.Tensor, torch.Tensor]:
    """Return posterior mean, posterior variance, and measurement weight."""

    if observed.shape != prior_mean.shape or observed.shape != prior_variance.shape:
        raise ValueError("observed values and prior moments must have the same shape")
    if noise_variance < 0:
        raise ValueError("noise variance must be non-negative")
    prior_variance = prior_variance.clamp_min(0.0)
    noise = torch.as_tensor(noise_variance, dtype=observed.dtype, device=observed.device)
    denominator = (prior_variance + noise).clamp_min(1.0e-12)
    weight = prior_variance / denominator
    if noise_variance == 0.0:
        weight = torch.ones_like(weight)
    mean = weight * observed + (1.0 - weight) * prior_mean
    variance = prior_variance * noise / denominator
    return mean, variance, weight


__all__ = ["posterior_clean_observations"]
