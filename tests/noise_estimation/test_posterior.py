from __future__ import annotations

import torch

from rmdm_noise_estimation.posterior import posterior_clean_observations


def test_posterior_matches_gaussian_update() -> None:
    observed = torch.tensor([-60.0])
    prior_mean = torch.tensor([-70.0])
    prior_variance = torch.tensor([4.0])

    mean, variance, weight = posterior_clean_observations(
        observed, prior_mean, prior_variance, noise_variance=16.0
    )

    assert torch.allclose(mean, torch.tensor([-68.0]))
    assert torch.allclose(variance, torch.tensor([3.2]))
    assert torch.allclose(weight, torch.tensor([0.2]))


def test_zero_noise_keeps_measurement_even_with_zero_prior_variance() -> None:
    observed = torch.tensor([0.4, 0.8])
    prior_mean = torch.tensor([0.2, 0.1])
    prior_variance = torch.tensor([0.0, 1.0])

    mean, variance, weight = posterior_clean_observations(
        observed, prior_mean, prior_variance, noise_variance=0.0
    )

    assert torch.equal(mean, observed)
    assert torch.count_nonzero(variance) == 0
    assert torch.equal(weight, torch.ones_like(weight))
