from __future__ import annotations

import torch

from rmdm_noise_estimation.correction import corrected_sparse_batch


def test_corrected_batch_only_changes_observed_indices() -> None:
    observed_grid = torch.zeros((1, 1, 1, 2, 3))
    observed_grid.reshape(-1)[torch.tensor([1, 5])] = torch.tensor([0.8, 0.2])
    sparse = {"observed_rss": observed_grid, "sampling_mask": (observed_grid != 0).float()}
    vectors = {
        "flat_indices": torch.tensor([1, 5]),
        "observed": torch.tensor([0.8, 0.2]),
        "prior_mean": torch.tensor([0.4, 0.6]),
        "raw_variance": torch.tensor([0.01, 0.01]),
    }

    corrected, diagnostics = corrected_sparse_batch(
        sparse,
        vectors,
        {"scale": 1.0, "offset": 0.0, "floor": 1.0e-8},
        noise_variance=0.01,
    )

    assert torch.allclose(corrected["observed_rss"].reshape(-1)[[1, 5]], torch.tensor([0.6, 0.4]))
    assert torch.count_nonzero(corrected["observed_rss"].reshape(-1)[[0, 2, 3, 4]]) == 0
    assert diagnostics["mean_measurement_weight"] == 0.5


def test_constant_variance_and_soft_gate_are_supported() -> None:
    observed_grid = torch.zeros((1, 1, 1, 1, 2))
    observed_grid.reshape(-1)[1] = 0.8
    sparse = {"observed_rss": observed_grid}
    vectors = {
        "flat_indices": torch.tensor([1]),
        "observed": torch.tensor([0.8]),
        "prior_mean": torch.tensor([0.4]),
        "raw_variance": torch.tensor([100.0]),
    }

    corrected, diagnostics = corrected_sparse_batch(
        sparse,
        vectors,
        {"scale": 1.0, "offset": 0.0},
        noise_variance=0.01,
        prior_variance_mode="constant",
        constant_prior_variance=0.01,
        correction_alpha=0.5,
    )

    assert torch.allclose(corrected["observed_rss"].reshape(-1)[1], torch.tensor(0.7))
    assert diagnostics["prior_variance_mode"] == "constant"
    assert diagnostics["correction_alpha"] == 0.5
