from __future__ import annotations

import torch

from rmdm_noise_estimation.ensemble import ensemble_initial_noise


def test_ensemble_noise_is_paired_across_measurement_sigmas_by_construction() -> None:
    reference = torch.zeros((2, 1, 3, 3))
    kwargs = dict(
        frame_names=["a", "b"],
        rate=1.0,
        fold=2,
        members=[0, 1],
        seed=9,
        namespace="calibration",
    )
    first = ensemble_initial_noise(reference, **kwargs)
    second = ensemble_initial_noise(reference, **kwargs)
    different_fold = ensemble_initial_noise(reference, **{**kwargs, "fold": 3})

    assert torch.equal(first, second)
    assert not torch.equal(first, different_fold)
    assert not torch.equal(first[0], first[1])
