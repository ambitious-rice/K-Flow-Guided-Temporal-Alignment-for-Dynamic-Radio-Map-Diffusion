from __future__ import annotations

import torch

from rmdm_noise_estimation.assimilation import observation_gradient_update


def test_noise_floor_update_is_deterministic_and_stops_below_floor() -> None:
    gradient = torch.ones((2, 1, 1, 1, 2))
    loss = torch.tensor([0.01, 0.05])

    first = observation_gradient_update(
        gradient,
        loss,
        strength=0.5,
        normalization="rms",
        max_update=10.0,
        observation_noise_variance=0.01,
    )
    second = observation_gradient_update(
        gradient,
        loss,
        strength=0.5,
        normalization="rms",
        max_update=10.0,
        observation_noise_variance=0.01,
    )

    assert torch.equal(first, second)
    assert torch.count_nonzero(first[0]) == 0
    assert torch.allclose(first[1].square().mean().sqrt(), torch.tensor(0.1))


def test_normalization_modes_and_update_limit() -> None:
    gradient = torch.tensor([[1.0, -1.0], [2.0, -2.0]])
    loss = torch.tensor([0.01, 0.05])
    for mode, expected in (
        ("rms", [[0.0, 0.0], [0.1, -0.1]]),
        ("rms_noise_gate", [[0.0, 0.0], [0.05**0.5 / 2, -(0.05**0.5) / 2]]),
        ("none", [[0.5, -0.5], [1.0, -1.0]]),
    ):
        for limit in (10.0, 0.075):
            update = observation_gradient_update(
                gradient, loss, strength=0.5, normalization=mode,
                max_update=limit, observation_noise_variance=0.01,
            )
            assert torch.allclose(update, torch.tensor(expected).clamp(-limit, limit))
    assert torch.equal(gradient, torch.tensor([[1.0, -1.0], [2.0, -2.0]]))
