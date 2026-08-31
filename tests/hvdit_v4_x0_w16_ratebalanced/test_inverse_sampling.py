from __future__ import annotations

import torch

from rmdm_hvdit_v4_x0_w16_ratebalanced.inverse_sampling import (
    add_observation_noise,
    bound_mean_shift,
    helmholtz_loss,
    interior_free_mask,
    keyed_categorical,
    known_loss,
    log_p_over_q,
    observation_gradient_update,
    observation_loss,
)


def test_known_loss_normalizes_three_regions_independently() -> None:
    prediction = torch.tensor([[[[[2.0, 3.0], [4.0, 5.0]]]]])
    building = torch.tensor([[[[[1.0, 0.0], [0.0, 0.0]]]]])
    vehicle = torch.tensor([[[[[0.0, 1.0], [0.0, 0.0]]]]])
    sampling = torch.tensor([[[[[0.0, 0.0], [1.0, 1.0]]]]])
    observed = torch.tensor([[[[[0.0, 0.0], [3.0, 3.0]]]]])

    total, components = known_loss(prediction, building, vehicle, sampling, observed)

    assert torch.allclose(components["building"], torch.tensor([4.0]))
    assert torch.allclose(components["vehicle"], torch.tensor([9.0]))
    assert torch.allclose(components["sample"], torch.tensor([2.5]))
    assert torch.allclose(total, torch.tensor([15.5]))


def test_observation_loss_ignores_obstacles() -> None:
    prediction = torch.tensor([[[[[9.0, 3.0], [4.0, 5.0]]]]])
    sampling = torch.tensor([[[[[0.0, 1.0], [0.0, 0.0]]]]])
    observed = torch.tensor([[[[[0.0, 2.0], [0.0, 0.0]]]]])

    loss = observation_loss(prediction, sampling, observed)

    assert torch.allclose(loss, torch.tensor([1.0]))


def test_observation_gradient_update_supports_rms_and_raw_modes() -> None:
    gradient = torch.tensor([[[[[3.0, 4.0]]]]])
    loss = torch.tensor([4.0])

    normalized = observation_gradient_update(
        gradient, loss, strength=0.5, normalization="rms", max_update=10.0
    )
    raw = observation_gradient_update(
        gradient, loss, strength=0.5, normalization="none", max_update=10.0
    )

    assert torch.allclose(normalized.square().mean().sqrt(), torch.tensor(1.0))
    assert torch.allclose(raw, gradient * 0.5)


def test_observation_gradient_update_stops_at_measurement_noise_floor() -> None:
    gradient = torch.ones((2, 1, 1, 1, 2))
    loss = torch.tensor([0.01, 0.05])

    update = observation_gradient_update(
        gradient,
        loss,
        strength=0.5,
        normalization="rms",
        max_update=10.0,
        observation_noise_variance=0.01,
    )

    assert torch.count_nonzero(update[0]) == 0
    assert torch.allclose(
        update[1].square().mean().sqrt(),
        torch.tensor(0.1),
    )


def test_noise_gate_uses_full_observation_loss_above_noise_floor() -> None:
    gradient = torch.ones((2, 1, 1, 1, 2))
    loss = torch.tensor([0.01, 0.05])

    update = observation_gradient_update(
        gradient,
        loss,
        strength=0.5,
        normalization="rms_noise_gate",
        max_update=10.0,
        observation_noise_variance=0.01,
    )

    assert torch.count_nonzero(update[0]) == 0
    assert torch.allclose(
        update[1].square().mean().sqrt(),
        torch.tensor(0.5 * 0.05**0.5),
    )


def test_observation_noise_is_masked_and_reproducible() -> None:
    target = torch.full((1, 2, 1, 4, 4), 0.5)
    mask = torch.zeros_like(target)
    mask[..., 1, 2] = 1.0
    batch = {
        "target": target,
        "sampling_mask": mask,
        "observed_rss": mask * target,
        "video_id": ["scene"],
        "start": torch.tensor([16]),
    }

    first = add_observation_noise(batch, standard_deviation=0.05, rate=1.0, seed=7)
    second = add_observation_noise(batch, standard_deviation=0.05, rate=1.0, seed=7)

    assert torch.equal(first["observed_rss"], second["observed_rss"])
    assert torch.count_nonzero(first["observed_rss"] * (1.0 - mask)) == 0
    assert not torch.equal(first["observed_rss"], batch["observed_rss"])
    assert torch.equal(batch["observed_rss"], mask * target)


def test_interior_mask_erodes_border_and_obstacle_neighborhood() -> None:
    building = torch.zeros((1, 1, 1, 7, 7))
    vehicle = torch.zeros_like(building)
    building[..., 3, 3] = 1.0

    interior = interior_free_mask(building, vehicle)[0, 0, 0]

    assert not interior[0].any()
    assert not interior[:, 0].any()
    assert not interior[2:5, 2:5].any()
    assert interior[1, 1]


def test_constant_field_has_only_k_squared_residual_on_interior() -> None:
    prediction = torch.ones((2, 3, 1, 8, 8))
    obstacle = torch.zeros_like(prediction)

    loss = helmholtz_loss(prediction, obstacle, obstacle, wave_number=0.2)

    assert torch.allclose(loss, torch.full((2,), 0.2**4), atol=1.0e-7)


def test_proposal_ratio_matches_equal_covariance_gaussians() -> None:
    candidates = torch.tensor([[[[[[0.0]]]], [[[[2.0]]]]]])
    mean_p = torch.tensor([[[[[0.0]]]]])
    mean_q = torch.tensor([[[[[1.0]]]]])

    ratio = log_p_over_q(candidates, mean_p, mean_q, sigma=1.0)

    assert torch.allclose(ratio, torch.tensor([[0.5, -1.5]]))


def test_mean_shift_uses_single_window_mahalanobis_radius() -> None:
    shift = torch.tensor([[[[[3.0, 4.0]]]]])

    bounded = bound_mean_shift(shift, sigma=2.0, radius=1.0)

    assert torch.allclose(bounded.flatten(1).norm(dim=1), torch.tensor([2.0]))
    assert torch.allclose(bounded, shift * 0.4)


def test_keyed_categorical_is_reproducible() -> None:
    weights = torch.tensor([[0.0, 1.0, -1.0], [2.0, 0.0, 0.0]])
    kwargs = {
        "video_ids": ["a", "b"],
        "starts": [0, 16],
        "rate": 1.0,
        "seed": 7,
        "step_index": 35,
    }

    first = keyed_categorical(weights, **kwargs)
    second = keyed_categorical(weights, **kwargs)

    assert torch.equal(first, second)
