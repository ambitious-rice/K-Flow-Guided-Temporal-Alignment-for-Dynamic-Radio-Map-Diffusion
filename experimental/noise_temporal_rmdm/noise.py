"""Per-sample RSS observation-noise augmentation in normalized RSS units."""

from __future__ import annotations

from typing import Any

import torch

from rmdm.data import derive_seed

from .config import MeasurementNoiseConfig


NOISE_SAMPLER_VERSION = "noise-temporal-rmdm-measurement-mixture-v2"


def _starts(batch: dict[str, Any]) -> list[int]:
    values = batch["start"]
    return [int(value) for value in (values.detach().cpu().tolist() if torch.is_tensor(values) else values)]


def add_measurement_noise(
    sparse_batch: dict[str, Any],
    config: MeasurementNoiseConfig,
    *,
    epoch: int,
) -> dict[str, Any]:
    """Add Gaussian RSS observation error after sampling, without clipping.

    One ``sigma`` is drawn per sample/clip from the configured three-part
    mixture. Pixel/frame errors are independent conditional on that sigma and
    the implemented equation is ``Y=M*(X + sigma*eps)`` for normalized RSS X.
    """

    target = sparse_batch["target"]
    mask = sparse_batch["sampling_mask"]
    if target.ndim != 5 or target.shape != mask.shape:
        raise ValueError("target and sampling_mask must share [B,T,1,H,W]")
    video_ids = list(sparse_batch["video_id"])
    starts = _starts(sparse_batch)
    if len(video_ids) != target.shape[0] or len(starts) != target.shape[0]:
        raise ValueError("batch metadata does not match tensor batch size")
    selected: list[float] = []
    components: list[str] = []
    noises: list[torch.Tensor] = []
    for index, (video_id, start) in enumerate(zip(video_ids, starts)):
        choice_generator = torch.Generator(device="cpu").manual_seed(
            derive_seed(NOISE_SAMPLER_VERSION, config.seed, epoch, video_id, start, "component")
        )
        component_draw = float(torch.rand((), generator=choice_generator))
        if component_draw < config.clean_probability:
            component = "clean"
            sigma = 0.0
        elif component_draw < config.clean_probability + config.nominal_probability:
            component = "nominal"
            lower, upper = config.nominal_sigma_min, config.nominal_sigma_max
            sigma_generator = torch.Generator(device="cpu").manual_seed(
                derive_seed(NOISE_SAMPLER_VERSION, config.seed, epoch, video_id, start, "sigma")
            )
            sigma = lower + (upper - lower) * float(torch.rand((), generator=sigma_generator))
        else:
            component = "strong"
            lower, upper = config.strong_sigma_min, config.strong_sigma_max
            sigma_generator = torch.Generator(device="cpu").manual_seed(
                derive_seed(NOISE_SAMPLER_VERSION, config.seed, epoch, video_id, start, "sigma")
            )
            sigma = lower + (upper - lower) * float(torch.rand((), generator=sigma_generator))
        components.append(component)
        selected.append(sigma)
        noise_generator = torch.Generator(device=target.device).manual_seed(
            derive_seed(NOISE_SAMPLER_VERSION, config.seed, epoch, video_id, start, "noise")
        )
        noises.append(torch.randn(target[index].shape, dtype=target.dtype,
                                  device=target.device, generator=noise_generator))
    sigma = torch.tensor(selected, device=target.device, dtype=target.dtype)
    epsilon = torch.stack(noises)
    view = (target.shape[0],) + (1,) * (target.ndim - 1)
    observed = mask * (target + sigma.reshape(view) * epsilon)
    result = dict(sparse_batch)
    result.update({
        "observed_rss": observed,
        "measurement_noise_component": components,
        "measurement_standard_deviation": sigma,
        "measurement_variance": sigma.square(),
    })
    return result


def add_fixed_measurement_noise(
    sparse_batch: dict[str, Any],
    standard_deviation: float,
    *,
    seed: int,
) -> dict[str, Any]:
    """Deterministically add a known normalized noise level for validation."""

    if standard_deviation < 0:
        raise ValueError("standard_deviation must be non-negative")
    target = sparse_batch["target"]
    mask = sparse_batch["sampling_mask"]
    video_ids = list(sparse_batch["video_id"])
    starts = _starts(sparse_batch)
    noises = []
    for index, (video_id, start) in enumerate(zip(video_ids, starts)):
        generator = torch.Generator(device=target.device).manual_seed(
            derive_seed(NOISE_SAMPLER_VERSION, seed, video_id, start, f"fixed-{standard_deviation:.6f}")
        )
        noises.append(torch.randn(target[index].shape, device=target.device, dtype=target.dtype, generator=generator))
    epsilon = torch.stack(noises)
    observed = mask * (target + float(standard_deviation) * epsilon)
    result = dict(sparse_batch)
    sigma = torch.full(
        (target.shape[0],), float(standard_deviation), device=target.device, dtype=target.dtype
    )
    result["observed_rss"] = observed
    result["measurement_noise_component"] = ["fixed"] * target.shape[0]
    result["measurement_standard_deviation"] = sigma
    result["measurement_variance"] = sigma.square()
    return result
