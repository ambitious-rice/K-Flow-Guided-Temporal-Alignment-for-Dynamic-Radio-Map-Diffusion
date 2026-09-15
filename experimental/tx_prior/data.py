"""Observation-free batch construction for training and prior evaluation."""

from __future__ import annotations

from typing import Any

import torch

from rmdm.data import derive_seed

from .adapter import scene_prior_batch


def make_prior_training_batch(dense_batch: dict[str, Any]) -> dict[str, Any]:
    """Add inert compatibility fields without invoking ``SamplingPolicy``."""

    result = scene_prior_batch(dense_batch)
    reference = result["building"]
    result["sampling_rate"] = torch.zeros(
        reference.shape[:2], device=reference.device, dtype=reference.dtype
    )
    result["sampling_mode"] = ["scene_prior"] * reference.shape[0]
    result["extreme_frames"] = [()] * reference.shape[0]
    return result


def deterministic_prior_noise_like(
    target: torch.Tensor,
    frame_names: list[list[str]],
    *,
    seed: int,
) -> torch.Tensor:
    """Frame-keyed DDIM noise independent of any observation-rate label."""

    if target.ndim != 5 or len(frame_names) != target.shape[0]:
        raise ValueError("target/frame_names batch shape mismatch")
    samples = []
    for sample_index, names in enumerate(frame_names):
        if len(names) != target.shape[1]:
            raise ValueError("frame_names window size does not match target")
        frames = []
        for frame_index, name in enumerate(names):
            generator = torch.Generator(device=target.device)
            generator.manual_seed(derive_seed("tx-prior-ddim-noise-v1", int(seed), name))
            frames.append(torch.randn(
                target[sample_index, frame_index].shape,
                device=target.device,
                dtype=target.dtype,
                generator=generator,
            ))
        samples.append(torch.stack(frames))
    return torch.stack(samples)
