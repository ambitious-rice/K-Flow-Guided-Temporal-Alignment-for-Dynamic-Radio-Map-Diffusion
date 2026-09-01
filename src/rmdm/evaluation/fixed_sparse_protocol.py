"""Frame-keyed sparse observations and noise for reproducible paper tests."""

from __future__ import annotations

import hashlib
from typing import Any

import numpy as np
import torch

from rmdm.data.sampling import derive_seed


MASK_SAMPLER_VERSION = (
    "dynamic_sparse_v2_semantic_vehicle_blake2b_pcg64_without_replacement"
)
MASK_MANIFEST_SEED = 20260714
DDIM_NOISE_VERSION = "rmdm-paper-ddim-frame-noise-v1"
OBSERVATION_NOISE_VERSION = "rmdm-paper-observation-frame-noise-v1"


def frame_names_by_sample(
    batch: dict[str, Any],
    *,
    batch_size: int,
    window_size: int,
) -> list[list[str]]:
    """Undo PyTorch's default transposition of per-sample frame-name lists."""

    raw = batch.get("frame_names")
    if raw is None:
        raise KeyError("fixed paper protocol requires frame_names")
    if len(raw) != window_size:
        raise ValueError(
            f"expected {window_size} transposed frame-name positions, got {len(raw)}"
        )
    result = [
        [str(raw[frame_index][sample_index]) for frame_index in range(window_size)]
        for sample_index in range(batch_size)
    ]
    if any(len(names) != window_size for names in result):
        raise RuntimeError("frame-name reconstruction failed")
    return result


def apply_fixed_sparse_observations(
    dense_batch: dict[str, Any],
    *,
    rate: float,
    split: str,
    manifest_seed: int = MASK_MANIFEST_SEED,
) -> dict[str, Any]:
    """Apply the legacy manifest's exact frame-keyed BLAKE2b/PCG64 mask."""

    if rate <= 0 or not float(rate).is_integer():
        raise ValueError("paper-test sampling rates must be positive integers")
    required = ("building", "vehicle", "target")
    missing = [name for name in required if name not in dense_batch]
    if missing:
        raise KeyError(f"dense batch misses required keys: {missing}")
    building = dense_batch["building"]
    vehicle = dense_batch["vehicle"]
    target = dense_batch["target"]
    if building.ndim != 5 or building.shape != vehicle.shape or building.shape != target.shape:
        raise ValueError("building, vehicle and target must share [N,T,1,H,W]")

    batch_size, window_size = target.shape[:2]
    names = frame_names_by_sample(
        dense_batch,
        batch_size=batch_size,
        window_size=window_size,
    )
    valid_mask = ((building <= 0.5) & (vehicle <= 0.5)).to(target.dtype)
    sampling_mask = torch.zeros_like(target)
    rate_key = str(int(rate))

    for sample_index in range(batch_size):
        for frame_index in range(window_size):
            valid_flat = (
                torch.nonzero(
                    valid_mask[sample_index, frame_index, 0].reshape(-1) > 0.5,
                    as_tuple=False,
                )
                .flatten()
                .detach()
                .cpu()
                .numpy()
            )
            if valid_flat.size == 0:
                raise RuntimeError(f"no valid pixels in {names[sample_index][frame_index]}")
            count = max(1, int(round((float(rate) / 100.0) * valid_flat.size)))
            digest = hashlib.blake2b(
                (
                    f"{MASK_SAMPLER_VERSION}|{int(manifest_seed)}|{split}|"
                    f"{names[sample_index][frame_index]}|{rate_key}"
                ).encode("utf-8"),
                digest_size=16,
            ).digest()
            rng = np.random.default_rng(
                int.from_bytes(digest, byteorder="little", signed=False)
            )
            selected = rng.choice(valid_flat, size=count, replace=False)
            selected_tensor = torch.as_tensor(
                selected,
                dtype=torch.long,
                device=target.device,
            )
            sampling_mask[sample_index, frame_index, 0].view(-1)[
                selected_tensor
            ] = 1.0

    result = dict(dense_batch)
    result.update(
        {
            "valid_mask": valid_mask,
            "sampling_rate": torch.full(
                (batch_size, window_size),
                float(rate),
                device=target.device,
                dtype=torch.float32,
            ),
            "sampling_mask": sampling_mask,
            "observed_rss": sampling_mask * target,
            "sampling_mode": ["fixed_frame_manifest"] * batch_size,
            "extreme_frames": [()] * batch_size,
        }
    )
    return result


def deterministic_frame_noise_like(
    target: torch.Tensor,
    frame_names: list[list[str]],
    *,
    rate: float,
    seed: int,
    version: str = DDIM_NOISE_VERSION,
) -> torch.Tensor:
    """Generate identical noise for a physical frame under any windowing."""

    if target.ndim != 5:
        raise ValueError("target must have shape [N,T,C,H,W]")
    if len(frame_names) != target.shape[0]:
        raise ValueError("frame_names batch size does not match target")
    samples = []
    for sample_index, names in enumerate(frame_names):
        if len(names) != target.shape[1]:
            raise ValueError("frame_names window size does not match target")
        frames = []
        for frame_index, name in enumerate(names):
            generator = torch.Generator(device=target.device)
            generator.manual_seed(
                derive_seed(version, int(seed), name, f"{float(rate):.6f}")
            )
            frames.append(
                torch.randn(
                    target[sample_index, frame_index].shape,
                    device=target.device,
                    dtype=target.dtype,
                    generator=generator,
                )
            )
        samples.append(torch.stack(frames))
    return torch.stack(samples)


def add_fixed_observation_noise(
    sparse_batch: dict[str, Any],
    *,
    standard_deviation: float,
    rate: float,
    seed: int,
) -> dict[str, Any]:
    """Add frame-keyed Gaussian RSS noise without clipping."""

    if standard_deviation < 0:
        raise ValueError("standard_deviation must be non-negative")
    if standard_deviation == 0:
        return sparse_batch
    target = sparse_batch["target"]
    names = frame_names_by_sample(
        sparse_batch,
        batch_size=target.shape[0],
        window_size=target.shape[1],
    )
    noise = deterministic_frame_noise_like(
        target,
        names,
        rate=rate,
        seed=seed,
        version=OBSERVATION_NOISE_VERSION,
    ).float()
    result = dict(sparse_batch)
    result["observed_rss"] = sparse_batch["sampling_mask"] * (
        target.float() + float(standard_deviation) * noise
    )
    return result


__all__ = [
    "DDIM_NOISE_VERSION",
    "MASK_MANIFEST_SEED",
    "MASK_SAMPLER_VERSION",
    "OBSERVATION_NOISE_VERSION",
    "add_fixed_observation_noise",
    "apply_fixed_sparse_observations",
    "deterministic_frame_noise_like",
    "frame_names_by_sample",
]
