"""Deterministic observation corruption and held-out-point construction."""

from __future__ import annotations

from dataclasses import dataclass
from typing import Any, Sequence

import torch

from .config import AlignmentConfig, NoiseConfig


@dataclass(frozen=True)
class ObservationTrainingBatch:
    sparse_batch: dict[str, Any]
    input_mask: torch.Tensor
    heldout_mask: torch.Tensor
    noise_standard_deviation: torch.Tensor
    condition_dropped: torch.Tensor


def _stable_seed(base: int, *parts: object) -> int:
    value = int(base) & ((1 << 63) - 1)
    for byte in "|".join(str(part) for part in parts).encode("utf-8"):
        value ^= byte
        value = (value * 1_099_511_628_211) & ((1 << 63) - 1)
    return value


def _uniform(seed: int) -> float:
    generator = torch.Generator(device="cpu")
    generator.manual_seed(seed)
    return float(torch.rand((), generator=generator))


def _choose(values: Sequence[float], probabilities: Sequence[float], value: float) -> float:
    cumulative = 0.0
    for candidate, probability in zip(values, probabilities):
        cumulative += float(probability)
        if value < cumulative:
            return float(candidate)
    return float(values[-1])


def _starts(batch: dict[str, Any]) -> list[int]:
    starts = batch["start"]
    if torch.is_tensor(starts):
        return [int(value) for value in starts.detach().cpu().tolist()]
    return [int(value) for value in starts]


def prepare_observation_training_batch(
    sparse_batch: dict[str, Any],
    *,
    seed: int,
    epoch: int,
    noise_config: NoiseConfig,
    alignment_config: AlignmentConfig,
) -> ObservationTrainingBatch:
    target = sparse_batch["target"]
    original_mask = (sparse_batch["sampling_mask"] > 0.5).to(target.dtype)
    valid_mask = sparse_batch["valid_mask"] > 0.5
    frames = target.shape[1]
    video_ids = [str(value) for value in sparse_batch["video_id"]]
    starts = _starts(sparse_batch)

    input_mask = original_mask.clone()
    heldout_mask = torch.zeros_like(original_mask)
    observed_rss = torch.zeros_like(target)
    input_counts = original_mask.flatten(2).sum(2).cpu().tolist()
    noise_stds = []
    dropped = []

    for sample_index, (video_id, start) in enumerate(zip(video_ids, starts)):
        sample_seed = _stable_seed(seed, "observation-balance", epoch, video_id, start)
        sigma = _choose(
            noise_config.standard_deviations,
            noise_config.probabilities,
            _uniform(_stable_seed(sample_seed, "sigma")),
        )
        drop_condition = _uniform(_stable_seed(sample_seed, "drop-condition")) < float(
            alignment_config.condition_dropout_probability
        )
        noise_stds.append(sigma)
        dropped.append(drop_condition)

        for frame_index in range(frames):
            frame_seed = _stable_seed(sample_seed, "frame", frame_index)
            frame_target = target[sample_index, frame_index, 0]
            frame_mask = original_mask[sample_index, frame_index, 0]
            if not drop_condition:
                observation = frame_target
                if sigma > 0:
                    generator = torch.Generator(device=target.device)
                    generator.manual_seed(_stable_seed(frame_seed, "noise"))
                    noise = torch.randn(
                        frame_target.shape, dtype=target.dtype,
                        device=target.device, generator=generator,
                    )
                    observation = frame_target + sigma * noise
                observed_rss[sample_index, frame_index, 0] = frame_mask * observation

            available = torch.nonzero(
                (valid_mask[sample_index, frame_index, 0] & (frame_mask <= 0.5)).reshape(-1),
                as_tuple=False,
            ).flatten()
            input_count = input_counts[sample_index][frame_index]
            heldout_count = min(
                available.numel(),
                int(round(input_count * float(alignment_config.heldout_ratio))),
            )
            if heldout_count:
                generator = torch.Generator(device=target.device)
                generator.manual_seed(_stable_seed(frame_seed, "heldout"))
                selected = available[
                    torch.randperm(available.numel(), device=target.device, generator=generator)[
                        :heldout_count
                    ]
                ]
                heldout_mask[sample_index, frame_index, 0].view(-1)[selected] = 1.0

        if drop_condition:
            input_mask[sample_index].zero_()

    result = dict(sparse_batch)
    result["sampling_mask"] = input_mask
    result["observed_rss"] = observed_rss
    return ObservationTrainingBatch(
        sparse_batch=result,
        input_mask=input_mask,
        heldout_mask=heldout_mask,
        noise_standard_deviation=torch.tensor(noise_stds, dtype=torch.float32, device=target.device),
        condition_dropped=torch.tensor(dropped, dtype=torch.bool, device=target.device),
    )


def observation_alignment_weight(
    step: int,
    *,
    max_steps: int,
    config: AlignmentConfig,
) -> float:
    step = min(max(int(step), 0), int(max_steps))
    if config.warmup_steps > 0 and step < config.warmup_steps:
        return float(config.peak_weight) * step / config.warmup_steps
    if step <= config.hold_steps:
        return float(config.peak_weight)
    denominator = max(max_steps - config.hold_steps, 1)
    progress = (step - config.hold_steps) / denominator
    return float(config.peak_weight) + progress * (
        float(config.final_weight) - float(config.peak_weight)
    )


__all__ = [
    "ObservationTrainingBatch",
    "observation_alignment_weight",
    "prepare_observation_training_batch",
]
