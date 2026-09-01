"""Stateless W16 rate and spatial-mask sampling."""

from __future__ import annotations

import hashlib
from dataclasses import dataclass
from typing import Any, Sequence

import numpy as np
import torch

from rmdm.config import SamplingConfig


def derive_seed(*parts: object) -> int:
    digest = hashlib.blake2b("|".join(map(str, parts)).encode("utf-8"), digest_size=8).digest()
    return int.from_bytes(digest, "little", signed=False) & ((1 << 63) - 1)


def _choice(rng: np.random.Generator, values: Sequence[Any], probabilities: Sequence[float]) -> Any:
    index = int(rng.choice(len(values), p=np.asarray(probabilities, dtype=np.float64)))
    return values[index]


@dataclass(frozen=True)
class RateSample:
    rates: tuple[float, ...]
    mode: str
    extreme_frames: tuple[int, ...]


class SamplingPolicy:
    """Generate protocol-compliant rates and exact-budget masks."""

    SAMPLER_VERSION = "joint_sparse_blake2b_pcg64_without_replacement_v1"

    def __init__(self, config: SamplingConfig, *, split: str = "train") -> None:
        self.config = config
        self.split = str(split)
        self.epoch = 0

    def set_epoch(self, epoch: int) -> None:
        self.epoch = int(epoch)

    def sample_rates(
        self,
        *,
        video_id: str,
        start: int,
        window_size: int,
        fixed_rate: float | None = None,
    ) -> RateSample:
        if fixed_rate is not None:
            if fixed_rate <= 0:
                raise ValueError("fixed_rate must be positive")
            return RateSample((float(fixed_rate),) * window_size, "homogeneous_fixed", ())

        seed_parts = (self.SAMPLER_VERSION, self.config.seed, self.epoch, self.split, video_id, start)
        mode_rng = np.random.default_rng(derive_seed(*seed_parts, "mode"))
        homogeneous = bool(mode_rng.random() < self.config.homogeneous_probability)
        base_rng = np.random.default_rng(derive_seed(*seed_parts, "base"))
        base = float(_choice(base_rng, self.config.base_rates, self.config.base_probabilities))
        if homogeneous:
            return RateSample((base,) * window_size, "homogeneous", ())

        rates = []
        lower_rate = float(min(self.config.base_rates))
        upper_rate = float(max(self.config.base_rates))
        for frame_index in range(window_size):
            rng = np.random.default_rng(derive_seed(*seed_parts, "delta", frame_index))
            delta = float(_choice(rng, self.config.deltas, self.config.delta_probabilities))
            rates.append(float(np.clip(base + delta, lower_rate, upper_rate)))

        extreme_rng = np.random.default_rng(derive_seed(*seed_parts, "extreme-event"))
        extreme_frames: tuple[int, ...] = ()
        if extreme_rng.random() < self.config.extreme_probability_given_heterogeneous:
            count_rng = np.random.default_rng(derive_seed(*seed_parts, "extreme-count"))
            count = int(
                _choice(
                    count_rng,
                    self.config.extreme_frame_counts,
                    self.config.extreme_frame_count_probabilities,
                )
            )
            frame_rng = np.random.default_rng(derive_seed(*seed_parts, "extreme-frames"))
            extreme_frames = tuple(sorted(int(value) for value in frame_rng.choice(window_size, size=count, replace=False)))
            for frame_index in extreme_frames:
                rate_rng = np.random.default_rng(derive_seed(*seed_parts, "extreme-rate", frame_index))
                rates[frame_index] = float(
                    _choice(rate_rng, self.config.extreme_rates, self.config.extreme_rate_probabilities)
                )
        return RateSample(tuple(rates), "heterogeneous", extreme_frames)

    def __call__(self, dense_batch: dict[str, Any], *, fixed_rate: float | None = None) -> dict[str, Any]:
        required = ("building", "vehicle", "target", "video_id", "start")
        missing = [key for key in required if key not in dense_batch]
        if missing:
            raise KeyError(f"Dense batch misses required keys: {missing}")
        building = dense_batch["building"]
        vehicle = dense_batch["vehicle"]
        target = dense_batch["target"]
        if building.ndim != 5 or building.shape != vehicle.shape or building.shape != target.shape:
            raise ValueError("building, vehicle and target must share [N,T,1,H,W]")
        batch_size, window_size = building.shape[:2]
        video_ids = list(dense_batch["video_id"])
        starts = dense_batch["start"]
        if torch.is_tensor(starts):
            starts = starts.detach().cpu().tolist()
        else:
            starts = list(starts)
        if len(video_ids) != batch_size or len(starts) != batch_size:
            raise ValueError("video_id/start batch lengths do not match tensors")

        valid_mask = ((building <= 0.5) & (vehicle <= 0.5)).to(dtype=target.dtype)
        sampling_mask = torch.zeros_like(target)
        sampling_rates = torch.empty((batch_size, window_size), dtype=torch.float32, device=target.device)
        modes: list[str] = []
        extreme_frames: list[tuple[int, ...]] = []
        for batch_index, (video_id, start) in enumerate(zip(video_ids, starts)):
            rate_sample = self.sample_rates(
                video_id=str(video_id),
                start=int(start),
                window_size=window_size,
                fixed_rate=fixed_rate,
            )
            modes.append(rate_sample.mode)
            extreme_frames.append(rate_sample.extreme_frames)
            for frame_index, rate in enumerate(rate_sample.rates):
                sampling_rates[batch_index, frame_index] = rate
                valid_flat = torch.nonzero(
                    valid_mask[batch_index, frame_index, 0].reshape(-1) > 0.5,
                    as_tuple=False,
                ).flatten()
                if valid_flat.numel() == 0:
                    raise RuntimeError(f"No valid free-space pixels in {video_id} frame={int(start)+frame_index}")
                count = max(1, int(round((float(rate) / 100.0) * valid_flat.numel())))
                generator = torch.Generator(device=target.device)
                generator.manual_seed(
                    derive_seed(
                        self.SAMPLER_VERSION,
                        self.config.seed,
                        self.epoch,
                        self.split,
                        video_id,
                        int(start),
                        frame_index,
                        f"{rate:.6f}",
                        "spatial",
                    )
                )
                chosen = valid_flat[torch.randperm(valid_flat.numel(), generator=generator, device=target.device)[:count]]
                sampling_mask[batch_index, frame_index, 0].view(-1)[chosen] = 1.0

        result = dict(dense_batch)
        result.update(
            {
                "valid_mask": valid_mask,
                "sampling_rate": sampling_rates,
                "sampling_mask": sampling_mask,
                "observed_rss": sampling_mask * target,
                "sampling_mode": modes,
                "extreme_frames": extreme_frames,
            }
        )
        return result
