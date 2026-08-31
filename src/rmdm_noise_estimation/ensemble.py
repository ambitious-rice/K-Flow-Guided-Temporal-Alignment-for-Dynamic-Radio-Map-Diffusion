"""Condition-cached DDIM ensembles for held-out W16 predictions."""

from __future__ import annotations

from typing import Any

import torch
from diffusers import DDIMScheduler

from rmdm.data.sampling import derive_seed


ENSEMBLE_NOISE_VERSION = "w16-crossfit-ddim-ensemble-v1"


def ensemble_initial_noise(
    reference: torch.Tensor,
    frame_names: list[str],
    *,
    rate: float,
    fold: int,
    members: list[int],
    seed: int,
    namespace: str,
) -> torch.Tensor:
    """Return frame-keyed member noise, independent of measurement sigma."""

    if reference.ndim != 4 or len(frame_names) != reference.shape[0]:
        raise ValueError("reference must be [T,C,H,W] and match frame_names")
    samples = []
    for member in members:
        frames = []
        for frame_index, frame_name in enumerate(frame_names):
            generator = torch.Generator(device=reference.device)
            generator.manual_seed(
                derive_seed(
                    ENSEMBLE_NOISE_VERSION,
                    namespace,
                    seed,
                    frame_name,
                    f"{float(rate):.6f}",
                    fold,
                    member,
                )
            )
            frames.append(
                torch.randn(
                    reference[frame_index].shape,
                    device=reference.device,
                    dtype=reference.dtype,
                    generator=generator,
                )
            )
        samples.append(torch.stack(frames))
    return torch.stack(samples)


def _repeat_cache(cache: dict[str, torch.Tensor], repeats: int) -> dict[str, torch.Tensor]:
    return {name: value.repeat_interleave(repeats, dim=0) for name, value in cache.items()}


class CrossfitDDIMEnsemble:
    def __init__(self, diffusion_config: Any) -> None:
        self.scheduler = DDIMScheduler(
            num_train_timesteps=diffusion_config.train_timesteps,
            beta_schedule=diffusion_config.beta_schedule,
            prediction_type=diffusion_config.prediction_type,
            clip_sample=True,
            set_alpha_to_one=True,
            steps_offset=0,
        )

    @torch.no_grad()
    def _sample_chunk(
        self,
        model: Any,
        cache: dict[str, torch.Tensor],
        initial_noise: torch.Tensor,
        *,
        steps: int,
        accelerator: Any,
    ) -> torch.Tensor:
        sample = initial_noise
        cache_batch = next(iter(cache.values())).shape[0]
        if sample.shape[0] % cache_batch:
            raise ValueError("ensemble sample batch is not divisible by condition batch")
        repeated_cache = _repeat_cache(cache, sample.shape[0] // cache_batch)
        self.scheduler.set_timesteps(steps, device=sample.device)
        for timestep in self.scheduler.timesteps:
            times = torch.full(
                (sample.shape[0],), int(timestep), device=sample.device, dtype=torch.long
            )
            with accelerator.autocast():
                output = model.denoise(
                    self.scheduler.scale_model_input(sample, timestep),
                    times,
                    repeated_cache,
                )
            sample = self.scheduler.step(
                output,
                timestep,
                sample,
                eta=0.0,
                use_clipped_model_output=False,
                return_dict=False,
            )[0]
        return sample.clamp(0.0, 1.0)

    @torch.no_grad()
    def moments(
        self,
        model: Any,
        sparse_batch: dict[str, Any],
        *,
        frame_names: list[list[str]],
        rate: float,
        fold: int,
        members: int,
        member_batch_size: int,
        steps: int,
        seed: int,
        namespace: str,
        accelerator: Any,
    ) -> tuple[torch.Tensor, torch.Tensor]:
        batch_size = sparse_batch["target"].shape[0]
        if len(frame_names) != batch_size:
            raise ValueError("frame_names batch does not match sparse batch")
        if members < 2 or not 0 < member_batch_size <= members:
            raise ValueError("invalid ensemble size or member batch size")
        with accelerator.autocast():
            cache = model.encode_conditions(sparse_batch)
        total = torch.zeros_like(sparse_batch["target"], dtype=torch.float64)
        total_square = torch.zeros_like(total)
        for start in range(0, members, member_batch_size):
            indices = list(range(start, min(start + member_batch_size, members)))
            initial = torch.stack(
                [
                    ensemble_initial_noise(
                        sparse_batch["target"][sample_index],
                        frame_names[sample_index],
                        rate=rate,
                        fold=fold,
                        members=indices,
                        seed=seed,
                        namespace=namespace,
                    )
                    for sample_index in range(batch_size)
                ]
            ).flatten(0, 1)
            predictions = self._sample_chunk(
                model,
                cache,
                initial,
                steps=steps,
                accelerator=accelerator,
            ).to(torch.float64)
            predictions = predictions.reshape(batch_size, len(indices), *predictions.shape[1:])
            total += predictions.sum(dim=1)
            total_square += predictions.square().sum(dim=1)
        mean = total / members
        variance = (total_square - members * mean.square()) / (members - 1)
        return mean.float(), variance.clamp_min(0.0).float()


__all__ = ["ENSEMBLE_NOISE_VERSION", "CrossfitDDIMEnsemble", "ensemble_initial_noise"]
