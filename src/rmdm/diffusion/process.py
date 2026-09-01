"""Training-time forward diffusion for complete W-frame targets."""

from __future__ import annotations

from dataclasses import dataclass
from typing import Sequence

import torch
from diffusers import DDPMScheduler

from rmdm.config import DiffusionConfig


@dataclass(frozen=True)
class DiffusionTrainingBatch:
    noisy_target: torch.Tensor
    noise: torch.Tensor
    timesteps: torch.Tensor


class DiffusionProcess:
    def __init__(self, config: DiffusionConfig) -> None:
        self.config = config
        self.scheduler = DDPMScheduler(
            num_train_timesteps=config.train_timesteps,
            beta_schedule=config.beta_schedule,
            prediction_type=config.prediction_type,
            clip_sample=True,
        )

    def training_batch(
        self,
        target: torch.Tensor,
        *,
        generator: torch.Generator | None = None,
        seeds: Sequence[int] | None = None,
    ) -> DiffusionTrainingBatch:
        if target.ndim != 5:
            raise ValueError("target must be [N,T,C,H,W]")
        batch_size = target.shape[0]
        if seeds is not None:
            if generator is not None or len(seeds) != batch_size:
                raise ValueError("seeds must match batch size and cannot be combined with generator")
            timestep_parts = []
            noise_parts = []
            for batch_index, seed in enumerate(seeds):
                sample_generator = torch.Generator(device=target.device).manual_seed(int(seed))
                timestep_parts.append(
                    torch.randint(
                        0,
                        self.config.train_timesteps,
                        (1,),
                        device=target.device,
                        generator=sample_generator,
                        dtype=torch.long,
                    )
                )
                noise_parts.append(
                    torch.randn(
                        target[batch_index].shape,
                        device=target.device,
                        dtype=target.dtype,
                        generator=sample_generator,
                    )
                )
            timesteps = torch.cat(timestep_parts)
            noise = torch.stack(noise_parts)
        else:
            timesteps = torch.randint(
                0,
                self.config.train_timesteps,
                (batch_size,),
                device=target.device,
                generator=generator,
                dtype=torch.long,
            )
            noise = torch.randn(target.shape, device=target.device, dtype=target.dtype, generator=generator)
        noisy = self.scheduler.add_noise(target, noise, timesteps)
        return DiffusionTrainingBatch(noisy, noise, timesteps)

    def predict_x0(
        self,
        noisy_target: torch.Tensor,
        predicted_noise: torch.Tensor,
        timesteps: torch.Tensor,
    ) -> torch.Tensor:
        """Recover the un-clipped x0 estimate for training diagnostics."""

        if noisy_target.shape != predicted_noise.shape:
            raise ValueError("noisy_target and predicted_noise must have identical shapes")
        if timesteps.shape != (noisy_target.shape[0],):
            raise ValueError("timesteps must contain one value per batch item")
        alpha_bar = self.scheduler.alphas_cumprod.to(
            device=noisy_target.device,
            dtype=noisy_target.dtype,
        )[timesteps]
        shape = (noisy_target.shape[0],) + (1,) * (noisy_target.ndim - 1)
        alpha_bar = alpha_bar.reshape(shape)
        return (noisy_target - (1.0 - alpha_bar).sqrt() * predicted_noise) / alpha_bar.sqrt()
