"""DDIM sampling that computes Stage1 and condition features exactly once."""

from __future__ import annotations

from typing import Any

import torch
from diffusers import DDIMScheduler

from rmdm.config import DiffusionConfig
from rmdm.data.sampling import derive_seed


def deterministic_noise_like(
    target: torch.Tensor,
    *,
    video_ids: list[str],
    starts: list[int],
    rate: float,
    seed: int,
) -> torch.Tensor:
    if target.ndim != 5 or len(video_ids) != target.shape[0] or len(starts) != target.shape[0]:
        raise ValueError("target/video_ids/starts must describe the same [N,T,C,H,W] batch")
    samples = []
    for batch_index, (video_id, start) in enumerate(zip(video_ids, starts)):
        generator = torch.Generator(device=target.device)
        generator.manual_seed(derive_seed("joint-ddim-noise-v1", seed, video_id, start, f"{rate:.6f}"))
        samples.append(
            torch.randn(
                target[batch_index].shape,
                device=target.device,
                dtype=target.dtype,
                generator=generator,
            )
        )
    return torch.stack(samples)


class DDIMSampler:
    def __init__(self, config: DiffusionConfig) -> None:
        self.config = config
        self.scheduler = DDIMScheduler(
            num_train_timesteps=config.train_timesteps,
            beta_schedule=config.beta_schedule,
            prediction_type=config.prediction_type,
            clip_sample=True,
            set_alpha_to_one=True,
            steps_offset=0,
        )

    @torch.no_grad()
    def sample(
        self,
        model: Any,
        sparse_batch: dict[str, Any],
        *,
        initial_noise: torch.Tensor,
        steps: int | None = None,
        eta: float = 0.0,
    ) -> torch.Tensor:
        steps = int(steps or self.config.ddim_steps)
        if steps <= 0:
            raise ValueError("DDIM steps must be positive")
        if eta < 0.0:
            raise ValueError("DDIM eta must be non-negative")
        sample = initial_noise
        condition_cache = model.encode_conditions(sparse_batch)
        self.scheduler.set_timesteps(steps, device=sample.device)
        for timestep in self.scheduler.timesteps:
            timesteps = torch.full(
                (sample.shape[0],),
                int(timestep),
                device=sample.device,
                dtype=torch.long,
            )
            model_input = self.scheduler.scale_model_input(sample, timestep)
            predicted_noise = model.denoise(model_input, timesteps, condition_cache)
            sample = self.scheduler.step(
                predicted_noise,
                timestep,
                sample,
                eta=float(eta),
                use_clipped_model_output=False,
                return_dict=False,
            )[0]
        return sample.clamp(0.0, 1.0)
