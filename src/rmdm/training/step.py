"""One complete joint-denoising training batch."""

from __future__ import annotations

from dataclasses import dataclass
from typing import Any

import torch
import torch.nn.functional as F

from rmdm.data import SamplingPolicy, derive_seed
from rmdm.diffusion import DiffusionProcess


@dataclass(frozen=True)
class TrainingStepResult:
    loss: torch.Tensor
    predicted_noise: torch.Tensor
    target_noise: torch.Tensor
    sampling_rate_mean: torch.Tensor
    timesteps: torch.Tensor
    epsilon_mse_per_sample: torch.Tensor
    x0_mse_per_sample: torch.Tensor


def _diffusion_seeds(batch: dict[str, Any], *, training_seed: int, epoch: int) -> list[int]:
    starts = batch["start"]
    if torch.is_tensor(starts):
        starts = starts.detach().cpu().tolist()
    return [
        derive_seed("joint-training-diffusion-v1", training_seed, epoch, video_id, int(start))
        for video_id, start in zip(batch["video_id"], starts)
    ]


def training_step(
    model: torch.nn.Module,
    dense_batch: dict[str, Any],
    sampling_policy: SamplingPolicy,
    diffusion: DiffusionProcess,
    *,
    training_seed: int,
    epoch: int,
) -> TrainingStepResult:
    sparse_batch = sampling_policy(dense_batch)
    target = sparse_batch["target"]
    diffusion_batch = diffusion.training_batch(
        target,
        seeds=_diffusion_seeds(sparse_batch, training_seed=training_seed, epoch=epoch),
    )
    predicted_noise = model(diffusion_batch.noisy_target, diffusion_batch.timesteps, sparse_batch)
    loss = F.mse_loss(predicted_noise.float(), diffusion_batch.noise.float())
    with torch.no_grad():
        epsilon_mse_per_sample = (
            predicted_noise.detach().float() - diffusion_batch.noise.float()
        ).square().flatten(1).mean(dim=1)
        predicted_x0 = diffusion.predict_x0(
            diffusion_batch.noisy_target.float(),
            predicted_noise.detach().float(),
            diffusion_batch.timesteps,
        )
        x0_mse_per_sample = (predicted_x0 - target.float()).square().flatten(1).mean(dim=1)
    return TrainingStepResult(
        loss=loss,
        predicted_noise=predicted_noise,
        target_noise=diffusion_batch.noise,
        sampling_rate_mean=sparse_batch["sampling_rate"].mean(),
        timesteps=diffusion_batch.timesteps,
        epsilon_mse_per_sample=epsilon_mse_per_sample,
        x0_mse_per_sample=x0_mse_per_sample,
    )
