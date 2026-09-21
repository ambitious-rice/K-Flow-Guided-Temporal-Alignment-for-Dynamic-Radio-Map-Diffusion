"""The exact T1 objective: epsilon DDPM + clean cal + PINN."""

from __future__ import annotations

from dataclasses import dataclass
from typing import Any

import torch
import torch.nn.functional as F

from rmdm.data import SamplingPolicy, derive_seed
from rmdm.diffusion import DiffusionProcess
from utils import cal_pinn_components

from .config import ExperimentConfig
from .noise import add_measurement_noise


@dataclass(frozen=True)
class TrainingStepResult:
    loss: torch.Tensor
    diffusion_loss: torch.Tensor
    calibration_loss: torch.Tensor
    pinn_loss: torch.Tensor
    measurement_sigma_mean: torch.Tensor
    sampling_rate_mean: torch.Tensor
    epsilon_mse_per_sample: torch.Tensor
    x0_mse_per_sample: torch.Tensor


def _diffusion_seeds(batch: dict[str, Any], *, training_seed: int, epoch: int) -> list[int]:
    starts = batch["start"]
    starts = starts.detach().cpu().tolist() if torch.is_tensor(starts) else starts
    return [
        derive_seed("noise-temporal-rmdm-diffusion-v1", training_seed, epoch, video_id, int(start))
        for video_id, start in zip(batch["video_id"], starts)
    ]


def training_step(
    model: torch.nn.Module,
    dense_batch: dict[str, Any],
    sampling_policy: SamplingPolicy,
    diffusion: DiffusionProcess,
    config: ExperimentConfig,
    *,
    epoch: int,
) -> TrainingStepResult:
    sparse = add_measurement_noise(sampling_policy(dense_batch), config.measurement_noise, epoch=epoch)
    target = sparse["target"]
    diffusion_batch = diffusion.training_batch(
        target,
        seeds=_diffusion_seeds(sparse, training_seed=config.train.seed, epoch=epoch),
    )
    predicted_epsilon, cal = model(diffusion_batch.noisy_target, diffusion_batch.timesteps, sparse)
    if predicted_epsilon.shape != target.shape or cal.shape != target.shape:
        raise ValueError("predicted epsilon, cal, and target must have identical shapes")
    diffusion_loss = F.mse_loss(predicted_epsilon.float(), diffusion_batch.noise.float())
    # Calibration is always supervised by the clean target, never the noisy observations.
    calibration_loss = F.mse_loss(cal.float(), target.float())
    obstacle = ((sparse["building"] > 0.5) | (sparse["vehicle"] > 0.5)).to(cal.dtype)
    batch, time, _, height, width = cal.shape
    # Tx is intentionally unknown. Keep only the legacy equation and obstacle
    # terms; pass an all-zero placeholder and discard the source component.
    equation_loss, obstacle_loss, _ = cal_pinn_components(
        cal.reshape(batch * time, height, width),
        obstacle.reshape(batch * time, height, width),
        torch.zeros_like(cal).reshape(batch * time, height, width),
        k=config.loss.pinn_k,
    )
    pinn_loss = (equation_loss + obstacle_loss).mean().float()
    loss = (
        config.loss.diffusion_weight * diffusion_loss
        + config.loss.calibration_weight * calibration_loss
        + config.loss.pinn_weight * pinn_loss
    )
    with torch.no_grad():
        epsilon_mse = (predicted_epsilon.float() - diffusion_batch.noise.float()).square().flatten(1).mean(1)
        x0 = diffusion.predict_x0(
            diffusion_batch.noisy_target.float(), predicted_epsilon.detach().float(), diffusion_batch.timesteps
        )
        x0_mse = (x0 - target.float()).square().flatten(1).mean(1)
    return TrainingStepResult(
        loss=loss,
        diffusion_loss=diffusion_loss,
        calibration_loss=calibration_loss,
        pinn_loss=pinn_loss,
        measurement_sigma_mean=sparse["measurement_standard_deviation"].mean(),
        sampling_rate_mean=sparse["sampling_rate"].mean(),
        epsilon_mse_per_sample=epsilon_mse,
        x0_mse_per_sample=x0_mse,
    )
