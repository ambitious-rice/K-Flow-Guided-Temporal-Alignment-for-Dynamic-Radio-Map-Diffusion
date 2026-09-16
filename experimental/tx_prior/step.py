"""Standard epsilon-DDPM objective for the observation-free scene prior."""

from __future__ import annotations

from dataclasses import dataclass
from typing import Any

import torch
import torch.nn.functional as F

from rmdm.data import SamplingPolicy, derive_seed
from rmdm.diffusion import DiffusionProcess
from utils import cal_pinn_components


@dataclass(frozen=True)
class TrainingStepResult:
    loss: torch.Tensor
    diffusion_loss: torch.Tensor
    calibration_loss: torch.Tensor
    pinn_loss: torch.Tensor
    sampling_rate_mean: torch.Tensor
    epsilon_mse_per_sample: torch.Tensor
    x0_mse_per_sample: torch.Tensor


def _diffusion_seeds(
    batch: dict[str, Any],
    *,
    training_seed: int,
    epoch: int,
) -> list[int]:
    starts = batch["start"]
    starts = starts.detach().cpu().tolist() if torch.is_tensor(starts) else starts
    return [
        derive_seed(
            "rmdm-hvdit-v4-joint-diffusion-v1",
            "t1",
            training_seed,
            epoch,
            video_id,
            int(start),
        )
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
    pinn_k: float,
    pinn_weight: float,
    use_tx_source_supervision: bool = True,
) -> TrainingStepResult:
    """Train epsilon while retaining the existing HWM calibration/PINN losses."""

    sparse_batch = sampling_policy(dense_batch)
    target = sparse_batch["target"]
    diffusion_batch = diffusion.training_batch(
        target,
        seeds=_diffusion_seeds(
            sparse_batch,
            training_seed=training_seed,
            epoch=epoch,
        ),
    )
    prediction = model(
        diffusion_batch.noisy_target,
        diffusion_batch.timesteps,
        sparse_batch,
    )
    if not isinstance(prediction, (tuple, list)) or len(prediction) < 2:
        raise RuntimeError("The epsilon prior must return (predicted_epsilon, cal)")
    predicted_epsilon, cal = prediction[:2]
    if predicted_epsilon.shape != diffusion_batch.noise.shape or cal.shape != target.shape:
        raise ValueError("predicted epsilon, noise, cal, and target shapes must match")

    diffusion_loss = F.mse_loss(
        predicted_epsilon.float(),
        diffusion_batch.noise.float(),
    )
    calibration_loss = F.mse_loss(cal.float(), target.float())
    obstacle = (
        (sparse_batch["building"] > 0.5) | (sparse_batch["vehicle"] > 0.5)
    ).to(cal.dtype)
    batch, time, _, height, width = cal.shape
    flattened_cal = cal.reshape(batch * time, height, width)
    flattened_obstacle = obstacle.reshape(batch * time, height, width)
    flattened_tx = sparse_batch["tx"].to(cal.dtype).reshape(
        batch * time, height, width
    )
    if not use_tx_source_supervision:
        flattened_tx = torch.zeros_like(flattened_tx)
    equation_pde, obstacle_anchor, source_anchor = cal_pinn_components(
        flattened_cal,
        flattened_obstacle,
        flattened_tx,
        k=pinn_k,
    )
    pinn_loss = (
        equation_pde.mean() + obstacle_anchor.mean() + source_anchor.mean()
    ).float()
    loss = diffusion_loss + calibration_loss + float(pinn_weight) * pinn_loss

    with torch.no_grad():
        epsilon_error = (
            predicted_epsilon.detach().float() - diffusion_batch.noise.float()
        ).square().flatten(1).mean(1)
        predicted_x0 = diffusion.predict_x0(
            diffusion_batch.noisy_target.float(),
            predicted_epsilon.detach().float(),
            diffusion_batch.timesteps,
        )
        x0_error = (predicted_x0 - target.float()).square().flatten(1).mean(1)
    return TrainingStepResult(
        loss=loss,
        diffusion_loss=diffusion_loss,
        calibration_loss=calibration_loss,
        pinn_loss=pinn_loss,
        sampling_rate_mean=sparse_batch["sampling_rate"].mean(),
        epsilon_mse_per_sample=epsilon_error,
        x0_mse_per_sample=x0_error,
    )
