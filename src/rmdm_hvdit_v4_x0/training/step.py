"""Clean-x0 DDPM objective with V4's unchanged calibration and PINN branches."""

from __future__ import annotations

from dataclasses import dataclass
from typing import Any

import torch
import torch.nn.functional as F

from rmdm.data import SamplingPolicy, derive_seed
from rmdm.diffusion import DiffusionProcess
from utils import cal_pinn_components, full_image_hessian_charbonnier


@dataclass(frozen=True)
class TrainingStepResult:
    loss: torch.Tensor
    clean_data_loss: torch.Tensor
    observation_alignment_loss: torch.Tensor
    calibration_loss: torch.Tensor
    pinn_loss: torch.Tensor
    equation_regularizer_loss: torch.Tensor
    semantic_anchor_loss: torch.Tensor
    sampling_rate_mean: torch.Tensor
    x0_mse_per_sample: torch.Tensor
    derived_epsilon_mse_per_sample: torch.Tensor


def _diffusion_seeds(
    batch: dict[str, Any],
    *,
    training_seed: int,
    epoch: int,
) -> list[int]:
    """Match V4's noise/timestep stream exactly for a paired causal comparison."""

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


def _epsilon_from_x0(
    diffusion: DiffusionProcess,
    noisy_target: torch.Tensor,
    predicted_x0: torch.Tensor,
    timesteps: torch.Tensor,
) -> torch.Tensor:
    alpha_bar = diffusion.scheduler.alphas_cumprod.to(
        device=noisy_target.device,
        dtype=noisy_target.dtype,
    )[timesteps]
    shape = (noisy_target.shape[0],) + (1,) * (noisy_target.ndim - 1)
    alpha = alpha_bar.reshape(shape).sqrt()
    sigma = (1.0 - alpha_bar).reshape(shape).sqrt().clamp_min(1.0e-6)
    return (noisy_target - alpha * predicted_x0) / sigma


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
    regularizer_type: str = "pinn",
    regularizer_weight: float | None = None,
    hessian_epsilon: float = 1.0e-3,
    use_tx_source_supervision: bool = True,
    observation_alignment_weight: float = 0.0,
) -> TrainingStepResult:
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
    prediction = model(diffusion_batch.noisy_target, diffusion_batch.timesteps, sparse_batch)
    if not isinstance(prediction, (tuple, list)) or len(prediction) < 2:
        raise RuntimeError("The x0 pilot must return (predicted_x0, cal)")
    predicted_x0, cal = prediction[:2]
    if predicted_x0.shape != target.shape or cal.shape != target.shape:
        raise ValueError("predicted x0 and cal must match the full target shape")

    clean_data_loss = F.mse_loss(predicted_x0.float(), target.float())
    observation_mask = (sparse_batch["sampling_mask"] > 0.5).float()
    observation_squared_error = (
        (predicted_x0.float() - target.float()).square() * observation_mask
    ).flatten(1).sum(1)
    observation_count = observation_mask.flatten(1).sum(1).clamp_min(1.0)
    observation_alignment_loss = (observation_squared_error / observation_count).mean()
    calibration_loss = F.mse_loss(cal.float(), target.float())
    obstacle = ((sparse_batch["building"] > 0.5) | (sparse_batch["vehicle"] > 0.5)).to(cal.dtype)
    batch, time, _, height, width = cal.shape
    flattened_cal = cal.reshape(batch * time, height, width)
    flattened_obstacle = obstacle.reshape(batch * time, height, width)
    flattened_tx = sparse_batch["tx"].to(cal.dtype).reshape(batch * time, height, width)
    if not use_tx_source_supervision:
        flattened_tx = torch.zeros_like(flattened_tx)
    equation_pde, obstacle_anchor, source_anchor = cal_pinn_components(
        flattened_cal,
        flattened_obstacle,
        flattened_tx,
        k=pinn_k,
    )
    semantic_anchor_loss = (obstacle_anchor + source_anchor).mean().float()
    if regularizer_type == "pinn":
        equation_regularizer_loss = equation_pde.mean().float()
        spatial_regularizer_loss = (
            float(pinn_weight) * semantic_anchor_loss
            + float(pinn_weight) * equation_regularizer_loss
        )
    elif regularizer_type == "full_image_hessian_charbonnier":
        equation_regularizer_loss = full_image_hessian_charbonnier(
            flattened_cal,
            epsilon=hessian_epsilon,
        ).mean().float()
        if regularizer_weight is None:
            raise ValueError("full_image_hessian_charbonnier requires regularizer_weight")
        spatial_regularizer_loss = (
            float(pinn_weight) * semantic_anchor_loss
            + float(regularizer_weight) * equation_regularizer_loss
        )
    else:
        raise ValueError(f"Unsupported spatial regularizer: {regularizer_type!r}")
    loss = (
        clean_data_loss
        + float(observation_alignment_weight) * observation_alignment_loss
        + calibration_loss
        + spatial_regularizer_loss
    )

    with torch.no_grad():
        x0_error = (predicted_x0.detach().float() - target.float()).square().flatten(1).mean(1)
        predicted_epsilon = _epsilon_from_x0(
            diffusion,
            diffusion_batch.noisy_target.float(),
            predicted_x0.detach().float(),
            diffusion_batch.timesteps,
        )
        epsilon_error = (
            predicted_epsilon - diffusion_batch.noise.float()
        ).square().flatten(1).mean(1)
    return TrainingStepResult(
        loss=loss,
        clean_data_loss=clean_data_loss,
        observation_alignment_loss=observation_alignment_loss,
        calibration_loss=calibration_loss,
        pinn_loss=spatial_regularizer_loss,
        equation_regularizer_loss=equation_regularizer_loss,
        semantic_anchor_loss=semantic_anchor_loss,
        sampling_rate_mean=sparse_batch["sampling_rate"].mean(),
        x0_mse_per_sample=x0_error,
        derived_epsilon_mse_per_sample=epsilon_error,
    )
