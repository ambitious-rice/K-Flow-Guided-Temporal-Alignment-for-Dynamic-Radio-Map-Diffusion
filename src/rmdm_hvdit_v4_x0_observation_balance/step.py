"""Clean-x0 W1 step with balanced observation supervision."""

from __future__ import annotations

from dataclasses import dataclass
from typing import Any

import torch
import torch.nn.functional as F

from rmdm.data import SamplingPolicy, derive_seed
from rmdm.diffusion import DiffusionProcess
from utils import cal_pinn_components, full_image_hessian_charbonnier

from .augmentation import prepare_observation_training_batch
from .config import AlignmentConfig, NoiseConfig


@dataclass(frozen=True)
class ObservationBalanceStepResult:
    loss: torch.Tensor
    clean_data_loss: torch.Tensor
    clean_condition_data_loss: torch.Tensor
    noisy_condition_data_loss: torch.Tensor
    observation_alignment_loss: torch.Tensor
    heldout_loss: torch.Tensor
    calibration_loss: torch.Tensor
    spatial_regularizer_loss: torch.Tensor
    equation_regularizer_loss: torch.Tensor
    semantic_anchor_loss: torch.Tensor
    sampling_rate_mean: torch.Tensor
    observation_noise_std_mean: torch.Tensor
    noisy_condition_fraction: torch.Tensor
    condition_dropout_fraction: torch.Tensor
    derived_epsilon_mse: torch.Tensor


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


def _group_mean(values: torch.Tensor, selected: torch.Tensor) -> torch.Tensor:
    return torch.where(selected, values, 0.0).sum() / selected.sum().clamp_min(1)


def _masked_point_loss(squared_error: torch.Tensor, mask: torch.Tensor) -> torch.Tensor:
    counts = mask.flatten(1).sum(1)
    per_sample = (squared_error * mask).flatten(1).sum(1) / counts.clamp_min(1.0)
    return _group_mean(per_sample, counts > 0)


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
    branch_step: int,
    observation_alignment_weight: float,
    heldout_weight: float,
    noise_config: NoiseConfig,
    alignment_config: AlignmentConfig,
    pinn_k: float,
    pinn_weight: float,
    regularizer_type: str = "pinn",
    regularizer_weight: float | None = None,
    hessian_epsilon: float = 1.0e-3,
    use_tx_source_supervision: bool = True,
) -> ObservationBalanceStepResult:
    del branch_step
    sampled = sampling_policy(dense_batch)
    prepared = prepare_observation_training_batch(
        sampled,
        seed=training_seed,
        epoch=epoch,
        noise_config=noise_config,
        alignment_config=alignment_config,
    )
    sparse_batch = prepared.sparse_batch
    target = sparse_batch["target"]
    diffusion_batch = diffusion.training_batch(
        target,
        seeds=_diffusion_seeds(sparse_batch, training_seed=training_seed, epoch=epoch),
    )
    prediction = model(diffusion_batch.noisy_target, diffusion_batch.timesteps, sparse_batch)
    if not isinstance(prediction, (tuple, list)) or len(prediction) < 2:
        raise RuntimeError("the W1 x0 model must return (predicted_x0, cal)")
    predicted_x0, cal = prediction[:2]
    if predicted_x0.shape != target.shape or cal.shape != target.shape:
        raise ValueError("predicted_x0 and cal must match target")

    squared_error = (predicted_x0.float() - target.float()).square()
    error_per_sample = squared_error.flatten(1).mean(1)
    clean_data_loss = error_per_sample.mean()
    clean_group = prepared.noise_standard_deviation == 0
    noisy_group = prepared.noise_standard_deviation > 0
    clean_condition_data_loss = _group_mean(error_per_sample, clean_group)
    noisy_condition_data_loss = _group_mean(error_per_sample, noisy_group)
    observation_loss = _masked_point_loss(squared_error, prepared.input_mask)
    heldout_loss = _masked_point_loss(squared_error, prepared.heldout_mask)
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
        spatial_regularizer_loss = float(pinn_weight) * (
            semantic_anchor_loss + equation_regularizer_loss
        )
    elif regularizer_type == "full_image_hessian_charbonnier":
        if regularizer_weight is None:
            raise ValueError("full_image_hessian_charbonnier requires regularizer_weight")
        equation_regularizer_loss = full_image_hessian_charbonnier(
            flattened_cal,
            epsilon=hessian_epsilon,
        ).mean().float()
        spatial_regularizer_loss = (
            float(pinn_weight) * semantic_anchor_loss
            + float(regularizer_weight) * equation_regularizer_loss
        )
    else:
        raise ValueError(f"unsupported spatial regularizer: {regularizer_type!r}")

    loss = (
        clean_data_loss
        + float(observation_alignment_weight) * observation_loss
        + float(heldout_weight) * heldout_loss
        + calibration_loss
        + spatial_regularizer_loss
    )
    with torch.no_grad():
        predicted_epsilon = _epsilon_from_x0(
            diffusion,
            diffusion_batch.noisy_target.float(),
            predicted_x0.detach().float(),
            diffusion_batch.timesteps,
        )
        epsilon_mse = (
            predicted_epsilon - diffusion_batch.noise.float()
        ).square().flatten(1).mean(1).mean()

    return ObservationBalanceStepResult(
        loss=loss,
        clean_data_loss=clean_data_loss,
        clean_condition_data_loss=clean_condition_data_loss,
        noisy_condition_data_loss=noisy_condition_data_loss,
        observation_alignment_loss=observation_loss,
        heldout_loss=heldout_loss,
        calibration_loss=calibration_loss,
        spatial_regularizer_loss=spatial_regularizer_loss,
        equation_regularizer_loss=equation_regularizer_loss,
        semantic_anchor_loss=semantic_anchor_loss,
        sampling_rate_mean=sampled["sampling_rate"].mean(),
        observation_noise_std_mean=prepared.noise_standard_deviation.mean(),
        noisy_condition_fraction=noisy_group.float().mean(),
        condition_dropout_fraction=prepared.condition_dropped.float().mean(),
        derived_epsilon_mse=epsilon_mse,
    )


__all__ = ["ObservationBalanceStepResult", "training_step"]
