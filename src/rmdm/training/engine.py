"""One-epoch distributed training engine."""

from __future__ import annotations

import json
import time
from dataclasses import dataclass
from typing import Any

import torch

from rmdm.config import ExperimentConfig
from rmdm.data import SamplingPolicy
from rmdm.diffusion import DiffusionProcess

from .step import training_step


@dataclass(frozen=True)
class EpochResult:
    global_step: int
    mean_loss: float
    mean_sampling_rate: float
    batches: int
    elapsed_seconds: float
    reached_max_steps: bool
    timestep_buckets: list[dict[str, float | int]]


def train_one_epoch(
    accelerator: Any,
    model: torch.nn.Module,
    loader: Any,
    optimizer: torch.optim.Optimizer,
    lr_scheduler: Any,
    sampling_policy: SamplingPolicy,
    diffusion: DiffusionProcess,
    config: ExperimentConfig,
    *,
    epoch: int,
    global_step: int,
) -> EpochResult:
    model.train()
    sampling_policy.set_epoch(epoch)
    start_time = time.monotonic()
    sums = torch.zeros(3, dtype=torch.float64, device=accelerator.device)
    bucket_count = min(10, config.diffusion.train_timesteps)
    bucket_width = max((config.diffusion.train_timesteps + bucket_count - 1) // bucket_count, 1)
    bucket_sums = torch.zeros((bucket_count, 3), dtype=torch.float64, device=accelerator.device)
    optimizer.zero_grad(set_to_none=True)
    reached_max_steps = False
    for batch_index, dense_batch in enumerate(loader):
        with accelerator.accumulate(model):
            with accelerator.autocast():
                result = training_step(
                    model,
                    dense_batch,
                    sampling_policy,
                    diffusion,
                    training_seed=config.train.seed,
                    epoch=epoch,
                )
            accelerator.backward(result.loss)
            if accelerator.sync_gradients:
                accelerator.clip_grad_norm_(
                    [parameter for parameter in model.parameters() if parameter.requires_grad],
                    config.train.gradient_clip_norm,
                )
                optimizer.step()
                lr_scheduler.step()
                optimizer.zero_grad(set_to_none=True)
                global_step += 1
        sums[0] += result.loss.detach().double()
        sums[1] += result.sampling_rate_mean.detach().double()
        sums[2] += 1
        bucket_indices = torch.div(result.timesteps.detach(), bucket_width, rounding_mode="floor").clamp_max(
            bucket_count - 1
        )
        bucket_sums[:, 0].scatter_add_(
            0,
            bucket_indices,
            torch.ones_like(result.epsilon_mse_per_sample, dtype=torch.float64),
        )
        bucket_sums[:, 1].scatter_add_(0, bucket_indices, result.epsilon_mse_per_sample.detach().double())
        bucket_sums[:, 2].scatter_add_(0, bucket_indices, result.x0_mse_per_sample.detach().double())
        if accelerator.is_main_process and accelerator.sync_gradients and global_step % config.train.log_interval == 0:
            print(
                "[joint-train] "
                + json.dumps(
                    {
                        "epoch": epoch + 1,
                        "batch": batch_index,
                        "global_step": global_step,
                        "loss": float(result.loss.detach()),
                        "p_mean": float(result.sampling_rate_mean.detach()),
                        "timestep_mean": float(result.timesteps.float().mean()),
                        "epsilon_mse": float(result.epsilon_mse_per_sample.mean()),
                        "x0_mse": float(result.x0_mse_per_sample.mean()),
                        "lr": float(optimizer.param_groups[0]["lr"]),
                    }
                ),
                flush=True,
            )
        if config.train.max_steps and global_step >= config.train.max_steps:
            reached_max_steps = True
            break
    sums = accelerator.reduce(sums, reduction="sum")
    bucket_sums = accelerator.reduce(bucket_sums, reduction="sum")
    count = max(float(sums[2].item()), 1.0)
    timestep_buckets = []
    for index in range(bucket_count):
        sample_count = int(bucket_sums[index, 0].item())
        start = index * bucket_width
        end = min((index + 1) * bucket_width, config.diffusion.train_timesteps) - 1
        timestep_buckets.append(
            {
                "start": start,
                "end": end,
                "samples": sample_count,
                "epsilon_mse": float(bucket_sums[index, 1].item() / max(sample_count, 1)),
                "x0_mse": float(bucket_sums[index, 2].item() / max(sample_count, 1)),
            }
        )
    return EpochResult(
        global_step=global_step,
        mean_loss=float(sums[0].item() / count),
        mean_sampling_rate=float(sums[1].item() / count),
        batches=int(sums[2].item()),
        elapsed_seconds=time.monotonic() - start_time,
        reached_max_steps=reached_max_steps,
        timestep_buckets=timestep_buckets,
    )
