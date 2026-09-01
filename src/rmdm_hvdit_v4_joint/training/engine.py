"""Shared runner utilities without dependencies on the original RMDM runner."""

from __future__ import annotations

import json
import math
import os
from pathlib import Path
from typing import Any

import torch
from accelerate import Accelerator, DataLoaderConfiguration
from accelerate.utils import DistributedDataParallelKwargs, GradientAccumulationPlugin, set_seed


def write_json_atomic(path: str | Path, payload: dict[str, Any]) -> None:
    destination = Path(path).expanduser().resolve()
    destination.parent.mkdir(parents=True, exist_ok=True)
    temporary = destination.with_suffix(destination.suffix + ".tmp")
    temporary.write_text(json.dumps(payload, ensure_ascii=False, indent=2) + "\n", encoding="utf-8")
    os.replace(temporary, destination)


def append_jsonl(path: str | Path, payload: dict[str, Any]) -> None:
    destination = Path(path).expanduser().resolve()
    destination.parent.mkdir(parents=True, exist_ok=True)
    with destination.open("a", encoding="utf-8") as handle:
        handle.write(json.dumps(payload, ensure_ascii=False, sort_keys=True) + "\n")
        handle.flush()


def require_visible_physical_gpus(expected: list[int]) -> None:
    raw = os.environ.get("CUDA_VISIBLE_DEVICES", "")
    try:
        actual = [int(value.strip()) for value in raw.split(",") if value.strip()]
    except ValueError as error:
        raise RuntimeError(f"Invalid CUDA_VISIBLE_DEVICES={raw!r}") from error
    if actual != expected:
        raise RuntimeError(
            f"This command is authorized only for physical GPUs {expected}; "
            f"CUDA_VISIBLE_DEVICES={raw!r}"
        )


def make_accelerator(
    *,
    mixed_precision: str,
    gradient_accumulation_steps: int,
    even_batches: bool = True,
    data_seed: int,
) -> Accelerator:
    return Accelerator(
        mixed_precision=mixed_precision,
        gradient_accumulation_plugin=GradientAccumulationPlugin(
            num_steps=gradient_accumulation_steps,
            sync_with_dataloader=False,
        ),
        dataloader_config=DataLoaderConfiguration(
            even_batches=even_batches,
            use_seedable_sampler=True,
            data_seed=int(data_seed),
        ),
        # The exact legacy HWM contains a small number of registered branch
        # parameters that are not active in every forward path.  This mirrors
        # the original RMDM trainer's required DDP setting.
        kwargs_handlers=[DistributedDataParallelKwargs(find_unused_parameters=True)],
    )


def prepare_model_optimizer_loader(
    accelerator: Accelerator,
    model: torch.nn.Module,
    optimizer: torch.optim.Optimizer,
    loader: Any,
) -> tuple[Any, Any, Any]:
    """Prepare distributed components while keeping the LR scheduler unwrapped.

    The configured scheduler horizon is expressed in global optimizer updates.
    ``AcceleratedScheduler`` advances an un-split four-process loader four times
    per update, so schedulers must remain local and be stepped explicitly below.
    """

    return accelerator.prepare(model, optimizer, loader)


def step_scheduler_on_global_update(accelerator: Accelerator, scheduler: Any) -> None:
    """Advance a raw scheduler exactly once for each synchronized optimizer update."""

    if accelerator.sync_gradients:
        scheduler.step()


def require_scheduler_global_step(scheduler: Any, global_step: int) -> None:
    actual = int(scheduler.last_epoch)
    if actual != int(global_step):
        raise RuntimeError(
            f"LR scheduler drifted from global optimizer updates: scheduler={actual}, global_step={global_step}"
        )


def seed_everything(seed: int) -> None:
    set_seed(int(seed), device_specific=True)


def parameter_counts(model: torch.nn.Module) -> tuple[int, int]:
    trainable = sum(parameter.numel() for parameter in model.parameters() if parameter.requires_grad)
    total = sum(parameter.numel() for parameter in model.parameters())
    return trainable, total


def validate_parameter_contract(model: torch.nn.Module, minimum: int, maximum: int) -> tuple[int, int]:
    trainable, total = parameter_counts(model)
    if not minimum <= trainable <= maximum:
        raise ValueError(f"Trainable parameter count {trainable:,} is outside [{minimum:,}, {maximum:,}]")
    return trainable, total


def make_optimizer(
    model: torch.nn.Module,
    *,
    learning_rate: float,
    betas: list[float],
    epsilon: float,
    weight_decay: float,
) -> torch.optim.AdamW:
    parameters = [parameter for parameter in model.parameters() if parameter.requires_grad]
    return torch.optim.AdamW(
        parameters,
        lr=learning_rate,
        betas=(float(betas[0]), float(betas[1])),
        eps=float(epsilon),
        weight_decay=float(weight_decay),
    )


def cosine_scheduler(
    optimizer: torch.optim.Optimizer,
    *,
    total_steps: int,
    warmup_steps: int,
    base_learning_rate: float,
    min_learning_rate: float,
) -> torch.optim.lr_scheduler.LambdaLR:
    if total_steps <= 0 or not 0 <= warmup_steps < total_steps:
        raise ValueError("Invalid total_steps/warmup_steps")
    floor = float(min_learning_rate) / float(base_learning_rate)

    def multiplier(step: int) -> float:
        if warmup_steps and step < warmup_steps:
            return max(step, 1) / warmup_steps
        progress = (step - warmup_steps) / max(total_steps - warmup_steps, 1)
        return floor + (1.0 - floor) * 0.5 * (1.0 + math.cos(math.pi * min(max(progress, 0.0), 1.0)))

    return torch.optim.lr_scheduler.LambdaLR(optimizer, multiplier)
