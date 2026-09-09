"""Checkpoint IO for the isolated observation-balance branch."""

from __future__ import annotations

import os
from pathlib import Path
from typing import Any

import torch


def load_source_model(path: str | Path, model: torch.nn.Module) -> int:
    payload = torch.load(Path(path).expanduser().resolve(), map_location="cpu", weights_only=False)
    model.load_state_dict(payload["model"], strict=True)
    return int(payload.get("global_step", -1))


def load_training_checkpoint(
    path: str | Path,
    model: torch.nn.Module,
    optimizer: torch.optim.Optimizer,
    scheduler: Any,
) -> dict[str, Any]:
    payload = torch.load(Path(path).expanduser().resolve(), map_location="cpu", weights_only=False)
    model.load_state_dict(payload["model"], strict=True)
    optimizer.load_state_dict(payload["optimizer"])
    scheduler.load_state_dict(payload["scheduler"])
    return payload


def save_training_checkpoint(
    accelerator: Any,
    path: str | Path,
    model: torch.nn.Module,
    optimizer: torch.optim.Optimizer,
    scheduler: Any,
    *,
    branch_step: int,
    source_global_step: int,
    epoch: int,
    microbatches_consumed_in_epoch: int,
    source_checkpoint: str,
) -> None:
    accelerator.wait_for_everyone()
    if accelerator.is_main_process:
        destination = Path(path).expanduser().resolve()
        destination.parent.mkdir(parents=True, exist_ok=True)
        temporary = destination.with_suffix(destination.suffix + ".tmp")
        torch.save(
            {
                "format": "rmdm_w1_observation_balance_v1",
                "model": accelerator.unwrap_model(model).state_dict(),
                "optimizer": optimizer.state_dict(),
                "scheduler": scheduler.state_dict(),
                "branch_step": int(branch_step),
                "global_step": int(source_global_step + branch_step),
                "source_global_step": int(source_global_step),
                "epoch": int(epoch),
                "microbatches_consumed_in_epoch": int(microbatches_consumed_in_epoch),
                "source_checkpoint": str(source_checkpoint),
            },
            temporary,
        )
        os.replace(temporary, destination)
    accelerator.wait_for_everyone()


__all__ = ["load_source_model", "load_training_checkpoint", "save_training_checkpoint"]
