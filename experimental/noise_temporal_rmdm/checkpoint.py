"""Atomic, strict checkpoints for T1 and the future T16 continuation."""

from __future__ import annotations

import os
import random
from pathlib import Path
from typing import Any

import numpy as np
import torch

from . import ARCHITECTURE_ID


T1_SCHEMA = "noise_temporal_rmdm_t1_checkpoint_v2"
T16_INIT_SCHEMA = "noise_temporal_rmdm_t16_init_v2"


def _atomic_save(path: str | Path, payload: dict[str, Any]) -> None:
    destination = Path(path).expanduser().resolve()
    destination.parent.mkdir(parents=True, exist_ok=True)
    temporary = destination.with_suffix(destination.suffix + ".tmp")
    torch.save(payload, temporary)
    os.replace(temporary, destination)


def build_payload(
    model: torch.nn.Module,
    optimizer: torch.optim.Optimizer,
    scheduler: Any,
    config: Any,
    *,
    global_step: int,
    epoch: int,
    microbatches_consumed_in_epoch: int,
    source_provenance: dict[str, Any] | None = None,
    extra: dict[str, Any] | None = None,
) -> dict[str, Any]:
    return {
        "schema": T1_SCHEMA,
        "architecture_id": ARCHITECTURE_ID,
        "phase": "t1",
        "model": model.state_dict(),
        "optimizer": optimizer.state_dict(),
        "scheduler": scheduler.state_dict(),
        "global_step": int(global_step),
        "epoch": int(epoch),
        "microbatches_consumed_in_epoch": int(microbatches_consumed_in_epoch),
        "resolved_config": config.to_dict(),
        "source": source_provenance or {},
        "rng": {
            "python": random.getstate(),
            "numpy": np.random.get_state(),
            "torch": torch.get_rng_state(),
            "cuda": torch.cuda.get_rng_state_all() if torch.cuda.is_available() else [],
        },
        "extra": extra or {},
    }


def save(
    accelerator: Any,
    path: str | Path,
    model: torch.nn.Module,
    optimizer: torch.optim.Optimizer,
    scheduler: Any,
    config: Any,
    **progress: Any,
) -> None:
    accelerator.wait_for_everyone()
    if accelerator.is_main_process:
        payload = build_payload(
            accelerator.unwrap_model(model), optimizer, scheduler, config, **progress
        )
        _atomic_save(path, payload)
    accelerator.wait_for_everyone()


def load(
    path: str | Path,
    model: torch.nn.Module,
    optimizer: torch.optim.Optimizer | None = None,
    scheduler: Any | None = None,
) -> dict[str, Any]:
    payload = torch.load(Path(path).expanduser().resolve(), map_location="cpu", weights_only=False)
    if payload.get("schema") != T1_SCHEMA or payload.get("architecture_id") != ARCHITECTURE_ID:
        raise ValueError("checkpoint schema or architecture does not match noise-aware T1")
    if payload.get("phase") != "t1":
        raise ValueError("checkpoint phase is not t1")
    model.load_state_dict(payload["model"], strict=True)
    if optimizer is not None:
        optimizer.load_state_dict(payload["optimizer"])
    if scheduler is not None:
        scheduler.load_state_dict(payload["scheduler"])
    return payload
