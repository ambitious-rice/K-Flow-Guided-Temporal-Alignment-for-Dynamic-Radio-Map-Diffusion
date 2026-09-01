"""Atomic, resumable checkpoints for the x0 pilot."""

from __future__ import annotations

import os
import random
from pathlib import Path
from typing import Any

import numpy as np
import torch

from rmdm_hvdit_v4_x0 import ARCHITECTURE_ID


CHECKPOINT_SCHEMA = "rmdm_hvdit_v4_x0_t1_checkpoint_v1"


def _write_atomic(path: str | Path, payload: dict[str, Any]) -> None:
    destination = Path(path).expanduser().resolve()
    destination.parent.mkdir(parents=True, exist_ok=True)
    temporary = destination.with_suffix(destination.suffix + ".tmp")
    torch.save(payload, temporary)
    os.replace(temporary, destination)


def save_checkpoint(
    accelerator: Any,
    path: str | Path,
    model: torch.nn.Module,
    optimizer: torch.optim.Optimizer,
    scheduler: Any,
    config: Any,
    dependency_manifest: dict[str, Any],
    *,
    epoch: int,
    global_step: int,
    microbatches_consumed_in_epoch: int,
    validation_pending: bool,
    validation_path: str = "",
) -> None:
    accelerator.wait_for_everyone()
    if accelerator.is_main_process:
        payload = {
            "schema": CHECKPOINT_SCHEMA,
            "architecture_id": ARCHITECTURE_ID,
            "model": accelerator.unwrap_model(model).state_dict(),
            "optimizer": optimizer.state_dict(),
            "scheduler": scheduler.state_dict(),
            "epoch": int(epoch),
            "global_step": int(global_step),
            "microbatches_consumed_in_epoch": int(microbatches_consumed_in_epoch),
            "validation_pending": bool(validation_pending),
            "validation_path": validation_path,
            "resolved_config": config.to_dict(),
            "dependency_manifest": dependency_manifest,
            "dependency_manifest_sha256": dependency_manifest["manifest_sha256"],
            "rng": {
                "python": random.getstate(),
                "numpy": np.random.get_state(),
                "torch": torch.get_rng_state(),
            },
        }
        _write_atomic(path, payload)
    accelerator.wait_for_everyone()


def load_checkpoint(
    path: str | Path,
    model: torch.nn.Module,
    optimizer: torch.optim.Optimizer,
    scheduler: Any,
    dependency_manifest: dict[str, Any],
) -> dict[str, Any]:
    resolved = Path(path).expanduser().resolve()
    payload = torch.load(resolved, map_location="cpu", weights_only=False)
    if payload.get("schema") != CHECKPOINT_SCHEMA:
        raise ValueError(f"Unexpected x0 pilot checkpoint schema: {payload.get('schema')!r}")
    if payload.get("architecture_id") != ARCHITECTURE_ID:
        raise ValueError("x0 pilot checkpoint architecture mismatch")
    if payload.get("dependency_manifest_sha256") != dependency_manifest.get("manifest_sha256"):
        raise ValueError("x0 pilot dependency manifest drift detected; refusing resume")
    model.load_state_dict(payload["model"], strict=True)
    optimizer.load_state_dict(payload["optimizer"])
    scheduler.load_state_dict(payload["scheduler"])
    return payload
