"""Atomic full-state checkpoints for JointRMDM."""

from __future__ import annotations

import hashlib
import json
import os
import random
from functools import lru_cache
from pathlib import Path
from typing import Any

import numpy as np
import torch

from rmdm.config import ExperimentConfig


CHECKPOINT_SCHEMA = "rmdm_joint_w16_checkpoint_v1"


@lru_cache(maxsize=8)
def sha256_file(path: str | Path, chunk_size: int = 4 << 20) -> str:
    digest = hashlib.sha256()
    with Path(path).expanduser().resolve().open("rb") as handle:
        while chunk := handle.read(chunk_size):
            digest.update(chunk)
    return digest.hexdigest()


def write_json_atomic(path: str | Path, payload: dict[str, Any]) -> None:
    path = Path(path)
    path.parent.mkdir(parents=True, exist_ok=True)
    temporary = path.with_suffix(path.suffix + ".tmp")
    temporary.write_text(json.dumps(payload, ensure_ascii=False, indent=2) + "\n", encoding="utf-8")
    os.replace(temporary, path)


def save_checkpoint(
    accelerator: Any,
    path: str | Path,
    model: torch.nn.Module,
    optimizer: torch.optim.Optimizer,
    lr_scheduler: Any,
    config: ExperimentConfig,
    *,
    completed_epoch: int,
    global_step: int,
    extra: dict[str, Any] | None = None,
) -> None:
    accelerator.wait_for_everyone()
    if accelerator.is_main_process:
        path = Path(path).expanduser().resolve()
        path.parent.mkdir(parents=True, exist_ok=True)
        model_state = accelerator.unwrap_model(model).state_dict()
        payload = {
            "schema": CHECKPOINT_SCHEMA,
            "model": model_state,
            "optimizer": optimizer.state_dict(),
            "lr_scheduler": lr_scheduler.state_dict(),
            "completed_epoch": int(completed_epoch),
            "global_step": int(global_step),
            "config": config.to_dict(),
            "stage1_checkpoint_sha256": sha256_file(config.stage1.checkpoint),
            "rng": {
                "python": random.getstate(),
                "numpy": np.random.get_state(),
                "torch": torch.get_rng_state(),
            },
            "extra": extra or {},
        }
        temporary = path.with_suffix(path.suffix + ".tmp")
        torch.save(payload, temporary)
        os.replace(temporary, path)
    accelerator.wait_for_everyone()


def load_checkpoint(
    path: str | Path,
    model: torch.nn.Module,
    optimizer: torch.optim.Optimizer | None = None,
) -> dict[str, Any]:
    path = Path(path).expanduser().resolve()
    payload = torch.load(path, map_location="cpu", weights_only=False)
    if payload.get("schema") != CHECKPOINT_SCHEMA:
        raise ValueError(f"Unsupported checkpoint schema in {path}: {payload.get('schema')}")
    model.load_state_dict(payload["model"], strict=True)
    if optimizer is not None:
        optimizer.load_state_dict(payload["optimizer"])
    return payload
