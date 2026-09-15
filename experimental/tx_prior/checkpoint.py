"""Atomic checkpoints owned by the single tx_prior run."""

from __future__ import annotations

import os
import random
from pathlib import Path
from typing import Any

import numpy as np
import torch

SCHEMA = "tx_prior_w1_checkpoint_v1"


def save(
    accelerator: Any,
    path: Path,
    model: Any,
    optimizer: Any,
    scheduler: Any,
    config: Any,
    **progress: Any,
) -> None:
    accelerator.wait_for_everyone()
    if accelerator.is_main_process:
        path.parent.mkdir(parents=True, exist_ok=True)
        temporary = path.with_suffix(path.suffix + ".tmp")
        payload = {
            "schema": SCHEMA,
            "model": accelerator.unwrap_model(model).state_dict(),
            "optimizer": optimizer.state_dict(),
            "scheduler": scheduler.state_dict(),
            "resolved_config": config.to_dict(),
            "rng": {
                "python": random.getstate(),
                "numpy": np.random.get_state(),
                "torch": torch.get_rng_state(),
                "cuda": torch.cuda.get_rng_state_all() if torch.cuda.is_available() else [],
            },
            **progress,
        }
        torch.save(payload, temporary)
        os.replace(temporary, path)
    accelerator.wait_for_everyone()


def load(
    path: str | Path,
    model: Any,
    optimizer: Any,
    scheduler: Any,
) -> dict[str, Any]:
    payload = torch.load(
        Path(path).expanduser().resolve(), map_location="cpu", weights_only=False
    )
    if payload.get("schema") != SCHEMA:
        raise ValueError(f"unexpected tx_prior checkpoint schema: {payload.get('schema')!r}")
    model.load_state_dict(payload["model"], strict=True)
    optimizer.load_state_dict(payload["optimizer"])
    scheduler.load_state_dict(payload["scheduler"])
    rng = payload.get("rng")
    if not rng:
        raise ValueError("tx_prior checkpoint misses RNG state")
    random.setstate(rng["python"])
    np.random.set_state(rng["numpy"])
    torch.set_rng_state(rng["torch"])
    if torch.cuda.is_available() and rng["cuda"]:
        torch.cuda.set_rng_state_all(rng["cuda"])
    return payload
