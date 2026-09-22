"""Atomic, strict checkpoints for T1 and the future T16 continuation."""

from __future__ import annotations

import os
import random
import sys
from pathlib import Path
from typing import Any

import numpy as np
import torch

from . import T1_ARCHITECTURE_ID, T16_ARCHITECTURE_ID


T1_SCHEMA = "noise_temporal_rmdm_t1_checkpoint_v2"
T16_INIT_SCHEMA = "noise_temporal_rmdm_t16_init_v2"
T16_SCHEMA = "noise_temporal_rmdm_t16_checkpoint_v1"


def _contract(phase: str) -> tuple[str, str]:
    if phase == "t1":
        return T1_SCHEMA, T1_ARCHITECTURE_ID
    if phase == "t16":
        return T16_SCHEMA, T16_ARCHITECTURE_ID
    raise ValueError(f"unsupported checkpoint phase: {phase}")


def _load_payload(path: str | Path) -> dict[str, Any]:
    """Load local NumPy-2 checkpoints on the remote NumPy-1 environment."""

    resolved = Path(path).expanduser().resolve()
    try:
        return torch.load(resolved, map_location="cpu", weights_only=False)
    except ModuleNotFoundError as error:
        if error.name not in {"numpy._core", "numpy._core.multiarray"}:
            raise
        sys.modules.setdefault("numpy._core", np.core)
        sys.modules.setdefault("numpy._core.multiarray", np.core.multiarray)
        return torch.load(resolved, map_location="cpu", weights_only=False)


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
    phase = str(config.runtime.phase)
    schema, architecture_id = _contract(phase)
    return {
        "schema": schema,
        "architecture_id": architecture_id,
        "phase": phase,
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
    expected_phase: str = "t1",
) -> dict[str, Any]:
    payload = _load_payload(path)
    schema, architecture_id = _contract(expected_phase)
    if payload.get("schema") != schema or payload.get("architecture_id") != architecture_id:
        raise ValueError(f"checkpoint schema or architecture does not match noise-aware {expected_phase}")
    if payload.get("phase") != expected_phase:
        raise ValueError(f"checkpoint phase is not {expected_phase}")
    model.load_state_dict(payload["model"], strict=True)
    if optimizer is not None:
        optimizer.load_state_dict(payload["optimizer"])
    if scheduler is not None:
        scheduler.load_state_dict(payload["scheduler"])
    return payload


def initialize_t16_from_t1(path: str | Path, model: torch.nn.Module) -> dict[str, Any]:
    """Strictly copy every T1 tensor and leave only declared temporal tensors new."""

    payload = _load_payload(path)
    if (
        payload.get("schema") != T1_SCHEMA
        or payload.get("architecture_id") != T1_ARCHITECTURE_ID
        or payload.get("phase") != "t1"
    ):
        raise ValueError("T16 initialization requires a noise-aware T1 checkpoint")
    source = payload.get("model")
    if not isinstance(source, dict):
        raise ValueError("T1 checkpoint has no model state")
    target_keys = set(model.state_dict())
    source_keys = set(source)
    unexpected = sorted(source_keys - target_keys)
    expected_new = sorted(target_keys - source_keys)
    if unexpected or not expected_new or any(
        not key.startswith("temporal_hook.stages.") for key in expected_new
    ):
        raise ValueError(
            f"T1/T16 state contract mismatch; unexpected={unexpected}, new={expected_new[:8]}"
        )
    incompatible = model.load_state_dict(source, strict=False)
    if sorted(incompatible.missing_keys) != expected_new or incompatible.unexpected_keys:
        raise RuntimeError("T1-to-T16 state loading did not match the declared temporal extension")
    for name, module in model.named_modules():
        if name.endswith(".output"):
            if not torch.count_nonzero(module.weight).eq(0) or not torch.count_nonzero(module.bias).eq(0):
                raise RuntimeError(f"temporal output projection is not zero initialized: {name}")
    return {
        "schema": T16_INIT_SCHEMA,
        "source_checkpoint": str(Path(path).expanduser().resolve()),
        "source_global_step": int(payload["global_step"]),
        "copied_tensors": len(source_keys),
        "new_temporal_tensors": len(expected_new),
    }
