"""Atomic, phase-specific HV-DiT v4 joint checkpoints."""

from __future__ import annotations

import os
import random
from pathlib import Path
from typing import Any

import numpy as np
import torch

from rmdm_hvdit_v4_joint import ARCHITECTURE_ID
from rmdm_hvdit_v4_joint.config import ExperimentConfig
from rmdm_hvdit_v4_joint.provenance import assert_dependency_match


T1_CHECKPOINT_SCHEMA = "rmdm_hvdit_v4_joint_t1_checkpoint_v2"
W16_INIT_SCHEMA = "rmdm_hvdit_v4_joint_w16_init_v2"
W16_CHECKPOINT_SCHEMA = "rmdm_hvdit_v4_joint_w16_checkpoint_v2"
W16_SELECTION_CANDIDATE_SCHEMA = "rmdm_hvdit_v4_joint_w16_selection_candidate_v2"


def write_torch_atomic(path: str | Path, payload: dict[str, Any]) -> None:
    destination = Path(path).expanduser().resolve()
    destination.parent.mkdir(parents=True, exist_ok=True)
    temporary = destination.with_suffix(destination.suffix + ".tmp")
    torch.save(payload, temporary)
    os.replace(temporary, destination)


def phase_schema(phase: str) -> str:
    if phase == "t1":
        return T1_CHECKPOINT_SCHEMA
    if phase == "w16":
        return W16_CHECKPOINT_SCHEMA
    raise ValueError(f"Unsupported training phase: {phase}")


def save_training_checkpoint(
    accelerator: Any,
    path: str | Path,
    model: torch.nn.Module,
    optimizer: torch.optim.Optimizer,
    scheduler: Any,
    config: ExperimentConfig,
    dependency_manifest: dict[str, Any],
    *,
    phase: str,
    completed_epoch: int,
    global_step: int,
    extra: dict[str, Any] | None = None,
) -> None:
    accelerator.wait_for_everyone()
    if accelerator.is_main_process:
        payload = {
            "schema": phase_schema(phase),
            "architecture_id": ARCHITECTURE_ID,
            "phase": phase,
            "model": accelerator.unwrap_model(model).state_dict(),
            "optimizer": optimizer.state_dict(),
            "scheduler": scheduler.state_dict(),
            "completed_epoch": int(completed_epoch),
            "global_step": int(global_step),
            "resolved_config": config.to_dict(),
            "dependency_manifest": dependency_manifest,
            "dependency_manifest_sha256": dependency_manifest["manifest_sha256"],
            "rng": {
                "python": random.getstate(),
                "numpy": np.random.get_state(),
                "torch": torch.get_rng_state(),
            },
            "extra": extra or {},
        }
        write_torch_atomic(path, payload)
    accelerator.wait_for_everyone()


def load_training_checkpoint(
    path: str | Path,
    model: torch.nn.Module,
    *,
    phase: str,
    dependency_manifest: dict[str, Any],
    optimizer: torch.optim.Optimizer | None = None,
    scheduler: Any | None = None,
) -> dict[str, Any]:
    resolved = Path(path).expanduser().resolve()
    payload = torch.load(resolved, map_location="cpu", weights_only=False)
    if payload.get("schema") != phase_schema(phase):
        raise ValueError(f"Checkpoint schema/phase mismatch in {resolved}: {payload.get('schema')!r}")
    if payload.get("architecture_id") != ARCHITECTURE_ID or payload.get("phase") != phase:
        raise ValueError(f"Checkpoint architecture mismatch in {resolved}")
    assert_dependency_match(payload["dependency_manifest"], dependency_manifest)
    model.load_state_dict(payload["model"], strict=True)
    if optimizer is not None:
        optimizer.load_state_dict(payload["optimizer"])
    if scheduler is not None:
        scheduler.load_state_dict(payload["scheduler"])
    return payload
