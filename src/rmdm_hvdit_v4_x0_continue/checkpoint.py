"""Audited source migration and resumable continuation checkpoints."""

from __future__ import annotations

import os
import random
from pathlib import Path
from typing import Any

import numpy as np
import torch

from . import ARCHITECTURE_ID


SOURCE_SCHEMA = "rmdm_hvdit_v4_x0_t1_checkpoint_v1"
SOURCE_ARCHITECTURE_ID = "rmdm_hvdit_v4_x0_ddpm_sample_pilot_v1"
CONTINUATION_SCHEMA = "rmdm_hvdit_v4_x0_continue_checkpoint_v1"

ALLOWED_SOURCE_CONFIG_DIFFERENCES = {
    "t1_train.max_steps",
    "t1_train.per_gpu_batch_size",
    "t1_train.validation_first_step",
    "t1_train.patience_validations",
    "pipeline.output_root",
    "pipeline.allowed_physical_gpus",
    "pipeline.allow_gpu_co_tenancy",
    "pipeline.free_memory_mib",
    "pipeline.lock_file",
}


def _flatten(value: Any, prefix: str = "") -> dict[str, Any]:
    if not isinstance(value, dict):
        return {prefix: value}
    result: dict[str, Any] = {}
    for key, item in value.items():
        path = f"{prefix}.{key}" if prefix else str(key)
        result.update(_flatten(item, path))
    return result


def _validate_source_config(
    source: dict[str, Any],
    target: dict[str, Any],
    required_differences: set[str] | None = None,
) -> None:
    source_flat = _flatten(source)
    target_flat = _flatten(target)
    if set(source_flat) != set(target_flat):
        raise ValueError("Source and continuation config schemas differ")
    differences = {
        key for key in source_flat if source_flat[key] != target_flat[key]
    }
    unexpected = differences - ALLOWED_SOURCE_CONFIG_DIFFERENCES
    if unexpected:
        raise ValueError(f"Continuation changes training semantics at {sorted(unexpected)}")
    required = required_differences or {
        "t1_train.max_steps",
        "t1_train.validation_first_step",
        "pipeline.output_root",
        "pipeline.allowed_physical_gpus",
        "pipeline.allow_gpu_co_tenancy",
        "pipeline.free_memory_mib",
        "pipeline.lock_file",
    }
    if differences != required:
        raise ValueError(f"Unexpected continuation difference set: {sorted(differences)}")


def load_source_checkpoint(
    path: str | Path,
    model: torch.nn.Module,
    optimizer: torch.optim.Optimizer,
    scheduler: Any,
    config: Any,
    required_config_differences: set[str] | None = None,
) -> dict[str, Any]:
    resolved = Path(path).expanduser().resolve()
    payload = torch.load(resolved, map_location="cpu", weights_only=False)
    if payload.get("schema") != SOURCE_SCHEMA or payload.get("architecture_id") != SOURCE_ARCHITECTURE_ID:
        raise ValueError("Continuation source is not the completed V4-W1 x0 pilot")
    if (
        int(payload.get("global_step", -1)) != 10_000
        or int(payload.get("scheduler", {}).get("last_epoch", -1)) != 10_000
        or bool(payload.get("validation_pending", True))
    ):
        raise ValueError("Continuation source must be the fully validated step-10k checkpoint")
    source_config = payload.get("resolved_config")
    if not isinstance(source_config, dict):
        raise ValueError("Source checkpoint lacks its resolved configuration")
    _validate_source_config(
        source_config,
        config.to_dict(),
        required_differences=required_config_differences,
    )
    validation_path = Path(payload.get("validation_path", "")).expanduser().resolve()
    if not validation_path.is_file():
        raise FileNotFoundError(f"Source Stage-A result is absent: {validation_path}")
    model.load_state_dict(payload["model"], strict=True)
    optimizer.load_state_dict(payload["optimizer"])
    scheduler.load_state_dict(payload["scheduler"])
    return payload


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
    extra: dict[str, Any],
) -> None:
    accelerator.wait_for_everyone()
    if accelerator.is_main_process:
        payload = {
            "schema": CONTINUATION_SCHEMA,
            "architecture_id": ARCHITECTURE_ID,
            "model": accelerator.unwrap_model(model).state_dict(),
            "optimizer": optimizer.state_dict(),
            "scheduler": scheduler.state_dict(),
            "epoch": int(epoch),
            "global_step": int(global_step),
            "microbatches_consumed_in_epoch": int(microbatches_consumed_in_epoch),
            "validation_pending": bool(validation_pending),
            "extra": extra,
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


def load_continuation_checkpoint(
    path: str | Path,
    model: torch.nn.Module,
    optimizer: torch.optim.Optimizer,
    scheduler: Any,
    dependency_manifest: dict[str, Any],
) -> dict[str, Any]:
    payload = torch.load(Path(path).expanduser().resolve(), map_location="cpu", weights_only=False)
    if payload.get("schema") != CONTINUATION_SCHEMA or payload.get("architecture_id") != ARCHITECTURE_ID:
        raise ValueError("Not an x0 continuation checkpoint")
    if payload.get("dependency_manifest_sha256") != dependency_manifest.get("manifest_sha256"):
        raise ValueError("Continuation dependency drift detected; refusing resume")
    model.load_state_dict(payload["model"], strict=True)
    optimizer.load_state_dict(payload["optimizer"])
    scheduler.load_state_dict(payload["scheduler"])
    return payload


def load_objective_finetune_checkpoint(
    path: str | Path,
    model: torch.nn.Module,
    optimizer: torch.optim.Optimizer,
    scheduler: Any,
) -> dict[str, Any]:
    """Load a validated W1 continuation as the start of a new objective branch."""

    payload = torch.load(Path(path).expanduser().resolve(), map_location="cpu", weights_only=False)
    if payload.get("schema") != CONTINUATION_SCHEMA or payload.get("architecture_id") != ARCHITECTURE_ID:
        raise ValueError("Objective fine-tuning requires a W1 continuation checkpoint")
    if bool(payload.get("validation_pending", True)):
        raise ValueError("Objective fine-tuning requires a fully validated checkpoint")
    model.load_state_dict(payload["model"], strict=True)
    optimizer.load_state_dict(payload["optimizer"])
    scheduler.load_state_dict(payload["scheduler"])
    return payload
