"""Audited checkpoint migration for execution-only GPU placement changes."""

from __future__ import annotations

import copy
import math
from pathlib import Path
from typing import Any

import torch

from rmdm_hvdit_v4_joint import ARCHITECTURE_ID
from rmdm_hvdit_v4_joint.config import ExperimentConfig
from rmdm_hvdit_v4_joint.provenance import (
    build_dependency_manifest,
    canonical_hash,
    sha256_file,
)
from rmdm_hvdit_v4_joint.training.checkpoint import (
    T1_CHECKPOINT_SCHEMA,
    write_torch_atomic,
)
from rmdm_hvdit_v4_joint.training.engine import write_json_atomic
from rmdm_hvdit_v4_joint.training.execution import (
    AUTHORIZED_T1_GPU_PROFILES,
    build_execution_profile,
    checkpoint_samples_consumed_in_epoch,
)


MIGRATION_SCHEMA = "rmdm_hvdit_v4_joint_execution_migration_v1"
ALLOWED_ISOLATED_FILE_CHANGES = frozenset(
    {
        "configs/hvdit_v4_joint/t1_to_w16_4gpu.yaml",
        "configs/hvdit_v4_joint/execution_8gpu_to_4gpu_20260720.yaml",
        "src/rmdm_hvdit_v4_joint/cli/audit.py",
        "src/rmdm_hvdit_v4_joint/cli/ddp_smoke.py",
        "src/rmdm_hvdit_v4_joint/cli/evaluate.py",
        "src/rmdm_hvdit_v4_joint/cli/run_pipeline.py",
        "src/rmdm_hvdit_v4_joint/cli/scheduled_supervisor.py",
        "src/rmdm_hvdit_v4_joint/cli/timed_supervisor.py",
        "src/rmdm_hvdit_v4_joint/config.py",
        "src/rmdm_hvdit_v4_joint/training/execution.py",
        "src/rmdm_hvdit_v4_joint/training/runner_t1.py",
        "src/rmdm_hvdit_v4_joint/training/runner_w16.py",
        "src/rmdm_hvdit_v4_joint/transfer/migrate_execution.py",
        "tests/hvdit_v4_joint/test_execution.py",
        "tests/hvdit_v4_joint/test_protocol.py",
    }
)

ALLOWED_CONFIG_CHANGES = {
    "t1_train.max_steps": {"source": 50_000, "target": 80_000},
    "t1_train.lr_schedule_steps": {"source": None, "target": 50_000},
    "t1_train.patience_validations": {"source": 4, "target": 2},
}


def _flatten(mapping: dict[str, Any], prefix: str = "") -> dict[str, Any]:
    flattened: dict[str, Any] = {}
    for key, value in mapping.items():
        name = f"{prefix}.{key}" if prefix else key
        if isinstance(value, dict):
            flattened.update(_flatten(value, name))
        else:
            flattened[name] = value
    return flattened


def verify_continuation_config_change(
    source: dict[str, Any],
    target: dict[str, Any],
) -> dict[str, dict[str, Any]]:
    source_flat = _flatten(source)
    target_flat = _flatten(target)
    changes = {
        key: {"source": source_flat.get(key), "target": target_flat.get(key)}
        for key in source_flat.keys() | target_flat.keys()
        if source_flat.get(key) != target_flat.get(key)
    }
    # The first migration extends the legacy 50k run.  Later migrations may
    # change only execution/recovery code while keeping the already-approved
    # 80k experiment config byte-for-byte equivalent.
    if changes and changes != ALLOWED_CONFIG_CHANGES:
        raise ValueError(f"Continuation config changed outside the authorized boundary: {changes}")
    return changes


def _verify_manifest_integrity(manifest: dict[str, Any]) -> None:
    expected = manifest.get("manifest_sha256")
    unsigned = dict(manifest)
    unsigned.pop("manifest_sha256", None)
    actual = canonical_hash(unsigned)
    if not expected or expected != actual:
        raise ValueError(
            "Dependency manifest is internally inconsistent: "
            f"expected={expected!r}, actual={actual!r}"
        )


def verify_execution_only_dependency_change(
    source: dict[str, Any],
    target: dict[str, Any],
) -> list[str]:
    _verify_manifest_integrity(source)
    _verify_manifest_integrity(target)
    for key in (
        "schema",
        "architecture_id",
        "hwm",
        "sf_reference_checkpoint",
        "evaluation_subset",
        "formal_test_subset",
        "files",
        "runtime",
    ):
        if source.get(key) != target.get(key):
            raise ValueError(f"Dependency boundary changed outside execution control: {key}")
    if source.get("config", {}).get("path") != target.get("config", {}).get("path"):
        raise ValueError("Config path changed across the continuation migration")
    if source.get("git", {}).get("commit") != target.get("git", {}).get("commit"):
        raise ValueError("Git commit changed across the execution migration")
    source_files = source.get("isolated_files", {})
    target_files = target.get("isolated_files", {})
    changed = sorted(
        key
        for key in source_files.keys() | target_files.keys()
        if source_files.get(key) != target_files.get(key)
    )
    unexpected = set(changed) - ALLOWED_ISOLATED_FILE_CHANGES
    if unexpected:
        raise ValueError(
            "Model, data, diffusion, or optimization implementation drifted: "
            f"{sorted(unexpected)}"
        )
    return changed


def migrate_t1_execution_checkpoint(
    source: str | Path,
    destination: str | Path,
    config: ExperimentConfig,
    *,
    config_path: str | Path,
    repository_root: str | Path,
    expected_global_step: int,
) -> dict[str, Any]:
    source_path = Path(source).expanduser().resolve()
    destination_path = Path(destination).expanduser().resolve()
    source_sha256 = sha256_file(source_path)
    payload = torch.load(source_path, map_location="cpu", weights_only=False)
    if (
        payload.get("schema") != T1_CHECKPOINT_SCHEMA
        or payload.get("architecture_id") != ARCHITECTURE_ID
        or payload.get("phase") != "t1"
    ):
        raise ValueError("Source is not an HV-DiT v4 joint T1 checkpoint")
    global_step = int(payload.get("global_step", -1))
    if global_step != int(expected_global_step):
        raise ValueError(
            f"Expected audited step {expected_global_step}, found step {global_step}"
        )
    target_config = config.to_dict()
    config_changes = verify_continuation_config_change(payload.get("resolved_config", {}), target_config)
    scheduler_step = int(payload.get("scheduler", {}).get("last_epoch", -1))
    if scheduler_step != global_step:
        raise ValueError("Source scheduler is not aligned with its global optimizer step")
    optimizer_lr = float(payload["optimizer"]["param_groups"][0]["lr"])
    scheduler_lr = float(payload["scheduler"]["_last_lr"][0])
    if not math.isclose(optimizer_lr, scheduler_lr, rel_tol=0.0, abs_tol=1.0e-15):
        raise ValueError("Source optimizer and scheduler learning rates do not match")
    source_train = payload["resolved_config"]["t1_train"]
    if config_changes:
        if int(config.t1_train.lr_schedule_steps) != int(source_train["max_steps"]):
            raise ValueError("The continuation must preserve the original 50k LR schedule horizon")
        if global_step >= int(config.t1_train.lr_schedule_steps):
            raise ValueError("The scheduler-horizon migration must occur before the original 50k endpoint")
    elif int(source_train.get("lr_schedule_steps", -1)) != int(config.t1_train.lr_schedule_steps):
        raise ValueError("An execution-only remigration cannot change the LR schedule horizon")

    source_manifest = payload["dependency_manifest"]
    if payload.get("dependency_manifest_sha256") != source_manifest.get("manifest_sha256"):
        raise ValueError("Source checkpoint dependency hash is inconsistent")
    target_manifest = build_dependency_manifest(
        config,
        config_path=config_path,
        repository_root=repository_root,
    )
    isolated_changes = verify_execution_only_dependency_change(source_manifest, target_manifest)

    samples_consumed, source_execution = checkpoint_samples_consumed_in_epoch(payload)
    target_eight = build_execution_profile(
        physical_gpus=list(range(8)),
        actual_world_size=8,
        per_gpu_batch_size=config.t1_train.per_gpu_batch_size,
        gradient_accumulation_steps=1,
        effective_global_batch_size=config.t1_train.effective_global_batch_size,
        authorized_profiles=AUTHORIZED_T1_GPU_PROFILES,
    )
    target_four = build_execution_profile(
        physical_gpus=[4, 5, 6, 7],
        actual_world_size=4,
        per_gpu_batch_size=config.t1_train.per_gpu_batch_size,
        gradient_accumulation_steps=2,
        effective_global_batch_size=config.t1_train.effective_global_batch_size,
        authorized_profiles=AUTHORIZED_T1_GPU_PROFILES,
    )
    for profile in (target_eight, target_four):
        if samples_consumed % profile.global_microbatch_size:
            raise ValueError(
                "Checkpoint data cursor is not exactly representable by an authorized target: "
                f"samples={samples_consumed}, profile={profile.to_dict()}"
            )

    extra = copy.deepcopy(payload.get("extra", {}))
    migration = {
        "schema": MIGRATION_SCHEMA,
        "source_checkpoint": str(source_path),
        "source_checkpoint_sha256": source_sha256,
        "source_dependency_manifest_sha256": source_manifest["manifest_sha256"],
        "target_dependency_manifest_sha256": target_manifest["manifest_sha256"],
        "target_config_sha256": sha256_file(config_path),
        "target_physical_gpus": list(config.pipeline.allowed_physical_gpus),
        "global_step": global_step,
        "completed_epoch": int(payload.get("completed_epoch", -1)),
        "scheduler_step": scheduler_step,
        "optimizer_learning_rate": optimizer_lr,
        "samples_consumed_in_epoch": samples_consumed,
        "source_execution_profile": source_execution,
        "eight_gpu_execution_profile": target_eight.to_dict(),
        "four_gpu_execution_profile": target_four.to_dict(),
        "isolated_file_changes": isolated_changes,
        "config_changes": config_changes,
        "model_state_preserved": True,
        "optimizer_state_preserved": True,
        "scheduler_state_preserved": True,
        "rng_state_preserved": True,
        "resolved_config_updated_with_authorized_changes": True,
    }
    extra["samples_consumed_in_epoch"] = samples_consumed
    extra["microbatches_consumed_in_epoch"] = (
        samples_consumed // target_eight.global_microbatch_size
    )
    extra["execution_profile"] = target_eight.to_dict()
    extra.setdefault("execution_migrations", []).append(migration)
    payload["dependency_manifest"] = target_manifest
    payload["dependency_manifest_sha256"] = target_manifest["manifest_sha256"]
    payload["resolved_config"] = copy.deepcopy(target_config)
    payload["extra"] = extra
    payload["migration"] = migration
    write_torch_atomic(destination_path, payload)
    return {
        **migration,
        "destination_checkpoint": str(destination_path),
        "destination_checkpoint_sha256": sha256_file(destination_path),
    }


def migrate_pipeline_state(
    path: str | Path,
    report: dict[str, Any],
) -> dict[str, Any]:
    state_path = Path(path).expanduser().resolve()
    state = __import__("json").loads(state_path.read_text(encoding="utf-8"))
    source_hash = report["source_dependency_manifest_sha256"]
    target_hash = report["target_dependency_manifest_sha256"]
    state_source_hash = state.get("dependency_manifest_sha256")
    prior_targets_from_same_source = {
        migration.get("target_dependency_manifest_sha256")
        for migration in state.get("execution_migrations", [])
        if migration.get("source_dependency_manifest_sha256") == source_hash
    }
    if state_source_hash != source_hash and state_source_hash not in prior_targets_from_same_source:
        raise ValueError(
            "Pipeline state does not match the source dependency manifest: "
            f"state={state_source_hash!r}, source={source_hash!r}, "
            f"authorized_prior_targets={sorted(value for value in prior_targets_from_same_source if value)!r}"
        )
    state["dependency_manifest_sha256"] = target_hash
    state["config_sha256"] = report["target_config_sha256"]
    state["authorized_physical_gpus"] = list(report["target_physical_gpus"])
    state.setdefault("execution_migrations", []).append(
        {
            "schema": MIGRATION_SCHEMA,
            "pipeline_state_source_dependency_manifest_sha256": state_source_hash,
            "source_dependency_manifest_sha256": source_hash,
            "target_dependency_manifest_sha256": target_hash,
            "source_checkpoint": report["source_checkpoint"],
            "source_checkpoint_sha256": report["source_checkpoint_sha256"],
            "destination_checkpoint": report["destination_checkpoint"],
            "destination_checkpoint_sha256": report["destination_checkpoint_sha256"],
            "global_step": report["global_step"],
            "samples_consumed_in_epoch": report["samples_consumed_in_epoch"],
        }
    )
    state.setdefault("phases", {})["t1_training_and_gate"] = {
        "status": "pending_execution_resume",
        "resume_checkpoint": report["destination_checkpoint"],
        "global_step": report["global_step"],
    }
    # Both checks are placement-sensitive: the environment must see all eight
    # devices, and W16's chosen microbatch must fit the least-free target GPU.
    state["phases"].pop("environment_audit", None)
    state["phases"].pop("cuda_smoke", None)
    state.pop("resource_wait", None)
    state["state"] = "running"
    state["current_phase"] = "t1_training_and_gate"
    write_json_atomic(state_path, state)
    return state
