"""Auditable 2-D-to-3-D inflation with no permissive state loading."""

from __future__ import annotations

import hashlib
from pathlib import Path
from typing import Any

import torch

from rmdm_hvdit_v4_joint import ARCHITECTURE_ID
from rmdm_hvdit_v4_joint.config import ExperimentConfig
from rmdm_hvdit_v4_joint.model import build_t1_system, build_w16_system
from rmdm_hvdit_v4_joint.provenance import canonical_hash, sha256_file
from rmdm_hvdit_v4_joint.training.checkpoint import T1_CHECKPOINT_SCHEMA, W16_INIT_SCHEMA, write_torch_atomic


TEMPORAL_STEM_KEYS = {
    "denoiser.input_stem.dense_projection.weight",
    "denoiser.input_stem.observation_projection.weight",
    "denoiser.condition_stem.dense_projection.weight",
    "denoiser.condition_stem.observation_projection.weight",
}
TEMPORAL_MERGE_KEYS = {
    "denoiser.hierarchy_merge.projection.weight",
    "denoiser.condition_merge.projection.weight",
}
TEMPORAL_EXPAND_KEY = "denoiser.hierarchy_expand.projection.weight"
TEMPORAL_OUTPUT_WEIGHT = "denoiser.output_head.token_projection.weight"
TEMPORAL_OUTPUT_BIAS = "denoiser.output_head.token_projection.bias"


def _tensor_sha256(value: torch.Tensor) -> str:
    contiguous = value.detach().cpu().contiguous()
    return hashlib.sha256(contiguous.numpy().tobytes()).hexdigest()


def _inflate_merge(value: torch.Tensor, target_shape: torch.Size) -> torch.Tensor:
    """Each W16 temporal half receives 0.5 times the T1 spatial merge."""

    if value.ndim != 2 or target_shape != torch.Size((value.shape[0], 2 * value.shape[1])):
        raise ValueError(f"Unexpected merge inflation {tuple(value.shape)} -> {tuple(target_shape)}")
    inflated = torch.cat((0.5 * value, 0.5 * value), dim=1)
    if inflated.shape != target_shape:
        raise ValueError(f"Inflated stem has shape {tuple(inflated.shape)}, expected {tuple(target_shape)}")
    return inflated


def _inflate_stem(value: torch.Tensor, target_shape: torch.Size) -> torch.Tensor:
    """A repeated, losslessly packed temporal pair preserves the T1 token."""

    if value.ndim != 2 or target_shape != torch.Size((value.shape[0], 2 * value.shape[1])):
        raise ValueError(f"Unexpected stem inflation {tuple(value.shape)} -> {tuple(target_shape)}")
    inflated = torch.cat((0.5 * value, 0.5 * value), dim=1)
    if inflated.shape != target_shape:
        raise ValueError(f"Inflated stem has shape {tuple(inflated.shape)}, expected {tuple(target_shape)}")
    return inflated


def _inflate_expand(value: torch.Tensor, target_shape: torch.Size) -> torch.Tensor:
    """Both W16 temporal outputs receive a full copy of the T1 spatial expand."""

    if value.ndim != 2 or target_shape != torch.Size((2 * value.shape[0], value.shape[1])):
        raise ValueError(f"Unexpected expand inflation {tuple(value.shape)} -> {tuple(target_shape)}")
    duplicated = torch.cat((value, value), dim=0)
    if duplicated.shape != target_shape:
        raise ValueError(f"Duplicated output has shape {tuple(duplicated.shape)}, expected {tuple(target_shape)}")
    return duplicated


def _inflate_output(value: torch.Tensor, target_shape: torch.Size) -> torch.Tensor:
    """Both frames receive the exact T1 token-to-feature projection."""

    if value.ndim not in (1, 2) or target_shape != torch.Size((2 * value.shape[0], *value.shape[1:])):
        raise ValueError(f"Unexpected output inflation {tuple(value.shape)} -> {tuple(target_shape)}")
    duplicated = torch.cat((value, value), dim=0)
    if duplicated.shape != target_shape:
        raise ValueError(f"Inflated output has shape {tuple(duplicated.shape)}, expected {tuple(target_shape)}")
    return duplicated


def _inflate_state(
    source: dict[str, torch.Tensor],
    target_template: dict[str, torch.Tensor],
) -> tuple[dict[str, torch.Tensor], list[dict[str, Any]]]:
    if set(source) != set(target_template):
        missing = sorted(set(target_template) - set(source))
        unexpected = sorted(set(source) - set(target_template))
        raise KeyError(f"T1/W16 state keys differ; missing={missing}, unexpected={unexpected}")
    result: dict[str, torch.Tensor] = {}
    records: list[dict[str, Any]] = []
    for key in sorted(target_template):
        source_value = source[key].detach().cpu()
        target_value = target_template[key].detach().cpu()
        if key in TEMPORAL_STEM_KEYS:
            transferred = _inflate_stem(source_value, target_value.shape)
            rule = "temporal_stem_halves_each_half_t1_packed_projection"
        elif key in TEMPORAL_MERGE_KEYS:
            transferred = _inflate_merge(source_value, target_value.shape)
            rule = "temporal_halves_each_half_spatial_merge"
        elif key == TEMPORAL_EXPAND_KEY:
            transferred = _inflate_expand(source_value, target_value.shape)
            rule = "temporal_halves_each_full_spatial_expand"
        elif key in {TEMPORAL_OUTPUT_WEIGHT, TEMPORAL_OUTPUT_BIAS}:
            transferred = _inflate_output(source_value, target_value.shape)
            rule = "temporal_decoder_frames_each_exact_t1_projection"
        else:
            if source_value.shape != target_value.shape:
                raise ValueError(
                    f"Key {key!r} is not an approved transform but changes shape "
                    f"{tuple(source_value.shape)} -> {tuple(target_value.shape)}"
                )
            transferred = source_value.clone()
            rule = "exact_copy"
        result[key] = transferred
        records.append(
            {
                "key": key,
                "rule": rule,
                "source_shape": list(source_value.shape),
                "target_shape": list(transferred.shape),
                "source_sha256": _tensor_sha256(source_value),
                "target_sha256": _tensor_sha256(transferred),
            }
        )
    return result, records


def inflate_t1_checkpoint(
    config: ExperimentConfig,
    *,
    t1_checkpoint: str | Path,
    output_path: str | Path,
) -> dict[str, Any]:
    """Create and immediately re-audit a CPU W16 initialization artifact."""

    source_path = Path(t1_checkpoint).expanduser().resolve()
    payload = torch.load(source_path, map_location="cpu", weights_only=False)
    if payload.get("schema") != T1_CHECKPOINT_SCHEMA or payload.get("phase") != "t1":
        raise ValueError(f"Not an HV-DiT v4 joint T1 checkpoint: {source_path}")
    if payload.get("architecture_id") != ARCHITECTURE_ID:
        raise ValueError("T1 checkpoint architecture_id does not match current HV-DiT v4 joint")

    t1_model = build_t1_system(config, attention_backend="reference")
    t1_model.load_state_dict(payload["model"], strict=True)
    w16_model = build_w16_system(config, attention_backend="reference")
    inflated, records = _inflate_state(t1_model.state_dict(), w16_model.state_dict())
    w16_model.load_state_dict(inflated, strict=True)

    audit = {
        "schema": "rmdm_hvdit_v4_joint_inflation_audit_v2",
        "source_checkpoint": str(source_path),
        "source_checkpoint_sha256": sha256_file(source_path),
        "source_schema": payload["schema"],
        "target_schema": W16_INIT_SCHEMA,
        "unexpected_missing_keys": [],
        "declared_new_initialization": [],
        "record_count": len(records),
        "records": records,
    }
    audit["audit_sha256"] = canonical_hash(audit)
    artifact = {
        "schema": W16_INIT_SCHEMA,
        "architecture_id": ARCHITECTURE_ID,
        "phase": "w16_init",
        "model": w16_model.state_dict(),
        "resolved_config": config.to_dict(),
        "dependency_manifest": payload["dependency_manifest"],
        "dependency_manifest_sha256": payload["dependency_manifest_sha256"],
        "source_t1_checkpoint": str(source_path),
        "source_t1_checkpoint_sha256": audit["source_checkpoint_sha256"],
        "inflation_audit": audit,
        "inflation_audit_sha256": audit["audit_sha256"],
    }
    destination = Path(output_path).expanduser().resolve()
    write_torch_atomic(destination, artifact)

    reloaded = torch.load(destination, map_location="cpu", weights_only=False)
    if reloaded.get("schema") != W16_INIT_SCHEMA or reloaded.get("inflation_audit_sha256") != audit["audit_sha256"]:
        raise RuntimeError("Serialized W16 initialization failed schema/hash audit")
    verifier = build_w16_system(config, attention_backend="reference")
    verifier.load_state_dict(reloaded["model"], strict=True)
    for key, expected in inflated.items():
        actual = verifier.state_dict()[key]
        if not torch.equal(actual, expected):
            raise RuntimeError(f"Serialized W16 tensor changed during audit: {key}")
    return audit


def load_w16_initialization(
    path: str | Path,
    model: torch.nn.Module,
    *,
    expected_dependency_manifest_sha256: str,
) -> dict[str, Any]:
    resolved = Path(path).expanduser().resolve()
    payload = torch.load(resolved, map_location="cpu", weights_only=False)
    if payload.get("schema") != W16_INIT_SCHEMA or payload.get("architecture_id") != ARCHITECTURE_ID:
        raise ValueError(f"Unsupported W16 initialization: {resolved}")
    if payload.get("dependency_manifest_sha256") != expected_dependency_manifest_sha256:
        raise ValueError("W16 initialization dependency manifest does not match this run")
    model.load_state_dict(payload["model"], strict=True)
    return payload
