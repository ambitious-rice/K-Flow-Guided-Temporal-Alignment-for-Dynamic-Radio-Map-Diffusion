"""Fail-closed T1 gate before temporal inflation."""

from __future__ import annotations

import json
from pathlib import Path
from typing import Any

from rmdm_hvdit_v4_joint.config import ExperimentConfig
from rmdm_hvdit_v4_joint.provenance import sha256_file


def _load(path: str | Path) -> dict[str, Any]:
    with Path(path).expanduser().resolve().open("r", encoding="utf-8") as handle:
        payload = json.load(handle)
    if not isinstance(payload, dict):
        raise TypeError(f"Expected JSON object: {path}")
    return payload


def _reference_rates(reference: dict[str, Any]) -> tuple[dict[str, float], float]:
    if reference.get("schema") == "rmdm_hvdit_v4_joint_evaluation_v1":
        if reference.get("ddim_steps") != 20 or reference.get("subset_stage") != "stage_a":
            raise ValueError("T1 reference must be Stage-A DDIM20")
        rates = {
            rate: float(reference["rates"][rate]["metrics"]["full_image"]["nmse"])
            for rate in ("1", "2", "3")
        }
    elif str(reference.get("schema", "")).startswith("rmdm_sf_sparse_eval_summary"):
        records = {str(int(item["rate_percent"])): item for item in reference.get("per_rate", [])}
        if any(rate not in records for rate in ("1", "2", "3")):
            raise ValueError("Fixed-SF reference does not contain p1/p2/p3")
        if any(int(records[rate].get("ddim_steps", -1)) != 20 for rate in ("1", "2", "3")):
            raise ValueError("Fixed-SF reference is not DDIM20")
        if reference.get("subset_stage") != "stage_a":
            raise ValueError("Fixed-SF reference is not Stage-A")
        rates = {
            rate: float(records[rate]["metrics_by_domain"]["full_image"]["nmse"])
            for rate in ("1", "2", "3")
        }
    else:
        raise ValueError(f"Unsupported fixed-SF reference schema: {reference.get('schema')!r}")
    return rates, sum(rates.values()) / 3.0


def validate_t1_reference(config: ExperimentConfig, path: str | Path | None = None) -> dict[str, Any]:
    reference_file = Path(path or config.evaluation.t1_reference_summary).expanduser().resolve()
    reference = _load(reference_file)
    if str(reference.get("schema", "")).startswith("rmdm_sf_sparse_eval_summary"):
        reference_rates = [int(value) for value in reference.get("rates_percent", [])]
        if reference.get("split") != "val" or not {1, 2, 3}.issubset(reference_rates):
            raise ValueError("Fixed-SF reference must be val and contain p1/p2/p3")
        expected_checkpoint = Path(config.evaluation.sf_reference_checkpoint).expanduser().resolve()
        actual_checkpoint = Path(reference.get("checkpoint", "")).expanduser().resolve()
        if actual_checkpoint != expected_checkpoint:
            raise ValueError(
                f"Fixed-SF reference checkpoint mismatch: expected {expected_checkpoint}, got {actual_checkpoint}"
            )
        expected_subset = Path(config.evaluation.subset_manifest).expanduser().resolve()
        actual_subset = Path(reference.get("subset_manifest", "")).expanduser().resolve()
        if actual_subset != expected_subset:
            raise ValueError(f"Fixed-SF reference subset mismatch: expected {expected_subset}, got {actual_subset}")
        expected_mask_manifest = Path(config.evaluation.sf_mask_manifest).expanduser().resolve()
        actual_mask_manifest = Path(reference.get("manifest", "")).expanduser().resolve()
        if actual_mask_manifest != expected_mask_manifest:
            raise ValueError(
                f"Fixed-SF mask manifest mismatch: expected {expected_mask_manifest}, got {actual_mask_manifest}"
            )
        subset_sha256 = sha256_file(expected_subset)
        records = reference.get("per_rate", [])
        for record in records:
            subset = record.get("subset") or {}
            if (
                int(record.get("checkpoint_epoch", -1)) != 9
                or int(record.get("evaluated_frames", -1)) != 3_000
                or subset.get("stage") != "stage_a"
                or int(subset.get("video_count", -1)) != 30
                or subset.get("sha256") != subset_sha256
            ):
                raise ValueError("Fixed-SF reference does not match epoch9 / 30-video Stage-A")
    rates, macro = _reference_rates(reference)
    return {
        "path": str(reference_file),
        "sha256": sha256_file(reference_file),
        "rates": rates,
        "macro": macro,
    }


def _validate_t1_result(
    config: ExperimentConfig,
    result: dict[str, Any],
    *,
    ablated: bool,
) -> None:
    expected_manifest = Path(config.evaluation.subset_manifest).expanduser().resolve()
    actual_manifest = Path(result.get("manifest", "")).expanduser().resolve()
    checks = (
        result.get("schema") == "rmdm_hvdit_v4_joint_evaluation_v1",
        result.get("variant") == "t1",
        result.get("subset_stage") == "stage_a",
        result.get("split") == "val",
        int(result.get("video_count", -1)) == 30,
        int(result.get("window_count", -1)) == 3_000,
        int(result.get("scored_frames_per_video", -1)) == 100,
        int(result.get("ddim_steps", -1)) == 20,
        not bool(result.get("full100")),
        bool(result.get("raw_observations_ablated")) is ablated,
        actual_manifest == expected_manifest,
        set(result.get("rates", {})) == {"1", "2", "3"},
        all(int(result.get("rates", {}).get(rate, {}).get("scored_frames", -1)) == 3_000 for rate in ("1", "2", "3")),
    )
    if not all(checks):
        label = "ablated" if ablated else "full"
        raise ValueError(f"{label} T1 result violates the immutable 30-video/all100/DDIM20 Stage-A contract")


def evaluate_t1_gate(
    config: ExperimentConfig,
    *,
    full_result: dict[str, Any],
    ablated_result: dict[str, Any],
    reference_path: str | Path | None = None,
) -> dict[str, Any]:
    _validate_t1_result(config, full_result, ablated=False)
    _validate_t1_result(config, ablated_result, ablated=True)
    reference_file = Path(reference_path or config.evaluation.t1_reference_summary).expanduser().resolve()
    reference_validation = validate_t1_reference(config, reference_file)
    reference_rates = reference_validation["rates"]
    reference_macro = float(reference_validation["macro"])
    full_rates = {
        rate: float(full_result["rates"][rate]["metrics"]["full_image"]["nmse"])
        for rate in ("1", "2", "3")
    }
    ablated_rates = {
        rate: float(ablated_result["rates"][rate]["metrics"]["full_image"]["nmse"])
        for rate in ("1", "2", "3")
    }
    full_macro = sum(full_rates.values()) / 3.0
    checks = {
        "within_fixed_sf_tolerance": full_macro <= reference_macro * (1.0 + config.evaluation.t1_reference_tolerance),
        "observation_response_p1_to_p3": full_rates["1"] > full_rates["2"] > full_rates["3"],
        "raw_inputs_beat_ablation_at_each_rate": all(full_rates[rate] < ablated_rates[rate] for rate in ("1", "2", "3")),
    }
    if not config.evaluation.require_monotonic_observation_response:
        checks["observation_response_p1_to_p3"] = True
    if not config.evaluation.require_ablation_improvement_each_rate:
        checks["raw_inputs_beat_ablation_at_each_rate"] = True
    return {
        "schema": "rmdm_hvdit_v4_joint_t1_gate_v1",
        "passed": all(checks.values()),
        "checks": checks,
        "tolerance": config.evaluation.t1_reference_tolerance,
        "reference_path": str(reference_file),
        "reference_sha256": reference_validation["sha256"],
        "reference_rates": reference_rates,
        "reference_macro": reference_macro,
        "t1_rates": full_rates,
        "t1_macro": full_macro,
        "ablated_rates": ablated_rates,
    }
