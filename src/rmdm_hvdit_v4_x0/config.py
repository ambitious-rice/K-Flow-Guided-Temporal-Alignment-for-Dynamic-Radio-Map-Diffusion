"""Strict configuration boundary for the isolated V4-W1 x0 pilot."""

from __future__ import annotations

import copy
from pathlib import Path

import yaml

from rmdm_hvdit_v4_joint.config import (
    ExperimentConfig,
    _from_mapping,
)


def _validate_pilot(config: ExperimentConfig) -> None:
    """Validate the pilot and reuse every unchanged V4 invariant.

    A temporary baseline-shaped copy is passed through V4's original validator.
    The real pilot object remains sample-prediction, 10k-only, and GPU 0-3.
    """

    if config.diffusion.prediction_type != "sample":
        raise ValueError("The x0 pilot must use DDIM prediction_type='sample'")
    if (
        config.diffusion.train_timesteps != 1_000
        or config.diffusion.beta_schedule != "linear"
        or config.diffusion.ddim_steps != 20
    ):
        raise ValueError("The x0 pilot preserves the V4 linear/1000-step/DDIM20 contract")
    if config.t1_train.max_steps != 10_000:
        raise ValueError("The exploratory x0 pilot must stop exactly at step 10,000")
    if config.t1_train.validation_first_step != 10_000:
        raise ValueError("The exploratory x0 pilot validates exactly at step 10,000")
    if config.t1_train.lr_schedule_steps != 50_000:
        raise ValueError("The x0 pilot must preserve V4's first-10k learning-rate trajectory")
    no_tx = not config.model.use_explicit_tx_condition
    allowed_profiles = (
        {
            (tuple([2, 3, 4, 5]), 32, 2),
            (tuple([4, 5, 6, 7]), 32, 2),
            (tuple(range(8)), 16, 2),
        }
        if no_tx
        else {(tuple([0, 1, 2, 3]), 32, 2)}
    )
    actual_profile = (
        tuple(config.pipeline.allowed_physical_gpus),
        config.t1_train.per_gpu_batch_size,
        config.t1_train.gradient_accumulation_steps,
    )
    if (
        actual_profile not in allowed_profiles
        or config.t1_train.effective_global_batch_size != 256
    ):
        raise ValueError("The x0 pilot execution profile must preserve global batch 256")
    expected_gpus = list(actual_profile[0])
    expected_root = (
        "runs/rmdm_hvdit_v4_x0_no_tx_strict"
        if no_tx and not config.model.use_tx_source_supervision
        else "runs/rmdm_hvdit_v4_x0_no_tx"
        if no_tx
        else "runs/rmdm_hvdit_v4_x0"
    )
    if config.pipeline.allowed_physical_gpus != expected_gpus:
        raise ValueError(f"The x0 pilot is authorized only on physical GPUs {expected_gpus}")
    if config.pipeline.allow_gpu_co_tenancy:
        raise ValueError("The x0 pilot requires exclusive use of GPUs 0-3")
    if config.pipeline.output_root != expected_root:
        raise ValueError(f"The x0 pilot must write only below {expected_root}")
    if config.evaluation.rates != [1.0, 2.0, 3.0] or config.evaluation.ddim_steps != 20:
        raise ValueError("The x0 pilot evaluation is fixed to Stage-A p1/p2/p3 and DDIM20")

    baseline = copy.deepcopy(config)
    baseline.diffusion.prediction_type = "epsilon"
    baseline.t1_train.max_steps = 80_000
    baseline.pipeline.output_root = "runs/rmdm_hvdit_v4_joint"
    baseline.pipeline.allowed_physical_gpus = expected_gpus if no_tx else [4, 5, 6, 7]
    baseline.pipeline.allow_gpu_co_tenancy = False if no_tx else True
    if no_tx and len(expected_gpus) == 8:
        baseline.w16_train.default_gradient_accumulation_steps = 4
    baseline.pipeline.lock_file = "runs/rmdm_hvdit_v4_joint/.pipeline.lock"
    baseline.validate()


def load_config(path: str | Path) -> ExperimentConfig:
    resolved = Path(path).expanduser().resolve()
    with resolved.open("r", encoding="utf-8") as handle:
        payload = yaml.safe_load(handle) or {}
    if not isinstance(payload, dict):
        raise TypeError(f"Configuration root must be a mapping: {resolved}")
    config = _from_mapping(ExperimentConfig, payload)
    _validate_pilot(config)
    return config


__all__ = ["ExperimentConfig", "load_config"]
