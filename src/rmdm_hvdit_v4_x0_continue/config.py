"""Strict continuation configuration without changing the completed 10k pilot."""

from __future__ import annotations

import copy
from pathlib import Path

import yaml

from rmdm_hvdit_v4_joint.config import ExperimentConfig, _from_mapping


def _validate_continuation(config: ExperimentConfig) -> None:
    if (
        config.diffusion.prediction_type != "sample"
        or config.diffusion.train_timesteps != 1_000
        or config.diffusion.beta_schedule != "linear"
        or config.diffusion.ddim_steps != 20
    ):
        raise ValueError("x0 continuation must preserve sample/linear/1000-step/DDIM20")
    train = config.t1_train
    validation_schedule = (
        train.validation_first_step,
        train.validation_every_steps,
    )
    standard_schedule = (
        train.max_steps == 50_000
        and validation_schedule in {(15_000, 5_000), (26_000, 1_000)}
        and train.early_stop_min_step == 25_000
    )
    full_observation_epoch_schedule = (
        (
            train.max_steps == 33_000
            and validation_schedule == (28_000, 1_000)
            and train.early_stop_min_step == 33_000
        )
        or (
            train.max_steps == 40_000
            and validation_schedule == (29_000, 2_000)
            and train.early_stop_min_step == 40_000
        )
        or (
            train.max_steps == 40_000
            and validation_schedule == (2_000, 2_000)
            and train.early_stop_min_step == 40_000
        )
    )
    if (
        train.lr_schedule_steps != 50_000
        or not (standard_schedule or full_observation_epoch_schedule)
        or train.patience_validations not in {2, 3}
    ):
        raise ValueError("x0 continuation requires the standard schedule or 1k objective-finetune validation")
    no_tx = not config.model.use_explicit_tx_condition
    expected_gpus = list(config.pipeline.allowed_physical_gpus)
    allowed_no_tx_profiles = {
        (tuple([0, 1, 2, 3]), 32, 2),
        (tuple([4, 5, 6, 7]), 32, 2),
        (tuple(range(8)), 16, 2),
    }
    actual_profile = (
        tuple(expected_gpus),
        train.per_gpu_batch_size,
        train.gradient_accumulation_steps,
    )
    if no_tx and actual_profile not in allowed_no_tx_profiles:
        raise ValueError(f"unsupported no-Tx continuation profile: {actual_profile}")
    if train.effective_global_batch_size != 256:
        raise ValueError("x0 continuation must preserve effective global batch 256")
    if not no_tx and expected_gpus != [4, 5, 6, 7]:
        raise ValueError("Tx-conditioned continuation requires GPUs 4-7")
    expected_root = (
        "runs/rmdm_hvdit_v4_x0_no_tx_strict_continue"
        if no_tx and not config.model.use_tx_source_supervision
        else "runs/rmdm_hvdit_v4_x0_no_tx_continue"
        if no_tx
        else "runs/rmdm_hvdit_v4_x0_continue"
    )
    if not no_tx and not config.pipeline.allow_gpu_co_tenancy:
        raise ValueError("GPU 6/7 retain small unrelated jobs; safe co-tenancy must remain explicit")
    if no_tx and config.pipeline.allow_gpu_co_tenancy:
        raise ValueError("The no-Tx continuation requires exclusive use of GPUs 0-3")
    if config.pipeline.output_root != expected_root:
        raise ValueError(f"Continuation output must stay below {expected_root}")
    if config.evaluation.rates != [1.0, 2.0, 3.0] or config.evaluation.ddim_steps != 20:
        raise ValueError("Continuation validation is fixed to Stage-A p1/p2/p3 DDIM20")

    baseline = copy.deepcopy(config)
    baseline.diffusion.prediction_type = "epsilon"
    baseline.t1_train.max_steps = 80_000
    baseline.t1_train.validation_first_step = 10_000
    baseline.pipeline.output_root = "runs/rmdm_hvdit_v4_joint"
    baseline.pipeline.lock_file = "runs/rmdm_hvdit_v4_joint/.pipeline.lock"
    if no_tx and len(expected_gpus) == 8:
        baseline.w16_train.default_gradient_accumulation_steps = 4
    baseline.validate()


def load_config(path: str | Path) -> ExperimentConfig:
    resolved = Path(path).expanduser().resolve()
    with resolved.open("r", encoding="utf-8") as handle:
        payload = yaml.safe_load(handle) or {}
    if not isinstance(payload, dict):
        raise TypeError(f"Configuration root must be a mapping: {resolved}")
    config = _from_mapping(ExperimentConfig, payload)
    _validate_continuation(config)
    return config


__all__ = ["ExperimentConfig", "load_config"]
