"""Stage-A/Stage-B validation wrappers."""

from __future__ import annotations

from typing import Any

from rmdm.config import ExperimentConfig
from rmdm.evaluation import evaluate_rates


def should_run_stage_a(config: ExperimentConfig, completed_epoch: int) -> bool:
    start = config.evaluation.validate_from_epoch
    interval = config.evaluation.validate_every_epochs
    return completed_epoch >= start and (completed_epoch - start) % interval == 0


def run_stage_a(accelerator: Any, model: Any, config: ExperimentConfig) -> dict[str, Any]:
    return evaluate_rates(
        accelerator,
        model,
        config,
        subset_stage="stage_a",
        rates=config.evaluation.stage_a_rates,
        ddim_steps=config.evaluation.stage_a_ddim_steps,
    )


def run_stage_b(accelerator: Any, model: Any, config: ExperimentConfig) -> dict[str, Any]:
    return evaluate_rates(
        accelerator,
        model,
        config,
        subset_stage="stage_b_extra",
        rates=config.evaluation.stage_a_rates,
        ddim_steps=config.evaluation.stage_b_ddim_steps,
    )

