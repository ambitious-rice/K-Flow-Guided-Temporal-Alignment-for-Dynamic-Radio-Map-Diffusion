"""Configuration for the isolated W1 observation-balance fine-tuning branch."""

from __future__ import annotations

from dataclasses import dataclass, field
from pathlib import Path
from typing import Any

import yaml


@dataclass(frozen=True)
class NoiseConfig:
    standard_deviations: list[float] = field(default_factory=lambda: [0.0, 0.01, 0.03, 0.05])
    probabilities: list[float] = field(default_factory=lambda: [0.50, 0.20, 0.15, 0.15])


@dataclass(frozen=True)
class AlignmentConfig:
    warmup_steps: int = 500
    hold_steps: int = 4_000
    peak_weight: float = 0.25
    final_weight: float = 0.05
    heldout_ratio: float = 0.25
    heldout_weight: float = 0.10
    condition_dropout_probability: float = 0.10


@dataclass(frozen=True)
class OptimizerConfig:
    learning_rate: float = 1.0e-4
    min_learning_rate: float = 1.0e-5
    warmup_steps: int = 250
    betas: list[float] = field(default_factory=lambda: [0.9, 0.95])
    epsilon: float = 1.0e-8
    weight_decay: float = 1.0e-2
    gradient_clip_norm: float = 1.0


@dataclass(frozen=True)
class FastDecisionConfig:
    minimum_clean_relative_improvement: float = 0.05
    maximum_noisy_relative_regression: float = 0.03
    full_test_improvement_increment: float = 0.02
    plateau_minimum_step: int = 4_000
    plateau_improvement: float = 0.005
    plateau_evaluations: int = 3
    no_eligible_stop_step: int = 6_000
    excessive_noisy_regression: float = 0.08
    excessive_noisy_evaluations: int = 2


@dataclass(frozen=True)
class EvaluationConfig:
    suite: str = "configs/evaluation/observation_balance_validation_v1.yaml"
    manifest: str = "manifests/dynamic_sparse_v2_semantic_vehicle/val_subset_v1.json"
    output_root: str = "runs/observation_balance_validation"
    baseline_id: str = "original_w1"
    gpus: list[int] = field(default_factory=lambda: list(range(8)))
    python: str = "/data/fzj/conda_envs/RMDM_HVDIT_V2/bin/python"
    base_port: int = 29700


@dataclass(frozen=True)
class ObservationBalanceConfig:
    run_id: str = "w1_observation_balance_v1"
    seed: int = 20260717
    model_config: str = "configs/hvdit_v4_x0_no_tx_continue/t1_from10k_to50k_4gpu.yaml"
    source_checkpoint: str = (
        "runs/rmdm_hvdit_v4_x0_no_tx_continue/t1_from10k_to50k/checkpoints/best.pth"
    )
    output_dir: str = "runs/rmdm_hvdit_v4_x0_observation_balance/w1_v1"
    max_steps: int = 12_000
    segment_steps: int = 1_000
    log_every_steps: int = 20
    mixed_precision: str = "bf16"
    noise: NoiseConfig = field(default_factory=NoiseConfig)
    alignment: AlignmentConfig = field(default_factory=AlignmentConfig)
    optimizer: OptimizerConfig = field(default_factory=OptimizerConfig)
    fast_decision: FastDecisionConfig = field(default_factory=FastDecisionConfig)
    evaluation: EvaluationConfig = field(default_factory=EvaluationConfig)


def _mapping(value: Any, name: str) -> dict[str, Any]:
    if not isinstance(value, dict):
        raise TypeError(f"{name} must be a mapping")
    return value


def load_observation_balance_config(path: str | Path) -> ObservationBalanceConfig:
    with Path(path).expanduser().resolve().open("r", encoding="utf-8") as handle:
        values = _mapping(yaml.safe_load(handle) or {}, "configuration")
    for name, cls in {
        "noise": NoiseConfig,
        "alignment": AlignmentConfig,
        "optimizer": OptimizerConfig,
        "fast_decision": FastDecisionConfig,
        "evaluation": EvaluationConfig,
    }.items():
        if name in values:
            values[name] = cls(**_mapping(values[name], name))
    config = ObservationBalanceConfig(**values)
    _validate(config)
    return config


def _validate(config: ObservationBalanceConfig) -> None:
    if config.max_steps <= 0 or config.segment_steps <= 0:
        raise ValueError("max_steps and segment_steps must be positive")
    if config.max_steps % config.segment_steps:
        raise ValueError("max_steps must be divisible by segment_steps")
    noise = config.noise
    if not noise.standard_deviations or len(noise.standard_deviations) != len(noise.probabilities):
        raise ValueError("noise values and probabilities must have equal non-zero lengths")
    if any(value < 0 for value in noise.standard_deviations):
        raise ValueError("noise standard deviations must be non-negative")
    if any(value < 0 for value in noise.probabilities) or abs(sum(noise.probabilities) - 1.0) > 1e-6:
        raise ValueError("noise probabilities must be non-negative and sum to one")
    alignment = config.alignment
    if not 0 <= alignment.condition_dropout_probability < 1:
        raise ValueError("condition dropout probability must be in [0, 1)")
    if not 0 <= alignment.heldout_ratio <= 1:
        raise ValueError("heldout ratio must be in [0, 1]")
    if min(alignment.peak_weight, alignment.final_weight, alignment.heldout_weight) < 0:
        raise ValueError("loss weights must be non-negative")
    if not 0 <= alignment.warmup_steps <= alignment.hold_steps <= config.max_steps:
        raise ValueError("alignment steps must satisfy 0 <= warmup <= hold <= max_steps")
    optimizer = config.optimizer
    if not 0 < optimizer.min_learning_rate <= optimizer.learning_rate:
        raise ValueError("learning rates must satisfy 0 < min <= base")
    if len(optimizer.betas) != 2:
        raise ValueError("optimizer betas must contain two values")
    if not config.evaluation.gpus or len(set(config.evaluation.gpus)) != len(config.evaluation.gpus):
        raise ValueError("evaluation.gpus must be non-empty and unique")


__all__ = [
    "AlignmentConfig",
    "FastDecisionConfig",
    "ObservationBalanceConfig",
    "load_observation_balance_config",
]
