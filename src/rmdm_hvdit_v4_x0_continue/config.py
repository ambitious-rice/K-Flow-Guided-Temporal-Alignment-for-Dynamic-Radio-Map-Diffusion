"""Strict continuation configuration without changing the completed 10k pilot."""

from __future__ import annotations

from pathlib import Path

import yaml

from rmdm_hvdit_v4_joint.config import ExperimentConfig, _from_mapping


def _validate_continuation(config: ExperimentConfig) -> None:
    train = config.t1_train
    config.validate()
    if config.diffusion.prediction_type != "sample":
        raise ValueError("x0 continuation requires diffusion.prediction_type=sample")
    if train.validation_every_steps <= 0:
        raise ValueError("validation_every_steps must be positive")


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
