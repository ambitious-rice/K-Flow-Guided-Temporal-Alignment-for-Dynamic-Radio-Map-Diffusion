"""Small configuration compatibility layer for the single tx_prior task."""

from __future__ import annotations

from pathlib import Path

import yaml

from rmdm_hvdit_v4_joint.config import ExperimentConfig, _from_mapping


def load_config(path: str | Path, *, smoke: bool) -> ExperimentConfig:
    resolved = Path(path).expanduser().resolve()
    with resolved.open("r", encoding="utf-8") as handle:
        payload = yaml.safe_load(handle) or {}
    config = _from_mapping(ExperimentConfig, payload)
    config.validate()
    train = config.t1_train
    if not config.model.use_explicit_tx_condition:
        raise ValueError("tx_prior requires explicit TX conditioning")
    if config.diffusion.prediction_type != "sample":
        raise ValueError("tx_prior is a clean-x0 prediction experiment")
    expected = len(config.pipeline.allowed_physical_gpus) * train.per_gpu_batch_size * train.gradient_accumulation_steps
    if expected != train.effective_global_batch_size:
        raise ValueError(f"effective global batch is {expected}, configured {train.effective_global_batch_size}")
    if config.pipeline.output_root != "runs/tx_prior":
        raise ValueError("tx_prior output_root must remain runs/tx_prior")
    if smoke and train.max_steps > 2:
        raise ValueError("smoke entry permits at most two optimizer steps")
    if not smoke and train.max_steps <= 2:
        raise ValueError("formal entry rejects smoke-sized max_steps")
    return config
