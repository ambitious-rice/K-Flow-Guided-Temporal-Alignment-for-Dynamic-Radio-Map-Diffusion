"""Shared recipe: x0/epsilon differ only in prediction_type and output path."""
from dataclasses import asdict, dataclass, field
from pathlib import Path

import yaml

from experimental.noise_temporal_rmdm.config import (
    DataConfig, MeasurementNoiseConfig, SamplingConfig, DiffusionConfig,
    TrainConfig, _from_mapping,
)
from rmdm_hvdit_v4_joint.config import ModelConfig


@dataclass
class Training(TrainConfig):
    max_steps: int = 40000
    per_gpu_batch_size: int = 16
    gradient_accumulation_steps: int = 4
    learning_rate: float = 0.0002
    warmup_steps: int = 2000
    min_learning_rate: float = 0.00002
    ema_decay: float = 0.999


@dataclass
class Loss:
    calibration: float = 1.0
    equation: float = 1.0
    obstacle: float = 1.0
    source: float = 1.0
    clean_observation: float = 1.0
    pinn_k: float = 0.2


@dataclass
class Evaluation:
    every_steps: int = 4000
    patience: int = 3
    batch_size: int = 8
    rates: list[float] = field(default_factory=lambda: [1, 2, 3])
    sigmas: list[float] = field(default_factory=lambda: [0, 0.01, 0.03, 0.05, 0.07, 0.09])
    # Identical physical frames for W1 and W16; starts below refer to clips.
    starts: list[int] = field(default_factory=lambda: [0, 48])
    manifest: str = "configs/manifests/noise_temporal_rmdm_clean16_val.json"


@dataclass
class Config:
    data: DataConfig = field(default_factory=DataConfig)
    sampling: SamplingConfig = field(default_factory=SamplingConfig)
    measurement_noise: MeasurementNoiseConfig = field(default_factory=MeasurementNoiseConfig)
    model: ModelConfig = field(default_factory=lambda: ModelConfig(use_explicit_tx_condition=False))
    diffusion: DiffusionConfig = field(default_factory=lambda: DiffusionConfig(prediction_type="sample"))
    train: Training = field(default_factory=Training)
    loss: Loss = field(default_factory=Loss)
    evaluation: Evaluation = field(default_factory=Evaluation)
    embedding_width: int = 512
    source_masks: str = "/dev/shm/noise_hvdit_source_masks.npy"
    output: str = "runs/noise_hvdit/w1_x0"

    def to_dict(self):
        return asdict(self)


def load_config(path):
    return _from_mapping(Config, yaml.safe_load(Path(path).read_text()) or {})
