"""Typed configuration for the W16 joint-denoising pilot."""

from __future__ import annotations

from dataclasses import asdict, dataclass, field, fields, is_dataclass
from pathlib import Path
from typing import Any, TypeVar, get_type_hints

import yaml


@dataclass
class DataConfig:
    root: str = "/data/fzj/CARLA_0.9.15/datasets/DynamicRadioMap/M20_Formal075_RadioMapSeerPack"
    split_file: str = "/data/fzj/CARLA_0.9.15/configs/dynamic_radio/multi20_formal_scene_split.json"
    image_size: int = 128
    window_size: int = 16
    cache_size: int = 8
    workers: int = 4
    tx_heatmap_sigma_px: float = 1.5


@dataclass
class SamplingConfig:
    seed: int = 20260717
    homogeneous_probability: float = 0.5
    base_rates: list[float] = field(default_factory=lambda: list(range(1, 11)))
    base_probabilities: list[float] = field(
        default_factory=lambda: [0.20, 0.20, 0.20, 0.10, 0.10, 0.05, 0.05, 0.04, 0.03, 0.03]
    )
    deltas: list[float] = field(default_factory=lambda: [-2, -1, 0, 1, 2])
    delta_probabilities: list[float] = field(default_factory=lambda: [0.10, 0.20, 0.40, 0.20, 0.10])
    extreme_probability_given_heterogeneous: float = 0.20
    extreme_frame_counts: list[int] = field(default_factory=lambda: [1, 2])
    extreme_frame_count_probabilities: list[float] = field(default_factory=lambda: [0.80, 0.20])
    extreme_rates: list[float] = field(default_factory=lambda: [0.1, 0.2, 0.5])
    extreme_rate_probabilities: list[float] = field(default_factory=lambda: [0.20, 0.30, 0.50])


@dataclass
class Stage1Config:
    checkpoint: str = "runs/rmdm_sf_sparse_v2_fullimage_obstacle_e1trial/epoch_009.pth"
    trainable: bool = False
    chunk_size: int = 16


@dataclass
class ModelConfig:
    patch_size: int = 4
    high_dim: int = 256
    high_heads: int = 4
    high_encoder_blocks: int = 2
    high_decoder_blocks: int = 2
    window_attention_size: int = 8
    bottleneck_dim: int = 384
    bottleneck_heads: int = 6
    bottleneck_blocks: int = 8
    mlp_ratio: float = 4.0
    qk_norm: bool = True
    dropout: float = 0.0
    attention_dropout: float = 0.0
    gradient_checkpointing: bool = True


@dataclass
class DiffusionConfig:
    train_timesteps: int = 1000
    beta_schedule: str = "linear"
    prediction_type: str = "epsilon"
    ddim_steps: int = 20


@dataclass
class TrainConfig:
    seed: int = 20260717
    epochs: int = 40
    per_gpu_batch_size: int = 1
    gradient_accumulation_steps: int = 4
    learning_rate: float = 1.0e-4
    betas: list[float] = field(default_factory=lambda: [0.9, 0.95])
    weight_decay: float = 0.01
    gradient_clip_norm: float = 1.0
    mixed_precision: str = "bf16"
    warmup_steps: int = 500
    warmup_epochs: float = 0.0
    min_learning_rate: float = 1.0e-5
    log_interval: int = 20
    output_dir: str = "runs/rmdm_joint_w16_pilot"
    resume_from: str = ""
    max_steps: int = 0
    expected_trainable_parameters: int = 0
    expected_total_parameters: int = 0


@dataclass
class EvaluationConfig:
    subset_manifest: str = "manifests/dynamic_sparse_v2_semantic_vehicle/val_subset_v1.json"
    starts: list[int] = field(default_factory=lambda: [0, 16, 32, 48, 64, 80])
    stage_a_rates: list[float] = field(default_factory=lambda: [1, 2, 3])
    full_rates: list[float] = field(default_factory=lambda: [1, 2, 3, 5, 8, 10])
    validate_from_epoch: int = 2
    validate_every_epochs: int = 2
    early_stop_min_epoch: int = 0
    patience_validations: int = 3
    stage_a_ddim_steps: int = 20
    stage_b_ddim_steps: int = 20
    stage_a_top_k: int = 3


@dataclass
class ExperimentConfig:
    data: DataConfig = field(default_factory=DataConfig)
    sampling: SamplingConfig = field(default_factory=SamplingConfig)
    stage1: Stage1Config = field(default_factory=Stage1Config)
    model: ModelConfig = field(default_factory=ModelConfig)
    diffusion: DiffusionConfig = field(default_factory=DiffusionConfig)
    train: TrainConfig = field(default_factory=TrainConfig)
    evaluation: EvaluationConfig = field(default_factory=EvaluationConfig)

    def validate(self) -> None:
        if self.data.image_size != 128:
            raise ValueError("The DynamicRadioMap contract requires image_size=128")
        if self.data.window_size <= 0 or self.data.window_size > 100:
            raise ValueError("window_size must be in [1, 100]")
        if self.data.image_size % self.model.patch_size:
            raise ValueError("patch_size must divide image_size")
        if (self.data.image_size // self.model.patch_size) % 2:
            raise ValueError("The high-resolution token grid must be divisible by 2")
        if self.model.high_dim % self.model.high_heads:
            raise ValueError("high_dim must be divisible by high_heads")
        if self.model.bottleneck_dim % self.model.bottleneck_heads:
            raise ValueError("bottleneck_dim must be divisible by bottleneck_heads")
        if self.model.window_attention_size <= 0:
            raise ValueError("window_attention_size must be positive")
        high_grid = self.data.image_size // self.model.patch_size
        if high_grid % self.model.window_attention_size:
            raise ValueError("window_attention_size must divide the high-resolution token grid")
        if any(start < 0 or start + self.data.window_size > 100 for start in self.evaluation.starts):
            raise ValueError("Every evaluation start must define a complete window in 100 frames")
        expected = list(range(0, 96, self.data.window_size))
        if self.data.window_size == 16 and self.evaluation.starts != expected:
            raise ValueError(f"W16 pilot evaluation starts must be {expected}")
        _validate_distribution("base", self.sampling.base_rates, self.sampling.base_probabilities)
        _validate_distribution("delta", self.sampling.deltas, self.sampling.delta_probabilities)
        _validate_distribution(
            "extreme frame count",
            self.sampling.extreme_frame_counts,
            self.sampling.extreme_frame_count_probabilities,
        )
        _validate_distribution("extreme rate", self.sampling.extreme_rates, self.sampling.extreme_rate_probabilities)
        if not 0.0 <= self.sampling.homogeneous_probability <= 1.0:
            raise ValueError("homogeneous_probability must be in [0, 1]")
        if not 0.0 <= self.sampling.extreme_probability_given_heterogeneous <= 1.0:
            raise ValueError("extreme_probability_given_heterogeneous must be in [0, 1]")
        if self.train.epochs <= 0 or self.train.per_gpu_batch_size <= 0:
            raise ValueError("epochs and per_gpu_batch_size must be positive")
        if self.train.gradient_accumulation_steps <= 0:
            raise ValueError("gradient_accumulation_steps must be positive")
        if self.train.warmup_steps < 0 or self.train.warmup_epochs < 0:
            raise ValueError("warmup_steps and warmup_epochs must be non-negative")
        if self.train.warmup_steps and self.train.warmup_epochs:
            raise ValueError("Set only one of warmup_steps or warmup_epochs")
        if self.train.expected_trainable_parameters < 0 or self.train.expected_total_parameters < 0:
            raise ValueError("expected parameter counts must be non-negative")
        if self.train.mixed_precision not in {"no", "fp16", "bf16"}:
            raise ValueError("mixed_precision must be one of no/fp16/bf16")
        if self.evaluation.validate_from_epoch <= 0 or self.evaluation.validate_every_epochs <= 0:
            raise ValueError("validation epochs must be positive")
        if not 0 <= self.evaluation.early_stop_min_epoch <= self.train.epochs:
            raise ValueError("early_stop_min_epoch must be in [0, epochs]")
        if self.evaluation.patience_validations <= 0:
            raise ValueError("patience_validations must be positive")

    def to_dict(self) -> dict[str, Any]:
        return asdict(self)


T = TypeVar("T")


def _from_mapping(cls: type[T], values: dict[str, Any]) -> T:
    known = {item.name: item for item in fields(cls)}
    type_hints = get_type_hints(cls)
    unknown = set(values) - set(known)
    if unknown:
        raise KeyError(f"Unknown {cls.__name__} keys: {sorted(unknown)}")
    kwargs: dict[str, Any] = {}
    for name, value in values.items():
        field_type = type_hints.get(name, known[name].type)
        if is_dataclass(field_type) and isinstance(value, dict):
            kwargs[name] = _from_mapping(field_type, value)
        else:
            kwargs[name] = value
    return cls(**kwargs)


def _validate_distribution(name: str, values: list[Any], probabilities: list[float]) -> None:
    if not values or len(values) != len(probabilities):
        raise ValueError(f"{name} values and probabilities must have the same non-zero length")
    if any(probability < 0 for probability in probabilities):
        raise ValueError(f"{name} probabilities must be non-negative")
    if abs(sum(probabilities) - 1.0) > 1.0e-6:
        raise ValueError(f"{name} probabilities must sum to 1, got {sum(probabilities)}")


def load_config(path: str | Path) -> ExperimentConfig:
    path = Path(path).expanduser().resolve()
    with path.open("r", encoding="utf-8") as handle:
        payload = yaml.safe_load(handle) or {}
    if not isinstance(payload, dict):
        raise TypeError(f"Configuration root must be a mapping: {path}")
    config = _from_mapping(ExperimentConfig, payload)
    config.validate()
    return config
