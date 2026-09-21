"""Strict configuration for the noise-aware single-frame RMDM stage."""

from __future__ import annotations

from dataclasses import asdict, dataclass, field, fields, is_dataclass
from pathlib import Path
from typing import Any, TypeVar, get_type_hints

import yaml


@dataclass
class DataConfig:
    root: str = "/data_p6/fzj/resources/RMDM/datasets/extracted/DynamicRadioMap/M20_Formal075_RadioMapSeerPack"
    split_file: str = "configs/splits/m20_formal075_clean16_scene_split.json"
    image_size: int = 128
    frames_per_video: int = 100
    workers: int = 8
    prefetch_factor: int = 2
    cache_size: int = 8
    packed_cache_root: str = ""


@dataclass
class SamplingConfig:
    seed: int = 20260717
    homogeneous_probability: float = 1.0
    base_rates: list[float] = field(default_factory=lambda: list(range(1, 11)))
    base_probabilities: list[float] = field(default_factory=lambda: [0.1] * 10)
    deltas: list[float] = field(default_factory=lambda: [-2, -1, 0, 1, 2])
    delta_probabilities: list[float] = field(default_factory=lambda: [0.1, 0.2, 0.4, 0.2, 0.1])
    extreme_probability_given_heterogeneous: float = 0.0
    extreme_frame_counts: list[int] = field(default_factory=lambda: [1, 2])
    extreme_frame_count_probabilities: list[float] = field(default_factory=lambda: [0.8, 0.2])
    extreme_rates: list[float] = field(default_factory=lambda: [0.1, 0.2, 0.5])
    extreme_rate_probabilities: list[float] = field(default_factory=lambda: [0.2, 0.3, 0.5])


@dataclass
class MeasurementNoiseConfig:
    clean_probability: float = 0.20
    nominal_probability: float = 0.65
    strong_probability: float = 0.15
    nominal_sigma_min: float = 0.0
    nominal_sigma_max: float = 0.05
    strong_sigma_min: float = 0.05
    strong_sigma_max: float = 0.09
    reference_variance: float = 0.0081
    seed: int = 20260921


@dataclass
class ModelConfig:
    model_channels: int = 96
    channel_multipliers: list[int] = field(default_factory=lambda: [1, 1, 2, 3, 4])
    residual_blocks_per_level: int = 2
    attention_levels: list[int] = field(default_factory=lambda: [3])
    attention_heads: int = 4
    use_scale_shift_norm: bool = True
    resblock_updown: bool = False
    hwm_base_features: int = 48
    hwm_channel_multipliers: list[int] = field(default_factory=lambda: [1, 2, 4, 8])
    hwm_blocks_per_level: int = 2
    variance_embedding_dim: int = 512
    variance_mlp_width: int = 512
    dropout: float = 0.0
    gradient_checkpointing: bool = False
    expected_trainable_parameters_min: int = 50_000_000
    expected_trainable_parameters_max: int = 180_000_000


@dataclass
class DiffusionConfig:
    train_timesteps: int = 1000
    beta_schedule: str = "linear"
    prediction_type: str = "epsilon"
    ddim_steps: int = 20


@dataclass
class LossConfig:
    diffusion_weight: float = 1.0
    calibration_weight: float = 1.0
    pinn_weight: float = 1.0
    pinn_k: float = 0.2


@dataclass
class TrainConfig:
    seed: int = 20260921
    max_steps: int = 40_000
    per_gpu_batch_size: int = 4
    gradient_accumulation_steps: int = 8
    learning_rate: float = 1.0e-4
    betas: list[float] = field(default_factory=lambda: [0.9, 0.95])
    epsilon: float = 1.0e-8
    weight_decay: float = 1.0e-2
    gradient_clip_norm: float = 1.0
    mixed_precision: str = "bf16"
    warmup_steps: int = 1_000
    min_learning_rate: float = 1.0e-5
    log_every_steps: int = 20
    checkpoint_every_steps: int = 1_000
    resume_from: str = ""


@dataclass
class ValidationConfig:
    subset_manifest: str = "configs/manifests/noise_temporal_rmdm_clean16_val.json"
    included_scenes: list[str] = field(default_factory=lambda: [
        "town02_opt_junction_0298", "town10_junction_0532"
    ])
    excluded_scenes: list[str] = field(default_factory=lambda: [
        "town01_opt_junction_0087", "town05_opt_junction_0053",
        "town05_opt_junction_0838", "town05_opt_junction_1427",
    ])
    rates: list[float] = field(default_factory=lambda: [1.0, 3.0])
    noise_standard_deviations: list[float] = field(default_factory=lambda: [0.0, 0.03, 0.05, 0.09])
    frame_starts: list[int] = field(default_factory=lambda: [0, 50])
    batch_size: int = 4
    ddim_steps: int = 20
    every_steps: int = 5_000


@dataclass
class FinalTestConfig:
    subset_manifest: str = "configs/manifests/noise_temporal_rmdm_clean16_test.json"
    included_scenes: list[str] = field(default_factory=lambda: [
        "town04_opt_junction_0053", "town05_opt_junction_0396"
    ])
    excluded_scenes: list[str] = field(default_factory=lambda: [
        "town01_opt_junction_0087", "town05_opt_junction_0053",
        "town05_opt_junction_0838", "town05_opt_junction_1427",
    ])
    rates: list[float] = field(default_factory=lambda: [1.0, 2.0, 3.0])
    noise_standard_deviations: list[float] = field(default_factory=lambda: [
        0.0, 0.01, 0.03, 0.05, 0.07, 0.09
    ])
    frame_starts: list[int] = field(default_factory=lambda: [0, 25, 50, 75])
    batch_size: int = 4
    ddim_steps: int = 20


@dataclass
class RuntimeConfig:
    output_root: str = "runs/noise_temporal_rmdm"


@dataclass
class ExperimentConfig:
    data: DataConfig = field(default_factory=DataConfig)
    sampling: SamplingConfig = field(default_factory=SamplingConfig)
    measurement_noise: MeasurementNoiseConfig = field(default_factory=MeasurementNoiseConfig)
    model: ModelConfig = field(default_factory=ModelConfig)
    diffusion: DiffusionConfig = field(default_factory=DiffusionConfig)
    loss: LossConfig = field(default_factory=LossConfig)
    train: TrainConfig = field(default_factory=TrainConfig)
    validation: ValidationConfig = field(default_factory=ValidationConfig)
    final_test: FinalTestConfig = field(default_factory=FinalTestConfig)
    runtime: RuntimeConfig = field(default_factory=RuntimeConfig)

    def validate(self) -> None:
        if self.data.image_size <= 0 or self.data.frames_per_video <= 0:
            raise ValueError("data dimensions must be positive")
        _distribution("sampling", self.sampling.base_rates, self.sampling.base_probabilities)
        noise = self.measurement_noise
        _distribution(
            "measurement-noise components",
            ["clean", "nominal", "strong"],
            [noise.clean_probability, noise.nominal_probability, noise.strong_probability],
        )
        if not 0 <= noise.nominal_sigma_min <= noise.nominal_sigma_max:
            raise ValueError("invalid nominal sigma interval")
        if not 0 <= noise.strong_sigma_min <= noise.strong_sigma_max:
            raise ValueError("invalid strong sigma interval")
        if noise.nominal_sigma_max != noise.strong_sigma_min:
            raise ValueError("nominal and strong sigma intervals must meet at one boundary")
        if noise.reference_variance <= 0:
            raise ValueError("reference_variance must be positive")
        if self.diffusion.prediction_type != "epsilon":
            raise ValueError("noise_temporal_rmdm requires epsilon prediction")
        if self.model.model_channels <= 0 or self.model.hwm_base_features <= 0:
            raise ValueError("model widths must be positive")
        widths = [self.model.model_channels * value for value in self.model.channel_multipliers]
        if any(width % self.model.attention_heads for index, width in enumerate(widths)
               if index in self.model.attention_levels):
            raise ValueError("attention widths must be divisible by attention_heads")
        if self.model.residual_blocks_per_level <= 0 or self.model.hwm_blocks_per_level <= 0:
            raise ValueError("blocks per level must be positive")
        if self.train.max_steps <= 0 or self.train.checkpoint_every_steps <= 0:
            raise ValueError("training steps must be positive")
        if self.train.mixed_precision not in {"no", "fp16", "bf16"}:
            raise ValueError("mixed_precision must be no/fp16/bf16")
        if set(self.validation.included_scenes) & set(self.validation.excluded_scenes):
            raise ValueError("validation included/excluded scenes overlap")
        if any(value < 0 for value in self.validation.noise_standard_deviations):
            raise ValueError("validation noise standard deviations must be non-negative")
        if set(self.final_test.included_scenes) & set(self.final_test.excluded_scenes):
            raise ValueError("test included/excluded scenes overlap")
        if any(value < 0 for value in self.final_test.noise_standard_deviations):
            raise ValueError("test noise standard deviations must be non-negative")
        if self.validation.ddim_steps <= 0 or self.final_test.ddim_steps <= 0:
            raise ValueError("evaluation DDIM steps must be positive")

    def to_dict(self) -> dict[str, Any]:
        return asdict(self)


T = TypeVar("T")


def _from_mapping(cls: type[T], values: dict[str, Any]) -> T:
    known = {item.name: item for item in fields(cls)}
    unknown = set(values) - set(known)
    if unknown:
        raise KeyError(f"Unknown {cls.__name__} keys: {sorted(unknown)}")
    hints = get_type_hints(cls)
    kwargs: dict[str, Any] = {}
    for name, value in values.items():
        kind = hints.get(name, known[name].type)
        kwargs[name] = _from_mapping(kind, value) if is_dataclass(kind) and isinstance(value, dict) else value
    return cls(**kwargs)


def _distribution(name: str, values: list[Any], probabilities: list[float]) -> None:
    if not values or len(values) != len(probabilities):
        raise ValueError(f"{name} values/probabilities must have equal non-zero lengths")
    if any(value < 0 for value in probabilities) or abs(sum(probabilities) - 1.0) > 1e-6:
        raise ValueError(f"{name} probabilities must be non-negative and sum to one")


def load_config(path: str | Path) -> ExperimentConfig:
    resolved = Path(path).expanduser().resolve()
    with resolved.open("r", encoding="utf-8") as handle:
        payload = yaml.safe_load(handle) or {}
    if not isinstance(payload, dict):
        raise TypeError("configuration root must be a mapping")
    config = _from_mapping(ExperimentConfig, payload)
    config.validate()
    return config
