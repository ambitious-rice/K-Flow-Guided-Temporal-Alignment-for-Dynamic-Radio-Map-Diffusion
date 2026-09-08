"""Strict, self-contained configuration for the jointly trained HV-DiT V4."""

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
    frames_per_video: int = 100
    cache_size: int = 8
    workers: int = 8
    tx_heatmap_sigma_px: float = 1.5


@dataclass
class SamplingConfig:
    """Attributes intentionally match the read-only ``rmdm.data.SamplingPolicy`` boundary."""

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
class Stage1Config:
    base_features: int = 32
    trainable: bool = True
    chunk_size: int = 16
    pinn_k: float = 0.2
    pinn_weight: float = 1.0


@dataclass
class ModelConfig:
    use_explicit_tx_condition: bool = True
    use_tx_source_supervision: bool = True
    spatial_patch_size: int = 4
    temporal_patch_size: int = 2
    local_dim: int = 384
    global_dim: int = 768
    head_dim: int = 64
    local_depth: int = 2
    global_depth: int = 11
    local_kernel: list[int] = field(default_factory=lambda: [3, 7, 7])
    feedforward_multiplier: int = 3
    mapping_depth: int = 1
    mapping_width: int = 768
    mapping_feedforward_multiplier: int = 3
    dropout: float = 0.0
    gradient_checkpointing: bool = True
    local_attention_backend: str = "natten"
    rope_axis_dims: list[int] = field(default_factory=lambda: [16, 24, 24])
    decoder_token_channels: int = 192
    decoder_stage_channels: list[int] = field(default_factory=lambda: [128, 64])
    decoder_blocks_per_stage: int = 1
    expected_trainable_parameters_min: int = 126_000_000
    expected_trainable_parameters_max: int = 140_000_000


@dataclass
class DiffusionConfig:
    train_timesteps: int = 1000
    beta_schedule: str = "linear"
    prediction_type: str = "epsilon"
    ddim_steps: int = 20


@dataclass
class T1TrainConfig:
    seed: int = 20260717
    max_steps: int = 80_000
    per_gpu_batch_size: int = 32
    gradient_accumulation_steps: int = 2
    effective_global_batch_size: int = 256
    learning_rate: float = 5.0e-4
    betas: list[float] = field(default_factory=lambda: [0.9, 0.95])
    epsilon: float = 1.0e-8
    weight_decay: float = 1.0e-2
    gradient_clip_norm: float = 1.0
    mixed_precision: str = "bf16"
    gradient_checkpointing: bool = False
    warmup_steps: int = 2_000
    lr_schedule_steps: int = 50_000
    min_learning_rate: float = 5.0e-5
    validation_first_step: int = 10_000
    validation_every_steps: int = 5_000
    early_stop_min_step: int = 25_000
    patience_validations: int = 2
    log_every_steps: int = 20
    checkpoint_every_steps: int = 1_000
    resume_from: str = ""


@dataclass
class W16TrainConfig:
    seed: int = 20260717
    epochs: int = 200
    updates_per_epoch: int = 328
    max_steps: int = 65_600
    effective_global_batch_size: int = 32
    microbatch_candidates: list[int] = field(default_factory=lambda: [1, 2, 4])
    default_per_gpu_batch_size: int = 1
    default_gradient_accumulation_steps: int = 8
    learning_rate: float = 1.0e-4
    betas: list[float] = field(default_factory=lambda: [0.9, 0.95])
    epsilon: float = 1.0e-8
    weight_decay: float = 1.0e-2
    gradient_clip_norm: float = 1.0
    mixed_precision: str = "bf16"
    warmup_epochs: float = 2.0
    min_learning_rate: float = 1.0e-5
    validation_first_epoch: int = 10
    validation_every_epochs: int = 10
    early_stop_min_epoch: int = 100
    patience_validations: int = 5
    stage_b_top_k: int = 3
    log_every_steps: int = 20
    resume_from: str = ""


@dataclass
class EvaluationConfig:
    subset_manifest: str = "manifests/dynamic_sparse_v2_semantic_vehicle/val_subset_v1.json"
    formal_test_manifest: str = "manifests/dynamic_sparse_v2_semantic_vehicle/motivation_test_subset_v1.json"
    factorized_baseline_run_dir: str = "runs/rmdm_joint_w16_unified_large120m_b8_uniform_base_p1_p10_no_extreme"
    sf_mask_manifest: str = "manifests/dynamic_sparse_v2_semantic_vehicle/sparse_masks_val.json"
    sf_reference_checkpoint: str = "runs/rmdm_sf_sparse_v2_fullimage_obstacle_e1trial/epoch_009.pth"
    sf_reference_per_gpu_batch_size: int = 16
    rates: list[float] = field(default_factory=lambda: [1.0, 2.0, 3.0])
    formal_test_rates: list[float] = field(default_factory=lambda: [1.0, 2.0, 3.0, 5.0, 8.0, 10.0])
    ddim_steps: int = 20
    t1_all_frame_starts: bool = True
    w16_starts: list[int] = field(default_factory=lambda: [0, 16, 32, 48, 64, 80])
    full100_extra_start: int = 84
    t1_reference_summary: str = (
        "runs/rmdm_sf_sparse_v2_fullimage_obstacle_e1trial/"
        "epochwise_validation/stage_a/epoch_009/summary_val.json"
    )
    t1_reference_tolerance: float = 0.05
    require_monotonic_observation_response: bool = True
    require_ablation_improvement_each_rate: bool = True
    t1_evaluation_batch_size: int = 4
    w16_evaluation_batch_size: int = 1


@dataclass
class PipelineConfig:
    output_root: str = "runs/rmdm_hvdit_v4_joint"
    environment_path: str = "/data/fzj/conda_envs/RMDM_HVDIT_V2"
    allowed_physical_gpus: list[int] = field(default_factory=lambda: [4, 5, 6, 7])
    allow_gpu_co_tenancy: bool = True
    wait_for_gpus: bool = True
    gpu_poll_seconds: int = 60
    free_memory_mib: int = 22_000
    consecutive_free_polls: int = 1
    lock_file: str = "runs/rmdm_hvdit_v4_joint/.pipeline.lock"
    default_through: str = "w16_validation"


@dataclass
class ExperimentConfig:
    data: DataConfig = field(default_factory=DataConfig)
    sampling: SamplingConfig = field(default_factory=SamplingConfig)
    stage1: Stage1Config = field(default_factory=Stage1Config)
    model: ModelConfig = field(default_factory=ModelConfig)
    diffusion: DiffusionConfig = field(default_factory=DiffusionConfig)
    t1_train: T1TrainConfig = field(default_factory=T1TrainConfig)
    w16_train: W16TrainConfig = field(default_factory=W16TrainConfig)
    evaluation: EvaluationConfig = field(default_factory=EvaluationConfig)
    pipeline: PipelineConfig = field(default_factory=PipelineConfig)

    def validate(self) -> None:
        model = self.model
        if self.data.image_size <= 0 or self.data.frames_per_video <= 0:
            raise ValueError("image_size and frames_per_video must be positive")
        if model.temporal_patch_size <= 0 or model.spatial_patch_size <= 0:
            raise ValueError("patch sizes must be positive")
        if min(model.local_dim, model.global_dim, model.head_dim) <= 0:
            raise ValueError("model dimensions must be positive")
        if model.local_dim % model.head_dim or model.global_dim % model.head_dim:
            raise ValueError("feature dimensions must be divisible by head_dim")
        if self.stage1.chunk_size <= 0:
            raise ValueError("stage1.chunk_size must be positive")
        _distribution("base", self.sampling.base_rates, self.sampling.base_probabilities)
        if not 0.0 <= self.sampling.homogeneous_probability <= 1.0:
            raise ValueError("homogeneous_probability must be in [0, 1]")
        if self.diffusion.train_timesteps <= 0 or self.diffusion.ddim_steps <= 0:
            raise ValueError("diffusion step counts must be positive")
        if self.diffusion.prediction_type not in {"epsilon", "sample"}:
            raise ValueError("prediction_type must be epsilon or sample")
        world_size = len(self.pipeline.allowed_physical_gpus)
        if world_size <= 0 or len(set(self.pipeline.allowed_physical_gpus)) != world_size:
            raise ValueError("allowed_physical_gpus must be a non-empty unique list")
        if self.t1_train.per_gpu_batch_size <= 0 or self.t1_train.gradient_accumulation_steps <= 0:
            raise ValueError("T1 batch size and accumulation must be positive")
        if self.t1_train.max_steps <= 0 or self.t1_train.lr_schedule_steps <= 0:
            raise ValueError("T1 step counts must be positive")
        if self.t1_train.checkpoint_every_steps <= 0:
            raise ValueError("T1 checkpoint_every_steps must be positive")
        if not self.evaluation.rates:
            raise ValueError("at least one evaluation rate is required")

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
        field_type = hints.get(name, known[name].type)
        kwargs[name] = _from_mapping(field_type, value) if is_dataclass(field_type) and isinstance(value, dict) else value
    return cls(**kwargs)


def _distribution(name: str, values: list[Any], probabilities: list[float]) -> None:
    if not values or len(values) != len(probabilities):
        raise ValueError(f"{name} distribution lengths do not match")
    if any(value < 0 for value in probabilities) or abs(sum(probabilities) - 1.0) > 1.0e-6:
        raise ValueError(f"{name} probabilities must be non-negative and sum to one")


def _phase_batch(name: str, microbatch: int, accumulation: int, expected: int, *, world_size: int) -> None:
    actual = int(microbatch) * int(accumulation) * int(world_size)
    if actual != expected:
        raise ValueError(f"{name} effective batch is {actual}, expected {expected}")


def load_config(path: str | Path) -> ExperimentConfig:
    resolved = Path(path).expanduser().resolve()
    with resolved.open("r", encoding="utf-8") as handle:
        payload = yaml.safe_load(handle) or {}
    if not isinstance(payload, dict):
        raise TypeError(f"Configuration root must be a mapping: {resolved}")
    config = _from_mapping(ExperimentConfig, payload)
    config.validate()
    return config
