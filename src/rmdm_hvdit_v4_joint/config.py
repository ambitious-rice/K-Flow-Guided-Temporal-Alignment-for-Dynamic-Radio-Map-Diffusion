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
        if self.data.image_size != 128 or self.data.frames_per_video != 100:
            raise ValueError("HV-DiT V4 is locked to 128x128, 100-frame DynamicRadioMap videos")
        model = self.model
        if (model.temporal_patch_size, model.spatial_patch_size) != (2, 4):
            raise ValueError("The confirmed patch contract is T/H/W = 2/4/4 for W16")
        if (model.local_dim, model.global_dim, model.head_dim) != (384, 768, 64):
            raise ValueError("The V4 width contract is local/global/head = 384/768/64")
        if model.local_depth != 2 or model.global_depth != 11:
            raise ValueError("The V4 depth contract is local/global = 2/11")
        if model.local_kernel != [3, 7, 7]:
            raise ValueError("The confirmed W16 neighborhood kernel is [3, 7, 7]")
        if model.local_dim % model.head_dim or model.global_dim % model.head_dim:
            raise ValueError("feature dimensions must be divisible by head_dim")
        if model.rope_axis_dims != [16, 24, 24]:
            raise ValueError("The separable RoPE head allocation is locked to [16,24,24]")
        if (
            model.decoder_token_channels != 192
            or model.decoder_stage_channels != [128, 64]
            or model.decoder_blocks_per_stage != 1
        ):
            raise ValueError("The V4 pixel decoder contract is locked to 192→128→64 with one ResBlock per stage")
        if (
            model.feedforward_multiplier,
            model.mapping_depth,
            model.mapping_width,
            model.mapping_feedforward_multiplier,
        ) != (3, 1, 768, 3):
            raise ValueError("The confirmed GEGLU/mapping parameterization has drifted")
        if model.dropout != 0.0 or not model.gradient_checkpointing:
            raise ValueError("W16 requires dropout=0 and gradient checkpointing")
        if self.t1_train.gradient_checkpointing:
            raise ValueError("The measured B32 T1 profile disables gradient checkpointing")
        if model.local_attention_backend != "natten":
            raise ValueError("Production configuration must use NATTEN; reference is test-only")
        if (
            self.stage1.base_features != 32
            or not self.stage1.trainable
            or self.stage1.chunk_size <= 0
            or self.stage1.pinn_k != 0.2
            or self.stage1.pinn_weight != 1.0
        ):
            raise ValueError("V4 requires an exact trainable-from-scratch RMDM HWM and k/weight=0.2/1.0")
        _distribution("base", self.sampling.base_rates, self.sampling.base_probabilities)
        if self.sampling.base_rates != list(range(1, 11)) or any(
            abs(value - 0.1) > 1.0e-9 for value in self.sampling.base_probabilities
        ):
            raise ValueError("Training rates are locked to uniform homogeneous p1-p10")
        if self.sampling.homogeneous_probability != 1.0:
            raise ValueError("T1 and W16 training are locked to homogeneous per-window sampling")
        if self.sampling.extreme_probability_given_heterogeneous != 0.0:
            raise ValueError("Extreme-rate injection is disabled for this experiment")
        expected_gpu_profiles = (
            ([4, 5, 6, 7],)
            if model.use_explicit_tx_condition
            else ([0, 1, 2, 3], [2, 3, 4, 5], [4, 5, 6, 7], list(range(8)))
        )
        if self.pipeline.allowed_physical_gpus not in expected_gpu_profiles:
            if model.use_explicit_tx_condition:
                raise ValueError("The canonical full pipeline is authorized only on physical GPUs 4-7")
            raise ValueError(
                "The no-Tx ablation is authorized on GPUs 0-3, GPUs 2-5, or all eight GPUs"
            )
        if model.use_explicit_tx_condition and not self.pipeline.allow_gpu_co_tenancy:
            raise ValueError("GPU 6/7 retain small unrelated jobs; V4 must use explicit safe co-tenancy")
        if not model.use_explicit_tx_condition and self.pipeline.allow_gpu_co_tenancy:
            raise ValueError("The no-Tx ablation requires exclusive use of GPUs 2-5")
        if self.pipeline.consecutive_free_polls != 1:
            raise ValueError("The confirmed 4-7 placement starts after one sufficient-memory poll")
        if (
            self.diffusion.train_timesteps != 1_000
            or self.diffusion.beta_schedule != "linear"
            or self.diffusion.prediction_type != "epsilon"
            or self.diffusion.ddim_steps != 20
        ):
            raise ValueError("The unchanged epsilon/linear/DDIM20 diffusion contract has drifted")
        world_size = len(self.pipeline.allowed_physical_gpus)
        _phase_batch(
            "T1",
            self.t1_train.per_gpu_batch_size,
            self.t1_train.gradient_accumulation_steps,
            self.t1_train.effective_global_batch_size,
            world_size=world_size,
        )
        if not self.t1_train.warmup_steps < self.t1_train.lr_schedule_steps <= self.t1_train.max_steps:
            raise ValueError("T1 LR schedule must finish after warmup and no later than max_steps")
        if self.t1_train.checkpoint_every_steps <= 0:
            raise ValueError("T1 checkpoint_every_steps must be positive")
        _phase_batch(
            "W16 default",
            self.w16_train.default_per_gpu_batch_size,
            self.w16_train.default_gradient_accumulation_steps,
            self.w16_train.effective_global_batch_size,
            world_size=world_size,
        )
        if self.w16_train.max_steps != self.w16_train.epochs * self.w16_train.updates_per_epoch:
            raise ValueError("W16 max_steps must equal epochs * updates_per_epoch")
        if self.w16_train.microbatch_candidates != [1, 2, 4]:
            raise ValueError("W16 smoke candidates are locked to [1,2,4]")
        for candidate in self.w16_train.microbatch_candidates:
            if self.w16_train.effective_global_batch_size % (world_size * candidate):
                raise ValueError(
                    f"W16 microbatch {candidate} cannot preserve global batch "
                    f"{self.w16_train.effective_global_batch_size}"
                )
        if self.evaluation.w16_starts != [0, 16, 32, 48, 64, 80]:
            raise ValueError("First96 W16 validation starts are immutable")
        if self.evaluation.rates != [1.0, 2.0, 3.0] or self.evaluation.ddim_steps != 20:
            raise ValueError("Stage-A is locked to p1/p2/p3 and DDIM20")
        if self.evaluation.formal_test_rates != [1.0, 2.0, 3.0, 5.0, 8.0, 10.0]:
            raise ValueError("Formal test rates are locked to p1/p2/p3/p5/p8/p10")
        if not self.evaluation.sf_reference_checkpoint:
            raise ValueError("The fixed RMDM-SF reference checkpoint is required only for aligned validation")
        if not self.evaluation.t1_all_frame_starts or self.evaluation.full100_extra_start != 84:
            raise ValueError("T1 must score all frames and full100 must use the extra start=84 window")
        if self.evaluation.t1_evaluation_batch_size != 4 or self.evaluation.w16_evaluation_batch_size != 1:
            raise ValueError("Evaluation per-GPU batches are locked to T1/W16 = 4/1")
        if self.pipeline.default_through not in {"w16_validation", "formal_test"}:
            raise ValueError("pipeline.default_through must be w16_validation or formal_test")

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
