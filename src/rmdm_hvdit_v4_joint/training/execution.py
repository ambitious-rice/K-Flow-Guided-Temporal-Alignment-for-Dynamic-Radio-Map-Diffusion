"""Audited execution-only placement changes that preserve optimizer batch semantics."""

from __future__ import annotations

from dataclasses import asdict, dataclass
from datetime import datetime
from typing import Any, Mapping, Sequence


AUTHORIZED_T1_GPU_PROFILES = {
    (4, 5, 6, 7),
    tuple(range(8)),
}
AUTHORIZED_PIPELINE_GPU_PROFILES = {
    (0, 1, 2, 3),
    (4, 5, 6, 7),
    tuple(range(8)),
}


def parse_physical_gpus(value: str) -> list[int]:
    try:
        parsed = [int(item.strip()) for item in value.split(",") if item.strip()]
    except ValueError as error:
        raise ValueError(f"Invalid physical GPU list: {value!r}") from error
    if not parsed or len(set(parsed)) != len(parsed):
        raise ValueError(f"Physical GPU list must be non-empty and unique: {value!r}")
    return parsed


def parse_wall_clock(value: str | None) -> datetime | None:
    if not value:
        return None
    parsed = datetime.fromisoformat(value)
    if parsed.tzinfo is None:
        raise ValueError("wall-clock pause time must include an explicit UTC offset")
    return parsed.astimezone()


@dataclass(frozen=True)
class ExecutionProfile:
    physical_gpus: tuple[int, ...]
    world_size: int
    per_gpu_batch_size: int
    gradient_accumulation_steps: int
    effective_global_batch_size: int

    @property
    def global_microbatch_size(self) -> int:
        return self.world_size * self.per_gpu_batch_size

    def to_dict(self) -> dict[str, Any]:
        return asdict(self)


def build_execution_profile(
    *,
    physical_gpus: Sequence[int],
    actual_world_size: int,
    per_gpu_batch_size: int,
    gradient_accumulation_steps: int,
    effective_global_batch_size: int,
    authorized_profiles: set[tuple[int, ...]],
) -> ExecutionProfile:
    gpus = tuple(int(value) for value in physical_gpus)
    if gpus not in authorized_profiles:
        raise ValueError(f"GPU execution profile {list(gpus)} is not authorized")
    profile = ExecutionProfile(
        physical_gpus=gpus,
        world_size=int(actual_world_size),
        per_gpu_batch_size=int(per_gpu_batch_size),
        gradient_accumulation_steps=int(gradient_accumulation_steps),
        effective_global_batch_size=int(effective_global_batch_size),
    )
    if profile.world_size != len(profile.physical_gpus):
        raise ValueError(
            f"Distributed world size {profile.world_size} differs from visible physical GPUs "
            f"{list(profile.physical_gpus)}"
        )
    if min(
        profile.world_size,
        profile.per_gpu_batch_size,
        profile.gradient_accumulation_steps,
        profile.effective_global_batch_size,
    ) <= 0:
        raise ValueError("Execution batch dimensions must all be positive")
    represented = (
        profile.world_size
        * profile.per_gpu_batch_size
        * profile.gradient_accumulation_steps
    )
    if represented != profile.effective_global_batch_size:
        raise ValueError(
            "Execution profile changes optimizer batch semantics: "
            f"{profile.world_size} * {profile.per_gpu_batch_size} * "
            f"{profile.gradient_accumulation_steps} = {represented}, expected "
            f"{profile.effective_global_batch_size}"
        )
    return profile


def checkpoint_samples_consumed_in_epoch(
    payload: Mapping[str, Any],
) -> tuple[int, dict[str, Any]]:
    extra = payload.get("extra", {})
    if "samples_consumed_in_epoch" in extra:
        samples = int(extra["samples_consumed_in_epoch"])
        source = dict(extra.get("execution_profile", {}))
        if not source:
            raise ValueError("Checkpoint records samples_consumed_in_epoch without an execution profile")
        return samples, source

    resolved = payload.get("resolved_config", {})
    train = resolved.get("t1_train", {})
    pipeline = resolved.get("pipeline", {})
    source_world = len(pipeline.get("allowed_physical_gpus", []))
    source_batch = int(train.get("per_gpu_batch_size", 0))
    source_accumulation = int(train.get("gradient_accumulation_steps", 0))
    offset = int(extra.get("microbatches_consumed_in_epoch", 0))
    if min(source_world, source_batch, source_accumulation) <= 0:
        raise ValueError("Legacy checkpoint lacks a valid source execution profile")
    source = {
        "physical_gpus": tuple(int(value) for value in pipeline["allowed_physical_gpus"]),
        "world_size": source_world,
        "per_gpu_batch_size": source_batch,
        "gradient_accumulation_steps": source_accumulation,
        "effective_global_batch_size": int(train.get("effective_global_batch_size", 0)),
    }
    return offset * source_world * source_batch, source


def convert_resume_microbatch_offset(
    payload: Mapping[str, Any],
    target: ExecutionProfile,
) -> tuple[int, int, dict[str, Any]]:
    samples, source = checkpoint_samples_consumed_in_epoch(payload)
    if samples < 0 or samples % target.global_microbatch_size:
        raise ValueError(
            "Checkpoint data cursor cannot be represented exactly by the target global microbatch: "
            f"samples={samples}, target_microbatch={target.global_microbatch_size}"
        )
    return samples // target.global_microbatch_size, samples, source
