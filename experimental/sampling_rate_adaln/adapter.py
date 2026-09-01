"""Continuous sampling-rate conditioning for the historical W1 DiT runtime."""

from __future__ import annotations

import math
import types
from typing import Any, Mapping

import torch
from torch import nn


class ContinuousSamplingRateConditioner(nn.Module):
    """Map an observed percentage to the shared DiT conditioning width.

    SamplingPolicy reports percentages (for example ``1`` for 1%).  The log
    transform is continuous and leaves values outside the 1--10% training range
    unclipped, so later rate changes do not turn this path into a discrete
    lookup table.  The final projection is zero-initialized: before adaptation,
    inserting this module is exactly a no-op for a loaded W1 checkpoint.
    """

    def __init__(self, width: int) -> None:
        super().__init__()
        self.input = nn.Linear(1, width, bias=True)
        self.output = nn.Linear(width, width, bias=True)
        nn.init.normal_(self.input.weight, mean=0.0, std=0.02)
        nn.init.zeros_(self.input.bias)
        nn.init.zeros_(self.output.weight)
        nn.init.zeros_(self.output.bias)

    def forward(self, sampling_rate_percent: torch.Tensor) -> torch.Tensor:
        if sampling_rate_percent.ndim != 1:
            raise ValueError("sampling_rate must be a [B] tensor of percentages")
        rate_fraction = sampling_rate_percent.float().clamp_min(1.0e-4) / 100.0
        # 1% maps to 0 and 10% maps to 1, but the input is not clamped.
        normalized_log_rate = torch.log(rate_fraction / 0.01) / math.log(10.0)
        value = torch.nn.functional.silu(self.input(normalized_log_rate.unsqueeze(-1)))
        return self.output(value)


def _forward_with_sampling_rate(
    self: nn.Module,
    noisy_target: torch.Tensor,
    timesteps: torch.Tensor,
    cache: Mapping[str, torch.Tensor],
) -> torch.Tensor:
    """Historical JointTokenDenoiser.forward with one additive AdaLN condition."""

    if timesteps.shape != (noisy_target.shape[0],):
        raise ValueError("timesteps must contain one value per video")
    required = ("hwm_gate", "condition_high", "condition_low", "sampling_rate")
    missing = [name for name in required if name not in cache]
    if missing:
        raise KeyError(f"encoded condition cache misses {missing}")
    dense, observation = self.build_inputs(noisy_target, cache)
    state = self.input_stem(dense, observation)
    # Imported by the legacy JointTokenDenoiser module; its source remains
    # untouched.  This method deliberately reuses its existing operation.
    from rmdm_hvdit_v4_joint.model.joint import apply_detached_hwm_gate, grid_coordinates, merge_coordinates

    state = apply_detached_hwm_gate(state, cache["hwm_gate"])
    condition_high = cache["condition_high"]
    condition_low = cache["condition_low"]
    if condition_high.shape != state.shape:
        raise ValueError("high-resolution condition pyramid does not match main tokens")

    timestep_condition = self.timestep_mapping(timesteps).to(state.dtype)
    rate_condition = self.sampling_rate_conditioner(cache["sampling_rate"]).to(state.dtype)
    timestep_condition = timestep_condition + rate_condition
    local_modulation = self.local_time_modulation(timestep_condition)
    fine_coordinates = grid_coordinates(state.shape[1], state.shape[2], state.shape[3], device=state.device)
    for block in self.local_encoder:
        state = self._run_transformer(block, state, fine_coordinates, local_modulation, condition_high)

    processed_skip = state
    state = self.hierarchy_merge(state)
    if condition_low.shape != state.shape:
        raise ValueError("low-resolution condition pyramid does not match bottleneck tokens")
    coarse_coordinates = merge_coordinates(fine_coordinates, temporal_factor=self.temporal_factor)
    global_modulation = self.global_time_modulation(timestep_condition)
    for block in self.global_bottleneck:
        state = self._run_transformer(block, state, coarse_coordinates, global_modulation, condition_low)

    state = self.processed_token_skip(self.hierarchy_expand(state), processed_skip, timestep_condition)
    for block in self.local_decoder:
        state = self._run_transformer(block, state, fine_coordinates, local_modulation, condition_high)
    return self.output_head(state, timestep_condition)


def install_sampling_rate_conditioning(system: nn.Module, *, freeze_backbone: bool = True) -> nn.Module:
    """Attach the zero-initialized rate branch to a W1 ``HvditSystem``.

    ``sampling_rate`` already exists in the sparse batch produced by
    ``SamplingPolicy``.  It is carried into the DDIM-cacheable condition cache
    and added to the existing timestep mapping before both local and global
    residual-modulation projections.
    """

    if hasattr(system.denoiser, "sampling_rate_conditioner"):
        raise ValueError("sampling-rate conditioning is already installed")
    if freeze_backbone:
        system.requires_grad_(False)
    width = int(system.denoiser.timestep_mapping.output_norm.scale.numel())
    system.denoiser.sampling_rate_conditioner = ContinuousSamplingRateConditioner(width)
    system.denoiser.forward = types.MethodType(_forward_with_sampling_rate, system.denoiser)

    original_encode_conditions = system.encode_conditions

    def encode_conditions_with_rate(self: nn.Module, sparse_batch: dict[str, Any]) -> dict[str, torch.Tensor]:
        cache = original_encode_conditions(sparse_batch)
        rate = sparse_batch.get("sampling_rate")
        if not torch.is_tensor(rate):
            raise KeyError("sparse batch misses tensor sampling_rate")
        if rate.shape != (cache["condition_high"].shape[0],):
            raise ValueError("sampling_rate must contain one value per W1 sample")
        cache["sampling_rate"] = rate.to(device=cache["condition_high"].device)
        return cache

    system.encode_conditions = types.MethodType(encode_conditions_with_rate, system)
    return system
