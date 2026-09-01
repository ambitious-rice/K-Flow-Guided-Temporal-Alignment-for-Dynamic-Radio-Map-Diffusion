"""Hierarchical DiT with scale-preserving input and a local pixel decoder."""

from __future__ import annotations

from collections.abc import Mapping

import torch
from torch import nn
from torch.utils.checkpoint import checkpoint

from rmdm_hvdit_v4_joint.config import ModelConfig

from .blocks import GlobalTransformerLayer, LocalTransformerLayer
from .common import SharedTimeModulation, TimestepMapping
from .hierarchy import ControlledTokenSkip, SpaceTimeExpand, SpaceTimeMerge, merge_coordinates
from .patching import PixelDecoderHead, T1DoubleStem, W16DoubleStem, apply_detached_hwm_gate
from .rope import grid_coordinates


class JointTokenDenoiser(nn.Module):
    """One local level, one lossy hierarchy, and a joint global bottleneck."""

    def __init__(
        self,
        config: ModelConfig,
        *,
        frames: int,
        attention_backend: str | None = None,
        gradient_checkpointing: bool | None = None,
    ) -> None:
        super().__init__()
        if frames not in (1, 16):
            raise ValueError("the explicit model variants are T1 and W16")
        self.config = config
        self.frames = int(frames)
        self.temporal_patch = 1 if frames == 1 else config.temporal_patch_size
        self.temporal_factor = 1 if frames == 1 else 2
        self.gradient_checkpointing = (
            config.gradient_checkpointing if gradient_checkpointing is None else bool(gradient_checkpointing)
        )
        backend = attention_backend or config.local_attention_backend
        local_dim, global_dim = config.local_dim, config.global_dim

        stem_type = T1DoubleStem if frames == 1 else W16DoubleStem
        if frames == 1:
            self.input_stem = stem_type(4, local_dim, config.spatial_patch_size)
            self.condition_stem = stem_type(3, local_dim, config.spatial_patch_size)
        else:
            self.input_stem = stem_type(
                4,
                local_dim,
                config.temporal_patch_size,
                config.spatial_patch_size,
            )
            self.condition_stem = stem_type(
                3,
                local_dim,
                config.temporal_patch_size,
                config.spatial_patch_size,
            )

        self.timestep_mapping = TimestepMapping(
            config.mapping_width,
            config.mapping_depth,
            config.mapping_width * config.mapping_feedforward_multiplier,
            config.dropout,
        )
        self.local_time_modulation = SharedTimeModulation(config.mapping_width, local_dim)
        self.global_time_modulation = SharedTimeModulation(config.mapping_width, global_dim)

        local_kernel = (config.local_kernel[1], config.local_kernel[2]) if frames == 1 else tuple(config.local_kernel)
        local_kwargs = dict(
            dim=local_dim,
            head_dim=config.head_dim,
            hidden_dim=local_dim * config.feedforward_multiplier,
            kernel_size=local_kernel,
            rope_axis_dims=tuple(config.rope_axis_dims),
            backend=backend,
            dropout=config.dropout,
        )
        self.local_encoder = nn.ModuleList(
            [LocalTransformerLayer(**local_kwargs) for _ in range(config.local_depth)]
        )
        self.hierarchy_merge = SpaceTimeMerge(
            local_dim,
            global_dim,
            temporal_factor=self.temporal_factor,
        )
        self.condition_merge = SpaceTimeMerge(
            local_dim,
            global_dim,
            temporal_factor=self.temporal_factor,
        )
        self.global_bottleneck = nn.ModuleList(
            [
                GlobalTransformerLayer(
                    global_dim,
                    config.head_dim,
                    global_dim * config.feedforward_multiplier,
                    rope_axis_dims=tuple(config.rope_axis_dims),
                    dropout=config.dropout,
                )
                for _ in range(config.global_depth)
            ]
        )
        self.hierarchy_expand = SpaceTimeExpand(
            global_dim,
            local_dim,
            temporal_factor=self.temporal_factor,
        )
        self.processed_token_skip = ControlledTokenSkip(local_dim, config.mapping_width)
        self.local_decoder = nn.ModuleList(
            [LocalTransformerLayer(**local_kwargs) for _ in range(config.local_depth)]
        )
        self.output_head = PixelDecoderHead(
            local_dim,
            config.mapping_width,
            temporal_patch=self.temporal_patch,
            spatial_patch=config.spatial_patch_size,
            token_channels=config.decoder_token_channels,
            stage_channels=tuple(config.decoder_stage_channels),
            blocks_per_stage=config.decoder_blocks_per_stage,
        )

    def _run_transformer(
        self,
        block: nn.Module,
        state: torch.Tensor,
        coordinates: torch.Tensor,
        modulation: torch.Tensor,
        condition_tokens: torch.Tensor,
    ) -> torch.Tensor:
        if self.gradient_checkpointing and self.training and torch.is_grad_enabled():
            return checkpoint(
                block,
                state,
                coordinates,
                modulation,
                condition_tokens,
                use_reentrant=False,
            )
        return block(state, coordinates, modulation, condition_tokens)

    def _validate_raw_conditions(self, raw: Mapping[str, torch.Tensor]) -> None:
        required = ("building", "tx", "vehicle", "observed_rss", "sampling_mask")
        missing = [name for name in required if name not in raw]
        if missing:
            raise KeyError(f"raw condition cache misses {missing}")
        reference = raw["building"]
        if reference.ndim != 5 or reference.shape[1:3] != (self.frames, 1):
            raise ValueError(f"conditions must be [B,{self.frames},1,H,W]")
        for name in required:
            if raw[name].shape != reference.shape:
                raise ValueError(f"condition {name!r} shape differs from building")

    def encode_raw_conditions(self, raw: Mapping[str, torch.Tensor]) -> tuple[torch.Tensor, torch.Tensor]:
        """Create the independent, DDIM-cacheable raw-condition pyramid."""

        self._validate_raw_conditions(raw)
        dense = torch.cat(
            (
                raw["building"] + 10.0 * raw["tx"],
                raw["tx"],
                raw["vehicle"],
            ),
            dim=2,
        )
        observation = torch.cat((raw["observed_rss"], raw["sampling_mask"]), dim=2)
        high = self.condition_stem(dense, observation)
        return high, self.condition_merge(high)

    def build_inputs(
        self,
        noisy_target: torch.Tensor,
        cache: Mapping[str, torch.Tensor],
    ) -> tuple[torch.Tensor, torch.Tensor]:
        if noisy_target.ndim != 5 or noisy_target.shape[1:3] != (self.frames, 1):
            raise ValueError(f"expected noisy target [B,{self.frames},1,H,W]")
        self._validate_raw_conditions(cache)
        dense = torch.cat(
            (
                noisy_target,
                cache["building"] + 10.0 * cache["tx"],
                cache["tx"],
                cache["vehicle"],
            ),
            dim=2,
        )
        observation = torch.cat((cache["observed_rss"], cache["sampling_mask"]), dim=2)
        return dense, observation

    def forward(self, noisy_target: torch.Tensor, timesteps: torch.Tensor, cache: Mapping[str, torch.Tensor]) -> torch.Tensor:
        if timesteps.shape != (noisy_target.shape[0],):
            raise ValueError("timesteps must contain one value per video")
        required = ("hwm_gate", "condition_high", "condition_low")
        missing = [name for name in required if name not in cache]
        if missing:
            raise KeyError(f"encoded condition cache misses {missing}")
        dense, observation = self.build_inputs(noisy_target, cache)
        state = self.input_stem(dense, observation)
        state = apply_detached_hwm_gate(state, cache["hwm_gate"])
        condition_high = cache["condition_high"]
        condition_low = cache["condition_low"]
        if condition_high.shape != state.shape:
            raise ValueError("high-resolution condition pyramid does not match main tokens")

        timestep_condition = self.timestep_mapping(timesteps).to(state.dtype)
        local_modulation = self.local_time_modulation(timestep_condition)
        fine_coordinates = grid_coordinates(
            state.shape[1], state.shape[2], state.shape[3], device=state.device
        )
        for block in self.local_encoder:
            state = self._run_transformer(
                block,
                state,
                fine_coordinates,
                local_modulation,
                condition_high,
            )

        processed_skip = state
        state = self.hierarchy_merge(state)
        if condition_low.shape != state.shape:
            raise ValueError("low-resolution condition pyramid does not match bottleneck tokens")
        coarse_coordinates = merge_coordinates(fine_coordinates, temporal_factor=self.temporal_factor)
        global_modulation = self.global_time_modulation(timestep_condition)
        for block in self.global_bottleneck:
            state = self._run_transformer(
                block,
                state,
                coarse_coordinates,
                global_modulation,
                condition_low,
            )

        state = self.processed_token_skip(
            self.hierarchy_expand(state),
            processed_skip,
            timestep_condition,
        )
        for block in self.local_decoder:
            state = self._run_transformer(
                block,
                state,
                fine_coordinates,
                local_modulation,
                condition_high,
            )
        return self.output_head(state, timestep_condition)
