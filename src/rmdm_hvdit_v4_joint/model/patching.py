"""Scale-preserving patch stems, detached HWM gating, and pixel decoding."""

from __future__ import annotations

import math

import torch
import torch.nn.functional as F
from torch import nn

from .common import RMSNorm, modulate, normal_linear, xavier_linear


def _xavier_conv(layer: nn.Conv2d) -> nn.Conv2d:
    nn.init.xavier_uniform_(layer.weight)
    if layer.bias is not None:
        nn.init.zeros_(layer.bias)
    return layer


def _zero_conv(layer: nn.Conv2d) -> nn.Conv2d:
    nn.init.zeros_(layer.weight)
    if layer.bias is not None:
        nn.init.zeros_(layer.bias)
    return layer


def _space_to_depth(value: torch.Tensor, patch_size: int) -> torch.Tensor:
    """Losslessly move every spatial patch value into the channel axis."""

    if value.ndim != 4:
        raise ValueError("SpaceToDepth requires [N,C,H,W]")
    if value.shape[-2] % patch_size or value.shape[-1] % patch_size:
        raise ValueError("spatial dimensions must be divisible by patch_size")
    return F.pixel_unshuffle(value, patch_size)


class T1DoubleStem(nn.Module):
    """Single-frame dual stem with no normalization before branch fusion."""

    def __init__(self, dense_channels: int, dim: int, patch_size: int) -> None:
        super().__init__()
        self.patch_size = int(patch_size)
        patch_area = self.patch_size**2
        self.dense_projection = xavier_linear(
            nn.Linear(dense_channels * patch_area, dim, bias=False)
        )
        self.observation_projection = xavier_linear(
            nn.Linear(2 * patch_area, dim, bias=False)
        )
        self.fusion = xavier_linear(nn.Linear(2 * dim, dim, bias=False))

    def forward(self, dense: torch.Tensor, observation: torch.Tensor) -> torch.Tensor:
        if dense.ndim != 5 or observation.ndim != 5 or dense.shape[1] != 1 or observation.shape[1] != 1:
            raise ValueError("T1 stems require [B,1,C,H,W]")
        dense_patch = _space_to_depth(dense[:, 0], self.patch_size).permute(0, 2, 3, 1)
        observation_patch = _space_to_depth(observation[:, 0], self.patch_size).permute(0, 2, 3, 1)
        dense_token = self.dense_projection(dense_patch)
        observation_token = self.observation_projection(observation_patch)
        return self.fusion(torch.cat((dense_token, observation_token), dim=-1)).unsqueeze(1)


class W16DoubleStem(nn.Module):
    """Two-frame tubelet stem that packs values without temporal averaging."""

    def __init__(self, dense_channels: int, dim: int, temporal_patch: int, spatial_patch: int) -> None:
        super().__init__()
        self.temporal_patch = int(temporal_patch)
        self.spatial_patch = int(spatial_patch)
        patch_volume = self.temporal_patch * self.spatial_patch**2
        self.dense_projection = xavier_linear(
            nn.Linear(dense_channels * patch_volume, dim, bias=False)
        )
        self.observation_projection = xavier_linear(
            nn.Linear(2 * patch_volume, dim, bias=False)
        )
        self.fusion = xavier_linear(nn.Linear(2 * dim, dim, bias=False))

    def _pack(self, value: torch.Tensor) -> torch.Tensor:
        batch, frames, channels, height, width = value.shape
        if frames % self.temporal_patch:
            raise ValueError("frame count must be divisible by temporal_patch")
        token_time = frames // self.temporal_patch
        # The flattened input channel order is [frame0 channels, frame1 channels].
        # PixelUnshuffle then stores all 4x4 values for each of those channels.
        paired = value.reshape(
            batch,
            token_time,
            self.temporal_patch,
            channels,
            height,
            width,
        ).reshape(batch * token_time, self.temporal_patch * channels, height, width)
        packed = _space_to_depth(paired, self.spatial_patch).permute(0, 2, 3, 1)
        return packed.reshape(batch, token_time, packed.shape[1], packed.shape[2], packed.shape[3])

    def forward(self, dense: torch.Tensor, observation: torch.Tensor) -> torch.Tensor:
        if dense.ndim != 5 or observation.ndim != 5 or dense.shape[1] != 16 or observation.shape[1] != 16:
            raise ValueError("W16 stems require [B,16,C,H,W]")
        dense_token = self.dense_projection(self._pack(dense))
        observation_token = self.observation_projection(self._pack(observation))
        return self.fusion(torch.cat((dense_token, observation_token), dim=-1))


def apply_detached_hwm_gate(tokens: torch.Tensor, hwm_gate: torch.Tensor) -> torch.Tensor:
    """Adapt the original RMDM post-first-layer gate to the patch token grid."""

    if tokens.ndim != 5 or hwm_gate.ndim != 5 or hwm_gate.shape[0] != tokens.shape[0] or hwm_gate.shape[2] != 1:
        raise ValueError("HWM gate must be [B,T,1,H,W] for a [B,T',H',W',D] token state")
    pooled = F.adaptive_avg_pool3d(
        hwm_gate.detach().permute(0, 2, 1, 3, 4).float(),
        output_size=tokens.shape[1:4],
    ).permute(0, 2, 3, 4, 1).to(tokens.dtype)
    return tokens * (1.0 + pooled)


def _group_count(channels: int) -> int:
    return math.gcd(32, int(channels))


class TimestepConditionedResBlock(nn.Module):
    """Convolutional residual block with per-video timestep shift/scale."""

    def __init__(self, channels: int, condition_dim: int) -> None:
        super().__init__()
        groups = _group_count(channels)
        self.norm1 = nn.GroupNorm(groups, channels, eps=1.0e-6)
        self.conv1 = _xavier_conv(nn.Conv2d(channels, channels, kernel_size=3, padding=1))
        self.norm2 = nn.GroupNorm(groups, channels, eps=1.0e-6)
        self.time_projection = normal_linear(nn.Linear(condition_dim, 2 * channels, bias=True))
        # The block initially preserves its input while its residual branch learns.
        self.conv2 = _zero_conv(nn.Conv2d(channels, channels, kernel_size=3, padding=1))

    def forward(self, state: torch.Tensor, timestep_condition: torch.Tensor) -> torch.Tensor:
        if state.ndim != 4 or timestep_condition.shape[0] != state.shape[0]:
            raise ValueError("pixel ResBlock expects [N,C,H,W] and one condition per frame")
        hidden = self.conv1(F.silu(self.norm1(state)))
        shift, scale = self.time_projection(F.silu(timestep_condition)).chunk(2, dim=-1)
        hidden = self.norm2(hidden)
        hidden = hidden * (1.0 + scale[:, :, None, None]) + shift[:, :, None, None]
        return state + self.conv2(F.silu(hidden))


class PixelShuffleStage(nn.Module):
    def __init__(
        self,
        in_channels: int,
        out_channels: int,
        condition_dim: int,
        blocks: int,
    ) -> None:
        super().__init__()
        self.expand = _xavier_conv(
            nn.Conv2d(in_channels, 4 * out_channels, kernel_size=3, padding=1)
        )
        self.shuffle = nn.PixelShuffle(2)
        self.blocks = nn.ModuleList(
            [TimestepConditionedResBlock(out_channels, condition_dim) for _ in range(blocks)]
        )

    def forward(self, state: torch.Tensor, timestep_condition: torch.Tensor) -> torch.Tensor:
        state = self.shuffle(self.expand(state))
        for block in self.blocks:
            state = block(state, timestep_condition)
        return state


class PixelDecoderHead(nn.Module):
    """Restore time first, then decode 32→64→128 with local convolutions."""

    def __init__(
        self,
        dim: int,
        condition_dim: int,
        *,
        temporal_patch: int,
        spatial_patch: int,
        token_channels: int,
        stage_channels: tuple[int, ...],
        blocks_per_stage: int,
    ) -> None:
        super().__init__()
        if spatial_patch != 2 ** len(stage_channels):
            raise ValueError("decoder stages must exactly invert the spatial patch size")
        if blocks_per_stage <= 0:
            raise ValueError("pixel decoder requires at least one ResBlock per stage")
        self.temporal_patch = int(temporal_patch)
        self.token_channels = int(token_channels)
        self.norm = RMSNorm(dim)
        self.time_projection = normal_linear(nn.Linear(condition_dim, 2 * dim, bias=True))
        self.token_projection = xavier_linear(
            nn.Linear(dim, self.temporal_patch * self.token_channels, bias=True)
        )
        stages: list[nn.Module] = []
        in_channels = self.token_channels
        for out_channels in stage_channels:
            stages.append(
                PixelShuffleStage(
                    in_channels,
                    int(out_channels),
                    condition_dim,
                    blocks_per_stage,
                )
            )
            in_channels = int(out_channels)
        self.stages = nn.ModuleList(stages)
        self.final_norm = nn.GroupNorm(_group_count(in_channels), in_channels, eps=1.0e-6)
        self.final_projection = _zero_conv(
            nn.Conv2d(in_channels, 1, kernel_size=3, padding=1)
        )

    def forward(self, state: torch.Tensor, timestep_condition: torch.Tensor) -> torch.Tensor:
        if state.ndim != 5 or timestep_condition.shape[0] != state.shape[0]:
            raise ValueError("pixel decoder expects [B,T,H,W,D] and one condition per video")
        shift, scale = self.time_projection(F.silu(timestep_condition)).chunk(2, dim=-1)
        features = self.token_projection(modulate(self.norm(state), shift, scale))
        batch, token_time, height, width, _ = features.shape
        frame_count = token_time * self.temporal_patch
        # Restore the physical time axis before any spatial convolution.
        features = (
            features.reshape(
                batch,
                token_time,
                height,
                width,
                self.temporal_patch,
                self.token_channels,
            )
            .permute(0, 1, 4, 5, 2, 3)
            .reshape(batch * frame_count, self.token_channels, height, width)
        )
        frame_condition = (
            timestep_condition[:, None, :]
            .expand(batch, frame_count, timestep_condition.shape[-1])
            .reshape(batch * frame_count, timestep_condition.shape[-1])
        )
        for stage in self.stages:
            features = stage(features, frame_condition)
        epsilon = self.final_projection(F.silu(self.final_norm(features)))
        return epsilon.reshape(batch, frame_count, 1, epsilon.shape[-2], epsilon.shape[-1])
