"""Two-scale W16 denoiser with a seven-channel unified pixel input."""

from __future__ import annotations

from collections.abc import Mapping

import torch
from torch import nn
from torch.utils.checkpoint import checkpoint

from rmdm.config import ModelConfig
from rmdm.models.blocks import PatchExpand, PatchMerge
from rmdm.models.embeddings import TimestepEmbedder, spatiotemporal_position

from .blocks import UnifiedSTDiTBlock


UNIFIED_INPUT_CHANNELS = (
    "noisy_target",
    "building",
    "tx",
    "vehicle",
    "observed_rss",
    "sampling_mask",
    "prior",
)


class UnifiedJointDenoiser(nn.Module):
    """Joint epsilon denoiser whose conditions enter through patch embedding."""

    def __init__(self, config: ModelConfig) -> None:
        super().__init__()
        self.config = config
        self.patch_size = int(config.patch_size)
        self.patch_embed = nn.Conv2d(
            len(UNIFIED_INPUT_CHANNELS),
            config.high_dim,
            kernel_size=self.patch_size,
            stride=self.patch_size,
        )
        self.time_embed = TimestepEmbedder(config.bottleneck_dim)
        common = dict(
            mlp_ratio=config.mlp_ratio,
            qk_norm=config.qk_norm,
            dropout=config.dropout,
            attention_dropout=config.attention_dropout,
        )
        self.high_encoder = nn.ModuleList(
            [
                UnifiedSTDiTBlock(
                    config.high_dim,
                    config.high_heads,
                    config.bottleneck_dim,
                    window_size=config.window_attention_size,
                    shifted=bool(index % 2),
                    **common,
                )
                for index in range(config.high_encoder_blocks)
            ]
        )
        self.state_merge = PatchMerge(config.high_dim, config.bottleneck_dim)
        self.bottleneck = nn.ModuleList(
            [
                UnifiedSTDiTBlock(
                    config.bottleneck_dim,
                    config.bottleneck_heads,
                    config.bottleneck_dim,
                    window_size=None,
                    shifted=False,
                    **common,
                )
                for _ in range(config.bottleneck_blocks)
            ]
        )
        self.state_expand = PatchExpand(config.bottleneck_dim, config.high_dim)
        self.high_decoder = nn.ModuleList(
            [
                UnifiedSTDiTBlock(
                    config.high_dim,
                    config.high_heads,
                    config.bottleneck_dim,
                    window_size=config.window_attention_size,
                    shifted=bool(index % 2),
                    **common,
                )
                for index in range(config.high_decoder_blocks)
            ]
        )
        self.output_norm = nn.LayerNorm(config.high_dim)
        self.output = nn.Linear(config.high_dim, self.patch_size * self.patch_size)
        nn.init.zeros_(self.output.weight)
        nn.init.zeros_(self.output.bias)

    @staticmethod
    def build_unified_input(
        noisy_target: torch.Tensor,
        condition_cache: Mapping[str, torch.Tensor],
    ) -> torch.Tensor:
        """Build ``[X_tau, B, Tx, V, Y, M, P]`` without hidden preprocessing."""

        if noisy_target.ndim != 5 or noisy_target.shape[2] != 1:
            raise ValueError("noisy_target must be [N,T,1,H,W]")
        missing = [name for name in UNIFIED_INPUT_CHANNELS[1:] if name not in condition_cache]
        if missing:
            raise KeyError(f"Condition cache misses unified inputs: {missing}")
        values = [noisy_target]
        for name in UNIFIED_INPUT_CHANNELS[1:]:
            value = condition_cache[name]
            if value.shape != noisy_target.shape:
                raise ValueError(
                    f"Unified input {name!r} has shape {tuple(value.shape)}, "
                    f"expected {tuple(noisy_target.shape)}"
                )
            values.append(value.to(device=noisy_target.device, dtype=noisy_target.dtype))
        return torch.cat(values, dim=2)

    def _block(
        self,
        block: nn.Module,
        state: torch.Tensor,
        time_embedding: torch.Tensor,
    ) -> torch.Tensor:
        if self.config.gradient_checkpointing and self.training and torch.is_grad_enabled():
            return checkpoint(block, state, time_embedding, use_reentrant=False)
        return block(state, time_embedding)

    def forward(
        self,
        noisy_target: torch.Tensor,
        timesteps: torch.Tensor,
        condition_cache: Mapping[str, torch.Tensor],
    ) -> torch.Tensor:
        if timesteps.shape != (noisy_target.shape[0],):
            raise ValueError(
                f"timesteps must have shape {(noisy_target.shape[0],)}, got {tuple(timesteps.shape)}"
            )
        model_input = self.build_unified_input(noisy_target, condition_cache)
        batch, time, _, height, width = model_input.shape
        embedded = self.patch_embed(model_input.reshape(batch * time, len(UNIFIED_INPUT_CHANNELS), height, width))
        high_h, high_w = embedded.shape[-2:]
        state = embedded.reshape(batch, time, self.config.high_dim, high_h, high_w).permute(0, 1, 3, 4, 2)
        state = state + spatiotemporal_position(
            time, high_h, high_w, self.config.high_dim, device=state.device, dtype=state.dtype
        )
        time_embedding = self.time_embed(timesteps).to(dtype=state.dtype)
        for block in self.high_encoder:
            state = self._block(block, state, time_embedding)
        skip = state
        state = self.state_merge(state)
        state = state + spatiotemporal_position(
            time,
            state.shape[2],
            state.shape[3],
            self.config.bottleneck_dim,
            device=state.device,
            dtype=state.dtype,
        )
        for block in self.bottleneck:
            state = self._block(block, state, time_embedding)
        state = self.state_expand(state) + skip
        for block in self.high_decoder:
            state = self._block(block, state, time_embedding)
        patches = self.output(self.output_norm(state))
        return (
            patches.reshape(batch, time, high_h, high_w, self.patch_size, self.patch_size)
            .permute(0, 1, 2, 4, 3, 5)
            .reshape(batch, time, 1, height, width)
        )
