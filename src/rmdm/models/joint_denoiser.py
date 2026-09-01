"""Two-scale U-shaped joint spatio-temporal DiT denoiser."""

from __future__ import annotations

import torch
from torch import nn
from torch.utils.checkpoint import checkpoint

from rmdm.config import ModelConfig

from .blocks import PatchExpand, PatchMerge, STDiTBlock
from .embeddings import TimestepEmbedder, spatiotemporal_position


class JointDenoiser(nn.Module):
    def __init__(self, config: ModelConfig) -> None:
        super().__init__()
        self.config = config
        self.patch_size = int(config.patch_size)
        self.patch_embed = nn.Conv2d(1, config.high_dim, kernel_size=self.patch_size, stride=self.patch_size)
        self.time_embed = TimestepEmbedder(config.bottleneck_dim)
        common = dict(
            mlp_ratio=config.mlp_ratio,
            qk_norm=config.qk_norm,
            dropout=config.dropout,
            attention_dropout=config.attention_dropout,
        )
        self.high_encoder = nn.ModuleList(
            [
                STDiTBlock(
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
                STDiTBlock(
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
                STDiTBlock(
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

    def _block(self, block: nn.Module, state: torch.Tensor, condition: torch.Tensor, time: torch.Tensor) -> torch.Tensor:
        if self.config.gradient_checkpointing and self.training and torch.is_grad_enabled():
            return checkpoint(block, state, condition, time, use_reentrant=False)
        return block(state, condition, time)

    def forward(
        self,
        noisy_target: torch.Tensor,
        timesteps: torch.Tensor,
        condition_cache: dict[str, torch.Tensor],
    ) -> torch.Tensor:
        if noisy_target.ndim != 5 or noisy_target.shape[2] != 1:
            raise ValueError("noisy_target must be [N,T,1,H,W]")
        batch, time, _, height, width = noisy_target.shape
        if timesteps.shape != (batch,):
            raise ValueError(f"timesteps must have shape {(batch,)}, got {tuple(timesteps.shape)}")
        embedded = self.patch_embed(noisy_target.reshape(batch * time, 1, height, width))
        high_h, high_w = embedded.shape[-2:]
        state = embedded.reshape(batch, time, self.config.high_dim, high_h, high_w).permute(0, 1, 3, 4, 2)
        state = state + spatiotemporal_position(
            time, high_h, high_w, self.config.high_dim, device=state.device, dtype=state.dtype
        )
        condition_high = condition_cache["high"]
        condition_low = condition_cache["low"]
        if condition_high.shape != state.shape:
            raise ValueError(f"High condition shape {condition_high.shape} does not match state {state.shape}")
        time_embedding = self.time_embed(timesteps).to(dtype=state.dtype)
        for block in self.high_encoder:
            state = self._block(block, state, condition_high, time_embedding)
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
        if condition_low.shape != state.shape:
            raise ValueError(f"Low condition shape {condition_low.shape} does not match state {state.shape}")
        for block in self.bottleneck:
            state = self._block(block, state, condition_low, time_embedding)
        state = self.state_expand(state) + skip
        for block in self.high_decoder:
            state = self._block(block, state, condition_high, time_embedding)
        patches = self.output(self.output_norm(state))
        output = (
            patches.reshape(batch, time, high_h, high_w, self.patch_size, self.patch_size)
            .permute(0, 1, 2, 4, 3, 5)
            .reshape(batch, time, 1, height, width)
        )
        return output
