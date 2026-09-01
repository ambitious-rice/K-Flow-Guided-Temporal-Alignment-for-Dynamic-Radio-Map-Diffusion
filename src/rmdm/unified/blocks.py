"""Condition-fusion-free ST-DiT blocks for the unified-input model."""

from __future__ import annotations

import torch
from torch import nn

from rmdm.models.blocks import FeedForward, SpatialAttention, TemporalAttention, modulate


class UnifiedSTDiTBlock(nn.Module):
    """Spatial/temporal/MLP block conditioned only by diffusion timestep."""

    def __init__(
        self,
        dim: int,
        heads: int,
        time_dim: int,
        *,
        window_size: int | None,
        shifted: bool,
        mlp_ratio: float,
        qk_norm: bool,
        dropout: float,
        attention_dropout: float,
    ) -> None:
        super().__init__()
        self.dim = int(dim)
        self.spatial_norm = nn.LayerNorm(dim, elementwise_affine=False)
        self.temporal_norm = nn.LayerNorm(dim, elementwise_affine=False)
        self.mlp_norm = nn.LayerNorm(dim, elementwise_affine=False)
        self.spatial = SpatialAttention(
            dim,
            heads,
            window_size=window_size,
            shifted=shifted,
            qk_norm=qk_norm,
            attention_dropout=attention_dropout,
            projection_dropout=dropout,
        )
        self.temporal = TemporalAttention(
            dim,
            heads,
            qk_norm=qk_norm,
            attention_dropout=attention_dropout,
            projection_dropout=dropout,
        )
        self.mlp = FeedForward(dim, mlp_ratio, dropout)
        self.modulation = nn.Sequential(nn.SiLU(), nn.Linear(time_dim, 9 * dim))
        nn.init.zeros_(self.modulation[-1].weight)
        nn.init.zeros_(self.modulation[-1].bias)

    def forward(self, state: torch.Tensor, time_embedding: torch.Tensor) -> torch.Tensor:
        values = self.modulation(time_embedding).chunk(9, dim=-1)
        shift_s, scale_s, gate_s, shift_t, scale_t, gate_t, shift_m, scale_m, gate_m = values
        spatial = self.spatial(modulate(self.spatial_norm(state), shift_s, scale_s))
        state = state + gate_s[:, None, None, None, :] * spatial
        temporal = self.temporal(modulate(self.temporal_norm(state), shift_t, scale_t))
        state = state + gate_t[:, None, None, None, :] * temporal
        mlp = self.mlp(modulate(self.mlp_norm(state), shift_m, scale_m))
        return state + gate_m[:, None, None, None, :] * mlp
