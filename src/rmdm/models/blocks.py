"""Spatio-temporal DiT building blocks."""

from __future__ import annotations

import torch
import torch.nn.functional as F
from torch import nn


def modulate(value: torch.Tensor, shift: torch.Tensor, scale: torch.Tensor) -> torch.Tensor:
    return value * (1 + scale[:, None, None, None, :]) + shift[:, None, None, None, :]


class MultiHeadSelfAttention(nn.Module):
    def __init__(
        self,
        dim: int,
        heads: int,
        *,
        qk_norm: bool = True,
        attention_dropout: float = 0.0,
        projection_dropout: float = 0.0,
    ) -> None:
        super().__init__()
        if dim % heads:
            raise ValueError("dim must be divisible by heads")
        self.heads = int(heads)
        self.head_dim = dim // heads
        self.qkv = nn.Linear(dim, 3 * dim)
        self.q_norm = nn.LayerNorm(self.head_dim, elementwise_affine=False) if qk_norm else nn.Identity()
        self.k_norm = nn.LayerNorm(self.head_dim, elementwise_affine=False) if qk_norm else nn.Identity()
        self.projection = nn.Linear(dim, dim)
        self.attention_dropout = float(attention_dropout)
        self.projection_dropout = nn.Dropout(projection_dropout)

    def forward(self, value: torch.Tensor, attention_mask: torch.Tensor | None = None) -> torch.Tensor:
        batch, tokens, dim = value.shape
        qkv = self.qkv(value).reshape(batch, tokens, 3, self.heads, self.head_dim).permute(2, 0, 3, 1, 4)
        query, key, val = qkv.unbind(0)
        query = self.q_norm(query)
        key = self.k_norm(key)
        attended = F.scaled_dot_product_attention(
            query,
            key,
            val,
            attn_mask=attention_mask,
            dropout_p=self.attention_dropout if self.training else 0.0,
        )
        attended = attended.transpose(1, 2).reshape(batch, tokens, dim)
        return self.projection_dropout(self.projection(attended))


class SpatialAttention(nn.Module):
    def __init__(
        self,
        dim: int,
        heads: int,
        *,
        window_size: int | None,
        shifted: bool,
        qk_norm: bool,
        attention_dropout: float,
        projection_dropout: float,
    ) -> None:
        super().__init__()
        self.window_size = int(window_size) if window_size is not None else None
        self.shifted = bool(shifted)
        self.attention = MultiHeadSelfAttention(
            dim,
            heads,
            qk_norm=qk_norm,
            attention_dropout=attention_dropout,
            projection_dropout=projection_dropout,
        )

    def forward(self, value: torch.Tensor) -> torch.Tensor:
        batch, time, height, width, dim = value.shape
        if self.window_size is None:
            flat = value.reshape(batch * time, height * width, dim)
            return self.attention(flat).reshape(batch, time, height, width, dim)
        window = self.window_size
        if height % window or width % window:
            raise ValueError(f"window_size={window} must divide grid {height}x{width}")
        flat = value.reshape(batch * time, height, width, dim)
        shift = window // 2 if self.shifted else 0
        if shift:
            flat = torch.roll(flat, shifts=(-shift, -shift), dims=(1, 2))
        windows = (
            flat.reshape(batch * time, height // window, window, width // window, window, dim)
            .permute(0, 1, 3, 2, 4, 5)
            .reshape(-1, window * window, dim)
        )
        attention_mask = None
        if shift:
            region = torch.zeros((1, height, width, 1), device=value.device, dtype=torch.int64)
            height_slices = (slice(0, -window), slice(-window, -shift), slice(-shift, None))
            width_slices = (slice(0, -window), slice(-window, -shift), slice(-shift, None))
            label = 0
            for height_slice in height_slices:
                for width_slice in width_slices:
                    region[:, height_slice, width_slice, :] = label
                    label += 1
            region_windows = (
                region.reshape(1, height // window, window, width // window, window, 1)
                .permute(0, 1, 3, 2, 4, 5)
                .reshape(-1, window * window)
            )
            differences = region_windows[:, None, :] - region_windows[:, :, None]
            attention_mask = torch.zeros_like(differences, dtype=value.dtype).masked_fill(differences != 0, float("-inf"))
            attention_mask = attention_mask.repeat(batch * time, 1, 1).unsqueeze(1)
        windows = self.attention(windows, attention_mask)
        flat = (
            windows.reshape(batch * time, height // window, width // window, window, window, dim)
            .permute(0, 1, 3, 2, 4, 5)
            .reshape(batch * time, height, width, dim)
        )
        if shift:
            flat = torch.roll(flat, shifts=(shift, shift), dims=(1, 2))
        return flat.reshape(batch, time, height, width, dim)


class TemporalAttention(nn.Module):
    def __init__(
        self,
        dim: int,
        heads: int,
        *,
        qk_norm: bool,
        attention_dropout: float,
        projection_dropout: float,
    ) -> None:
        super().__init__()
        self.attention = MultiHeadSelfAttention(
            dim,
            heads,
            qk_norm=qk_norm,
            attention_dropout=attention_dropout,
            projection_dropout=projection_dropout,
        )

    def forward(self, value: torch.Tensor) -> torch.Tensor:
        batch, time, height, width, dim = value.shape
        sequence = value.permute(0, 2, 3, 1, 4).reshape(batch * height * width, time, dim)
        sequence = self.attention(sequence)
        return sequence.reshape(batch, height, width, time, dim).permute(0, 3, 1, 2, 4)


class FeedForward(nn.Module):
    def __init__(self, dim: int, mlp_ratio: float, dropout: float) -> None:
        super().__init__()
        hidden = int(round(dim * mlp_ratio))
        self.network = nn.Sequential(nn.Linear(dim, hidden), nn.GELU(approximate="tanh"), nn.Dropout(dropout), nn.Linear(hidden, dim))

    def forward(self, value: torch.Tensor) -> torch.Tensor:
        return self.network(value)


class ConditionFusion(nn.Module):
    def __init__(self, dim: int) -> None:
        super().__init__()
        self.state_norm = nn.LayerNorm(dim)
        self.condition_norm = nn.LayerNorm(dim)
        self.gate = nn.Linear(2 * dim, dim)
        self.condition_projection = nn.Linear(dim, dim)
        nn.init.zeros_(self.condition_projection.weight)
        nn.init.zeros_(self.condition_projection.bias)

    def forward(self, state: torch.Tensor, condition: torch.Tensor) -> torch.Tensor:
        state_norm = self.state_norm(state)
        condition_norm = self.condition_norm(condition)
        gate = torch.sigmoid(self.gate(torch.cat([state_norm, condition_norm], dim=-1)))
        return state + gate * self.condition_projection(condition_norm)


class STDiTBlock(nn.Module):
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
        self.condition_fusion = ConditionFusion(dim)
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

    def forward(self, state: torch.Tensor, condition: torch.Tensor, time_embedding: torch.Tensor) -> torch.Tensor:
        state = self.condition_fusion(state, condition)
        values = self.modulation(time_embedding).chunk(9, dim=-1)
        shift_s, scale_s, gate_s, shift_t, scale_t, gate_t, shift_m, scale_m, gate_m = values
        spatial = self.spatial(modulate(self.spatial_norm(state), shift_s, scale_s))
        state = state + gate_s[:, None, None, None, :] * spatial
        temporal = self.temporal(modulate(self.temporal_norm(state), shift_t, scale_t))
        state = state + gate_t[:, None, None, None, :] * temporal
        mlp = self.mlp(modulate(self.mlp_norm(state), shift_m, scale_m))
        return state + gate_m[:, None, None, None, :] * mlp


class PatchMerge(nn.Module):
    def __init__(self, in_dim: int, out_dim: int) -> None:
        super().__init__()
        self.projection = nn.Linear(4 * in_dim, out_dim)

    def forward(self, value: torch.Tensor) -> torch.Tensor:
        batch, time, height, width, dim = value.shape
        if height % 2 or width % 2:
            raise ValueError("PatchMerge requires even spatial dimensions")
        grouped = (
            value.reshape(batch, time, height // 2, 2, width // 2, 2, dim)
            .permute(0, 1, 2, 4, 3, 5, 6)
            .reshape(batch, time, height // 2, width // 2, 4 * dim)
        )
        return self.projection(grouped)


class PatchExpand(nn.Module):
    def __init__(self, in_dim: int, out_dim: int) -> None:
        super().__init__()
        self.out_dim = int(out_dim)
        self.projection = nn.Linear(in_dim, 4 * out_dim)

    def forward(self, value: torch.Tensor) -> torch.Tensor:
        batch, time, height, width, _ = value.shape
        expanded = self.projection(value).reshape(batch, time, height, width, 2, 2, self.out_dim)
        return expanded.permute(0, 1, 2, 4, 3, 5, 6).reshape(batch, time, height * 2, width * 2, self.out_dim)
