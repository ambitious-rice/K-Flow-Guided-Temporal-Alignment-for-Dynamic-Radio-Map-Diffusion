"""Wan-style DiT blocks with zero-initialized raw-condition residuals."""

from __future__ import annotations

import math

import torch
from torch import nn

from .attention import GlobalSpaceTimeSelfAttention, NeighborhoodSelfAttention
from .common import GEGLU, RMSNorm, broadcast_channels, modulate, xavier_linear, zero_linear


class FeedForward(nn.Module):
    def __init__(self, dim: int, hidden_dim: int, dropout: float) -> None:
        super().__init__()
        self.up = GEGLU(dim, hidden_dim)
        self.dropout = nn.Dropout(dropout)
        self.down = xavier_linear(nn.Linear(hidden_dim, dim, bias=False))

    def forward(self, state: torch.Tensor) -> torch.Tensor:
        return self.down(self.dropout(self.up(state)))


class _ConditionedDiTBlock(nn.Module):
    """Own residual modulation while delegating only the attention transform."""

    def __init__(self, dim: int, hidden_dim: int, dropout: float) -> None:
        super().__init__()
        self.dim = int(dim)
        self.attention_norm = RMSNorm(dim)
        self.feedforward_norm = RMSNorm(dim)
        self.condition_norm = RMSNorm(dim)
        # This is the only zero-initialized path inside a block.  It prevents a
        # new condition highway while retaining a first-step gradient through
        # the non-zero timestep residual gate.
        self.condition_projection = zero_linear(nn.Linear(dim, dim, bias=False))
        self.feedforward = FeedForward(dim, hidden_dim, dropout)
        self.modulation = nn.Parameter(torch.empty(6, dim))
        nn.init.normal_(self.modulation, mean=0.0, std=1.0 / math.sqrt(dim))

    def _forward(
        self,
        state: torch.Tensor,
        coordinates: torch.Tensor,
        shared_modulation: torch.Tensor,
        condition_tokens: torch.Tensor,
    ) -> torch.Tensor:
        if condition_tokens.shape != state.shape:
            raise ValueError("per-block condition tokens must match the state grid and width")
        if shared_modulation.shape != (state.shape[0], 6, self.dim):
            raise ValueError("shared timestep modulation must be [B,6,D]")
        values = shared_modulation + self.modulation.to(shared_modulation.dtype).unsqueeze(0)
        shift_attention, scale_attention, gate_attention, shift_ffn, scale_ffn, gate_ffn = values.unbind(1)

        condition_residual = self.condition_projection(self.condition_norm(condition_tokens))
        state = state + broadcast_channels(torch.tanh(gate_attention), state) * condition_residual

        attention_input = modulate(self.attention_norm(state), shift_attention, scale_attention)
        state = state + broadcast_channels(gate_attention, state) * self._attention(attention_input, coordinates)
        ffn_input = modulate(self.feedforward_norm(state), shift_ffn, scale_ffn)
        return state + broadcast_channels(gate_ffn, state) * self.feedforward(ffn_input)


class LocalTransformerLayer(_ConditionedDiTBlock):
    def __init__(
        self,
        dim: int,
        head_dim: int,
        hidden_dim: int,
        *,
        kernel_size: tuple[int, ...],
        rope_axis_dims: tuple[int, int, int],
        backend: str,
        dropout: float,
    ) -> None:
        super().__init__(dim, hidden_dim, dropout)
        self.attention = NeighborhoodSelfAttention(
            dim,
            head_dim,
            kernel_size=kernel_size,
            rope_axis_dims=rope_axis_dims,
            backend=backend,
            dropout=dropout,
        )

    def _attention(self, state: torch.Tensor, coordinates: torch.Tensor) -> torch.Tensor:
        return self.attention(state, coordinates)

    def forward(
        self,
        state: torch.Tensor,
        coordinates: torch.Tensor,
        shared_modulation: torch.Tensor,
        condition_tokens: torch.Tensor,
    ) -> torch.Tensor:
        return self._forward(state, coordinates, shared_modulation, condition_tokens)


class GlobalTransformerLayer(_ConditionedDiTBlock):
    def __init__(
        self,
        dim: int,
        head_dim: int,
        hidden_dim: int,
        *,
        rope_axis_dims: tuple[int, int, int],
        dropout: float,
    ) -> None:
        super().__init__(dim, hidden_dim, dropout)
        self.attention = GlobalSpaceTimeSelfAttention(
            dim,
            head_dim,
            rope_axis_dims=rope_axis_dims,
            dropout=dropout,
        )

    def _attention(self, state: torch.Tensor, coordinates: torch.Tensor) -> torch.Tensor:
        return self.attention(state, coordinates)

    def forward(
        self,
        state: torch.Tensor,
        coordinates: torch.Tensor,
        shared_modulation: torch.Tensor,
        condition_tokens: torch.Tensor,
    ) -> torch.Tensor:
        return self._forward(state, coordinates, shared_modulation, condition_tokens)
