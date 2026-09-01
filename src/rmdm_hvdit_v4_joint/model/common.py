"""Normalization, timestep mapping, and initialization primitives for V4.

The V4 transformer follows the Wan/DiT residual-modulation pattern: a shared
timestep projection supplies shift/scale/gate values, while every block owns a
small learned modulation table. Attention and feed-forward outputs are live at
initialization; explicitly controlled condition paths, pixel-decoder residual
tails, and the final epsilon projection are zero initialized.
"""

from __future__ import annotations

import math

import torch
import torch.nn.functional as F
from torch import nn


def zero_linear(layer: nn.Linear) -> nn.Linear:
    nn.init.zeros_(layer.weight)
    if layer.bias is not None:
        nn.init.zeros_(layer.bias)
    return layer


def xavier_linear(layer: nn.Linear) -> nn.Linear:
    nn.init.xavier_uniform_(layer.weight)
    if layer.bias is not None:
        nn.init.zeros_(layer.bias)
    return layer


def normal_linear(layer: nn.Linear, *, std: float = 0.02) -> nn.Linear:
    nn.init.normal_(layer.weight, mean=0.0, std=float(std))
    if layer.bias is not None:
        nn.init.zeros_(layer.bias)
    return layer


class RMSNorm(nn.Module):
    def __init__(self, dim: int, *, eps: float = 1.0e-6, affine: bool = True) -> None:
        super().__init__()
        self.scale = nn.Parameter(torch.ones(dim)) if affine else None
        self.eps = float(eps)

    def forward(self, value: torch.Tensor) -> torch.Tensor:
        normalized = F.rms_norm(value.float(), (value.shape[-1],), eps=self.eps).to(value.dtype)
        if self.scale is None:
            return normalized
        return normalized * self.scale.to(value.dtype)


class GEGLU(nn.Module):
    def __init__(self, in_dim: int, hidden_dim: int) -> None:
        super().__init__()
        self.projection = xavier_linear(nn.Linear(in_dim, 2 * hidden_dim, bias=False))

    def forward(self, value: torch.Tensor) -> torch.Tensor:
        value, gate = self.projection(value).chunk(2, dim=-1)
        return value * F.gelu(gate, approximate="tanh")


def sinusoidal_timestep_embedding(timesteps: torch.Tensor, dim: int) -> torch.Tensor:
    half = dim // 2
    frequencies = torch.exp(
        -math.log(10_000.0)
        * torch.arange(half, device=timesteps.device, dtype=torch.float32)
        / max(half, 1)
    )
    phase = timesteps.float().reshape(-1, 1) * frequencies.reshape(1, -1)
    embedding = torch.cat((phase.cos(), phase.sin()), dim=-1)
    if dim % 2:
        embedding = torch.cat((embedding, torch.zeros_like(embedding[:, :1])), dim=-1)
    return embedding


class MappingBlock(nn.Module):
    def __init__(self, dim: int, hidden_dim: int, dropout: float) -> None:
        super().__init__()
        self.norm = RMSNorm(dim)
        self.up = GEGLU(dim, hidden_dim)
        self.dropout = nn.Dropout(dropout)
        self.down = normal_linear(nn.Linear(hidden_dim, dim, bias=False))

    def forward(self, value: torch.Tensor) -> torch.Tensor:
        return value + self.down(self.dropout(self.up(self.norm(value))))


class TimestepMapping(nn.Module):
    """Map integer diffusion steps to the shared conditioning width."""

    def __init__(self, width: int, depth: int, hidden_dim: int, dropout: float) -> None:
        super().__init__()
        self.frequency_dim = 256
        self.input = normal_linear(nn.Linear(self.frequency_dim, width, bias=False))
        self.input_norm = RMSNorm(width)
        self.blocks = nn.ModuleList([MappingBlock(width, hidden_dim, dropout) for _ in range(depth)])
        self.output_norm = RMSNorm(width)

    def forward(self, timesteps: torch.Tensor) -> torch.Tensor:
        value = self.input(sinusoidal_timestep_embedding(timesteps, self.frequency_dim).to(self.input.weight.dtype))
        value = self.input_norm(value)
        for block in self.blocks:
            value = block(value)
        return self.output_norm(value)


class SharedTimeModulation(nn.Module):
    """One Wan-style timestep projection shared by all blocks of a width."""

    def __init__(self, condition_dim: int, dim: int) -> None:
        super().__init__()
        self.dim = int(dim)
        self.projection = normal_linear(nn.Linear(condition_dim, 6 * dim, bias=True))

    def forward(self, condition: torch.Tensor) -> torch.Tensor:
        return self.projection(F.silu(condition)).reshape(condition.shape[0], 6, self.dim)


def broadcast_channels(value: torch.Tensor, reference: torch.Tensor) -> torch.Tensor:
    """Broadcast ``[B,D]`` over every token axis of ``reference``."""

    if value.ndim != 2 or value.shape[0] != reference.shape[0] or value.shape[-1] != reference.shape[-1]:
        raise ValueError("channel modulation must be [B,D] and match the token state")
    return value.reshape(value.shape[0], *((1,) * (reference.ndim - 2)), value.shape[-1])


def modulate(value: torch.Tensor, shift: torch.Tensor, scale: torch.Tensor) -> torch.Tensor:
    return value * (1.0 + broadcast_channels(scale, value)) + broadcast_channels(shift, value)
