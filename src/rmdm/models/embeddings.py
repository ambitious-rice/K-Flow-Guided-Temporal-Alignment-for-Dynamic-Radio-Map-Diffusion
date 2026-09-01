"""Fixed spatial/temporal positions and diffusion timestep embeddings."""

from __future__ import annotations

import math

import torch
from torch import nn


def sinusoidal_embedding(values: torch.Tensor, dim: int, max_period: float = 10_000.0) -> torch.Tensor:
    if dim <= 0:
        raise ValueError("embedding dim must be positive")
    half = dim // 2
    frequencies = torch.exp(
        -math.log(max_period)
        * torch.arange(half, device=values.device, dtype=torch.float32)
        / max(half, 1)
    )
    phases = values.to(dtype=torch.float32).reshape(-1, 1) * frequencies.reshape(1, -1)
    embedding = torch.cat([torch.cos(phases), torch.sin(phases)], dim=-1)
    if dim % 2:
        embedding = torch.cat([embedding, torch.zeros_like(embedding[:, :1])], dim=-1)
    return embedding


def spatiotemporal_position(
    time: int,
    height: int,
    width: int,
    dim: int,
    *,
    device: torch.device,
    dtype: torch.dtype,
) -> torch.Tensor:
    if dim % 4:
        raise ValueError("position embedding dim must be divisible by 4")
    temporal_dim = dim // 2
    spatial_axis_dim = dim // 4
    t = sinusoidal_embedding(torch.arange(time, device=device), temporal_dim)
    y = sinusoidal_embedding(torch.arange(height, device=device), spatial_axis_dim)
    x = sinusoidal_embedding(torch.arange(width, device=device), spatial_axis_dim)
    temporal = torch.cat(
        [t[:, None, None, :], torch.zeros(time, 1, 1, dim - temporal_dim, device=device)], dim=-1
    )
    spatial = torch.cat(
        [
            torch.zeros(height, width, temporal_dim, device=device),
            y[:, None, :].expand(height, width, spatial_axis_dim),
            x[None, :, :].expand(height, width, spatial_axis_dim),
        ],
        dim=-1,
    )
    return (temporal + spatial.unsqueeze(0)).unsqueeze(0).to(dtype=dtype)


class TimestepEmbedder(nn.Module):
    def __init__(self, hidden_dim: int, frequency_dim: int = 256) -> None:
        super().__init__()
        self.frequency_dim = int(frequency_dim)
        self.mlp = nn.Sequential(
            nn.Linear(self.frequency_dim, hidden_dim),
            nn.SiLU(),
            nn.Linear(hidden_dim, hidden_dim),
        )

    def forward(self, timesteps: torch.Tensor) -> torch.Tensor:
        return self.mlp(sinusoidal_embedding(timesteps, self.frequency_dim).to(dtype=self.mlp[0].weight.dtype))

