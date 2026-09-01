"""Single-stage space-time hierarchy and its controlled processed-token skip."""

from __future__ import annotations

import torch
import torch.nn.functional as F
from torch import nn

from .common import RMSNorm, broadcast_channels, xavier_linear, zero_linear


def _group(value: torch.Tensor, temporal_factor: int) -> torch.Tensor:
    batch, time, height, width, dim = value.shape
    if time % temporal_factor or height % 2 or width % 2:
        raise ValueError("space-time merge factors must divide the token grid")
    return (
        value.reshape(batch, time // temporal_factor, temporal_factor, height // 2, 2, width // 2, 2, dim)
        .permute(0, 1, 3, 5, 2, 4, 6, 7)
        .reshape(batch, time // temporal_factor, height // 2, width // 2, temporal_factor * 4 * dim)
    )


class SpaceTimeMerge(nn.Module):
    def __init__(self, in_dim: int, out_dim: int, *, temporal_factor: int) -> None:
        super().__init__()
        if temporal_factor not in (1, 2):
            raise ValueError("temporal_factor must be one for T1 or two for W16")
        self.temporal_factor = int(temporal_factor)
        self.in_dim = int(in_dim)
        self.projection = xavier_linear(
            nn.Linear(4 * temporal_factor * in_dim, out_dim, bias=False)
        )

    def forward(self, value: torch.Tensor) -> torch.Tensor:
        return self.projection(_group(value, self.temporal_factor))


class SpaceTimeExpand(nn.Module):
    def __init__(self, in_dim: int, out_dim: int, *, temporal_factor: int) -> None:
        super().__init__()
        if temporal_factor not in (1, 2):
            raise ValueError("temporal_factor must be one for T1 or two for W16")
        self.temporal_factor = int(temporal_factor)
        self.out_dim = int(out_dim)
        self.projection = xavier_linear(
            nn.Linear(in_dim, 4 * temporal_factor * out_dim, bias=False)
        )

    def forward(self, value: torch.Tensor) -> torch.Tensor:
        batch, time, height, width, _ = value.shape
        expanded = self.projection(value).reshape(
            batch, time, height, width, self.temporal_factor, 2, 2, self.out_dim
        )
        return expanded.permute(0, 1, 4, 2, 5, 3, 6, 7).reshape(
            batch,
            time * self.temporal_factor,
            height * 2,
            width * 2,
            self.out_dim,
        )


class ControlledTokenSkip(nn.Module):
    """Fuse only processed pre-merge tokens through a bounded zero-init gate."""

    def __init__(self, dim: int, condition_dim: int) -> None:
        super().__init__()
        self.norm = RMSNorm(dim)
        self.projection = xavier_linear(nn.Linear(dim, dim, bias=False))
        self.gate = zero_linear(nn.Linear(condition_dim, dim, bias=False))

    def forward(
        self,
        decoder: torch.Tensor,
        skip: torch.Tensor,
        timestep_condition: torch.Tensor,
    ) -> torch.Tensor:
        if decoder.shape != skip.shape:
            raise ValueError("decoder and processed-token skip must have identical shapes")
        gate = torch.tanh(self.gate(F.silu(timestep_condition))).to(decoder.dtype)
        return decoder + broadcast_channels(gate, decoder) * self.projection(self.norm(skip))


def merge_coordinates(coordinates: torch.Tensor, *, temporal_factor: int) -> torch.Tensor:
    """Use group centers by averaging the exact fine-grid coordinates."""

    if coordinates.ndim != 4 or coordinates.shape[-1] != 3:
        raise ValueError("coordinates must be [T,H,W,3]")
    return _group(coordinates.unsqueeze(0), temporal_factor)[0].reshape(
        coordinates.shape[0] // temporal_factor,
        coordinates.shape[1] // 2,
        coordinates.shape[2] // 2,
        temporal_factor * 4,
        3,
    ).mean(dim=-2)
