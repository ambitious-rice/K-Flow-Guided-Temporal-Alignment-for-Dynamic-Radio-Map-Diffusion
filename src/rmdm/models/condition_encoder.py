"""Six-channel aligned condition encoder."""

from __future__ import annotations

import torch
from torch import nn

from .blocks import PatchMerge
from .embeddings import spatiotemporal_position


class ConditionEncoder(nn.Module):
    def __init__(self, high_dim: int, bottleneck_dim: int, patch_size: int = 4) -> None:
        super().__init__()
        self.high_dim = int(high_dim)
        self.bottleneck_dim = int(bottleneck_dim)
        self.patch_size = int(patch_size)
        self.stem = nn.Sequential(
            nn.Conv2d(6, 64, kernel_size=3, padding=1),
            nn.SiLU(),
            nn.Conv2d(64, high_dim, kernel_size=patch_size, stride=patch_size),
        )
        self.merge = PatchMerge(high_dim, bottleneck_dim)

    def forward(self, conditions: torch.Tensor) -> dict[str, torch.Tensor]:
        if conditions.ndim != 5 or conditions.shape[2] != 6:
            raise ValueError("conditions must be [N,T,6,H,W]")
        batch, time, _, height, width = conditions.shape
        encoded = self.stem(conditions.reshape(batch * time, 6, height, width))
        high_h, high_w = encoded.shape[-2:]
        high = encoded.reshape(batch, time, self.high_dim, high_h, high_w).permute(0, 1, 3, 4, 2)
        high = high + spatiotemporal_position(
            time, high_h, high_w, self.high_dim, device=high.device, dtype=high.dtype
        )
        low = self.merge(high)
        low = low + spatiotemporal_position(
            time, low.shape[2], low.shape[3], self.bottleneck_dim, device=low.device, dtype=low.dtype
        )
        return {"high": high, "low": low}

