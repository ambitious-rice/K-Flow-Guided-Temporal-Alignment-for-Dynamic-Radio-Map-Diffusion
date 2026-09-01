"""Legacy HWM/cal branch as a trainable JointRMDM submodule."""

from __future__ import annotations

from contextlib import nullcontext
from typing import Any

import torch
from torch import nn


class Stage1Prior(nn.Module):
    def __init__(self, hwm: nn.Module, *, trainable: bool = False, chunk_size: int = 16) -> None:
        super().__init__()
        if chunk_size <= 0:
            raise ValueError("chunk_size must be positive")
        self.hwm = hwm
        self.chunk_size = int(chunk_size)
        self.trainable = True
        self.set_trainable(trainable)

    def set_trainable(self, trainable: bool) -> None:
        self.trainable = bool(trainable)
        for parameter in self.hwm.parameters():
            parameter.requires_grad_(self.trainable)
        self.hwm.train(self.training if self.trainable else False)

    def train(self, mode: bool = True):
        super().train(mode)
        self.hwm.train(mode if self.trainable else False)
        return self

    def _conditions(self, batch: dict[str, Any]) -> torch.Tensor:
        required = ("building", "tx", "vehicle", "observed_rss", "sampling_mask")
        missing = [name for name in required if name not in batch]
        if missing:
            raise KeyError(f"Sparse batch misses Stage1 inputs: {missing}")
        conditions = torch.cat([batch[name] for name in required], dim=2)
        conditions = conditions.clone()
        conditions[:, :, 0] = conditions[:, :, 0] + 10.0 * conditions[:, :, 1]
        return conditions

    def forward(self, batch: dict[str, Any]) -> torch.Tensor:
        conditions = self._conditions(batch)
        if conditions.ndim != 5 or conditions.shape[2] != 5:
            raise ValueError("Stage1 conditions must be [N,T,5,H,W]")
        batch_size, time, _, height, width = conditions.shape
        flat = conditions.reshape(batch_size * time, 5, height, width)
        outputs = []
        context = nullcontext() if self.trainable else torch.no_grad()
        with context:
            for start in range(0, flat.shape[0], self.chunk_size):
                result = self.hwm(flat[start : start + self.chunk_size])
                if not isinstance(result, (tuple, list)) or len(result) < 2:
                    raise RuntimeError("Legacy HWM must return (anchors, cal)")
                outputs.append(result[1])
        prior = torch.cat(outputs, dim=0).reshape(batch_size, time, 1, height, width)
        return prior

