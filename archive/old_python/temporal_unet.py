"""Temporal wrappers for RMDM without modifying the original UNet.

The original RMDM model remains the single-frame baseline. This file adds a
video-capable wrapper that can load the original checkpoint into ``base_model``
and train only small temporal residual adapters.
"""

from __future__ import annotations

from typing import Any

import torch
import torch.nn as nn


class TemporalConvAdapter(nn.Module):
    """Zero-initialized temporal residual adapter for ``[B, F, C, H, W]`` tensors."""

    def __init__(self, channels: int, hidden_channels: int | None = None, kernel_size: int = 3):
        super().__init__()
        hidden = int(hidden_channels or channels)
        padding = int(kernel_size) // 2
        self.net = nn.Sequential(
            nn.Conv3d(channels, hidden, kernel_size=(kernel_size, 1, 1), padding=(padding, 0, 0)),
            nn.SiLU(),
            nn.Conv3d(hidden, channels, kernel_size=1),
        )
        nn.init.zeros_(self.net[-1].weight)
        nn.init.zeros_(self.net[-1].bias)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        if x.ndim != 5:
            raise ValueError(f"TemporalConvAdapter expects [B,F,C,H,W], got {tuple(x.shape)}")
        y = x.permute(0, 2, 1, 3, 4).contiguous()
        y = self.net(y)
        y = y.permute(0, 2, 1, 3, 4).contiguous()
        return x + y


class RMDMTemporalWrapper(nn.Module):
    """Video wrapper around the original RMDM UNet.

    Input modes:
    - image: ``sample`` is ``[B, C, H, W]`` and the call is delegated to the
      original RMDM model.
    - video: ``sample`` is ``[B, F, C, H, W]``. The wrapper applies temporal
      residual mixing to model inputs, flattens frames through the original
      model, then applies temporal residual mixing to the predicted noise and
      reconstruction branch.

    The temporal adapters are zero-initialized, so before training the wrapper is
    behaviorally aligned with the frame-by-frame baseline.
    """

    def __init__(
        self,
        base_model: nn.Module,
        input_channels: int = 4,
        out_channels: int = 1,
        temporal_hidden_channels: int = 16,
    ):
        super().__init__()
        self.base_model = base_model
        self.input_temporal = TemporalConvAdapter(input_channels, temporal_hidden_channels)
        self.noise_temporal = TemporalConvAdapter(out_channels, temporal_hidden_channels)
        self.cal_temporal = TemporalConvAdapter(out_channels, temporal_hidden_channels)

    def load_2d_state_dict(self, state_dict: dict[str, Any], strict: bool = True) -> None:
        self.base_model.load_state_dict(state_dict, strict=strict)

    def freeze_base_model(self) -> None:
        for param in self.base_model.parameters():
            param.requires_grad = False

    def forward(self, sample: torch.Tensor, timesteps: torch.Tensor):
        if sample.ndim == 4:
            return self.base_model(sample, timesteps)
        if sample.ndim != 5:
            raise ValueError(f"Expected [B,C,H,W] or [B,F,C,H,W], got {tuple(sample.shape)}")

        batch, frames, channels, height, width = sample.shape
        sample = self.input_temporal(sample)
        flat_sample = sample.reshape(batch * frames, channels, height, width)

        if timesteps.ndim == 0:
            flat_t = timesteps.reshape(1).repeat(batch * frames)
        elif timesteps.ndim == 1 and timesteps.shape[0] == batch:
            flat_t = timesteps[:, None].repeat(1, frames).reshape(batch * frames)
        elif timesteps.ndim == 2 and timesteps.shape[:2] == (batch, frames):
            flat_t = timesteps.reshape(batch * frames)
        elif timesteps.ndim == 1 and timesteps.shape[0] == batch * frames:
            flat_t = timesteps
        else:
            raise ValueError(f"Unexpected timestep shape {tuple(timesteps.shape)} for B={batch}, F={frames}")

        out = self.base_model(flat_sample, flat_t)
        if isinstance(out, tuple):
            noise, cal = out[0], out[1]
            noise = noise.reshape(batch, frames, *noise.shape[1:])
            cal = cal.reshape(batch, frames, *cal.shape[1:])
            noise = self.noise_temporal(noise)
            cal = self.cal_temporal(cal)
            return noise, cal

        out = out.reshape(batch, frames, *out.shape[1:])
        return self.noise_temporal(out)


def temporal_state_dict_from_2d(state_dict: dict[str, Any]) -> dict[str, Any]:
    """Prefix a 2D RMDM state dict for loading into ``RMDMTemporalWrapper``."""
    return {f"base_model.{key}": value for key, value in state_dict.items()}
