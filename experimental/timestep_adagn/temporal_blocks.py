"""Small temporal modules for RMDM video experiments."""

from __future__ import annotations

import _paths  # noqa: F401

import torch
import torch.nn as nn


class TemporalSelfAttention(nn.Module):
    """Residual self-attention over frames for ``[B, F, C, H, W]`` tensors.

    The residual gate is initialized to zero, so adding this block preserves the
    frame-by-frame model at initialization.
    """

    def __init__(
        self,
        channels: int,
        num_heads: int = 4,
        use_frame_positional_encoding: bool = False,
        max_frames: int = 128,
    ):
        super().__init__()
        channels = int(channels)
        heads = max(1, min(int(num_heads), channels))
        while channels % heads != 0 and heads > 1:
            heads -= 1
        self.norm = nn.LayerNorm(channels)
        self.attn = nn.MultiheadAttention(channels, heads, batch_first=True)
        self.gate = nn.Parameter(torch.zeros(()))
        self.frame_positional_encoding = None
        if use_frame_positional_encoding:
            self.frame_positional_encoding = nn.Parameter(torch.zeros(int(max_frames), channels))

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        if x.ndim != 5:
            raise ValueError(f"TemporalSelfAttention expects [B,F,C,H,W], got {tuple(x.shape)}")
        batch, frames, channels, height, width = x.shape
        tokens = x.permute(0, 3, 4, 1, 2).reshape(batch * height * width, frames, channels)
        attn_tokens = self.norm(tokens)
        if self.frame_positional_encoding is not None:
            if frames > self.frame_positional_encoding.shape[0]:
                raise ValueError(
                    f"frames={frames} exceeds max_frames={self.frame_positional_encoding.shape[0]} "
                    "for frame positional encoding"
                )
            pos = self.frame_positional_encoding[:frames].to(dtype=attn_tokens.dtype, device=attn_tokens.device)
            attn_tokens = attn_tokens + pos.unsqueeze(0)
        update, _ = self.attn(attn_tokens, attn_tokens, attn_tokens, need_weights=False)
        update = update.reshape(batch, height, width, frames, channels).permute(0, 3, 4, 1, 2)
        return x + self.gate.to(dtype=x.dtype) * update.to(dtype=x.dtype)


class TemporalConvAdapter(nn.Module):
    """Zero-initialized temporal residual convolution for ``[B, F, C, H, W]``."""

    def __init__(self, channels: int, hidden_channels: int | None = None, kernel_size: int = 3):
        super().__init__()
        hidden = int(hidden_channels or channels)
        padding = int(kernel_size) // 2
        self.net = nn.Sequential(
            nn.Conv3d(int(channels), hidden, kernel_size=(kernel_size, 1, 1), padding=(padding, 0, 0)),
            nn.SiLU(),
            nn.Conv3d(hidden, int(channels), kernel_size=1),
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
