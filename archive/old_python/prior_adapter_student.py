"""Frozen RMDM with temporal adapters on the highway/PINN prior branch.

This is intentionally separate from ``residual_kflow_student.py``. The goal is
to test whether making the RMDM highway prior temporally consistent is more
effective than correcting the final epsilon output.
"""

from __future__ import annotations

import os
import sys
from dataclasses import dataclass

import torch
from torch import nn
import torch.nn.functional as F

_UTILS_DIR = os.path.join(os.path.dirname(os.path.abspath(__file__)), "utils")
if _UTILS_DIR not in sys.path:
    sys.path.append(_UTILS_DIR)
from nn import timestep_embedding


def _group_count(channels: int, max_groups: int = 32) -> int:
    for groups in range(min(max_groups, channels), 0, -1):
        if channels % groups == 0:
            return groups
    return 1


class SeparablePriorTemporalAdapter(nn.Module):
    """Zero-init local 3D residual adapter for prior tensors [B,F,C,H,W]."""

    def __init__(self, channels: int, reduction: int = 4, min_hidden: int = 8):
        super().__init__()
        channels = int(channels)
        hidden = max(int(min_hidden), channels // max(int(reduction), 1))
        self.norm = nn.GroupNorm(_group_count(channels), channels)
        self.down = nn.Conv3d(channels, hidden, kernel_size=1)
        self.temporal = nn.Conv3d(hidden, hidden, kernel_size=(3, 1, 1), padding=(1, 0, 0))
        self.spatial = nn.Conv3d(hidden, hidden, kernel_size=(1, 3, 3), padding=(0, 1, 1))
        self.up = nn.Conv3d(hidden, channels, kernel_size=1)
        self.gate = nn.Parameter(torch.tensor(1.0))
        nn.init.zeros_(self.up.weight)
        nn.init.zeros_(self.up.bias)

    def forward(self, x: torch.Tensor) -> tuple[torch.Tensor, torch.Tensor]:
        if x.ndim != 5:
            raise ValueError(f"expected [B,F,C,H,W], got {tuple(x.shape)}")
        y = x.permute(0, 2, 1, 3, 4).contiguous()
        delta = self.down(self.norm(y))
        delta = F.silu(self.temporal(F.silu(delta)))
        delta = F.silu(self.spatial(delta))
        delta = self.up(delta)
        gate = torch.tanh(self.gate)
        out = y + gate * delta
        return out.permute(0, 2, 1, 3, 4).contiguous(), delta.detach().abs().mean()


@dataclass
class PriorAdapterStats:
    anchor_delta: torch.Tensor
    cal_delta: torch.Tensor
    gate_mean: torch.Tensor


class FrozenRMDMPriorAdapterStudent(nn.Module):
    """Frozen RMDM with temporal correction on ``highway_forward`` outputs."""

    def __init__(
        self,
        base_model: nn.Module,
        input_channels: int = 4,
        out_channels: int = 1,
        reduction: int = 4,
        min_hidden: int = 8,
        adapt_cal: bool = True,
    ):
        super().__init__()
        self.base_model = base_model
        self.unet = getattr(base_model, "unet", base_model)
        self.input_channels = int(input_channels)
        self.out_channels = int(out_channels)
        self.adapt_cal = bool(adapt_cal)

        anchor_channels = self._infer_anchor_channels()
        self.anchor_adapters = nn.ModuleList(
            [SeparablePriorTemporalAdapter(ch, reduction=reduction, min_hidden=min_hidden) for ch in anchor_channels]
        )
        self.cal_adapter = SeparablePriorTemporalAdapter(out_channels, reduction=1, min_hidden=min_hidden) if adapt_cal else None

        for param in self.base_model.parameters():
            param.requires_grad = False
        self.base_model.eval()

    def _infer_anchor_channels(self) -> list[int]:
        device = next(self.unet.parameters()).device
        dtype = next(self.unet.parameters()).dtype
        cond_channels = self.input_channels - self.out_channels
        with torch.no_grad():
            dummy = torch.zeros(
                1,
                cond_channels,
                int(self.unet.image_size),
                int(self.unet.image_size),
                device=device,
                dtype=dtype,
            )
            anch, _ = self.unet.highway_forward(dummy)
        anchors = anch if isinstance(anch, (list, tuple)) else [anch]
        return [int(item.shape[1]) for item in anchors]

    def train(self, mode: bool = True):
        super().train(mode)
        self.base_model.eval()
        return self

    def adapter_parameters(self):
        yield from self.anchor_adapters.parameters()
        if self.cal_adapter is not None:
            yield from self.cal_adapter.parameters()

    @staticmethod
    def _flat_timesteps(timesteps: torch.Tensor, batch: int, frames: int) -> torch.Tensor:
        if timesteps.ndim == 1 and timesteps.shape[0] == batch:
            return timesteps[:, None].repeat(1, frames).reshape(batch * frames)
        if timesteps.ndim == 1 and timesteps.shape[0] == batch * frames:
            return timesteps
        raise ValueError(f"unexpected timestep shape {tuple(timesteps.shape)} for B={batch}, F={frames}")

    @staticmethod
    def _to_sequence(value: torch.Tensor, batch: int, frames: int) -> torch.Tensor:
        return value.reshape(batch, frames, *value.shape[1:]).contiguous()

    def _adapt_anchor(self, anch, batch: int, frames: int):
        anchors = anch if isinstance(anch, (list, tuple)) else [anch]
        adapted = []
        deltas = []
        for idx, item in enumerate(anchors):
            seq = self._to_sequence(item, batch, frames)
            if idx < len(self.anchor_adapters):
                seq, delta = self.anchor_adapters[idx](seq)
                deltas.append(delta)
            adapted.append(seq.reshape(batch * frames, *seq.shape[2:]))
        if isinstance(anch, tuple):
            return tuple(adapted), deltas
        if isinstance(anch, list):
            return adapted, deltas
        return adapted[0], deltas

    @staticmethod
    def _apply_anchor_gate(h: torch.Tensor, anch) -> torch.Tensor:
        gate_map = None
        anchors = anch if isinstance(anch, (list, tuple)) else [anch]
        for anchor in anchors[:2]:
            resized = F.interpolate(anchor, size=h.shape[-2:], mode="bilinear", align_corners=False)
            mean = resized.mean(dim=1, keepdim=True)
            gate_map = mean if gate_map is None else gate_map + mean
        if gate_map is None:
            return h
        return h * (1.0 + torch.sigmoid(gate_map))

    def forward(self, conditions: torch.Tensor, noisy: torch.Tensor, timesteps: torch.Tensor):
        if conditions.ndim != 5 or noisy.ndim != 5:
            raise ValueError("conditions and noisy must be [B,F,C,H,W]")
        batch, frames, _, height, width = noisy.shape
        flat_t = self._flat_timesteps(timesteps, batch, frames)
        flat_sample = torch.cat([conditions, noisy], dim=2).reshape(
            batch * frames,
            conditions.shape[2] + noisy.shape[2],
            height,
            width,
        )

        with torch.no_grad():
            eps_base, cal_base = self.base_model(flat_sample, flat_t)

        flat_cond = flat_sample[:, :-self.out_channels]
        with torch.no_grad():
            raw_anch, raw_cal = self.unet.highway_forward(flat_cond)
        anch, anchor_deltas = self._adapt_anchor(raw_anch, batch, frames)
        cal_seq = self._to_sequence(raw_cal, batch, frames)
        cal_delta = cal_seq.new_tensor(0.0)
        if self.cal_adapter is not None:
            cal_seq, cal_delta = self.cal_adapter(cal_seq)

        hs = []
        emb = self.unet.time_embed(timestep_embedding(flat_t, self.unet.model_channels).to(self.unet.dtype))
        h = flat_sample.type(self.unet.dtype)
        for ind, module in enumerate(self.unet.input_blocks):
            if len(emb.size()) > 2:
                emb = emb.squeeze()
            h = module(h, emb)
            if ind == 0:
                h = self._apply_anchor_gate(h, anch)
            hs.append(h)
        h = self.unet.middle_block(h, emb)
        for module in self.unet.output_blocks:
            h = torch.cat([h, hs.pop()], dim=1)
            h = module(h, emb)
        h = h.type(flat_sample.dtype)
        eps = self.unet.out(h).reshape(batch, frames, self.out_channels, height, width)
        eps_base = eps_base.reshape(batch, frames, *eps_base.shape[1:]).contiguous()
        cal_base = cal_base.reshape(batch, frames, *cal_base.shape[1:]).contiguous()
        anchor_delta = torch.stack(anchor_deltas).mean() if anchor_deltas else eps.new_tensor(0.0)
        gates = [torch.tanh(adapter.gate).detach() for adapter in self.anchor_adapters]
        gate_mean = torch.stack(gates).mean() if gates else eps.new_tensor(0.0)
        return eps, cal_seq, {
            "eps_base": eps_base,
            "cal_base": cal_base,
            "anchor_delta": anchor_delta,
            "cal_delta": cal_delta,
            "gate_mean": gate_mean,
        }
