"""Temporal adapters inside the frozen RMDM HWM/PINN decoder.

This module keeps the original RMDM ``unet.py`` untouched. It manually forwards
the frozen ``base.unet.hwm`` branch and inserts zero-init separable 3D residual
adapters after selected decoder/localization blocks near the prior output.
"""

from __future__ import annotations

import torch
from torch import nn
import torch.nn.functional as F


def _group_count(channels: int, max_groups: int = 32) -> int:
    for groups in range(min(max_groups, channels), 0, -1):
        if channels % groups == 0:
            return groups
    return 1


class NoGateSeparable3DAdapter(nn.Module):
    """Residual adapter ``x + A(x)`` with zero-init output projection."""

    def __init__(self, channels: int, reduction: int = 4, min_hidden: int = 8):
        super().__init__()
        channels = int(channels)
        hidden = max(int(min_hidden), channels // max(int(reduction), 1))
        self.norm = nn.GroupNorm(_group_count(channels), channels)
        self.down = nn.Conv3d(channels, hidden, kernel_size=1)
        self.temporal = nn.Conv3d(hidden, hidden, kernel_size=(3, 1, 1), padding=(1, 0, 0))
        self.spatial = nn.Conv3d(hidden, hidden, kernel_size=(1, 3, 3), padding=(0, 1, 1))
        self.up = nn.Conv3d(hidden, channels, kernel_size=1)
        nn.init.zeros_(self.up.weight)
        nn.init.zeros_(self.up.bias)

    def forward(self, x: torch.Tensor, batch: int, frames: int) -> tuple[torch.Tensor, torch.Tensor]:
        if x.ndim != 4:
            raise ValueError(f"expected [B*F,C,H,W], got {tuple(x.shape)}")
        _, channels, height, width = x.shape
        seq = x.reshape(batch, frames, channels, height, width).permute(0, 2, 1, 3, 4).contiguous()
        y = self.down(self.norm(seq))
        y = F.silu(self.temporal(F.silu(y)))
        y = F.silu(self.spatial(y))
        delta = self.up(y)
        out = seq + delta
        out = out.permute(0, 2, 1, 3, 4).reshape(batch * frames, channels, height, width).contiguous()
        return out, delta.detach().abs().mean()


class HWMDecoderAdapterStudent(nn.Module):
    """Frozen RMDM with trainable no-gate temporal adapters in HWM decoder."""

    def __init__(
        self,
        base_model: nn.Module,
        adapter_indices: tuple[int, ...] = (-2, -1),
        reduction: int = 4,
        min_hidden: int = 8,
    ):
        super().__init__()
        self.base_model = base_model
        self.unet = getattr(base_model, "unet", base_model)
        self.hwm = self.unet.hwm
        num_blocks = len(self.hwm.conv_blocks_localization)
        resolved = []
        for idx in adapter_indices:
            real = idx if idx >= 0 else num_blocks + idx
            if real < 0 or real >= num_blocks:
                raise ValueError(f"adapter index {idx} resolves to {real}, outside {num_blocks} decoder blocks")
            resolved.append(real)
        self.adapter_indices = tuple(sorted(set(resolved)))
        self.adapters = nn.ModuleDict()
        for idx in self.adapter_indices:
            channels = int(self.hwm.conv_blocks_localization[idx][-1].output_channels)
            self.adapters[str(idx)] = NoGateSeparable3DAdapter(channels, reduction=reduction, min_hidden=min_hidden)

        for param in self.base_model.parameters():
            param.requires_grad = False
        self.base_model.eval()

    def train(self, mode: bool = True):
        super().train(mode)
        self.base_model.eval()
        return self

    def adapter_parameters(self):
        yield from self.adapters.parameters()

    @staticmethod
    def _flatten_conditions(conditions: torch.Tensor) -> tuple[torch.Tensor, int, int]:
        if conditions.ndim != 5:
            raise ValueError(f"expected conditions [B,F,C,H,W], got {tuple(conditions.shape)}")
        batch, frames, channels, height, width = conditions.shape
        return conditions.reshape(batch * frames, channels, height, width), batch, frames

    def forward_hwm(self, conditions: torch.Tensor):
        """Return adapted ``anchors, cal_seq, stats`` for ``conditions [B,F,C,H,W]``."""
        x, batch, frames = self._flatten_conditions(conditions)
        hwm = self.hwm
        skips = []
        seg_outputs = []
        anch_outputs = []
        stats = {}

        adapter_has_run = False
        with torch.no_grad():
            for d in range(len(hwm.conv_blocks_context) - 1):
                x = hwm.conv_blocks_context[d](x)
                skips.append(x)
                if not hwm.convolutional_pooling:
                    x = hwm.td[d](x)
            x = hwm.conv_blocks_context[-1](x)

        for u in range(len(hwm.tu)):
            grad_ctx = torch.enable_grad() if adapter_has_run else torch.no_grad()
            with grad_ctx:
                x = hwm.tu[u](x)
                skip = skips[-(u + 1)]
                x = torch.cat((x, skip), dim=1)
                x = hwm.conv_blocks_localization[u](x)
            if u in self.adapter_indices:
                x, delta = self.adapters[str(u)](x, batch, frames)
                adapter_has_run = True
                stats[f"adapter_delta_{u}"] = delta
            if hwm._deep_supervision:
                seg_outputs.append(hwm.final_nonlin(hwm.seg_outputs[u](x)))
            if hwm.anchor_out and (not hwm._deep_supervision):
                anch_outputs.append(x)

        if not seg_outputs:
            seg_outputs.append(hwm.final_nonlin(hwm.seg_outputs[0](x)))
        cal = seg_outputs[-1].reshape(batch, frames, *seg_outputs[-1].shape[1:]).contiguous()

        if hwm.anchor_out:
            anchors = tuple(
                upsample(anchor)
                for upsample, anchor in zip(list(hwm.upscale_logits_ops)[::-1], anch_outputs[:-1][::-1])
            )
        else:
            anchors = tuple()
        anchor_seq = tuple(anchor.reshape(batch, frames, *anchor.shape[1:]).contiguous() for anchor in anchors)
        if stats:
            stats["adapter_delta_mean"] = torch.stack(list(stats.values())).mean()
        else:
            stats["adapter_delta_mean"] = cal.new_tensor(0.0)
        return anchor_seq, cal, stats

    def forward_hwm_baseline(self, conditions: torch.Tensor):
        """Return frozen baseline HWM ``anchors, cal_seq, stats`` without adapters."""
        x, batch, frames = self._flatten_conditions(conditions)
        with torch.no_grad():
            anchors, cal = self.hwm(x, hs=None)
        if isinstance(anchors, torch.Tensor):
            anchor_seq = (anchors.reshape(batch, frames, *anchors.shape[1:]).contiguous(),)
        else:
            anchor_seq = tuple(anchor.reshape(batch, frames, *anchor.shape[1:]).contiguous() for anchor in anchors)
        cal_seq = cal.reshape(batch, frames, *cal.shape[1:]).contiguous()
        stats = {"adapter_delta_mean": cal_seq.new_tensor(0.0)}
        return anchor_seq, cal_seq, stats

    def forward_prior_only(self, conditions: torch.Tensor):
        _, cal, stats = self.forward_hwm(conditions)
        return cal, stats

    def forward(self, conditions: torch.Tensor):
        return self.forward_prior_only(conditions)
