"""Exact trainable RMDM HWM/cal branch, initialized from scratch."""

from __future__ import annotations

from typing import Any

import torch
import torch.nn.functional as F
from torch import nn


class TrainableHWM(nn.Module):
    """Run the legacy HWM framewise and expose only its exact gate and ``cal``."""

    def __init__(self, hwm: nn.Module, *, chunk_size: int, input_channels: int = 5) -> None:
        super().__init__()
        if chunk_size <= 0:
            raise ValueError("HWM chunk_size must be positive")
        if input_channels not in (3, 5):
            raise ValueError("HWM input_channels must be 3 (scene) or 5 (sparse)")
        self.hwm = hwm
        self.chunk_size = int(chunk_size)
        self.input_channels = int(input_channels)
        for parameter in self.hwm.parameters():
            parameter.requires_grad_(True)

    def conditions(self, batch: dict[str, Any]) -> torch.Tensor:
        required = ["building", "tx", "vehicle"]
        if self.input_channels == 5:
            required.extend(("observed_rss", "sampling_mask"))
        missing = [name for name in required if name not in batch]
        if missing:
            raise KeyError(f"sparse batch misses HWM inputs: {missing}")
        reference = batch["building"]
        for name in required:
            if batch[name].shape != reference.shape:
                raise ValueError(f"HWM input {name!r} shape differs from building")
        values = [
            batch["building"] + 10.0 * batch["tx"],
            batch["tx"],
            batch["vehicle"],
        ]
        if self.input_channels == 5:
            values.extend((batch["observed_rss"], batch["sampling_mask"]))
        return torch.cat(values, dim=2)

    def forward(self, batch: dict[str, Any]) -> dict[str, torch.Tensor]:
        conditions = self.conditions(batch)
        if conditions.ndim != 5 or conditions.shape[2] != self.input_channels:
            raise ValueError(
                f"HWM conditions must be [B,T,{self.input_channels},H,W]"
            )
        batch_size, time, _, height, width = conditions.shape
        flat = conditions.reshape(
            batch_size * time, self.input_channels, height, width
        )
        gates: list[torch.Tensor] = []
        calibrations: list[torch.Tensor] = []
        for start in range(0, flat.shape[0], self.chunk_size):
            result = self.hwm(flat[start : start + self.chunk_size])
            if not isinstance(result, (tuple, list)) or len(result) < 2:
                raise RuntimeError("legacy HWM must return (anchors, cal)")
            anchors, cal = result[0], result[1]
            if not isinstance(anchors, (tuple, list)) or len(anchors) < 2:
                raise RuntimeError("legacy HWM must expose at least its first two decoder anchors")
            if anchors[0].shape[1] != 32 or anchors[1].shape[1] != 64:
                raise RuntimeError("legacy HWM A1/A2 channel contract drifted")
            gate_map: torch.Tensor | None = None
            for anchor in anchors[:2]:
                resized = F.interpolate(anchor, size=(height, width), mode="bilinear", align_corners=False)
                mean = resized.mean(dim=1, keepdim=True)
                gate_map = mean if gate_map is None else gate_map + mean
            if gate_map is None:
                raise RuntimeError("legacy HWM produced no usable anchor")
            # This detach is intentional and reproduces UNetModel_newpreview:
            # diffusion gradients cannot train HWM through the spatial gate.
            gates.append(torch.sigmoid(gate_map).detach())
            calibrations.append(cal)
        return {
            "hwm_gate": torch.cat(gates, dim=0).reshape(batch_size, time, 1, height, width).contiguous(),
            "cal": torch.cat(calibrations, dim=0).reshape(batch_size, time, 1, height, width).contiguous(),
        }


def build_hwm_from_scratch(*, base_features: int, input_channels: int = 5) -> nn.Module:
    if base_features != 32:
        raise ValueError("the exact RMDM HWM contract requires 32 base features")
    from unet import Generic_UNet

    if input_channels not in (3, 5):
        raise ValueError("HWM input_channels must be 3 (scene) or 5 (sparse)")
    return Generic_UNet(
        input_channels,
        base_features,
        1,
        5,
        anchor_out=True,
        upscale_logits=True,
    )
