"""Temporal PINN-aware wrapper for RMDM.

This module leaves the original RMDM UNet implementation untouched. It wraps the
single-frame model, computes the highway/PINN prior for all frames in a clip,
passes that prior through zero-initialized temporal modules, and optionally
injects the temporally mixed prior back into the diffusion trunk.
"""

from __future__ import annotations

from typing import Any

import torch
import torch.nn as nn
import torch.nn.functional as F

from temporal_blocks import TemporalSelfAttention
from unet import timestep_embedding


class TemporalPINNUNet(nn.Module):
    """Video-capable RMDM wrapper with temporal PINN prior.

    ``prior_injection_mode``:
    - ``none``: keep the original frame-by-frame highway anchor gate, but do
      not inject temporally mixed highway features into the diffusion trunk.
    - ``uemb``: inject temporally mixed highway features with the same gate
      mechanism used by ``UNetModel_newpreview``.
    """

    def __init__(
        self,
        base_model: nn.Module,
        input_channels: int = 4,
        out_channels: int = 1,
        prior_injection_mode: str = "none",
        temporal_num_heads: int = 4,
        use_frame_positional_encoding: bool = True,
        max_frames: int = 128,
    ):
        super().__init__()
        if prior_injection_mode not in {"none", "uemb"}:
            raise ValueError("prior_injection_mode must be 'none' or 'uemb'")
        self.base_model = base_model
        self.input_channels = int(input_channels)
        self.out_channels = int(out_channels)
        self.prior_injection_mode = prior_injection_mode
        self.unet = getattr(base_model, "unet", base_model)
        self.use_frame_positional_encoding = bool(use_frame_positional_encoding)
        self.max_frames = int(max_frames)

        middle_channels = int(getattr(self.unet.middle_block[0], "channels", self.unet.model_channels))
        self.middle_temporal_attention = TemporalSelfAttention(
            middle_channels,
            num_heads=temporal_num_heads,
            use_frame_positional_encoding=self.use_frame_positional_encoding,
            max_frames=self.max_frames,
        )
        self.cal_temporal_attention = TemporalSelfAttention(
            out_channels,
            num_heads=1,
            use_frame_positional_encoding=self.use_frame_positional_encoding,
            max_frames=self.max_frames,
        )

        anch_channels = self._infer_anchor_channels()
        self.uemb_temporal_attention = nn.ModuleList(
            [
                TemporalSelfAttention(
                    channels,
                    num_heads=temporal_num_heads,
                    use_frame_positional_encoding=self.use_frame_positional_encoding,
                    max_frames=self.max_frames,
                )
                for channels in anch_channels
            ]
        )

    def _infer_anchor_channels(self) -> list[int]:
        if not hasattr(self.unet, "hwm"):
            return []
        device = next(self.unet.parameters()).device
        dtype = next(self.unet.parameters()).dtype
        cond_channels = max(1, self.input_channels - self.out_channels)
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
        if isinstance(anch, (list, tuple)):
            return [int(value.shape[1]) for value in anch]
        return [int(anch.shape[1])]

    def set_prior_injection_mode(self, mode: str) -> None:
        if mode not in {"none", "uemb"}:
            raise ValueError("prior_injection_mode must be 'none' or 'uemb'")
        self.prior_injection_mode = mode

    def load_2d_state_dict(self, state_dict: dict[str, Any], strict: bool = True) -> None:
        self.base_model.load_state_dict(state_dict, strict=strict)

    def freeze_base_model(self) -> None:
        for param in self.base_model.parameters():
            param.requires_grad = False

    def _flat_timesteps(self, timesteps: torch.Tensor, batch: int, frames: int) -> torch.Tensor:
        if timesteps.ndim == 0:
            return timesteps.reshape(1).repeat(batch * frames)
        if timesteps.ndim == 1 and timesteps.shape[0] == batch:
            return timesteps[:, None].repeat(1, frames).reshape(batch * frames)
        if timesteps.ndim == 2 and timesteps.shape[:2] == (batch, frames):
            return timesteps.reshape(batch * frames)
        if timesteps.ndim == 1 and timesteps.shape[0] == batch * frames:
            return timesteps
        raise ValueError(f"Unexpected timestep shape {tuple(timesteps.shape)} for B={batch}, F={frames}")

    def _reshape_prior(self, value: torch.Tensor, batch: int, frames: int) -> torch.Tensor:
        return value.reshape(batch, frames, *value.shape[1:]).contiguous()

    def _temporalize_anchor(self, anch: torch.Tensor | list[torch.Tensor] | tuple[torch.Tensor], batch: int, frames: int):
        if isinstance(anch, (list, tuple)):
            out = []
            for idx, value in enumerate(anch):
                seq = self._reshape_prior(value, batch, frames)
                if idx < len(self.uemb_temporal_attention):
                    seq = self.uemb_temporal_attention[idx](seq)
                out.append(seq.reshape(batch * frames, *seq.shape[2:]))
            return type(anch)(out)
        seq = self._reshape_prior(anch, batch, frames)
        if self.uemb_temporal_attention:
            seq = self.uemb_temporal_attention[0](seq)
        return seq.reshape(batch * frames, *seq.shape[2:])

    def _compute_temporal_prior(self, flat_cond: torch.Tensor, batch: int, frames: int, temporalize_anchor: bool = True):
        anch, cal = self.unet.highway_forward(flat_cond)
        cal_seq = self._reshape_prior(cal, batch, frames)
        cal_seq = self.cal_temporal_attention(cal_seq)
        temporal_anch = self._temporalize_anchor(anch, batch, frames) if temporalize_anchor else None
        return anch, temporal_anch, cal_seq

    def forward_prior_only(self, conditions: torch.Tensor) -> torch.Tensor:
        """Compute only the temporal PINN ``cal`` prior from ``[B,F,C,H,W]`` conditions."""
        if conditions.ndim != 5:
            raise ValueError(f"forward_prior_only expects [B,F,C,H,W], got {tuple(conditions.shape)}")
        batch, frames, channels, height, width = conditions.shape
        flat_cond = conditions.reshape(batch * frames, channels, height, width)
        _, _, cal_seq = self._compute_temporal_prior(flat_cond, batch, frames, temporalize_anchor=False)
        return cal_seq

    def _apply_anchor_gate(self, h: torch.Tensor, anch, detach_gate: bool) -> torch.Tensor:
        gate_map = None
        anchors = anch if isinstance(anch, (list, tuple)) else [anch]
        for anchor in anchors[:2]:
            anchor = F.interpolate(anchor, size=h.shape[-2:], mode="bilinear", align_corners=False)
            anchor = anchor.mean(dim=1, keepdim=True)
            gate_map = anchor if gate_map is None else gate_map + anchor
        if gate_map is None:
            return h
        gate = torch.sigmoid(gate_map)
        if detach_gate:
            gate = gate.detach()
        return h * (1 + gate)

    def forward(self, sample: torch.Tensor, timesteps: torch.Tensor):
        if sample.ndim == 4:
            return self.base_model(sample, timesteps)
        if sample.ndim != 5:
            raise ValueError(f"Expected [B,C,H,W] or [B,F,C,H,W], got {tuple(sample.shape)}")

        batch, frames, channels, height, width = sample.shape
        flat_sample = sample.reshape(batch * frames, channels, height, width)
        flat_t = self._flat_timesteps(timesteps, batch, frames)

        flat_cond = flat_sample[:, :-self.out_channels, ...]
        raw_anch, temporal_anch, cal_seq = self._compute_temporal_prior(flat_cond, batch, frames)

        hs = []
        emb = self.unet.time_embed(timestep_embedding(flat_t, self.unet.model_channels).to(self.unet.dtype))
        h = flat_sample.type(self.unet.dtype)

        for ind, module in enumerate(self.unet.input_blocks):
            if len(emb.size()) > 2:
                emb = emb.squeeze()
            h = module(h, emb)
            if ind == 0:
                if self.prior_injection_mode == "uemb":
                    h = self._apply_anchor_gate(h, temporal_anch, detach_gate=False)
                else:
                    h = self._apply_anchor_gate(h, raw_anch, detach_gate=True)
            hs.append(h)

        h = self.unet.middle_block(h, emb)
        h_seq = h.reshape(batch, frames, *h.shape[1:])
        h = self.middle_temporal_attention(h_seq).reshape_as(h)

        for module in self.unet.output_blocks:
            h = torch.cat([h, hs.pop()], dim=1)
            h = module(h, emb)
        h = h.type(flat_sample.dtype)
        out = self.unet.out(h).reshape(batch, frames, self.out_channels, height, width)
        return out, cal_seq


def temporal_pinn_state_dict_from_2d(state_dict: dict[str, Any]) -> dict[str, Any]:
    """Prefix a 2D RMDM state dict for loading into ``TemporalPINNUNet``."""
    return {f"base_model.{key}": value for key, value in state_dict.items()}
