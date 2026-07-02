"""Dual decoder temporal adapter variants for RMDM.

This file intentionally keeps the original RMDM UNet untouched. It mirrors the
current dual decoder adapter wrapper, but swaps the adapter block so architecture
probes can be trained/evaluated without overloading the baseline implementation.
"""

from __future__ import annotations

import torch
from torch import nn
import torch.nn.functional as F

from hwm_decoder_adapter_student import _group_count
from utils.nn import timestep_embedding


class MultiScaleTemporalAdapter(nn.Module):
    """Residual multi-scale temporal adapter with zero-init output projection."""

    def __init__(self, channels: int, reduction: int = 1, min_hidden: int = 16):
        super().__init__()
        channels = int(channels)
        hidden = max(int(min_hidden), channels // max(int(reduction), 1))
        self.norm = nn.GroupNorm(_group_count(channels), channels)
        self.down = nn.Conv3d(channels, hidden, kernel_size=1)
        self.temporal3 = nn.Conv3d(hidden, hidden, kernel_size=(3, 1, 1), padding=(1, 0, 0))
        self.temporal5 = nn.Conv3d(hidden, hidden, kernel_size=(5, 1, 1), padding=(2, 0, 0))
        self.temporal_dilated = nn.Conv3d(
            hidden,
            hidden,
            kernel_size=(3, 1, 1),
            padding=(2, 0, 0),
            dilation=(2, 1, 1),
        )
        self.fuse = nn.Conv3d(hidden * 3, hidden, kernel_size=1)
        self.spatial = nn.Conv3d(hidden, hidden, kernel_size=(1, 3, 3), padding=(0, 1, 1))
        self.up = nn.Conv3d(hidden, channels, kernel_size=1)
        nn.init.zeros_(self.up.weight)
        nn.init.zeros_(self.up.bias)

    def forward(
        self,
        x: torch.Tensor,
        batch: int,
        frames: int,
        k2: torch.Tensor | None = None,
        timesteps: torch.Tensor | None = None,
    ) -> tuple[torch.Tensor, torch.Tensor]:
        if x.ndim != 4:
            raise ValueError(f"expected [B*F,C,H,W], got {tuple(x.shape)}")
        _, channels, height, width = x.shape
        seq = x.reshape(batch, frames, channels, height, width).permute(0, 2, 1, 3, 4).contiguous()
        y = F.silu(self.down(self.norm(seq)))
        y3 = F.silu(self.temporal3(y))
        y5 = F.silu(self.temporal5(y))
        yd = F.silu(self.temporal_dilated(y))
        y = F.silu(self.fuse(torch.cat([y3, y5, yd], dim=1)))
        y = F.silu(self.spatial(y))
        delta = self.up(y)
        out = seq + delta
        out = out.permute(0, 2, 1, 3, 4).reshape(batch * frames, channels, height, width).contiguous()
        return out, delta.detach().abs().mean()


class KeyPathGatedTemporalAdapter(nn.Module):
    """Residual adapter whose correction is spatially gated by high-K2 regions."""

    def __init__(
        self,
        channels: int,
        reduction: int = 1,
        min_hidden: int = 16,
        k2_quantile: float = 0.70,
        gate_floor: float = 0.05,
        gate_dilate: int = 2,
    ):
        super().__init__()
        channels = int(channels)
        hidden = max(int(min_hidden), channels // max(int(reduction), 1))
        self.k2_quantile = float(k2_quantile)
        self.gate_floor = float(gate_floor)
        self.gate_dilate = int(gate_dilate)
        self.norm = nn.GroupNorm(_group_count(channels), channels)
        self.down = nn.Conv3d(channels, hidden, kernel_size=1)
        self.temporal = nn.Conv3d(hidden, hidden, kernel_size=(3, 1, 1), padding=(1, 0, 0))
        self.spatial = nn.Conv3d(hidden, hidden, kernel_size=(1, 3, 3), padding=(0, 1, 1))
        self.up = nn.Conv3d(hidden, channels, kernel_size=1)
        nn.init.zeros_(self.up.weight)
        nn.init.zeros_(self.up.bias)

    def _key_gate(self, k2: torch.Tensor, height: int, width: int, dtype: torch.dtype) -> torch.Tensor:
        if k2 is None:
            raise ValueError("KeyPathGatedTemporalAdapter requires k2 [B,F,1,H,W]")
        if k2.ndim != 5:
            raise ValueError(f"k2 must be [B,F,1,H,W], got {tuple(k2.shape)}")
        gate = F.interpolate(
            k2.reshape(k2.shape[0] * k2.shape[1], 1, *k2.shape[-2:]).to(dtype=dtype),
            size=(height, width),
            mode="bilinear",
            align_corners=False,
        ).reshape(k2.shape[0], k2.shape[1], 1, height, width)
        flat = gate.flatten(2)
        thresh = torch.quantile(flat, self.k2_quantile, dim=2, keepdim=True).view(k2.shape[0], k2.shape[1], 1, 1, 1)
        gate = (gate >= thresh).to(dtype=dtype)
        if self.gate_dilate > 0:
            radius = self.gate_dilate
            gate = F.max_pool2d(
                gate.reshape(k2.shape[0] * k2.shape[1], 1, height, width),
                kernel_size=2 * radius + 1,
                stride=1,
                padding=radius,
            ).reshape(k2.shape[0], k2.shape[1], 1, height, width)
        gate = gate.permute(0, 2, 1, 3, 4).contiguous()
        if self.gate_floor > 0:
            gate = gate * (1.0 - self.gate_floor) + self.gate_floor
        return gate

    def forward(
        self,
        x: torch.Tensor,
        batch: int,
        frames: int,
        k2: torch.Tensor | None = None,
        timesteps: torch.Tensor | None = None,
    ) -> tuple[torch.Tensor, torch.Tensor]:
        if x.ndim != 4:
            raise ValueError(f"expected [B*F,C,H,W], got {tuple(x.shape)}")
        _, channels, height, width = x.shape
        seq = x.reshape(batch, frames, channels, height, width).permute(0, 2, 1, 3, 4).contiguous()
        y = self.down(self.norm(seq))
        y = F.silu(self.temporal(F.silu(y)))
        y = F.silu(self.spatial(y))
        delta = self.up(y) * self._key_gate(k2, height, width, seq.dtype)
        out = seq + delta
        out = out.permute(0, 2, 1, 3, 4).reshape(batch * frames, channels, height, width).contiguous()
        return out, delta.detach().abs().mean()


class KeyPathLocalTemporalAttentionAdapter(nn.Module):
    """K2-guided local temporal attention adapter.

    Attention is local over a small temporal neighborhood at each spatial
    location. It avoids global frame mixing while letting key-path pixels choose
    useful adjacent-frame features.
    """

    def __init__(
        self,
        channels: int,
        reduction: int = 1,
        min_hidden: int = 16,
        k2_quantile: float = 0.70,
        gate_floor: float = 0.05,
        gate_dilate: int = 2,
        radius: int = 2,
    ):
        super().__init__()
        channels = int(channels)
        hidden = max(int(min_hidden), channels // max(int(reduction), 1))
        self.k2_quantile = float(k2_quantile)
        self.gate_floor = float(gate_floor)
        self.gate_dilate = int(gate_dilate)
        self.radius = int(radius)
        self.norm = nn.GroupNorm(_group_count(channels), channels)
        self.down = nn.Conv3d(channels, hidden, kernel_size=1)
        self.q = nn.Conv3d(hidden, hidden, kernel_size=1)
        self.k = nn.Conv3d(hidden, hidden, kernel_size=1)
        self.v = nn.Conv3d(hidden, hidden, kernel_size=1)
        self.attn_out = nn.Conv3d(hidden, hidden, kernel_size=1)
        self.spatial = nn.Conv3d(hidden, hidden, kernel_size=(1, 3, 3), padding=(0, 1, 1))
        self.up = nn.Conv3d(hidden, channels, kernel_size=1)
        nn.init.zeros_(self.up.weight)
        nn.init.zeros_(self.up.bias)

    def _key_gate(self, k2: torch.Tensor, height: int, width: int, dtype: torch.dtype) -> torch.Tensor:
        if k2 is None:
            raise ValueError("KeyPathLocalTemporalAttentionAdapter requires k2 [B,F,1,H,W]")
        gate = F.interpolate(
            k2.reshape(k2.shape[0] * k2.shape[1], 1, *k2.shape[-2:]).to(dtype=dtype),
            size=(height, width),
            mode="bilinear",
            align_corners=False,
        ).reshape(k2.shape[0], k2.shape[1], 1, height, width)
        flat = gate.flatten(2)
        thresh = torch.quantile(flat, self.k2_quantile, dim=2, keepdim=True).view(k2.shape[0], k2.shape[1], 1, 1, 1)
        gate = (gate >= thresh).to(dtype=dtype)
        if self.gate_dilate > 0:
            radius = self.gate_dilate
            gate = F.max_pool2d(
                gate.reshape(k2.shape[0] * k2.shape[1], 1, height, width),
                kernel_size=2 * radius + 1,
                stride=1,
                padding=radius,
            ).reshape(k2.shape[0], k2.shape[1], 1, height, width)
        gate = gate.permute(0, 2, 1, 3, 4).contiguous()
        if self.gate_floor > 0:
            gate = gate * (1.0 - self.gate_floor) + self.gate_floor
        return gate

    @staticmethod
    def _shift_time(x: torch.Tensor, offset: int) -> torch.Tensor:
        if offset == 0:
            return x
        out = torch.empty_like(x)
        if offset > 0:
            out[:, :, :offset] = x[:, :, :1]
            out[:, :, offset:] = x[:, :, :-offset]
        else:
            step = -offset
            out[:, :, -step:] = x[:, :, -1:]
            out[:, :, :-step] = x[:, :, step:]
        return out

    def forward(
        self,
        x: torch.Tensor,
        batch: int,
        frames: int,
        k2: torch.Tensor | None = None,
        timesteps: torch.Tensor | None = None,
    ) -> tuple[torch.Tensor, torch.Tensor]:
        if x.ndim != 4:
            raise ValueError(f"expected [B*F,C,H,W], got {tuple(x.shape)}")
        _, channels, height, width = x.shape
        seq = x.reshape(batch, frames, channels, height, width).permute(0, 2, 1, 3, 4).contiguous()
        gate = self._key_gate(k2, height, width, seq.dtype)
        y = F.silu(self.down(self.norm(seq) * gate))
        q = self.q(y)
        k = self.k(y)
        v = self.v(y)
        offsets = list(range(-self.radius, self.radius + 1))
        scores = []
        values = []
        scale = float(q.shape[1]) ** -0.5
        for offset in offsets:
            kk = self._shift_time(k, offset)
            vv = self._shift_time(v, offset)
            scores.append((q * kk).sum(dim=1, keepdim=True) * scale)
            values.append(vv)
        attn = torch.softmax(torch.cat(scores, dim=1), dim=1)
        mixed = sum(attn[:, i : i + 1] * values[i] for i in range(len(values)))
        y = F.silu(self.attn_out(mixed))
        y = F.silu(self.spatial(y))
        delta = self.up(y) * gate
        out = seq + delta
        out = out.permute(0, 2, 1, 3, 4).reshape(batch * frames, channels, height, width).contiguous()
        return out, delta.detach().abs().mean()


class AdaptiveAlphaTemporalAdapter(nn.Module):
    """Separable temporal adapter with timestep-conditioned residual scale."""

    def __init__(
        self,
        channels: int,
        reduction: int = 1,
        min_hidden: int = 16,
        diffusion_steps: int = 1000,
    ):
        super().__init__()
        channels = int(channels)
        hidden = max(int(min_hidden), channels // max(int(reduction), 1))
        self.diffusion_steps = int(diffusion_steps)
        self.norm = nn.GroupNorm(_group_count(channels), channels)
        self.down = nn.Conv3d(channels, hidden, kernel_size=1)
        self.temporal = nn.Conv3d(hidden, hidden, kernel_size=(3, 1, 1), padding=(1, 0, 0))
        self.spatial = nn.Conv3d(hidden, hidden, kernel_size=(1, 3, 3), padding=(0, 1, 1))
        self.up = nn.Conv3d(hidden, channels, kernel_size=1)
        self.alpha = nn.Sequential(
            nn.Linear(1, 16),
            nn.SiLU(),
            nn.Linear(16, 1),
        )
        nn.init.zeros_(self.up.weight)
        nn.init.zeros_(self.up.bias)
        nn.init.zeros_(self.alpha[-1].weight)
        nn.init.zeros_(self.alpha[-1].bias)

    def _alpha_t(self, timesteps: torch.Tensor | None, batch: int, frames: int, height: int, width: int, dtype: torch.dtype, device: torch.device) -> torch.Tensor:
        if timesteps is None:
            alpha = torch.full((batch, 1), 0.5, device=device, dtype=dtype)
        else:
            t = timesteps.to(device=device, dtype=dtype).view(batch, 1) / max(float(self.diffusion_steps - 1), 1.0)
            alpha = torch.sigmoid(self.alpha(t))
        return alpha.view(batch, 1, 1, 1, 1).expand(batch, 1, frames, height, width)

    def forward(
        self,
        x: torch.Tensor,
        batch: int,
        frames: int,
        k2: torch.Tensor | None = None,
        timesteps: torch.Tensor | None = None,
    ) -> tuple[torch.Tensor, torch.Tensor]:
        if x.ndim != 4:
            raise ValueError(f"expected [B*F,C,H,W], got {tuple(x.shape)}")
        _, channels, height, width = x.shape
        seq = x.reshape(batch, frames, channels, height, width).permute(0, 2, 1, 3, 4).contiguous()
        y = self.down(self.norm(seq))
        y = F.silu(self.temporal(F.silu(y)))
        y = F.silu(self.spatial(y))
        delta = self.up(y)
        alpha = self._alpha_t(timesteps, batch, frames, height, width, seq.dtype, seq.device)
        out = seq + alpha * delta
        out = out.permute(0, 2, 1, 3, 4).reshape(batch * frames, channels, height, width).contiguous()
        return out, (alpha.detach() * delta.detach().abs()).mean()


def build_adapter(
    variant: str,
    channels: int,
    reduction: int,
    min_hidden: int,
    keypath_quantile: float = 0.70,
    keypath_gate_floor: float = 0.05,
    keypath_gate_dilate: int = 2,
) -> nn.Module:
    if variant == "multiscale":
        return MultiScaleTemporalAdapter(channels, reduction=reduction, min_hidden=min_hidden)
    if variant == "keypath":
        return KeyPathGatedTemporalAdapter(
            channels,
            reduction=reduction,
            min_hidden=min_hidden,
            k2_quantile=keypath_quantile,
            gate_floor=keypath_gate_floor,
            gate_dilate=keypath_gate_dilate,
        )
    if variant == "keypath_attn":
        return KeyPathLocalTemporalAttentionAdapter(
            channels,
            reduction=reduction,
            min_hidden=min_hidden,
            k2_quantile=keypath_quantile,
            gate_floor=keypath_gate_floor,
            gate_dilate=keypath_gate_dilate,
        )
    if variant == "adaptive_alpha":
        return AdaptiveAlphaTemporalAdapter(channels, reduction=reduction, min_hidden=min_hidden)
    raise ValueError(f"unknown adapter variant: {variant}")


class HWMDecoderAdapterVariant(nn.Module):
    """HWM decoder adapter wrapper for architecture variants."""

    def __init__(
        self,
        base_model: nn.Module,
        variant: str,
        adapter_indices: tuple[int, ...] = (-2, -1),
        reduction: int = 1,
        min_hidden: int = 16,
        keypath_quantile: float = 0.70,
        keypath_gate_floor: float = 0.05,
        keypath_gate_dilate: int = 2,
    ):
        super().__init__()
        self.base_model = base_model
        self.unet = getattr(base_model, "unet", base_model)
        self.hwm = self.unet.hwm
        self.variant = str(variant)
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
            self.adapters[str(idx)] = build_adapter(
                self.variant,
                channels,
                reduction,
                min_hidden,
                keypath_quantile,
                keypath_gate_floor,
                keypath_gate_dilate,
            )
        for param in self.base_model.parameters():
            param.requires_grad = False
        self.base_model.eval()

    def forward_hwm(
        self,
        conditions: torch.Tensor,
        k2: torch.Tensor | None = None,
        timesteps: torch.Tensor | None = None,
    ):
        if conditions.ndim != 5:
            raise ValueError(f"expected conditions [B,F,C,H,W], got {tuple(conditions.shape)}")
        batch, frames, channels, height, width = conditions.shape
        x = conditions.reshape(batch * frames, channels, height, width)
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
                x, delta = self.adapters[str(u)](x, batch, frames, k2=k2, timesteps=timesteps)
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
        stats["adapter_delta_mean"] = torch.stack(list(stats.values())).mean() if stats else cal.new_tensor(0.0)
        return anchor_seq, cal, stats


class DualDecoderAdapterVariantStudent(nn.Module):
    """RMDM with variant temporal adapters in HWM and diffusion decoders."""

    def __init__(
        self,
        base_model: nn.Module,
        variant: str,
        hwm_adapter_indices: tuple[int, ...] = (-2, -1),
        stage2_adapter_indices: tuple[int, ...] = (-2, -1),
        adapter_reduction: int = 1,
        adapter_min_hidden: int = 16,
        detach_anchor_gate: bool = False,
        train_base: bool = False,
        keypath_quantile: float = 0.70,
        keypath_gate_floor: float = 0.05,
        keypath_gate_dilate: int = 2,
    ):
        super().__init__()
        self.base_model = base_model
        self.unet = getattr(base_model, "unet", base_model)
        self.variant = str(variant)
        self.hwm_adapter = HWMDecoderAdapterVariant(
            base_model,
            variant=self.variant,
            adapter_indices=hwm_adapter_indices,
            reduction=adapter_reduction,
            min_hidden=adapter_min_hidden,
            keypath_quantile=keypath_quantile,
            keypath_gate_floor=keypath_gate_floor,
            keypath_gate_dilate=keypath_gate_dilate,
        )
        num_blocks = len(self.unet.output_blocks)
        resolved = []
        for idx in stage2_adapter_indices:
            real = idx if idx >= 0 else num_blocks + idx
            if real < 0 or real >= num_blocks:
                raise ValueError(f"stage2 adapter index {idx} resolves to {real}, outside {num_blocks}")
            resolved.append(real)
        self.stage2_adapter_indices = tuple(sorted(set(resolved)))
        self.stage2_adapters = nn.ModuleDict()
        for idx in self.stage2_adapter_indices:
            channels = self._output_block_channels(self.unet.output_blocks[idx])
            self.stage2_adapters[str(idx)] = build_adapter(
                self.variant,
                channels,
                adapter_reduction,
                adapter_min_hidden,
                keypath_quantile,
                keypath_gate_floor,
                keypath_gate_dilate,
            )
        self.detach_anchor_gate = bool(detach_anchor_gate)
        self.train_base = bool(train_base)
        for param in self.base_model.parameters():
            param.requires_grad = self.train_base
        for param in self.hwm_adapter.adapters.parameters():
            param.requires_grad = True
        for param in self.stage2_adapters.parameters():
            param.requires_grad = True
        if not self.train_base:
            self.base_model.eval()

    @staticmethod
    def _output_block_channels(block: nn.Module) -> int:
        for module in reversed(block):
            if hasattr(module, "out_channels"):
                return int(module.out_channels)
            if hasattr(module, "channels"):
                return int(module.channels)
        raise ValueError(f"cannot infer output channels for {block}")

    def train(self, mode: bool = True):
        super().train(mode)
        if not self.train_base:
            self.base_model.eval()
        return self

    def trainable_parameters(self):
        yield from self.hwm_adapter.adapters.parameters()
        yield from self.stage2_adapters.parameters()

    def forward(
        self,
        conditions: torch.Tensor,
        noisy: torch.Tensor,
        timesteps: torch.Tensor,
        k2: torch.Tensor | None = None,
    ):
        if conditions.ndim != 5 or noisy.ndim != 5:
            raise ValueError("conditions and noisy must be [B,F,C,H,W]")
        batch, frames, _, height, width = conditions.shape
        model = self.unet
        flat_t = timesteps[:, None].repeat(1, frames).reshape(batch * frames)
        flat_input = torch.cat([conditions, noisy], dim=2).reshape(batch * frames, conditions.shape[2] + noisy.shape[2], height, width)

        anch_seq, cal, hwm_stats = self.hwm_adapter.forward_hwm(conditions, k2=k2, timesteps=timesteps)
        anch = tuple(anchor.reshape(batch * frames, *anchor.shape[2:]).contiguous() for anchor in anch_seq)

        hs = []
        emb = model.time_embed(timestep_embedding(flat_t, model.model_channels))
        h = flat_input.type(model.dtype)
        for ind, module in enumerate(model.input_blocks):
            if len(emb.size()) > 2:
                emb = emb.squeeze()
            if ind == 0:
                h = module(h, emb)
                gate_map = None
                if isinstance(anch, (list, tuple)) and len(anch) > 0:
                    for anchor in anch[:2]:
                        resized = F.interpolate(anchor, size=h.shape[-2:], mode="bilinear", align_corners=False)
                        mean = resized.mean(dim=1, keepdim=True)
                        gate_map = mean if gate_map is None else gate_map + mean
                if gate_map is not None:
                    gate = torch.sigmoid(gate_map)
                    if self.detach_anchor_gate:
                        gate = gate.detach()
                    h = h * (1 + gate)
            else:
                h = module(h, emb)
            hs.append(h)
        h = model.middle_block(h, emb)
        stage2_deltas = []
        for idx, module in enumerate(model.output_blocks):
            h = torch.cat([h, hs.pop()], dim=1)
            h = module(h, emb)
            if idx in self.stage2_adapter_indices:
                h, delta = self.stage2_adapters[str(idx)](h, batch, frames, k2=k2, timesteps=timesteps)
                stage2_deltas.append(delta)
        eps = model.out(h.type(flat_input.dtype)).reshape(batch, frames, 1, height, width).contiguous()
        stats = dict(hwm_stats)
        stats["stage2_adapter_delta_mean"] = torch.stack(stage2_deltas).mean() if stage2_deltas else eps.new_tensor(0.0)
        return eps, cal, stats
