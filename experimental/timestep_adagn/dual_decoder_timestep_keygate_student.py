"""Timestep-conditioned dual decoder temporal adapters for frozen RMDM.

This is the deployable no-K2 branch: adapters may use the condition, noisy
sample, timestep, and internal features, but not oracle K2/key-path inputs.
"""

from __future__ import annotations

import _paths  # noqa: F401

import torch
from torch import nn
import torch.nn.functional as F

from hwm_decoder_adapter_student import HWMDecoderAdapterStudent, _group_count
from utils.nn import timestep_embedding


class TimestepAdaGNSeparable3DAdapter(nn.Module):
    """Residual temporal adapter with timestep-embedding AdaGN modulation.

    The timestep is sinusoidally encoded, passed through an MLP, and projected
    to per-channel ``scale, shift`` terms for the adapter input GroupNorm:
    ``AdaGN(h, t) = GN(h) * (1 + scale_t) + shift_t``.

    The time projection and adapter output projection are zero-initialized, so
    the adapter starts as the same no-op residual branch as the single-scale
    baseline while still giving timestep conditioning a real channel-wise path.
    """

    def __init__(
        self,
        channels: int,
        reduction: int = 1,
        min_hidden: int = 16,
        diffusion_steps: int = 1000,
        time_embed_dim: int = 128,
        time_hidden: int = 128,
    ):
        super().__init__()
        channels = int(channels)
        hidden = max(int(min_hidden), channels // max(int(reduction), 1))
        self.diffusion_steps = int(diffusion_steps)
        self.time_embed_dim = int(time_embed_dim)
        self.norm = nn.GroupNorm(_group_count(channels), channels)
        self.time_mlp = nn.Sequential(
            nn.Linear(self.time_embed_dim, int(time_hidden)),
            nn.SiLU(),
            nn.Linear(int(time_hidden), channels * 2),
        )
        self.down = nn.Conv3d(channels, hidden, kernel_size=1)
        self.temporal = nn.Conv3d(hidden, hidden, kernel_size=(3, 1, 1), padding=(1, 0, 0))
        self.spatial = nn.Conv3d(hidden, hidden, kernel_size=(1, 3, 3), padding=(0, 1, 1))
        self.up = nn.Conv3d(hidden, channels, kernel_size=1)
        nn.init.zeros_(self.time_mlp[-1].weight)
        nn.init.zeros_(self.time_mlp[-1].bias)
        nn.init.zeros_(self.up.weight)
        nn.init.zeros_(self.up.bias)

    def _scale_shift(
        self,
        timesteps: torch.Tensor,
        batch: int,
        dtype: torch.dtype,
        device: torch.device,
    ) -> tuple[torch.Tensor, torch.Tensor]:
        if timesteps.ndim != 1 or timesteps.shape[0] != batch:
            raise ValueError(f"timesteps must be [B], got {tuple(timesteps.shape)} for batch {batch}")
        weight = self.time_mlp[0].weight
        emb = timestep_embedding(timesteps.to(device=device), self.time_embed_dim)
        emb = emb.to(device=device, dtype=weight.dtype)
        scale, shift = self.time_mlp(emb).to(dtype=dtype).chunk(2, dim=1)
        return scale.view(batch, -1, 1, 1, 1), shift.view(batch, -1, 1, 1, 1)

    def forward(
        self,
        x: torch.Tensor,
        batch: int,
        frames: int,
        timesteps: torch.Tensor,
    ) -> tuple[torch.Tensor, torch.Tensor, torch.Tensor]:
        if x.ndim != 4:
            raise ValueError(f"expected [B*F,C,H,W], got {tuple(x.shape)}")
        _, channels, height, width = x.shape
        seq = x.reshape(batch, frames, channels, height, width).permute(0, 2, 1, 3, 4).contiguous()
        scale, shift = self._scale_shift(timesteps, batch, seq.dtype, seq.device)
        y = self.norm(seq) * (1.0 + scale) + shift
        y = self.down(y)
        y = F.silu(self.temporal(F.silu(y)))
        y = F.silu(self.spatial(y))
        delta = self.up(y)
        out = seq + delta
        out = out.permute(0, 2, 1, 3, 4).reshape(batch * frames, channels, height, width).contiguous()
        mod_mean = 0.5 * (scale.detach().abs().mean() + shift.detach().abs().mean())
        return out, delta.detach().abs().mean(), mod_mean


class HWMDecoderTimestepAdaGNAdapter(HWMDecoderAdapterStudent):
    """HWM decoder adapter wrapper using timestep-AdaGN residual adapters."""

    def __init__(
        self,
        base_model: nn.Module,
        adapter_indices: tuple[int, ...] = (-2, -1),
        reduction: int = 1,
        min_hidden: int = 16,
        diffusion_steps: int = 1000,
        time_embed_dim: int = 128,
        time_hidden: int = 128,
    ):
        super().__init__(
            base_model,
            adapter_indices=adapter_indices,
            reduction=reduction,
            min_hidden=min_hidden,
        )
        self.adapters = nn.ModuleDict()
        for idx in self.adapter_indices:
            channels = int(self.hwm.conv_blocks_localization[idx][-1].output_channels)
            self.adapters[str(idx)] = TimestepAdaGNSeparable3DAdapter(
                channels,
                reduction=reduction,
                min_hidden=min_hidden,
                diffusion_steps=diffusion_steps,
                time_embed_dim=time_embed_dim,
                time_hidden=time_hidden,
            )

    def forward_hwm(self, conditions: torch.Tensor, timesteps: torch.Tensor):
        """Return adapted ``anchors, cal_seq, stats`` for ``conditions [B,F,C,H,W]``."""
        x, batch, frames = self._flatten_conditions(conditions)
        hwm = self.hwm
        skips = []
        seg_outputs = []
        anch_outputs = []
        stats = {}
        mod_stats = {}

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
                x, delta, mod_mean = self.adapters[str(u)](x, batch, frames, timesteps)
                adapter_has_run = True
                stats[f"adapter_delta_{u}"] = delta
                mod_stats[f"adapter_mod_{u}"] = mod_mean
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
        stats["adapter_mod_mean"] = torch.stack(list(mod_stats.values())).mean() if mod_stats else cal.new_tensor(0.0)
        return anchor_seq, cal, stats


class DualDecoderTimestepScaleStudent(nn.Module):
    """Frozen RMDM with no-K2 timestep-AdaGN temporal adapters."""

    def __init__(
        self,
        base_model: nn.Module,
        hwm_adapter_indices: tuple[int, ...] = (-2, -1),
        stage2_adapter_indices: tuple[int, ...] = (-2, -1),
        adapter_reduction: int = 1,
        adapter_min_hidden: int = 16,
        diffusion_steps: int = 1000,
        alpha_hidden: int = 128,
        time_embed_dim: int = 128,
        detach_anchor_gate: bool = False,
        train_base: bool = False,
    ):
        super().__init__()
        self.base_model = base_model
        self.unet = getattr(base_model, "unet", base_model)
        self.hwm_adapter = HWMDecoderTimestepAdaGNAdapter(
            base_model,
            adapter_indices=hwm_adapter_indices,
            reduction=adapter_reduction,
            min_hidden=adapter_min_hidden,
            diffusion_steps=diffusion_steps,
            time_embed_dim=time_embed_dim,
            time_hidden=alpha_hidden,
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
            self.stage2_adapters[str(idx)] = TimestepAdaGNSeparable3DAdapter(
                channels,
                reduction=adapter_reduction,
                min_hidden=adapter_min_hidden,
                diffusion_steps=diffusion_steps,
                time_embed_dim=time_embed_dim,
                time_hidden=alpha_hidden,
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
        return DualDecoderTimestepScaleStudent._block_channels(block)

    @staticmethod
    def _block_channels(block: nn.Module) -> int:
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

    def load_hwm_adapter_checkpoint(self, path: str) -> None:
        payload = torch.load(path, map_location="cpu")
        self.hwm_adapter.adapters.load_state_dict(payload["adapters"], strict=True)

    def forward(self, conditions: torch.Tensor, noisy: torch.Tensor, timesteps: torch.Tensor):
        if conditions.ndim != 5 or noisy.ndim != 5:
            raise ValueError("conditions and noisy must be [B,F,C,H,W]")
        batch, frames, _, height, width = conditions.shape
        model = self.unet
        flat_t = timesteps[:, None].repeat(1, frames).reshape(batch * frames)
        flat_input = torch.cat([conditions, noisy], dim=2).reshape(batch * frames, conditions.shape[2] + noisy.shape[2], height, width)

        anch_seq, cal, hwm_stats = self.hwm_adapter.forward_hwm(conditions, timesteps)
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
        stage2_mods = []
        for idx, module in enumerate(model.output_blocks):
            h = torch.cat([h, hs.pop()], dim=1)
            h = module(h, emb)
            if idx in self.stage2_adapter_indices:
                h, delta, mod_mean = self.stage2_adapters[str(idx)](h, batch, frames, timesteps)
                stage2_deltas.append(delta)
                stage2_mods.append(mod_mean)
        eps = model.out(h.type(flat_input.dtype)).reshape(batch, frames, 1, height, width).contiguous()
        stats = dict(hwm_stats)
        stats["stage2_adapter_delta_mean"] = torch.stack(stage2_deltas).mean() if stage2_deltas else eps.new_tensor(0.0)
        stats["stage2_adapter_mod_mean"] = torch.stack(stage2_mods).mean() if stage2_mods else eps.new_tensor(0.0)
        return eps, cal, stats
