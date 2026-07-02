"""Stage-1 and Stage-2 decoder temporal adapters for frozen RMDM."""

from __future__ import annotations

import torch
from torch import nn
import torch.nn.functional as F

from hwm_decoder_adapter_student import HWMDecoderAdapterStudent, NoGateSeparable3DAdapter
from utils.nn import timestep_embedding


class DualDecoderAdapterStudent(nn.Module):
    """Frozen RMDM with temporal adapters in both prior and diffusion decoders."""

    def __init__(
        self,
        base_model: nn.Module,
        hwm_adapter_indices: tuple[int, ...] = (-2, -1),
        stage2_adapter_indices: tuple[int, ...] = (-2, -1),
        adapter_reduction: int = 1,
        adapter_min_hidden: int = 16,
        detach_anchor_gate: bool = False,
        train_base: bool = False,
    ):
        super().__init__()
        self.base_model = base_model
        self.unet = getattr(base_model, "unet", base_model)
        self.hwm_adapter = HWMDecoderAdapterStudent(
            base_model,
            adapter_indices=hwm_adapter_indices,
            reduction=adapter_reduction,
            min_hidden=adapter_min_hidden,
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
            self.stage2_adapters[str(idx)] = NoGateSeparable3DAdapter(channels, reduction=adapter_reduction, min_hidden=adapter_min_hidden)
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

        anch_seq, cal, hwm_stats = self.hwm_adapter.forward_hwm(conditions)
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
                h, delta = self.stage2_adapters[str(idx)](h, batch, frames)
                stage2_deltas.append(delta)
        eps = model.out(h.type(flat_input.dtype)).reshape(batch, frames, 1, height, width).contiguous()
        stats = dict(hwm_stats)
        stats["stage2_adapter_delta_mean"] = torch.stack(stage2_deltas).mean() if stage2_deltas else eps.new_tensor(0.0)
        return eps, cal, stats
