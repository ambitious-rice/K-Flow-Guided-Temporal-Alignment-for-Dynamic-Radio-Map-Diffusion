"""Joint Stage-1 HWM adapter and Stage-2 residual student for RMDM."""

from __future__ import annotations

import torch
from torch import nn
import torch.nn.functional as F

from hwm_decoder_adapter_student import HWMDecoderAdapterStudent
from residual_kflow_student import ResidualCorrectionHead
from utils.nn import timestep_embedding


class JointHWMResidualStudent(nn.Module):
    """Frozen RMDM with trainable HWM decoder adapters and output residual head.

    The original RMDM weights stay frozen. The HWM adapter replaces only the
    highway prior branch in a manual diffusion-trunk forward. Unlike the
    baseline RMDM forward, the anchor gate is not detached, so final diffusion
    losses can update the Stage-1 adapter.
    """

    def __init__(
        self,
        base_model: nn.Module,
        adapter_indices: tuple[int, ...] = (-2, -1),
        adapter_reduction: int = 1,
        adapter_min_hidden: int = 16,
        condition_channels: int = 3,
        head_hidden_channels: int = 32,
        head_num_layers: int = 2,
        residual_alpha: float = 0.03,
        diffusion_steps: int = 1000,
        use_frame_pos: bool = True,
        use_timestep_channel: bool = True,
        detach_anchor_gate: bool = False,
    ):
        super().__init__()
        self.base_model = base_model
        self.unet = getattr(base_model, "unet", base_model)
        self.hwm_adapter = HWMDecoderAdapterStudent(
            base_model,
            adapter_indices=adapter_indices,
            reduction=adapter_reduction,
            min_hidden=adapter_min_hidden,
        )
        self.residual_head = ResidualCorrectionHead(
            in_channels=int(condition_channels) + 1 + 1,
            hidden_channels=head_hidden_channels,
            num_layers=head_num_layers,
            use_frame_pos=use_frame_pos,
            use_timestep_channel=use_timestep_channel,
        )
        self.alpha = float(residual_alpha)
        self.diffusion_steps = int(diffusion_steps)
        self.detach_anchor_gate = bool(detach_anchor_gate)
        for param in self.base_model.parameters():
            param.requires_grad = False
        for param in self.hwm_adapter.adapters.parameters():
            param.requires_grad = True
        for param in self.residual_head.parameters():
            param.requires_grad = True
        self.base_model.eval()

    def train(self, mode: bool = True):
        super().train(mode)
        self.base_model.eval()
        return self

    def trainable_parameters(self):
        yield from self.hwm_adapter.adapters.parameters()
        yield from self.residual_head.parameters()

    def load_hwm_adapter_checkpoint(self, path: str) -> None:
        payload = torch.load(path, map_location="cpu")
        self.hwm_adapter.adapters.load_state_dict(payload["adapters"], strict=True)

    def forward_adapted_trunk(
        self,
        conditions: torch.Tensor,
        noisy: torch.Tensor,
        timesteps: torch.Tensor,
    ) -> tuple[torch.Tensor, torch.Tensor, dict[str, torch.Tensor]]:
        if conditions.ndim != 5 or noisy.ndim != 5:
            raise ValueError("conditions and noisy must be [B,F,C,H,W]")
        batch, frames, _, height, width = conditions.shape
        model = self.unet
        flat_t = timesteps[:, None].repeat(1, frames).reshape(batch * frames)
        flat_input = torch.cat([conditions, noisy], dim=2).reshape(
            batch * frames,
            conditions.shape[2] + noisy.shape[2],
            height,
            width,
        )
        anch_seq, cal, stats = self.hwm_adapter.forward_hwm(conditions)
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
        for module in model.output_blocks:
            h = torch.cat([h, hs.pop()], dim=1)
            h = module(h, emb)
        eps = model.out(h.type(flat_input.dtype)).reshape(batch, frames, 1, height, width).contiguous()
        return eps, cal, stats

    def forward(self, conditions: torch.Tensor, noisy: torch.Tensor, timesteps: torch.Tensor):
        eps_adapted, cal, stats = self.forward_adapted_trunk(conditions, noisy, timesteps)
        features = torch.cat([conditions, noisy, eps_adapted], dim=2)
        delta_eps = self.residual_head(features, timesteps, self.diffusion_steps)
        eps_final = eps_adapted + self.alpha * delta_eps
        extra = {
            "eps_adapted": eps_adapted,
            "delta_eps": delta_eps,
            "alpha": eps_final.new_tensor(self.alpha),
            **stats,
        }
        return eps_final, cal, extra
