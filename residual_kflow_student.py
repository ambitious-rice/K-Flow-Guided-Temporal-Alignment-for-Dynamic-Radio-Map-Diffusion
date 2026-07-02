"""Frozen-RMDM residual student for temporal consistency experiments."""

from __future__ import annotations

from dataclasses import dataclass
import os
import sys

import torch
from torch import nn
import torch.nn.functional as F

from lib.kflow_loss import compute_kflow, spatial_gradients_centered

_UTILS_DIR = os.path.join(os.path.dirname(os.path.abspath(__file__)), "utils")
if _UTILS_DIR not in sys.path:
    sys.path.append(_UTILS_DIR)
from nn import timestep_embedding


class ResidualCorrectionHead(nn.Module):
    """Small zero-init 3D CNN head that predicts ``delta_eps`` for a clip.

    The head sees frozen base predictions plus clip context, but the original
    RMDM trunk stays frozen. The final convolution is zero-initialized so the
    wrapper exactly matches the base model at step 0.
    """

    def __init__(
        self,
        in_channels: int,
        hidden_channels: int = 32,
        num_layers: int = 2,
        use_frame_pos: bool = True,
        use_timestep_channel: bool = True,
    ):
        super().__init__()
        self.use_frame_pos = bool(use_frame_pos)
        self.use_timestep_channel = bool(use_timestep_channel)
        extra = 0
        if self.use_frame_pos:
            extra += 2
        if self.use_timestep_channel:
            extra += 1
        total_in = int(in_channels) + extra
        layers: list[nn.Module] = []
        layers.extend((nn.Conv3d(total_in, hidden_channels, kernel_size=3, padding=1), nn.SiLU()))
        for _ in range(max(1, int(num_layers)) - 1):
            layers.extend((nn.Conv3d(hidden_channels, hidden_channels, kernel_size=3, padding=1), nn.SiLU()))
        layers.append(nn.Conv3d(hidden_channels, 1, kernel_size=3, padding=1))
        self.net = nn.Sequential(*layers)
        nn.init.zeros_(self.net[-1].weight)
        nn.init.zeros_(self.net[-1].bias)

    def forward(self, features: torch.Tensor, timesteps: torch.Tensor, diffusion_steps: int) -> torch.Tensor:
        """Return ``delta_eps`` with shape ``[B,F,1,H,W]``.

        ``features`` is ``[B,F,C,H,W]``. ``timesteps`` is clip-level ``[B]``.
        """
        if features.ndim != 5:
            raise ValueError(f"features must be [B,F,C,H,W], got {tuple(features.shape)}")
        batch, frames, _, height, width = features.shape
        extras = []
        if self.use_frame_pos:
            denom = max(frames - 1, 1)
            pos = torch.linspace(0.0, 1.0, frames, device=features.device, dtype=features.dtype)
            pos = pos.view(1, frames, 1, 1, 1).expand(batch, frames, 1, height, width)
            extras.extend((pos, 1.0 - pos))
        if self.use_timestep_channel:
            t = timesteps.to(device=features.device, dtype=features.dtype)
            t = t / max(float(diffusion_steps - 1), 1.0)
            t = t.view(batch, 1, 1, 1, 1).expand(batch, frames, 1, height, width)
            extras.append(t)
        if extras:
            features = torch.cat([features, *extras], dim=2)
        x = features.permute(0, 2, 1, 3, 4).contiguous()
        delta = self.net(x).permute(0, 2, 1, 3, 4).contiguous()
        return delta


def _group_count(channels: int, max_groups: int = 32) -> int:
    for groups in range(min(max_groups, channels), 0, -1):
        if channels % groups == 0:
            return groups
    return 1


class MiddleTemporalAdapter(nn.Module):
    """Local separable 3D residual adapter for frozen RMDM bottleneck features.

    The final projection is zero-initialized, so enabling the adapter keeps the
    wrapped model exactly equivalent to frozen RMDM at initialization.
    """

    def __init__(
        self,
        channels: int,
        reduction: int = 4,
        min_hidden: int = 32,
        use_frame_pos: bool = True,
        align_timestep: int = 300,
        timestep_gate: str = "linear",
    ):
        super().__init__()
        self.channels = int(channels)
        self.hidden_channels = max(int(min_hidden), self.channels // max(int(reduction), 1))
        self.use_frame_pos = bool(use_frame_pos)
        self.align_timestep = int(align_timestep)
        self.timestep_gate = str(timestep_gate)
        self.norm = nn.GroupNorm(_group_count(self.channels), self.channels)
        self.down = nn.Conv3d(self.channels, self.hidden_channels, kernel_size=1)
        self.temporal = nn.Conv3d(
            self.hidden_channels,
            self.hidden_channels,
            kernel_size=(3, 1, 1),
            padding=(1, 0, 0),
        )
        self.spatial = nn.Conv3d(
            self.hidden_channels,
            self.hidden_channels,
            kernel_size=(1, 3, 3),
            padding=(0, 1, 1),
        )
        self.up = nn.Conv3d(self.hidden_channels, self.channels, kernel_size=1)
        self.frame_pos_proj = nn.Conv3d(2, self.hidden_channels, kernel_size=1) if self.use_frame_pos else None
        self.layer_gate = nn.Parameter(torch.tensor(1.0))
        nn.init.zeros_(self.up.weight)
        nn.init.zeros_(self.up.bias)

    def _lambda_t(self, timesteps: torch.Tensor, frames: int, height: int, width: int, dtype: torch.dtype) -> torch.Tensor:
        t = timesteps.to(dtype=dtype)
        if self.timestep_gate == "none" or self.align_timestep <= 0:
            lam = torch.ones_like(t)
        elif self.timestep_gate == "hard":
            lam = (t <= float(self.align_timestep)).to(dtype=dtype)
        else:
            lam = (1.0 - t / float(self.align_timestep)).clamp(0.0, 1.0)
        return lam.view(-1, 1, 1, 1, 1).expand(-1, 1, frames, height, width)

    def forward(self, h: torch.Tensor, batch: int, frames: int, timesteps: torch.Tensor) -> tuple[torch.Tensor, dict[str, torch.Tensor]]:
        if h.ndim != 4:
            raise ValueError(f"middle adapter input must be [B*F,C,H,W], got {tuple(h.shape)}")
        _, channels, height, width = h.shape
        if channels != self.channels:
            raise ValueError(f"middle adapter expected {self.channels} channels, got {channels}")
        x = h.reshape(batch, frames, channels, height, width).permute(0, 2, 1, 3, 4).contiguous()
        y = self.down(self.norm(x))
        if self.frame_pos_proj is not None:
            denom = max(frames - 1, 1)
            pos = torch.linspace(0.0, 1.0, frames, device=h.device, dtype=h.dtype)
            pos = pos.view(1, 1, frames, 1, 1).expand(batch, 1, frames, height, width)
            pos = torch.cat([pos, 1.0 - pos], dim=1)
            y = y + self.frame_pos_proj(pos)
        y = F.silu(y)
        y = F.silu(self.temporal(y))
        y = F.silu(self.spatial(y))
        delta = self.up(y)
        lam = self._lambda_t(timesteps, frames, height, width, h.dtype)
        gate = torch.tanh(self.layer_gate)
        out = x + lam * gate * delta
        out = out.permute(0, 2, 1, 3, 4).reshape(batch * frames, channels, height, width).contiguous()
        stats = {
            "middle_gate": gate.detach(),
            "middle_lambda": lam[:, :, 0, 0, 0].mean().detach(),
            "middle_delta": delta.detach().abs().mean(),
        }
        return out, stats


class FrozenRMDMResidualStudent(nn.Module):
    """Frozen base RMDM plus bounded residual correction head."""

    def __init__(
        self,
        base_model: nn.Module,
        condition_channels: int = 3,
        hidden_channels: int = 32,
        head_num_layers: int = 2,
        alpha: float = 0.05,
        use_frame_pos: bool = True,
        use_timestep_channel: bool = True,
        diffusion_steps: int = 1000,
        use_output_head: bool = True,
        use_middle_adapter: bool = False,
        middle_adapter_reduction: int = 4,
        middle_adapter_min_hidden: int = 32,
        middle_adapter_align_timestep: int = 300,
        middle_adapter_timestep_gate: str = "linear",
    ):
        super().__init__()
        self.base_model = base_model
        self.unet = getattr(base_model, "unet", base_model)
        self.condition_channels = int(condition_channels)
        self.diffusion_steps = int(diffusion_steps)
        self.alpha = float(alpha)
        self.use_output_head = bool(use_output_head)
        self.use_middle_adapter = bool(use_middle_adapter)
        head_in = self.condition_channels + 1 + 1
        self.residual_head = ResidualCorrectionHead(
            in_channels=head_in,
            hidden_channels=hidden_channels,
            num_layers=head_num_layers,
            use_frame_pos=use_frame_pos,
            use_timestep_channel=use_timestep_channel,
        )
        self.middle_adapter = None
        if self.use_middle_adapter:
            middle_channels = self._infer_middle_channels(self.unet)
            self.middle_adapter = MiddleTemporalAdapter(
                middle_channels,
                reduction=middle_adapter_reduction,
                min_hidden=middle_adapter_min_hidden,
                use_frame_pos=use_frame_pos,
                align_timestep=middle_adapter_align_timestep,
                timestep_gate=middle_adapter_timestep_gate,
            )
        for param in self.base_model.parameters():
            param.requires_grad = False
        for param in self.residual_head.parameters():
            param.requires_grad = self.use_output_head
        if self.middle_adapter is not None:
            for param in self.middle_adapter.parameters():
                param.requires_grad = True
        self.base_model.eval()

    @staticmethod
    def _infer_middle_channels(base_model: nn.Module) -> int:
        channel_mult = getattr(base_model, "channel_mult", (1, 2, 4, 8))
        if isinstance(channel_mult, str):
            channel_mult = tuple(int(x) for x in channel_mult.split(",") if x)
        if hasattr(base_model, "model_channels"):
            return int(getattr(base_model, "model_channels")) * int(channel_mult[-1])
        first_middle = getattr(base_model, "middle_block")[0]
        return int(getattr(first_middle, "channels", getattr(first_middle, "out_channels")))

    def trainable_parameters(self):
        modules = []
        if self.use_output_head:
            modules.append(self.residual_head)
        if self.middle_adapter is not None:
            modules.append(self.middle_adapter)
        for module in modules:
            yield from module.parameters()

    def train(self, mode: bool = True):
        super().train(mode)
        self.base_model.eval()
        return self

    def _forward_with_middle_adapter(
        self,
        flat_input: torch.Tensor,
        flat_t: torch.Tensor,
        batch: int,
        frames: int,
        timesteps: torch.Tensor,
    ) -> torch.Tensor:
        model = self.unet
        hs = []
        with torch.no_grad():
            emb = model.time_embed(timestep_embedding(flat_t, model.model_channels))
            if model.num_classes is not None:
                raise NotImplementedError("class-conditional RMDM is not used in this wrapper")
            h = flat_input.type(model.dtype)
            c = h[:, :-1, ...]
            anch, _ = model.highway_forward(c)
            for ind, module in enumerate(model.input_blocks):
                if len(emb.size()) > 2:
                    emb = emb.squeeze()
                if ind == 0:
                    h = module(h, emb)
                    gate_map = None
                    if isinstance(anch, (list, tuple)) and len(anch) > 0:
                        for a in anch[:2]:
                            a_resized = F.interpolate(a, size=h.shape[-2:], mode="bilinear", align_corners=False)
                            a_mean = a_resized.mean(dim=1, keepdim=True)
                            gate_map = a_mean if gate_map is None else gate_map + a_mean
                    if gate_map is not None:
                        g = torch.sigmoid(gate_map).detach()
                        h = h * (1 + g)
                else:
                    h = module(h, emb)
                hs.append(h)
            h = model.middle_block(h, emb)
        h, adapter_stats = self.middle_adapter(h.detach(), batch, frames, timesteps)
        for module in model.output_blocks:
            skip = hs.pop().detach()
            h = torch.cat([h, skip], dim=1)
            h = module(h, emb.detach())
        h = h.type(flat_input.dtype)
        out = model.out(h)
        return out, adapter_stats

    def forward(self, conditions: torch.Tensor, noisy: torch.Tensor, timesteps: torch.Tensor):
        """Run frozen base and residual head.

        Args:
            conditions: ``[B,F,C,H,W]`` preprocessed RMDM conditions.
            noisy: ``[B,F,1,H,W]`` noisy RSS sample.
            timesteps: clip-level ``[B]`` diffusion timestep.
        """
        if conditions.ndim != 5 or noisy.ndim != 5:
            raise ValueError("conditions and noisy must be [B,F,C,H,W]")
        batch, frames = noisy.shape[:2]
        flat_t = timesteps[:, None].repeat(1, frames).reshape(batch * frames)
        flat_input = torch.cat([conditions, noisy], dim=2).reshape(
            batch * frames,
            conditions.shape[2] + noisy.shape[2],
            noisy.shape[-2],
            noisy.shape[-1],
        )
        with torch.no_grad():
            eps_base, cal = self.base_model(flat_input, flat_t)
        eps_base = eps_base.reshape(batch, frames, *eps_base.shape[1:]).contiguous()
        cal = cal.reshape(batch, frames, *cal.shape[1:]).contiguous()
        extra = {"eps_base": eps_base, "alpha": self.alpha}
        if self.middle_adapter is not None:
            eps_trunk, adapter_stats = self._forward_with_middle_adapter(flat_input, flat_t, batch, frames, timesteps)
            eps_final = eps_trunk.reshape(batch, frames, *eps_trunk.shape[1:]).contiguous()
            extra.update(adapter_stats)
        else:
            eps_final = eps_base
        if self.use_output_head:
            features = torch.cat([conditions, noisy, eps_base.detach()], dim=2)
            delta_eps = self.residual_head(features, timesteps, self.diffusion_steps)
            eps_final = eps_final + self.alpha * delta_eps
        else:
            delta_eps = torch.zeros_like(eps_base)
        extra["delta_eps"] = delta_eps
        return eps_final, cal, extra


def predict_x0_from_eps(noisy: torch.Tensor, eps: torch.Tensor, timesteps: torch.Tensor, alphas_cumprod: torch.Tensor) -> torch.Tensor:
    """Convert epsilon prediction to x0 for clip tensors."""
    batch, frames = noisy.shape[:2]
    flat_t = timesteps[:, None].repeat(1, frames).reshape(batch * frames)
    alpha = alphas_cumprod.to(device=noisy.device, dtype=noisy.dtype)[flat_t]
    while alpha.ndim < noisy.ndim - 1:
        alpha = alpha.unsqueeze(-1)
    alpha = alpha.reshape(batch, frames, *([1] * (noisy.ndim - 2)))
    return (noisy - torch.sqrt(1.0 - alpha) * eps) / torch.sqrt(alpha)


def _warp_scalar(prev: torch.Tensor, flow: torch.Tensor) -> torch.Tensor:
    """Warp ``prev`` [N,1,H,W] by pixel flow [N,2,H,W]."""
    n, _, height, width = prev.shape
    yy, xx = torch.meshgrid(
        torch.arange(height, device=prev.device, dtype=prev.dtype),
        torch.arange(width, device=prev.device, dtype=prev.dtype),
        indexing="ij",
    )
    x = xx.unsqueeze(0).expand(n, -1, -1) - flow[:, 0]
    y = yy.unsqueeze(0).expand(n, -1, -1) - flow[:, 1]
    x = 2.0 * x / max(width - 1, 1) - 1.0
    y = 2.0 * y / max(height - 1, 1) - 1.0
    grid = torch.stack([x, y], dim=-1)
    return F.grid_sample(prev, grid, mode="bilinear", padding_mode="border", align_corners=True)


@dataclass
class SafeKFlowConfig:
    delta: float = 1e-6
    clamp: float = 1.0
    hinge_margin: float = 0.0
    warp_error_threshold: float = 0.08
    rss_grad_max: float = 0.12
    rss_delta_max: float = 0.25
    k2_strong_threshold: float = 0.95
    k2_exclude_dilate: int = 2


@dataclass
class StaticKeyPathConfig:
    delta: float = 1e-6
    clamp: float = 1.0
    keypath_k2_quantile: float = 0.70
    static_flow_quantile: float = 0.70
    static_flow_threshold: float = 0.0
    margin: float = 0.005
    pair_stride: int = 1


@dataclass
class DynamicKeyPathConfig:
    delta: float = 1e-6
    clamp: float = 1.0
    keypath_k2_quantile: float = 0.70
    dynamic_flow_quantile: float = 0.85
    dynamic_flow_threshold: float = 0.0
    margin_low_scale: float = 0.5
    margin_high_scale: float = 2.0
    high_weight: float = 0.2
    pair_stride: int = 1


@dataclass
class KFlowStructureConfig:
    delta: float = 1e-6
    clamp: float = 1.0
    keypath_k2_quantile: float = 0.70
    dynamic_flow_quantile: float = 0.85
    dynamic_flow_threshold: float = 0.0
    pair_stride: int = 1
    smooth_l1_beta: float = 0.1


@dataclass
class GlobalTemporalResidualConfig:
    margin: float = 0.005
    pair_stride: int = 1


@dataclass
class GlobalDynamicRangeConfig:
    margin_low_scale: float = 0.5
    margin_high_scale: float = 2.0
    high_weight: float = 0.2
    pair_stride: int = 1


@dataclass
class KPathDeltaConfig:
    delta: float = 1e-6
    clamp: float = 1.0
    keypath_k2_quantile: float = 0.70
    static_flow_quantile: float = 0.70
    dynamic_flow_quantile: float = 0.85
    static_flow_threshold: float = 0.0
    dynamic_flow_threshold: float = 0.0
    margin: float = 0.005
    dynamic_lambda: float = 0.1
    pair_stride: int = 1


def safe_kflow_hinge_loss(
    student_x0: torch.Tensor,
    teacher_x0: torch.Tensor,
    target: torch.Tensor,
    k2: torch.Tensor | None = None,
    batch_mask: torch.Tensor | None = None,
    cfg: SafeKFlowConfig | None = None,
) -> tuple[torch.Tensor, dict[str, torch.Tensor]]:
    """Masked hinge k-flow loss.

    Penalizes only pixels where the student k-flow error exceeds the frozen
    teacher's error by ``hinge_margin``.
    """
    cfg = cfg or SafeKFlowConfig()
    if student_x0.shape[1] <= 1:
        zero = student_x0.new_tensor(0.0)
        return zero, {"mask_mean": zero.detach(), "teacher_error": zero.detach(), "student_error": zero.detach()}

    flow_student = compute_kflow(student_x0[:, :-1, 0], student_x0[:, 1:, 0], delta=cfg.delta)
    with torch.no_grad():
        flow_teacher = compute_kflow(teacher_x0[:, :-1, 0], teacher_x0[:, 1:, 0], delta=cfg.delta)
        flow_target = compute_kflow(target[:, :-1, 0], target[:, 1:, 0], delta=cfg.delta)
    if cfg.clamp and cfg.clamp > 0:
        flow_student = flow_student.clamp(-cfg.clamp, cfg.clamp)
        flow_teacher = flow_teacher.clamp(-cfg.clamp, cfg.clamp)
        flow_target = flow_target.clamp(-cfg.clamp, cfg.clamp)

    with torch.no_grad():
        batch, pairs, _, height, width = target[:, :-1].shape
        flat_prev = target[:, :-1].reshape(batch * pairs, 1, height, width)
        flat_next = target[:, 1:].reshape(batch * pairs, 1, height, width)
        flat_flow = flow_target.reshape(batch * pairs, 2, height, width)
        warp_err = (_warp_scalar(flat_prev, flat_flow) - flat_next).abs().reshape(batch, pairs, 1, height, width)
        mask = warp_err <= float(cfg.warp_error_threshold)

        dx0, dy0 = spatial_gradients_centered(target[:, :-1, 0])
        dx1, dy1 = spatial_gradients_centered(target[:, 1:, 0])
        grad = torch.maximum(torch.sqrt(dx0.square() + dy0.square() + 1e-12), torch.sqrt(dx1.square() + dy1.square() + 1e-12))
        mask = mask & (grad[:, :, None] <= float(cfg.rss_grad_max))
        mask = mask & ((target[:, 1:] - target[:, :-1]).abs() <= float(cfg.rss_delta_max))
        if k2 is not None:
            strong = (k2[:, :-1] > float(cfg.k2_strong_threshold)) | (k2[:, 1:] > float(cfg.k2_strong_threshold))
            if cfg.k2_exclude_dilate > 0:
                flat = strong.to(dtype=target.dtype).reshape(batch * pairs, 1, height, width)
                radius = int(cfg.k2_exclude_dilate)
                flat = F.max_pool2d(flat, kernel_size=2 * radius + 1, stride=1, padding=radius)
                strong = flat.reshape(batch, pairs, 1, height, width) > 0
            mask = mask & (~strong)
        if batch_mask is not None:
            mask = mask & batch_mask[:, None, None, None, None].to(device=mask.device, dtype=torch.bool)
        mask_f = mask.to(dtype=student_x0.dtype)

    student_err = (flow_student - flow_target).abs().mean(dim=2, keepdim=True)
    teacher_err = (flow_teacher - flow_target).abs().mean(dim=2, keepdim=True)
    hinge = F.relu(student_err - teacher_err - float(cfg.hinge_margin))
    denom = mask_f.sum().clamp_min(1.0)
    loss = (hinge * mask_f).sum() / denom
    stats = {
        "mask_mean": mask_f.mean().detach(),
        "teacher_error": ((teacher_err * mask_f).sum() / denom).detach(),
        "student_error": ((student_err * mask_f).sum() / denom).detach(),
    }
    return loss, stats


def global_temporal_residual_loss(
    student_x0: torch.Tensor,
    batch_mask: torch.Tensor | None = None,
    cfg: GlobalTemporalResidualConfig | None = None,
) -> tuple[torch.Tensor, dict[str, torch.Tensor]]:
    """Whole-image adjacent-frame residual consistency without k-flow or K2 masks."""
    cfg = cfg or GlobalTemporalResidualConfig()
    stride = max(1, int(cfg.pair_stride))
    if student_x0.ndim != 5 or student_x0.shape[1] <= stride:
        zero = student_x0.new_tensor(0.0)
        return zero, {"raw_diff": zero.detach(), "active_ratio": zero.detach()}

    pred0 = student_x0[:, :-stride]
    pred1 = student_x0[:, stride:]
    raw = (pred1 - pred0).abs()
    hinge = F.relu(raw - float(cfg.margin))

    if batch_mask is not None:
        mask = batch_mask[:, None, None, None, None].to(device=student_x0.device, dtype=student_x0.dtype)
        denom = mask.sum().clamp_min(1.0) * raw.shape[1] * raw.shape[2] * raw.shape[3] * raw.shape[4]
        loss = (hinge * mask).sum() / denom
        raw_mean = (raw * mask).sum() / denom
        active = ((hinge > 0).to(dtype=student_x0.dtype) * mask).sum() / denom
    else:
        loss = hinge.mean()
        raw_mean = raw.mean()
        active = (hinge > 0).to(dtype=student_x0.dtype).mean()

    stats = {
        "raw_diff": raw_mean.detach(),
        "active_ratio": active.detach(),
    }
    return loss, stats


def global_dynamic_range_loss(
    student_x0: torch.Tensor,
    target: torch.Tensor,
    batch_mask: torch.Tensor | None = None,
    cfg: GlobalDynamicRangeConfig | None = None,
) -> tuple[torch.Tensor, dict[str, torch.Tensor]]:
    """Apply the dynamic range loss to all pixels without K2/k-flow masks."""
    cfg = cfg or GlobalDynamicRangeConfig()
    stride = max(1, int(cfg.pair_stride))
    if student_x0.ndim != 5 or student_x0.shape[1] <= stride:
        zero = student_x0.new_tensor(0.0)
        return zero, {
            "gt_median": zero.detach(),
            "pred_diff": zero.detach(),
            "low_active_ratio": zero.detach(),
            "high_active_ratio": zero.detach(),
        }

    pred0 = student_x0[:, :-stride]
    pred1 = student_x0[:, stride:]
    target0 = target[:, :-stride]
    target1 = target[:, stride:]

    with torch.no_grad():
        gt_delta = (target1 - target0).abs()
        batch, pairs = gt_delta.shape[:2]
        gt_med = torch.zeros(batch, pairs, 1, 1, 1, device=target.device, dtype=target.dtype)
        for b in range(batch):
            for f in range(pairs):
                gt_med[b, f, 0, 0, 0] = gt_delta[b, f].float().median().to(dtype=target.dtype)
        lower = float(cfg.margin_low_scale) * gt_med
        upper = float(cfg.margin_high_scale) * gt_med
        if batch_mask is not None:
            mask = batch_mask[:, None, None, None, None].to(device=target.device, dtype=target.dtype)
            denom = mask.sum().clamp_min(1.0) * pred0.shape[1] * pred0.shape[2] * pred0.shape[3] * pred0.shape[4]
        else:
            mask = None
            denom = pred0.new_tensor(float(pred0.numel()))

    pred_delta = (pred1 - pred0).abs()
    low_hinge = F.relu(lower - pred_delta)
    high_hinge = F.relu(pred_delta - upper)
    if mask is not None:
        loss_low = (low_hinge * mask).sum() / denom
        loss_high = (high_hinge * mask).sum() / denom
        pred_mean = (pred_delta * mask).sum() / denom
        low_active = (((low_hinge > 0).to(dtype=student_x0.dtype) * mask).sum() / denom).detach()
        high_active = (((high_hinge > 0).to(dtype=student_x0.dtype) * mask).sum() / denom).detach()
    else:
        loss_low = low_hinge.mean()
        loss_high = high_hinge.mean()
        pred_mean = pred_delta.mean()
        low_active = (low_hinge > 0).to(dtype=student_x0.dtype).mean().detach()
        high_active = (high_hinge > 0).to(dtype=student_x0.dtype).mean().detach()
    loss = loss_low + float(cfg.high_weight) * loss_high
    stats = {
        "gt_median": gt_med.mean().detach(),
        "pred_diff": pred_mean.detach(),
        "low_active_ratio": low_active,
        "high_active_ratio": high_active,
    }
    return loss, stats


def kpath_delta_target_loss(
    student_x0: torch.Tensor,
    target: torch.Tensor,
    k2: torch.Tensor,
    batch_mask: torch.Tensor | None = None,
    cfg: KPathDeltaConfig | None = None,
) -> tuple[torch.Tensor, dict[str, torch.Tensor]]:
    """Combined static hinge and dynamic GT-delta matching on key paths.

    L = M_static * |d_pred - clip(d_pred, 0, margin)|
        + lambda * M_dynamic * |d_pred - d_gt|
    where d_pred and d_gt are adjacent-frame absolute differences.
    """
    cfg = cfg or KPathDeltaConfig()
    stride = max(1, int(cfg.pair_stride))
    if student_x0.ndim != 5 or student_x0.shape[1] <= stride:
        zero = student_x0.new_tensor(0.0)
        return zero, {
            "static_mask_mean": zero.detach(),
            "dynamic_mask_mean": zero.detach(),
            "static_loss": zero.detach(),
            "dynamic_loss": zero.detach(),
            "pred_dynamic": zero.detach(),
            "gt_dynamic": zero.detach(),
        }
    if k2 is None:
        raise ValueError("kpath_delta_target_loss requires k2 clips")

    pred0 = student_x0[:, :-stride]
    pred1 = student_x0[:, stride:]
    target0 = target[:, :-stride]
    target1 = target[:, stride:]
    k20 = k2[:, :-stride]
    k21 = k2[:, stride:]

    with torch.no_grad():
        key_strength = torch.maximum(k20, k21)
        key_mask = _pairwise_quantile_mask(key_strength, cfg.keypath_k2_quantile, keep_high=True)
        flow = compute_kflow(target0[:, :, 0], target1[:, :, 0], delta=cfg.delta)
        if cfg.clamp and cfg.clamp > 0:
            flow = flow.clamp(-cfg.clamp, cfg.clamp)
        flow_mag = torch.linalg.vector_norm(flow, dim=2, keepdim=True)
        if cfg.static_flow_threshold and cfg.static_flow_threshold > 0:
            static_flow_mask = flow_mag <= float(cfg.static_flow_threshold)
        else:
            static_flow_mask = _pairwise_quantile_mask(flow_mag, cfg.static_flow_quantile, keep_high=False)
        if cfg.dynamic_flow_threshold and cfg.dynamic_flow_threshold > 0:
            dynamic_flow_mask = flow_mag >= float(cfg.dynamic_flow_threshold)
        else:
            dynamic_flow_mask = _pairwise_quantile_mask(flow_mag, cfg.dynamic_flow_quantile, keep_high=True)
        static_mask = key_mask & static_flow_mask
        dynamic_mask = key_mask & dynamic_flow_mask
        if batch_mask is not None:
            batch_mask_f = batch_mask[:, None, None, None, None].to(device=target.device, dtype=torch.bool)
            static_mask = static_mask & batch_mask_f
            dynamic_mask = dynamic_mask & batch_mask_f
        static_mask_f = static_mask.to(dtype=student_x0.dtype)
        dynamic_mask_f = dynamic_mask.to(dtype=student_x0.dtype)

    pred_delta = (pred1 - pred0).abs()
    gt_delta = (target1 - target0).abs()
    static_map = (pred_delta - pred_delta.clamp(0.0, float(cfg.margin))).abs()
    dynamic_map = (pred_delta - gt_delta).abs()
    static_denom = static_mask_f.sum().clamp_min(1.0)
    dynamic_denom = dynamic_mask_f.sum().clamp_min(1.0)
    static_loss = (static_map * static_mask_f).sum() / static_denom
    dynamic_loss = (dynamic_map * dynamic_mask_f).sum() / dynamic_denom
    loss = static_loss + float(cfg.dynamic_lambda) * dynamic_loss
    stats = {
        "static_mask_mean": static_mask_f.mean().detach(),
        "dynamic_mask_mean": dynamic_mask_f.mean().detach(),
        "static_loss": static_loss.detach(),
        "dynamic_loss": dynamic_loss.detach(),
        "pred_dynamic": ((pred_delta * dynamic_mask_f).sum() / dynamic_denom).detach(),
        "gt_dynamic": ((gt_delta * dynamic_mask_f).sum() / dynamic_denom).detach(),
    }
    return loss, stats


def _pairwise_quantile_mask(values: torch.Tensor, quantile: float, keep_high: bool) -> torch.Tensor:
    """Per ``[B,F]`` quantile mask for ``values`` shaped ``[B,F,1,H,W]``."""
    q = min(max(float(quantile), 0.0), 0.999)
    batch, pairs = values.shape[:2]
    out = torch.zeros_like(values, dtype=torch.bool)
    for b in range(batch):
        for f in range(pairs):
            flat = values[b, f, 0].reshape(-1).float()
            threshold = torch.quantile(flat, q).to(device=values.device, dtype=values.dtype)
            if keep_high:
                out[b, f] = values[b, f] >= threshold
            else:
                out[b, f] = values[b, f] <= threshold
    return out


def static_keypath_consistency_loss(
    student_x0: torch.Tensor,
    target: torch.Tensor,
    k2: torch.Tensor,
    batch_mask: torch.Tensor | None = None,
    cfg: StaticKeyPathConfig | None = None,
) -> tuple[torch.Tensor, dict[str, torch.Tensor]]:
    """Directly keep static key-path pixels stable across adjacent frames.

    Mask:
      key_path = high K2 quantile
      static = low GT k-flow magnitude

    Loss:
      mean(mask * relu(|x0_t - x0_t+s| - margin))
    """
    cfg = cfg or StaticKeyPathConfig()
    stride = max(1, int(cfg.pair_stride))
    if student_x0.ndim != 5 or student_x0.shape[1] <= stride:
        zero = student_x0.new_tensor(0.0)
        return zero, {"mask_mean": zero.detach(), "raw_diff": zero.detach(), "active_ratio": zero.detach()}
    if k2 is None:
        raise ValueError("static_keypath_consistency_loss requires k2 clips")

    pred0 = student_x0[:, :-stride]
    pred1 = student_x0[:, stride:]
    target0 = target[:, :-stride]
    target1 = target[:, stride:]
    k20 = k2[:, :-stride]
    k21 = k2[:, stride:]

    with torch.no_grad():
        key_strength = torch.maximum(k20, k21)
        key_mask = _pairwise_quantile_mask(key_strength, cfg.keypath_k2_quantile, keep_high=True)
        flow = compute_kflow(target0[:, :, 0], target1[:, :, 0], delta=cfg.delta)
        if cfg.clamp and cfg.clamp > 0:
            flow = flow.clamp(-cfg.clamp, cfg.clamp)
        flow_mag = torch.linalg.vector_norm(flow, dim=2, keepdim=True)
        if cfg.static_flow_threshold and cfg.static_flow_threshold > 0:
            static_mask = flow_mag <= float(cfg.static_flow_threshold)
        else:
            static_mask = _pairwise_quantile_mask(flow_mag, cfg.static_flow_quantile, keep_high=False)
        mask = key_mask & static_mask
        if batch_mask is not None:
            mask = mask & batch_mask[:, None, None, None, None].to(device=mask.device, dtype=torch.bool)
        mask_f = mask.to(dtype=student_x0.dtype)

    raw = (pred1 - pred0).abs()
    hinge = F.relu(raw - float(cfg.margin))
    denom = mask_f.sum().clamp_min(1.0)
    loss = (hinge * mask_f).sum() / denom
    active = ((hinge > 0) & mask).to(dtype=student_x0.dtype)
    stats = {
        "mask_mean": mask_f.mean().detach(),
        "raw_diff": ((raw * mask_f).sum() / denom).detach(),
        "active_ratio": (active.sum() / denom).detach(),
    }
    return loss, stats


def dynamic_keypath_range_loss(
    student_x0: torch.Tensor,
    target: torch.Tensor,
    k2: torch.Tensor,
    batch_mask: torch.Tensor | None = None,
    cfg: DynamicKeyPathConfig | None = None,
) -> tuple[torch.Tensor, dict[str, torch.Tensor]]:
    """Keep dynamic key-path changes within a GT-derived plausible range.

    Mask:
      key_path = high K2 quantile
      dynamic = high GT k-flow magnitude

    For each clip pair, compute median GT frame delta inside the dynamic mask:
      lower = margin_low_scale * median(|gt_t+s - gt_t|)
      upper = margin_high_scale * median(|gt_t+s - gt_t|)

    Loss:
      mean(mask * relu(lower - |x0_t+s - x0_t|))
      + high_weight * mean(mask * relu(|x0_t+s - x0_t| - upper))
    """
    cfg = cfg or DynamicKeyPathConfig()
    stride = max(1, int(cfg.pair_stride))
    if student_x0.ndim != 5 or student_x0.shape[1] <= stride:
        zero = student_x0.new_tensor(0.0)
        return zero, {
            "mask_mean": zero.detach(),
            "gt_median": zero.detach(),
            "pred_diff": zero.detach(),
            "low_active_ratio": zero.detach(),
            "high_active_ratio": zero.detach(),
        }
    if k2 is None:
        raise ValueError("dynamic_keypath_range_loss requires k2 clips")

    pred0 = student_x0[:, :-stride]
    pred1 = student_x0[:, stride:]
    target0 = target[:, :-stride]
    target1 = target[:, stride:]
    k20 = k2[:, :-stride]
    k21 = k2[:, stride:]

    with torch.no_grad():
        key_strength = torch.maximum(k20, k21)
        key_mask = _pairwise_quantile_mask(key_strength, cfg.keypath_k2_quantile, keep_high=True)
        flow = compute_kflow(target0[:, :, 0], target1[:, :, 0], delta=cfg.delta)
        if cfg.clamp and cfg.clamp > 0:
            flow = flow.clamp(-cfg.clamp, cfg.clamp)
        flow_mag = torch.linalg.vector_norm(flow, dim=2, keepdim=True)
        if cfg.dynamic_flow_threshold and cfg.dynamic_flow_threshold > 0:
            dynamic_mask = flow_mag >= float(cfg.dynamic_flow_threshold)
        else:
            dynamic_mask = _pairwise_quantile_mask(flow_mag, cfg.dynamic_flow_quantile, keep_high=True)
        mask = key_mask & dynamic_mask
        if batch_mask is not None:
            mask = mask & batch_mask[:, None, None, None, None].to(device=mask.device, dtype=torch.bool)

        gt_delta = (target1 - target0).abs()
        batch, pairs = gt_delta.shape[:2]
        gt_med = torch.zeros(batch, pairs, 1, 1, 1, device=target.device, dtype=target.dtype)
        for b in range(batch):
            for f in range(pairs):
                selected = gt_delta[b, f][mask[b, f]]
                if selected.numel() > 0:
                    gt_med[b, f, 0, 0, 0] = selected.float().median().to(dtype=target.dtype)
        lower = float(cfg.margin_low_scale) * gt_med
        upper = float(cfg.margin_high_scale) * gt_med
        mask_f = mask.to(dtype=student_x0.dtype)

    pred_delta = (pred1 - pred0).abs()
    low_hinge = F.relu(lower - pred_delta)
    high_hinge = F.relu(pred_delta - upper)
    denom = mask_f.sum().clamp_min(1.0)
    loss_low = (low_hinge * mask_f).sum() / denom
    loss_high = (high_hinge * mask_f).sum() / denom
    loss = loss_low + float(cfg.high_weight) * loss_high
    low_active = ((low_hinge > 0) & mask).to(dtype=student_x0.dtype)
    high_active = ((high_hinge > 0) & mask).to(dtype=student_x0.dtype)
    stats = {
        "mask_mean": mask_f.mean().detach(),
        "gt_median": ((gt_med.expand_as(mask_f) * mask_f).sum() / denom).detach(),
        "pred_diff": ((pred_delta * mask_f).sum() / denom).detach(),
        "low_active_ratio": (low_active.sum() / denom).detach(),
        "high_active_ratio": (high_active.sum() / denom).detach(),
    }
    return loss, stats


def kflow_structure_loss(
    student_x0: torch.Tensor,
    target: torch.Tensor,
    k2: torch.Tensor,
    batch_mask: torch.Tensor | None = None,
    cfg: KFlowStructureConfig | None = None,
) -> tuple[torch.Tensor, dict[str, torch.Tensor]]:
    """Match dynamic key-path change structure to GT k-flow magnitude.

    This does not directly fit ``|GT[t+s] - GT[t]|``. Instead it compares
    normalized predicted adjacent-frame change magnitude with normalized GT
    k-flow magnitude inside dynamic key-path pixels.
    """
    cfg = cfg or KFlowStructureConfig()
    stride = max(1, int(cfg.pair_stride))
    if student_x0.ndim != 5 or student_x0.shape[1] <= stride:
        zero = student_x0.new_tensor(0.0)
        return zero, {
            "mask_mean": zero.detach(),
            "pred_mean": zero.detach(),
            "flow_mean": zero.detach(),
            "structure_l1": zero.detach(),
        }
    if k2 is None:
        raise ValueError("kflow_structure_loss requires k2 clips")

    pred0 = student_x0[:, :-stride]
    pred1 = student_x0[:, stride:]
    target0 = target[:, :-stride]
    target1 = target[:, stride:]
    k20 = k2[:, :-stride]
    k21 = k2[:, stride:]

    with torch.no_grad():
        key_strength = torch.maximum(k20, k21)
        key_mask = _pairwise_quantile_mask(key_strength, cfg.keypath_k2_quantile, keep_high=True)
        flow = compute_kflow(target0[:, :, 0], target1[:, :, 0], delta=cfg.delta)
        if cfg.clamp and cfg.clamp > 0:
            flow = flow.clamp(-cfg.clamp, cfg.clamp)
        flow_mag = torch.linalg.vector_norm(flow, dim=2, keepdim=True)
        if cfg.dynamic_flow_threshold and cfg.dynamic_flow_threshold > 0:
            dynamic_mask = flow_mag >= float(cfg.dynamic_flow_threshold)
        else:
            dynamic_mask = _pairwise_quantile_mask(flow_mag, cfg.dynamic_flow_quantile, keep_high=True)
        mask = key_mask & dynamic_mask
        if batch_mask is not None:
            mask = mask & batch_mask[:, None, None, None, None].to(device=mask.device, dtype=torch.bool)
        mask_f = mask.to(dtype=student_x0.dtype)
        denom = mask_f.sum().clamp_min(1.0)
        flow_mean = (flow_mag * mask_f).sum() / denom
        flow_norm = flow_mag / flow_mean.clamp_min(float(cfg.delta))

    pred_delta = (pred1 - pred0).abs()
    pred_mean = (pred_delta * mask_f).sum() / denom
    pred_norm = pred_delta / pred_mean.clamp_min(float(cfg.delta))
    diff = pred_norm - flow_norm.detach()
    loss_map = F.smooth_l1_loss(
        pred_norm,
        flow_norm.detach(),
        beta=float(cfg.smooth_l1_beta),
        reduction="none",
    )
    loss = (loss_map * mask_f).sum() / denom
    stats = {
        "mask_mean": mask_f.mean().detach(),
        "pred_mean": pred_mean.detach(),
        "flow_mean": flow_mean.detach(),
        "structure_l1": ((diff.abs() * mask_f).sum() / denom).detach(),
    }
    return loss, stats
