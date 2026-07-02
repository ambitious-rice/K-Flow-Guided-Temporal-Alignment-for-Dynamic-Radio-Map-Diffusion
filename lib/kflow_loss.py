"""k-flow losses for temporal RMDM PINN experiments."""

from __future__ import annotations

import torch
import torch.nn.functional as F


def spatial_gradients_centered(kappa: torch.Tensor) -> tuple[torch.Tensor, torch.Tensor]:
    """Return centered spatial gradients as ``(dx, dy)`` preserving shape."""
    dx = torch.zeros_like(kappa)
    dy = torch.zeros_like(kappa)
    if kappa.shape[-1] > 2:
        dx[..., 1:-1] = 0.5 * (kappa[..., 2:] - kappa[..., :-2])
    if kappa.shape[-1] > 1:
        dx[..., 0] = kappa[..., 1] - kappa[..., 0]
        dx[..., -1] = kappa[..., -1] - kappa[..., -2]
    if kappa.shape[-2] > 2:
        dy[..., 1:-1, :] = 0.5 * (kappa[..., 2:, :] - kappa[..., :-2, :])
    if kappa.shape[-2] > 1:
        dy[..., 0, :] = kappa[..., 1, :] - kappa[..., 0, :]
        dy[..., -1, :] = kappa[..., -1, :] - kappa[..., -2, :]
    return dx, dy


def compute_kflow(kappa_t: torch.Tensor, kappa_t1: torch.Tensor, delta: float = 1e-6) -> torch.Tensor:
    """Compute minimum-norm normal k-flow with output ``[..., 2, H, W]``."""
    if kappa_t.shape != kappa_t1.shape:
        raise ValueError(f"kappa_t and kappa_t1 shapes differ: {kappa_t.shape} vs {kappa_t1.shape}")
    if not torch.is_floating_point(kappa_t):
        raise TypeError(f"k-flow expects floating point tensors, got {kappa_t.dtype}")
    dx, dy = spatial_gradients_centered(kappa_t)
    delta_kappa = kappa_t1 - kappa_t
    eps = max(float(delta), torch.finfo(kappa_t.dtype).eps)
    denom = dx.square() + dy.square() + kappa_t.new_tensor(eps)
    flow = torch.stack((-(delta_kappa * dx) / denom, -(delta_kappa * dy) / denom), dim=-3)
    return torch.nan_to_num(flow, nan=0.0, posinf=0.0, neginf=0.0)


def pathway_mask_from_k2(k2: torch.Tensor, threshold: float = 0.03, dilate_radius: int = 5) -> torch.Tensor:
    """Build adjacent-frame path masks from K2 clip ``[B,F,1,H,W]`` in ``[0,1]``."""
    if k2.ndim != 5 or k2.shape[2] != 1:
        raise ValueError(f"k2 must have shape [B,F,1,H,W], got {tuple(k2.shape)}")
    mask = (k2[:, :-1] > float(threshold)) | (k2[:, 1:] > float(threshold))
    mask = mask.to(dtype=k2.dtype)
    radius = int(dilate_radius)
    if radius > 0:
        batch, pairs, channels, height, width = mask.shape
        flat = mask.reshape(batch * pairs, channels, height, width)
        kernel = 2 * radius + 1
        flat = F.max_pool2d(flat, kernel_size=kernel, stride=1, padding=radius)
        mask = flat.reshape(batch, pairs, channels, height, width)
    return mask


def masked_flow_l1(pred_flow: torch.Tensor, target_flow: torch.Tensor, mask: torch.Tensor | None = None) -> torch.Tensor:
    """Masked L1 flow loss. Flow shape is ``[B,F-1,2,H,W]``."""
    if pred_flow.shape != target_flow.shape:
        raise ValueError(f"flow shapes differ: {pred_flow.shape} vs {target_flow.shape}")
    diff = (pred_flow - target_flow).abs()
    if mask is None:
        return diff.mean()
    mask = mask.to(device=diff.device, dtype=diff.dtype)
    return (diff * mask).sum() / (mask.sum().clamp_min(1.0) * diff.shape[2])


def refine_mask_by_target_flow(
    mask: torch.Tensor | None,
    target_flow: torch.Tensor,
    flow_threshold: float = 0.0,
    flow_quantile: float = 0.0,
) -> torch.Tensor | None:
    """Keep only target-flow-active pixels inside an existing path mask.

    ``target_flow`` is ``[B,F-1,2,H,W]``. Quantile is computed per adjacent pair
    over pixels already selected by ``mask`` so the loss focuses on key
    propagation paths instead of static or cluttered support.
    """
    if mask is None or (flow_threshold <= 0 and flow_quantile <= 0):
        return mask
    flow_mag = torch.sqrt(target_flow[:, :, 0:1].square() + target_flow[:, :, 1:2].square() + 1e-12)
    refined = mask.to(device=target_flow.device, dtype=target_flow.dtype)
    if flow_threshold > 0:
        refined = refined * (flow_mag > float(flow_threshold)).to(dtype=refined.dtype)
    if flow_quantile > 0:
        q = min(max(float(flow_quantile), 0.0), 0.999)
        batch, pairs = refined.shape[:2]
        out = torch.zeros_like(refined)
        for b in range(batch):
            for f in range(pairs):
                support = refined[b, f] > 0
                if not torch.any(support):
                    continue
                values = flow_mag[b, f][support]
                threshold = torch.quantile(values.float(), q).to(dtype=flow_mag.dtype)
                out[b, f] = refined[b, f] * (flow_mag[b, f] >= threshold).to(dtype=refined.dtype)
        refined = out
    return refined


def temporal_kflow_loss(
    pred: torch.Tensor,
    target: torch.Tensor,
    path_mask: torch.Tensor | None = None,
    delta: float = 1e-6,
    clamp: float = 1.0,
    target_flow_threshold: float = 0.0,
    target_flow_quantile: float = 0.0,
) -> torch.Tensor:
    """Compare k-flow of predicted and target clips.

    ``pred`` and ``target`` are ``[B,F,1,H,W]`` RSS/cal clips in ``[0,1]``.
    """
    if pred.ndim != 5 or target.ndim != 5 or pred.shape[1] <= 1:
        return pred.new_tensor(0.0)
    pred_flow = compute_kflow(pred[:, :-1, 0], pred[:, 1:, 0], delta=delta)
    with torch.no_grad():
        target_flow = compute_kflow(target[:, :-1, 0], target[:, 1:, 0], delta=delta)
    if clamp and clamp > 0:
        pred_flow = pred_flow.clamp(-float(clamp), float(clamp))
        target_flow = target_flow.clamp(-float(clamp), float(clamp))
    path_mask = refine_mask_by_target_flow(
        path_mask,
        target_flow,
        flow_threshold=target_flow_threshold,
        flow_quantile=target_flow_quantile,
    )
    return masked_flow_l1(pred_flow, target_flow, path_mask)


def warmup_weight(step: int, target_weight: float, warmup_steps: int) -> float:
    if target_weight <= 0:
        return 0.0
    if warmup_steps <= 0:
        return float(target_weight)
    return float(target_weight) * min(1.0, max(0.0, float(step) / float(warmup_steps)))
