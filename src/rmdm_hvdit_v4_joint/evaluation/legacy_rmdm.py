"""Expose the legacy single-frame RMDM through the T1 sampling interface."""

from __future__ import annotations

from typing import Any

import torch
from torch import nn


class LegacyRMDMT1ProtocolAdapter(nn.Module):
    """Adapt RMDM-SF without changing its model-specific conditioning path.

    The surrounding evaluator owns the data, sparse-mask and DDIM-noise
    protocol.  This adapter only converts the shared ``[N,T,1,H,W]`` batch to
    the legacy RMDM ``[N,6,H,W]`` model input.
    """

    def __init__(self, model: nn.Module, *, without_tx: bool = False) -> None:
        super().__init__()
        self.model = model
        self.without_tx = bool(without_tx)

    def encode_conditions(self, sparse_batch: dict[str, Any]) -> torch.Tensor:
        names = ("building", "tx", "vehicle", "observed_rss", "sampling_mask")
        values = [sparse_batch[name] for name in names]
        reference = values[0]
        if reference.ndim != 5 or reference.shape[1:3] != (1, 1):
            raise ValueError("Legacy RMDM comparison requires [N,1,1,H,W] T1 conditions")
        if any(value.shape != reference.shape for value in values[1:]):
            raise ValueError("All T1 condition tensors must have identical shapes")
        conditions = torch.cat(values, dim=2)[:, 0].clone()
        if self.without_tx:
            conditions[:, 1].zero_()
        # Preserve the original RMDM building-plus-Tx preprocessing exactly.
        conditions[:, 0] = conditions[:, 0] + 10.0 * conditions[:, 1]
        return conditions

    def denoise(
        self,
        noisy_target: torch.Tensor,
        diffusion_step: torch.Tensor,
        condition_cache: torch.Tensor,
    ) -> torch.Tensor:
        if noisy_target.ndim != 5 or noisy_target.shape[1:3] != (1, 1):
            raise ValueError("Legacy RMDM comparison requires [N,1,1,H,W] noisy targets")
        model_input = torch.cat((condition_cache, noisy_target[:, 0]), dim=1)
        prediction = self.model(model_input, diffusion_step)
        if not isinstance(prediction, tuple) or not prediction:
            raise RuntimeError("Legacy RMDM must return (predicted_noise, calibration)")
        predicted_noise = prediction[0]
        expected = noisy_target[:, 0].shape
        if predicted_noise.shape != expected:
            raise RuntimeError(
                f"Legacy RMDM noise shape {tuple(predicted_noise.shape)} does not match {tuple(expected)}"
            )
        return predicted_noise.unsqueeze(1)
