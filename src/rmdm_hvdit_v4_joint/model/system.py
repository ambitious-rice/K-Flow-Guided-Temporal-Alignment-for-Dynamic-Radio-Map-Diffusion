"""Jointly trainable HWM/cal branch and V4 token-to-pixel denoising protocol."""

from __future__ import annotations

from typing import Any

import torch
from torch import nn

from rmdm_hvdit_v4_joint.config import ExperimentConfig

from .hwm import TrainableHWM, build_hwm_from_scratch
from .hvdit_t1 import HvditT1
from .hvdit_w16 import HvditW16


class HvditSystem(nn.Module):
    def __init__(
        self,
        hwm: TrainableHWM,
        denoiser: HvditT1 | HvditW16,
        *,
        use_explicit_tx_condition: bool = True,
    ) -> None:
        super().__init__()
        self.hwm = hwm
        self.denoiser = denoiser
        self.use_explicit_tx_condition = bool(use_explicit_tx_condition)

    @staticmethod
    def _raw_conditions(sparse_batch: dict[str, Any]) -> dict[str, torch.Tensor]:
        required = ("building", "tx", "vehicle", "observed_rss", "sampling_mask")
        missing = [name for name in required if name not in sparse_batch]
        if missing:
            raise KeyError(f"sparse batch misses model conditions: {missing}")
        return {name: sparse_batch[name] for name in required}

    def encode_conditions(self, sparse_batch: dict[str, Any]) -> dict[str, torch.Tensor]:
        condition_batch = sparse_batch
        if not self.use_explicit_tx_condition:
            condition_batch = dict(sparse_batch)
            condition_batch["tx"] = torch.zeros_like(sparse_batch["tx"])
        raw = self._raw_conditions(condition_batch)
        hwm_cache = self.hwm(condition_batch)
        high, low = self.denoiser.encode_raw_conditions(raw)
        return {
            **raw,
            **hwm_cache,
            "condition_high": high,
            "condition_low": low,
        }

    def ablate_raw_observations(self, cache: dict[str, torch.Tensor]) -> dict[str, torch.Tensor]:
        """Ablate direct raw-observation paths while retaining the fixed HWM cache.

        This keeps the diagnostic aligned with V3: it measures the direct dual
        stem/pyramid contribution rather than rerunning a second HWM condition.
        """

        result = dict(cache)
        result["observed_rss"] = torch.zeros_like(cache["observed_rss"])
        result["sampling_mask"] = torch.zeros_like(cache["sampling_mask"])
        high, low = self.denoiser.encode_raw_conditions(result)
        result["condition_high"] = high
        result["condition_low"] = low
        return result

    def denoise(
        self,
        noisy_target: torch.Tensor,
        diffusion_step: torch.Tensor,
        condition_cache: dict[str, torch.Tensor],
    ) -> torch.Tensor:
        return self.denoiser(noisy_target, diffusion_step, condition_cache)

    def forward(
        self,
        noisy_target: torch.Tensor,
        diffusion_step: torch.Tensor,
        sparse_batch: dict[str, Any],
    ) -> tuple[torch.Tensor, torch.Tensor]:
        cache = self.encode_conditions(sparse_batch)
        return self.denoise(noisy_target, diffusion_step, cache), cache["cal"]


def _hwm(config: ExperimentConfig) -> TrainableHWM:
    return TrainableHWM(
        build_hwm_from_scratch(base_features=config.stage1.base_features),
        chunk_size=config.stage1.chunk_size,
    )


def build_t1_system(config: ExperimentConfig, *, attention_backend: str | None = None) -> HvditSystem:
    return HvditSystem(
        _hwm(config),
        HvditT1(
            config.model,
            attention_backend=attention_backend,
            gradient_checkpointing=config.t1_train.gradient_checkpointing,
        ),
        use_explicit_tx_condition=config.model.use_explicit_tx_condition,
    )


def build_w16_system(config: ExperimentConfig, *, attention_backend: str | None = None) -> HvditSystem:
    return HvditSystem(
        _hwm(config),
        HvditW16(config.model, attention_backend=attention_backend),
        use_explicit_tx_condition=config.model.use_explicit_tx_condition,
    )
