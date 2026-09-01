"""Top-level Stage1-conditioned unified-input JointRMDM."""

from __future__ import annotations

from typing import Any

import torch
from torch import nn

from rmdm.config import ExperimentConfig
from rmdm.legacy import build_stage1_hwm
from rmdm.models.stage1_prior import Stage1Prior

from .unified_denoiser import UnifiedJointDenoiser


class UnifiedJointRMDM(nn.Module):
    """Compute Stage1 once, then denoise from a named seven-channel cache."""

    def __init__(self, stage1_prior: Stage1Prior, joint_denoiser: UnifiedJointDenoiser) -> None:
        super().__init__()
        self.stage1_prior = stage1_prior
        self.joint_denoiser = joint_denoiser

    def set_stage1_trainable(self, trainable: bool) -> None:
        self.stage1_prior.set_trainable(trainable)

    def encode_conditions(self, sparse_batch: dict[str, Any]) -> dict[str, torch.Tensor]:
        required = ("building", "tx", "vehicle", "observed_rss", "sampling_mask")
        missing = [name for name in required if name not in sparse_batch]
        if missing:
            raise KeyError(f"Sparse batch misses unified conditions: {missing}")
        prior = self.stage1_prior(sparse_batch)
        cache = {name: sparse_batch[name] for name in required}
        cache["prior"] = prior
        return cache

    @staticmethod
    def ablate_raw_observations(
        condition_cache: dict[str, torch.Tensor],
    ) -> dict[str, torch.Tensor]:
        """Return a cache with raw ``Y/M`` removed while preserving Stage1 ``P``."""

        required = ("observed_rss", "sampling_mask", "prior")
        missing = [name for name in required if name not in condition_cache]
        if missing:
            raise KeyError(f"Condition cache misses ablation inputs: {missing}")
        ablated = dict(condition_cache)
        ablated["observed_rss"] = torch.zeros_like(condition_cache["observed_rss"])
        ablated["sampling_mask"] = torch.zeros_like(condition_cache["sampling_mask"])
        return ablated

    def denoise(
        self,
        noisy_target: torch.Tensor,
        diffusion_step: torch.Tensor,
        condition_cache: dict[str, torch.Tensor],
    ) -> torch.Tensor:
        return self.joint_denoiser(noisy_target, diffusion_step, condition_cache)

    def forward(
        self,
        noisy_target: torch.Tensor,
        diffusion_step: torch.Tensor,
        sparse_batch: dict[str, Any],
    ) -> torch.Tensor:
        return self.denoise(noisy_target, diffusion_step, self.encode_conditions(sparse_batch))


def build_unified_joint_rmdm(config: ExperimentConfig) -> UnifiedJointRMDM:
    hwm = build_stage1_hwm(config.stage1.checkpoint)
    stage1 = Stage1Prior(hwm, trainable=config.stage1.trainable, chunk_size=config.stage1.chunk_size)
    denoiser = UnifiedJointDenoiser(config.model)
    return UnifiedJointRMDM(stage1, denoiser)
