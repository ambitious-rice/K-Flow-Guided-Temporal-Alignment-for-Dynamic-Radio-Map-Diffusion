"""Top-level Stage1-conditioned joint diffusion model."""

from __future__ import annotations

from typing import Any

import torch
from torch import nn

from rmdm.config import ExperimentConfig
from rmdm.legacy import build_stage1_hwm

from .condition_encoder import ConditionEncoder
from .joint_denoiser import JointDenoiser
from .stage1_prior import Stage1Prior


class JointRMDM(nn.Module):
    def __init__(
        self,
        stage1_prior: Stage1Prior,
        condition_encoder: ConditionEncoder,
        joint_denoiser: JointDenoiser,
    ) -> None:
        super().__init__()
        self.stage1_prior = stage1_prior
        self.condition_encoder = condition_encoder
        self.joint_denoiser = joint_denoiser

    def set_stage1_trainable(self, trainable: bool) -> None:
        self.stage1_prior.set_trainable(trainable)

    def encode_conditions(self, sparse_batch: dict[str, Any]) -> dict[str, torch.Tensor]:
        prior = self.stage1_prior(sparse_batch)
        condition = torch.cat(
            [
                sparse_batch["building"],
                sparse_batch["tx"],
                sparse_batch["vehicle"],
                sparse_batch["observed_rss"],
                sparse_batch["sampling_mask"],
                prior,
            ],
            dim=2,
        )
        cache = self.condition_encoder(condition)
        cache["prior"] = prior
        return cache

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


def build_joint_rmdm(config: ExperimentConfig) -> JointRMDM:
    hwm = build_stage1_hwm(config.stage1.checkpoint)
    stage1 = Stage1Prior(hwm, trainable=config.stage1.trainable, chunk_size=config.stage1.chunk_size)
    condition = ConditionEncoder(config.model.high_dim, config.model.bottleneck_dim, config.model.patch_size)
    denoiser = JointDenoiser(config.model)
    return JointRMDM(stage1, condition, denoiser)

