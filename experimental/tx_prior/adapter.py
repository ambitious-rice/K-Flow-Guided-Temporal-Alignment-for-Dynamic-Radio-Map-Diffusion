"""Isolate the TX-conditioned prior from every observation-conditioning path."""

from __future__ import annotations

from typing import Any, Mapping

import torch
from torch import nn

from rmdm_hvdit_v4_joint.model import HvditSystem
from rmdm_hvdit_v4_x0.model import build_t1_system


SCENE_KEYS = ("building", "tx", "vehicle")
OBSERVATION_KEYS = ("observed_rss", "sampling_mask")


def scene_prior_batch(batch: Mapping[str, Any]) -> dict[str, Any]:
    """Return a shallow batch copy with canonical, inert observation tensors.

    Keeping zero-valued keys preserves the unchanged five-channel HWM and dual
    stem shapes while making observations semantically absent.
    """

    missing = [name for name in SCENE_KEYS if name not in batch]
    if missing:
        raise KeyError(f"scene-prior batch misses conditions: {missing}")
    reference = batch["building"]
    if not torch.is_tensor(reference):
        raise TypeError("building must be a tensor")
    for name in SCENE_KEYS:
        value = batch[name]
        if not torch.is_tensor(value) or value.shape != reference.shape:
            raise ValueError(f"scene condition {name!r} shape differs from building")
    result = dict(batch)
    for name in OBSERVATION_KEYS:
        result[name] = torch.zeros_like(reference)
    return result


class ScenePriorSystem(HvditSystem):
    """V4 x0 system whose only external conditions are scene and explicit TX."""

    def __init__(self, hwm: nn.Module, denoiser: nn.Module) -> None:
        super().__init__(hwm, denoiser, use_explicit_tx_condition=True)

    def encode_conditions(self, batch: dict[str, Any]) -> dict[str, torch.Tensor]:
        return super().encode_conditions(scene_prior_batch(batch))

    def denoise(
        self,
        noisy_target: torch.Tensor,
        diffusion_step: torch.Tensor,
        condition_cache: dict[str, torch.Tensor],
    ) -> torch.Tensor:
        # Sanitize again so externally cached conditions cannot bypass the prior boundary.
        return super().denoise(
            noisy_target,
            diffusion_step,
            scene_prior_batch(condition_cache),
        )


def build_scene_prior_system(config: Any, *, attention_backend: str | None = None) -> ScenePriorSystem:
    if not config.model.use_explicit_tx_condition:
        raise ValueError("tx_prior requires model.use_explicit_tx_condition=true")
    base = build_t1_system(config, attention_backend=attention_backend)
    return ScenePriorSystem(base.hwm, base.denoiser)
