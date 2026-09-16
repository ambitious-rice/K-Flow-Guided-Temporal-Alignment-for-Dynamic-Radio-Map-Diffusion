"""Build the TX-conditioned prior without denoiser observation branches."""

from __future__ import annotations

from typing import Any, Mapping

import torch
import torch.nn.functional as F
from torch import nn

from rmdm_hvdit_v4_joint.model import HvditSystem
from rmdm_hvdit_v4_joint.model.hvdit_t1 import HvditT1
from rmdm_hvdit_v4_joint.model.hwm import TrainableHWM, build_hwm_from_scratch


SCENE_KEYS = ("building", "tx", "vehicle")
OBSERVATION_KEYS = ("observed_rss", "sampling_mask")


class T1SceneStem(nn.Module):
    """Patch-project dense scene channels without an observation projection/fusion."""

    def __init__(self, dense_channels: int, dim: int, patch_size: int) -> None:
        super().__init__()
        self.patch_size = int(patch_size)
        self.dense_projection = nn.Linear(
            dense_channels * self.patch_size**2,
            dim,
            bias=False,
        )
        nn.init.xavier_uniform_(self.dense_projection.weight)

    def forward(
        self,
        dense: torch.Tensor,
        observation: torch.Tensor | None = None,
    ) -> torch.Tensor:
        del observation
        if dense.ndim != 5 or dense.shape[1] != 1:
            raise ValueError("T1 scene stem requires [B,1,C,H,W]")
        if dense.shape[-2] % self.patch_size or dense.shape[-1] % self.patch_size:
            raise ValueError("spatial dimensions must be divisible by patch_size")
        packed = F.pixel_unshuffle(dense[:, 0], self.patch_size).permute(0, 2, 3, 1)
        return self.dense_projection(packed).unsqueeze(1)


class ScenePriorDenoiser(HvditT1):
    """T1 denoiser whose input and condition stems contain scene channels only."""

    def __init__(
        self,
        config: Any,
        *,
        attention_backend: str | None = None,
        gradient_checkpointing: bool | None = None,
    ) -> None:
        super().__init__(
            config,
            attention_backend=attention_backend,
            gradient_checkpointing=gradient_checkpointing,
        )
        self.input_stem = T1SceneStem(4, config.local_dim, config.spatial_patch_size)
        self.condition_stem = T1SceneStem(3, config.local_dim, config.spatial_patch_size)

    @staticmethod
    def _validate_scene_conditions(raw: Mapping[str, torch.Tensor]) -> None:
        missing = [name for name in SCENE_KEYS if name not in raw]
        if missing:
            raise KeyError(f"scene condition cache misses {missing}")
        reference = raw["building"]
        if reference.ndim != 5 or reference.shape[1:3] != (1, 1):
            raise ValueError("scene conditions must be [B,1,1,H,W]")
        for name in SCENE_KEYS:
            if raw[name].shape != reference.shape:
                raise ValueError(f"scene condition {name!r} shape differs from building")

    def encode_raw_conditions(
        self,
        raw: Mapping[str, torch.Tensor],
    ) -> tuple[torch.Tensor, torch.Tensor]:
        self._validate_scene_conditions(raw)
        dense = torch.cat(
            (
                raw["building"] + 10.0 * raw["tx"],
                raw["tx"],
                raw["vehicle"],
            ),
            dim=2,
        )
        high = self.condition_stem(dense)
        return high, self.condition_merge(high)

    def build_inputs(
        self,
        noisy_target: torch.Tensor,
        cache: Mapping[str, torch.Tensor],
    ) -> tuple[torch.Tensor, None]:
        if noisy_target.ndim != 5 or noisy_target.shape[1:3] != (1, 1):
            raise ValueError("expected noisy target [B,1,1,H,W]")
        self._validate_scene_conditions(cache)
        dense = torch.cat(
            (
                noisy_target,
                cache["building"] + 10.0 * cache["tx"],
                cache["tx"],
                cache["vehicle"],
            ),
            dim=2,
        )
        return dense, None


def scene_prior_batch(batch: Mapping[str, Any]) -> dict[str, Any]:
    """Return a shallow batch copy with canonical, inert observation tensors.

    Zero-valued compatibility keys remain available to generic cache and metric
    interfaces, but the scene HWM and denoiser stems never consume them.
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
    """V4 epsilon system whose only external conditions are scene and explicit TX."""

    def __init__(self, hwm: nn.Module, denoiser: nn.Module) -> None:
        super().__init__(hwm, denoiser, use_explicit_tx_condition=True)

    def encode_conditions(self, batch: dict[str, Any]) -> dict[str, torch.Tensor]:
        scene_batch = scene_prior_batch(batch)
        raw = {name: scene_batch[name] for name in SCENE_KEYS}
        hwm_cache = self.hwm(scene_batch)
        high, low = self.denoiser.encode_raw_conditions(raw)
        return {
            **raw,
            **hwm_cache,
            "condition_high": high,
            "condition_low": low,
        }

    def denoise(
        self,
        noisy_target: torch.Tensor,
        diffusion_step: torch.Tensor,
        condition_cache: dict[str, torch.Tensor],
    ) -> torch.Tensor:
        return super().denoise(noisy_target, diffusion_step, condition_cache)


def build_scene_prior_system(config: Any, *, attention_backend: str | None = None) -> ScenePriorSystem:
    if not config.model.use_explicit_tx_condition:
        raise ValueError("tx_prior requires model.use_explicit_tx_condition=true")
    hwm = TrainableHWM(
        build_hwm_from_scratch(
            base_features=config.stage1.base_features,
            input_channels=3,
        ),
        chunk_size=config.stage1.chunk_size,
        input_channels=3,
    )
    denoiser = ScenePriorDenoiser(
        config.model,
        attention_backend=attention_backend,
        gradient_checkpointing=config.t1_train.gradient_checkpointing,
    )
    return ScenePriorSystem(hwm, denoiser)
