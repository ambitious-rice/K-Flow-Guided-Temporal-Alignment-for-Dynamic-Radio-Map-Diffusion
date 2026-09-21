"""Noise-aware, Tx-blind T1 RMDM.

The observation path is confined to a variance-conditioned HWM branch. The
diffusion denoiser keeps the legacy ``UNetModel_newpreview`` topology and sees
only scene masks, the HWM calibration prior, normalized noise variance, and
the current diffusion state. No Tx tensor is accepted or used.
"""

from __future__ import annotations

from typing import Any

import torch
import torch.nn.functional as F
from torch import nn

from unet import UNetModel_newpreview

from .config import ExperimentConfig, ModelConfig


def _groups(channels: int) -> int:
    for groups in (32, 16, 8, 4, 2, 1):
        if channels % groups == 0:
            return groups
    return 1


class VarianceEmbedding(nn.Module):
    """Embed known observation variance ``v=sigma**2``."""

    def __init__(self, embedding_dim: int, hidden_dim: int, reference_variance: float) -> None:
        super().__init__()
        if reference_variance <= 0:
            raise ValueError("reference_variance must be positive")
        self.reference_variance = float(reference_variance)
        self.network = nn.Sequential(
            nn.Linear(3, hidden_dim),
            nn.SiLU(),
            nn.Linear(hidden_dim, hidden_dim),
            nn.SiLU(),
            nn.Linear(hidden_dim, embedding_dim),
        )

    def normalized_ratio(self, variance: torch.Tensor) -> torch.Tensor:
        return (variance.float() / self.reference_variance).clamp(max=1.0e6)

    def forward(self, variance: torch.Tensor) -> torch.Tensor:
        if variance.ndim == 2 and variance.shape[1] == 1:
            variance = variance[:, 0]
        if variance.ndim != 1:
            raise ValueError("measurement_variance must flatten to [N]")
        if not bool(torch.isfinite(variance).all()) or bool((variance < 0).any()):
            raise ValueError("measurement_variance must be finite and non-negative")
        ratio = self.normalized_ratio(variance)
        features = torch.stack((ratio, torch.log1p(ratio), (variance == 0).float()), dim=-1)
        return self.network(features)


class AdaptiveGroupNorm(nn.Module):
    def __init__(self, channels: int, embedding_dim: int) -> None:
        super().__init__()
        self.norm = nn.GroupNorm(_groups(channels), channels, affine=False)
        self.modulation = nn.Sequential(nn.SiLU(), nn.Linear(embedding_dim, 2 * channels))

    def forward(self, value: torch.Tensor, embedding: torch.Tensor) -> torch.Tensor:
        scale, shift = self.modulation(embedding).to(value.dtype).chunk(2, dim=1)
        return self.norm(value) * (1.0 + scale[:, :, None, None]) + shift[:, :, None, None]


class PlainResBlock(nn.Module):
    def __init__(self, in_channels: int, out_channels: int, dropout: float = 0.0) -> None:
        super().__init__()
        self.norm1 = nn.GroupNorm(_groups(in_channels), in_channels)
        self.conv1 = nn.Conv2d(in_channels, out_channels, 3, padding=1)
        self.norm2 = nn.GroupNorm(_groups(out_channels), out_channels)
        self.dropout = nn.Dropout(dropout)
        self.conv2 = nn.Conv2d(out_channels, out_channels, 3, padding=1)
        self.skip = nn.Identity() if in_channels == out_channels else nn.Conv2d(in_channels, out_channels, 1)

    def forward(self, value: torch.Tensor) -> torch.Tensor:
        hidden = self.conv1(F.silu(self.norm1(value)))
        hidden = self.conv2(self.dropout(F.silu(self.norm2(hidden))))
        return self.skip(value) + hidden


class VarianceResBlock(nn.Module):
    def __init__(self, channels: int, embedding_dim: int, dropout: float = 0.0) -> None:
        super().__init__()
        self.norm1 = AdaptiveGroupNorm(channels, embedding_dim)
        self.conv1 = nn.Conv2d(channels, channels, 3, padding=1)
        self.norm2 = AdaptiveGroupNorm(channels, embedding_dim)
        self.dropout = nn.Dropout(dropout)
        self.conv2 = nn.Conv2d(channels, channels, 3, padding=1)

    def forward(self, value: torch.Tensor, embedding: torch.Tensor) -> torch.Tensor:
        hidden = self.conv1(F.silu(self.norm1(value, embedding)))
        hidden = self.conv2(self.dropout(F.silu(self.norm2(hidden, embedding))))
        return value + hidden


class SceneEncoder(nn.Module):
    """Encode only building and current-frame vehicle masks."""

    def __init__(self, out_channels: int, dropout: float) -> None:
        super().__init__()
        self.network = nn.Sequential(
            nn.Conv2d(2, out_channels, 3, padding=1),
            PlainResBlock(out_channels, out_channels, dropout),
            PlainResBlock(out_channels, out_channels, dropout),
        )

    def forward(self, value: torch.Tensor) -> torch.Tensor:
        return self.network(value)


class VarianceConditionedEncoder(nn.Module):
    """The sole entry point for noisy RSS observations and their mask."""

    def __init__(self, out_channels: int, embedding_dim: int, dropout: float) -> None:
        super().__init__()
        self.input_projection = nn.Conv2d(2, out_channels, 3, padding=1)
        self.input_modulation = AdaptiveGroupNorm(out_channels, embedding_dim)
        self.input_conv = nn.Conv2d(out_channels, out_channels, 3, padding=1)
        self.blocks = nn.ModuleList([
            VarianceResBlock(out_channels, embedding_dim, dropout),
            VarianceResBlock(out_channels, embedding_dim, dropout),
        ])

    def forward(self, observation: torch.Tensor, variance_embedding: torch.Tensor) -> torch.Tensor:
        value = self.input_projection(observation)
        value = self.input_conv(F.silu(self.input_modulation(value, variance_embedding)))
        for block in self.blocks:
            value = block(value, variance_embedding)
        return value


class NoiseAwareHWM(nn.Module):
    """High-capacity condition branch producing a clean RSS calibration prior."""

    def __init__(self, config: ModelConfig) -> None:
        super().__init__()
        widths = [config.hwm_base_features * value for value in config.hwm_channel_multipliers]
        self.scene_stem = SceneEncoder(widths[0], config.dropout)
        self.observation_stem = VarianceConditionedEncoder(
            widths[0], config.variance_embedding_dim, config.dropout
        )
        self.encoder = nn.ModuleList([
            nn.Sequential(*[
                PlainResBlock(width, width, config.dropout)
                for _ in range(config.hwm_blocks_per_level)
            ]) for width in widths
        ])
        self.downsample = nn.ModuleList([
            nn.Conv2d(widths[index], widths[index + 1], 3, stride=2, padding=1)
            for index in range(len(widths) - 1)
        ])
        self.upsample = nn.ModuleList([
            nn.ConvTranspose2d(widths[index + 1], widths[index], 4, stride=2, padding=1)
            for index in range(len(widths) - 2, -1, -1)
        ])
        self.decoder = nn.ModuleList([
            nn.Sequential(
                PlainResBlock(2 * widths[index], widths[index], config.dropout),
                *[
                    PlainResBlock(widths[index], widths[index], config.dropout)
                    for _ in range(config.hwm_blocks_per_level - 1)
                ],
            ) for index in range(len(widths) - 2, -1, -1)
        ])
        self.calibration_head = nn.Sequential(
            nn.GroupNorm(_groups(widths[0]), widths[0]),
            nn.SiLU(),
            nn.Conv2d(widths[0], 1, 3, padding=1),
            nn.Sigmoid(),
        )

    def forward(
        self,
        scene: torch.Tensor,
        observation: torch.Tensor,
        variance_embedding: torch.Tensor,
    ) -> torch.Tensor:
        value = self.scene_stem(scene) + self.observation_stem(observation, variance_embedding)
        skips = []
        for index, blocks in enumerate(self.encoder):
            value = blocks(value)
            skips.append(value)
            if index < len(self.downsample):
                value = self.downsample[index](value)
        for module_index, level in enumerate(range(len(skips) - 2, -1, -1)):
            value = self.upsample[module_index](value)
            value = self.decoder[module_index](torch.cat((value, skips[level]), dim=1))
        return self.calibration_head(value)


class IdentityTemporalHook(nn.Module):
    """Stable [B,T,C,H,W] boundary for the later T16 stage."""

    def forward(self, stage: str, value: torch.Tensor) -> torch.Tensor:
        del stage
        return value


class LegacyDiffusionBackbone(nn.Module):
    """Exact legacy RMDM diffusion U-Net topology, without its old HWM."""

    def __init__(self, config: ModelConfig, image_size: int) -> None:
        super().__init__()
        attention_resolutions = tuple(2 ** int(level) for level in config.attention_levels)
        self.unet = UNetModel_newpreview(
            image_size=image_size,
            in_channels=5,
            model_channels=config.model_channels,
            out_channels=1,
            num_res_blocks=config.residual_blocks_per_level,
            attention_resolutions=attention_resolutions,
            dropout=config.dropout,
            channel_mult=tuple(config.channel_multipliers),
            num_classes=None,
            use_checkpoint=config.gradient_checkpointing,
            use_fp16=False,
            num_heads=config.attention_heads,
            num_head_channels=-1,
            num_heads_upsample=-1,
            use_scale_shift_norm=config.use_scale_shift_norm,
            resblock_updown=config.resblock_updown,
            use_new_attention_order=False,
            high_way=False,
        )

    def forward(self, value: torch.Tensor, timesteps: torch.Tensor) -> torch.Tensor:
        predicted_noise, _ = self.unet(value, timesteps)
        return predicted_noise


class NoiseAwareRMDM(nn.Module):
    """Framewise T1 model with explicit temporal-shaped caches and no Tx input."""

    def __init__(
        self,
        config: ModelConfig,
        *,
        image_size: int,
        reference_variance: float,
    ) -> None:
        super().__init__()
        self.variance_embedding = VarianceEmbedding(
            config.variance_embedding_dim, config.variance_mlp_width, reference_variance
        )
        self.hwm = NoiseAwareHWM(config)
        self.denoiser = LegacyDiffusionBackbone(config, image_size)
        self.temporal_hook: nn.Module = IdentityTemporalHook()

    @staticmethod
    def _flatten(value: torch.Tensor) -> torch.Tensor:
        return value.reshape(value.shape[0] * value.shape[1], *value.shape[2:])

    @staticmethod
    def _conditions(batch_data: dict[str, Any]) -> tuple[torch.Tensor, torch.Tensor, torch.Tensor]:
        required = ("building", "vehicle", "observed_rss", "sampling_mask", "measurement_variance")
        missing = [name for name in required if name not in batch_data]
        if missing:
            raise KeyError(f"noise-aware RMDM misses conditions: {missing}")
        reference = batch_data["building"]
        if reference.ndim != 5 or reference.shape[2] != 1:
            raise ValueError("image conditions must be [B,T,1,H,W]")
        for name in required[:-1]:
            if batch_data[name].shape != reference.shape:
                raise ValueError(f"condition {name!r} does not match building")
        scene = torch.cat((reference, batch_data["vehicle"]), dim=2)
        observation = torch.cat((batch_data["observed_rss"], batch_data["sampling_mask"]), dim=2)
        variance = batch_data["measurement_variance"]
        batch, time = reference.shape[:2]
        if variance.shape == (batch,):
            variance = variance[:, None].expand(batch, time)
        if variance.shape != (batch, time):
            raise ValueError(f"measurement_variance must be [B] or [B,T], got {tuple(variance.shape)}")
        return scene, observation, variance

    def encode_conditions(self, sparse_batch: dict[str, Any]) -> dict[str, Any]:
        scene, observation, variance = self._conditions(sparse_batch)
        batch, time, _, height, width = scene.shape
        variance_embedding = self.variance_embedding(variance.reshape(-1))
        cal = self.hwm(self._flatten(scene), self._flatten(observation), variance_embedding)
        cal = cal.reshape(batch, time, 1, height, width)
        cal = self.temporal_hook("calibration", cal)
        variance_ratio = self.variance_embedding.normalized_ratio(variance)
        variance_map = variance_ratio[:, :, None, None, None].expand(batch, time, 1, height, width)
        # The backbone never receives raw observations or Tx. Measurement
        # evidence enters through the variance-aware clean calibration prior.
        condition = torch.cat((scene, cal, variance_map.to(cal.dtype)), dim=2)
        condition = self.temporal_hook("condition", condition)
        return {"batch": batch, "time": time, "condition": condition, "cal": cal}

    def denoise(
        self,
        noisy_target: torch.Tensor,
        diffusion_step: torch.Tensor,
        condition_cache: dict[str, Any],
    ) -> torch.Tensor:
        if noisy_target.ndim != 5 or noisy_target.shape[2] != 1:
            raise ValueError("noisy_target must be [B,T,1,H,W]")
        batch, time = noisy_target.shape[:2]
        if (batch, time) != (condition_cache["batch"], condition_cache["time"]):
            raise ValueError("noisy target and condition cache B/T differ")
        if diffusion_step.shape != (batch,):
            raise ValueError("diffusion_step must contain one timestep per batch item")
        model_input = torch.cat((condition_cache["condition"], noisy_target), dim=2)
        output = self.denoiser(self._flatten(model_input), diffusion_step.repeat_interleave(time))
        output = output.reshape(batch, time, 1, *output.shape[-2:])
        return self.temporal_hook("denoised", output)

    def forward(
        self,
        noisy_target: torch.Tensor,
        diffusion_step: torch.Tensor,
        sparse_batch: dict[str, Any],
    ) -> tuple[torch.Tensor, torch.Tensor]:
        cache = self.encode_conditions(sparse_batch)
        return self.denoise(noisy_target, diffusion_step, cache), cache["cal"]


def build_model(config: ExperimentConfig) -> NoiseAwareRMDM:
    return NoiseAwareRMDM(
        config.model,
        image_size=config.data.image_size,
        reference_variance=config.measurement_noise.reference_variance,
    )


def parameter_counts(module: nn.Module) -> tuple[int, int]:
    trainable = sum(value.numel() for value in module.parameters() if value.requires_grad)
    total = sum(value.numel() for value in module.parameters())
    return trainable, total
