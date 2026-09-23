"""Original RMDM with an extra known-variance condition channel."""

from __future__ import annotations

import torch
from torch import nn

from train_sparse_dynamic_rmdm import build_model_config
from utils import build_unet_from_config


def build_model(train_args, *, reference_variance: float = 0.0081) -> "DirectNoiseRMDM":
    config = build_model_config(train_args)
    config["in_ch"] = 7
    return DirectNoiseRMDM(build_unet_from_config(config), reference_variance)


class DirectNoiseRMDM(nn.Module):
    def __init__(self, backbone: nn.Module, reference_variance: float) -> None:
        super().__init__()
        self.backbone = backbone
        self.reference_variance = reference_variance

    def encode_conditions(self, sparse_batch: dict) -> torch.Tensor:
        building = sparse_batch["building"][:, 0]
        zeros = torch.zeros_like(building)
        variance = sparse_batch["measurement_variance"].reshape(-1, 1, 1, 1)
        variance_map = (variance / self.reference_variance).expand_as(building)
        return torch.cat((building, zeros, sparse_batch["vehicle"][:, 0],
                          sparse_batch["observed_rss"][:, 0],
                          sparse_batch["sampling_mask"][:, 0], variance_map), dim=1)

    def forward(self, conditions: torch.Tensor, noisy_target: torch.Tensor,
                diffusion_step: torch.Tensor) -> tuple[torch.Tensor, torch.Tensor]:
        return self.backbone(torch.cat((conditions, noisy_target), dim=1), diffusion_step)[:2]

    def denoise(self, noisy_target: torch.Tensor, diffusion_step: torch.Tensor,
                condition_cache: torch.Tensor) -> torch.Tensor:
        return self.forward(condition_cache, noisy_target[:, 0], diffusion_step)[0].unsqueeze(1)


def load_original_weights(model: DirectNoiseRMDM, old_state: dict) -> None:
    """Inflate only the two input convolutions; zero variance reproduces old RMDM."""
    current = model.backbone.state_dict()
    for name, target in current.items():
        source = old_state[name]
        if source.shape == target.shape:
            current[name] = source
        elif name == "unet.input_blocks.0.0.weight":
            expanded = torch.zeros_like(target)
            expanded[:, :5] = source[:, :5]
            expanded[:, 6] = source[:, 5]
            current[name] = expanded
        elif name == "unet.hwm.conv_blocks_context.0.blocks.0.conv.weight":
            expanded = torch.zeros_like(target)
            expanded[:, :5] = source
            current[name] = expanded
        else:
            raise ValueError(f"Unexpected checkpoint tensor shape: {name}")
    model.backbone.load_state_dict(current, strict=True)
