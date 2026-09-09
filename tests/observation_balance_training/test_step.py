from __future__ import annotations

from types import SimpleNamespace

import torch

from rmdm_hvdit_v4_x0_observation_balance.config import AlignmentConfig, NoiseConfig
from rmdm_hvdit_v4_x0_observation_balance.step import training_step


class _Policy:
    def __call__(self, dense_batch):
        result = dict(dense_batch)
        mask = torch.zeros_like(result["target"])
        mask[..., 0, 0] = 1
        mask[..., 1, 1] = 1
        result.update(
            {
                "valid_mask": torch.ones_like(mask),
                "sampling_mask": mask,
                "observed_rss": mask * result["target"],
                "sampling_rate": torch.ones((result["target"].shape[0], 1)),
            }
        )
        return result


class _Diffusion:
    def __init__(self):
        self.scheduler = SimpleNamespace(alphas_cumprod=torch.tensor([0.9, 0.8]))

    def training_batch(self, target, *, seeds):
        del seeds
        return SimpleNamespace(
            noisy_target=target.clone(),
            timesteps=torch.ones(target.shape[0], dtype=torch.long),
            noise=torch.zeros_like(target),
        )


class _Model(torch.nn.Module):
    def __init__(self):
        super().__init__()
        self.scale = torch.nn.Parameter(torch.tensor(0.9))

    def forward(self, noisy_target, timesteps, sparse_batch):
        del timesteps, sparse_batch
        value = self.scale * noisy_target
        return value, value


def test_balanced_training_step_has_finite_gradient() -> None:
    target = torch.linspace(0.1, 1.0, 2 * 1 * 1 * 8 * 8).reshape(2, 1, 1, 8, 8)
    zeros = torch.zeros_like(target)
    dense = {
        "target": target,
        "building": zeros,
        "vehicle": zeros,
        "tx": zeros,
        "video_id": ["scene/a", "scene/b"],
        "start": torch.tensor([0, 1]),
    }
    model = _Model()
    result = training_step(
        model,
        dense,
        _Policy(),
        _Diffusion(),
        training_seed=7,
        epoch=0,
        branch_step=1,
        observation_alignment_weight=0.25,
        heldout_weight=0.1,
        noise_config=NoiseConfig(standard_deviations=[0.05], probabilities=[1.0]),
        alignment_config=AlignmentConfig(
            heldout_ratio=0.5,
            condition_dropout_probability=0.0,
        ),
        pinn_k=0.2,
        pinn_weight=0.0,
    )
    assert torch.isfinite(result.loss)
    assert result.observation_alignment_loss > 0
    assert result.heldout_loss > 0
    assert result.noisy_condition_fraction == 1
    result.loss.backward()
    assert model.scale.grad is not None
    assert torch.isfinite(model.scale.grad)
