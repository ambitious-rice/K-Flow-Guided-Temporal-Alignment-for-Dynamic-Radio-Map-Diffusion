from __future__ import annotations

import torch

from rmdm_hvdit_v4_x0_observation_balance.augmentation import (
    observation_alignment_weight,
    prepare_observation_training_batch,
)
from rmdm_hvdit_v4_x0_observation_balance.config import AlignmentConfig, NoiseConfig


def _batch() -> dict[str, object]:
    target = torch.arange(2 * 1 * 1 * 8 * 8, dtype=torch.float32).reshape(2, 1, 1, 8, 8) / 100
    mask = torch.zeros_like(target)
    mask[0, 0, 0].view(-1)[:8] = 1
    mask[1, 0, 0].view(-1)[8:16] = 1
    return {
        "target": target,
        "sampling_mask": mask,
        "observed_rss": mask * target,
        "valid_mask": torch.ones_like(target),
        "video_id": ["scene/a", "scene/b"],
        "start": torch.tensor([3, 7]),
    }


def test_observation_training_batch_is_repeatable_and_disjoint() -> None:
    noise = NoiseConfig(standard_deviations=[0.05], probabilities=[1.0])
    alignment = AlignmentConfig(
        heldout_ratio=0.25,
        condition_dropout_probability=0.0,
    )
    first = prepare_observation_training_batch(
        _batch(), seed=17, epoch=2, noise_config=noise, alignment_config=alignment
    )
    second = prepare_observation_training_batch(
        _batch(), seed=17, epoch=2, noise_config=noise, alignment_config=alignment
    )
    assert torch.equal(first.sparse_batch["observed_rss"], second.sparse_batch["observed_rss"])
    assert torch.equal(first.heldout_mask, second.heldout_mask)
    assert not torch.any((first.input_mask > 0.5) & (first.heldout_mask > 0.5))
    assert first.heldout_mask.flatten(1).sum(1).tolist() == [2.0, 2.0]


def test_condition_dropout_removes_only_model_observations() -> None:
    result = prepare_observation_training_batch(
        _batch(),
        seed=3,
        epoch=0,
        noise_config=NoiseConfig(standard_deviations=[0.0], probabilities=[1.0]),
        alignment_config=AlignmentConfig(condition_dropout_probability=0.999999),
    )
    assert torch.count_nonzero(result.input_mask) == 0
    assert torch.count_nonzero(result.sparse_batch["observed_rss"]) == 0
    assert torch.count_nonzero(result.heldout_mask) > 0
    assert torch.count_nonzero(result.sparse_batch["target"]) > 0


def test_alignment_weight_schedule() -> None:
    config = AlignmentConfig(
        warmup_steps=500,
        hold_steps=4000,
        peak_weight=0.25,
        final_weight=0.05,
    )
    assert observation_alignment_weight(0, max_steps=12000, config=config) == 0.0
    assert observation_alignment_weight(250, max_steps=12000, config=config) == 0.125
    assert observation_alignment_weight(500, max_steps=12000, config=config) == 0.25
    assert observation_alignment_weight(4000, max_steps=12000, config=config) == 0.25
    assert abs(observation_alignment_weight(12000, max_steps=12000, config=config) - 0.05) < 1e-12
