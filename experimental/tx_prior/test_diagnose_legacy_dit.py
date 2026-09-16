import pytest
import torch

from .diagnose_legacy_dit import old_config, zero_observations


def test_zero_observations_preserves_scene_and_target_without_mutation():
    batch = {key: torch.ones(2, 1, 1, 4, 4) for key in
             ("building", "tx", "vehicle", "target", "observed_rss", "sampling_mask")}
    zero = zero_observations(batch)
    for key in ("observed_rss", "sampling_mask", "sampling_rate"):
        assert torch.count_nonzero(zero[key]) == 0
    assert batch["observed_rss"].sum() > 0
    assert zero["target"] is batch["target"]
    assert zero["tx"] is batch["tx"]


def test_legacy_config_refuses_current_checkpoint():
    with pytest.raises(ValueError, match="historical x0"):
        old_config({"schema": "tx_prior_epsilon_scene_checkpoint_v1"}, None)
