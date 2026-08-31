from __future__ import annotations

import torch

from rmdm_noise_estimation.folds import assign_observation_folds, hide_fold


def test_frame_balanced_folds_are_reproducible_and_complete() -> None:
    mask = torch.zeros((1, 2, 1, 4, 5))
    mask[0, 0, 0].view(-1)[:11] = 1
    mask[0, 1, 0].view(-1)[:9] = 1
    names = [["frame-a", "frame-b"]]

    first = assign_observation_folds(mask, names, folds=4, seed=7)
    second = assign_observation_folds(mask, names, folds=4, seed=7)

    assert torch.equal(first, second)
    assert torch.all(first[mask > 0.5] >= 0)
    assert torch.all(first[mask <= 0.5] == -1)
    for frame in range(2):
        counts = torch.bincount(first[0, frame][first[0, frame] >= 0].long(), minlength=4)
        assert int(counts.max() - counts.min()) <= 1


def test_hide_fold_removes_value_and_mask_without_mutation() -> None:
    mask = torch.ones((1, 1, 1, 2, 4))
    values = torch.arange(8.0).reshape_as(mask)
    batch = {"sampling_mask": mask, "observed_rss": values}
    assignment = torch.tensor([[[[[0, 1, 2, 3], [0, 1, 2, 3]]]]])

    hidden = hide_fold(batch, assignment, 2)

    held = assignment == 2
    assert torch.count_nonzero(hidden["sampling_mask"][held]) == 0
    assert torch.count_nonzero(hidden["observed_rss"][held]) == 0
    assert torch.equal(batch["sampling_mask"], mask)
    assert torch.equal(batch["observed_rss"], values)
