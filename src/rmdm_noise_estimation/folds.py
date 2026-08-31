"""Deterministic, frame-balanced observation folds."""

from __future__ import annotations

from typing import Any

import torch

from rmdm.data.sampling import derive_seed


FOLD_VERSION = "w16-observation-crossfit-frame-balanced-v1"


def assign_observation_folds(
    sampling_mask: torch.Tensor,
    frame_names: list[list[str]],
    *,
    folds: int,
    seed: int,
) -> torch.Tensor:
    """Assign every observed pixel to one balanced fold within its frame."""

    if sampling_mask.ndim != 5 or sampling_mask.shape[2] != 1:
        raise ValueError("sampling_mask must have shape [N,T,1,H,W]")
    if folds < 2:
        raise ValueError("folds must be at least two")
    if len(frame_names) != sampling_mask.shape[0]:
        raise ValueError("frame_names batch does not match sampling_mask")

    assignment = torch.full_like(sampling_mask, -1, dtype=torch.int16)
    for sample_index, names in enumerate(frame_names):
        if len(names) != sampling_mask.shape[1]:
            raise ValueError("frame_names time dimension does not match sampling_mask")
        for frame_index, frame_name in enumerate(names):
            observed = torch.nonzero(
                sampling_mask[sample_index, frame_index, 0].reshape(-1) > 0.5,
                as_tuple=False,
            ).flatten()
            if observed.numel() < folds:
                raise ValueError(f"frame {frame_name} has fewer observations than folds")
            generator = torch.Generator(device=sampling_mask.device)
            generator.manual_seed(derive_seed(FOLD_VERSION, seed, frame_name))
            order = observed[
                torch.randperm(observed.numel(), generator=generator, device=observed.device)
            ]
            labels = torch.arange(order.numel(), device=order.device) % folds
            assignment[sample_index, frame_index, 0].view(-1)[order] = labels.to(torch.int16)
    return assignment


def hide_fold(sparse_batch: dict[str, Any], assignment: torch.Tensor, fold: int) -> dict[str, Any]:
    """Remove a fold from every observation path without mutating the input."""

    if assignment.shape != sparse_batch["sampling_mask"].shape:
        raise ValueError("fold assignment shape differs from sampling mask")
    held_out = assignment == int(fold)
    if not bool(held_out.any()):
        raise ValueError(f"fold {fold} contains no observations")
    result = dict(sparse_batch)
    keep = (~held_out).to(sparse_batch["sampling_mask"].dtype)
    result["sampling_mask"] = sparse_batch["sampling_mask"] * keep
    result["observed_rss"] = sparse_batch["observed_rss"] * keep
    return result


__all__ = ["FOLD_VERSION", "assign_observation_folds", "hide_fold"]
