"""Patch-wise cluster-aware fusion (PCAF).

PCAF clusters frozen dense RMDM priors independently in every spatial patch,
then constructs one fused sparse condition for every target frame.  Only
observed RSS values are transferred; dense priors are never copied into the
fused observation map.
"""

from __future__ import annotations

from dataclasses import dataclass
from functools import lru_cache
from itertools import combinations

import torch
import torch.nn.functional as F


@dataclass(frozen=True)
class PCAFDiagnostics:
    """Additive or per-target diagnostics produced by :func:`fuse_pcaf`."""

    original_observed_pixels: torch.Tensor  # [N,T]
    fused_observed_pixels: torch.Tensor  # [N,T]
    valid_pixels: torch.Tensor  # [N,T]
    singleton_patches: torch.Tensor  # [N,T]
    cluster_size_histogram: torch.Tensor  # [N,T,T+1]
    source_offset_histogram: torch.Tensor  # [N,T,2T-1]


@dataclass(frozen=True)
class PCAFResult:
    fused_rss: torch.Tensor  # [N,T,1,H,W]
    fused_mask: torch.Tensor  # [N,T,1,H,W]
    labels: torch.Tensor  # [N,R,T]
    medoids: torch.Tensor  # [N,R,K]
    diagnostics: PCAFDiagnostics


def _validate_inputs(
    prior: torch.Tensor,
    observed_rss: torch.Tensor,
    sampling_mask: torch.Tensor,
    valid_mask: torch.Tensor,
    *,
    patch_size: int,
    clusters: int,
    pooled_size: int,
) -> tuple[int, int, int, int]:
    if prior.ndim != 5:
        raise ValueError("PCAF inputs must be [N,T,1,H,W]")
    if any(value.shape != prior.shape for value in (observed_rss, sampling_mask, valid_mask)):
        raise ValueError("prior, observed_rss, sampling_mask and valid_mask must have identical shapes")
    batch, time, channels, height, width = prior.shape
    if channels != 1:
        raise ValueError("PCAF currently supports one-channel radio maps")
    if time < clusters or clusters <= 0:
        raise ValueError("clusters must be positive and no larger than the window length")
    if patch_size <= 0 or height % patch_size or width % patch_size:
        raise ValueError("patch_size must divide both spatial dimensions")
    if pooled_size <= 0 or pooled_size > patch_size:
        raise ValueError("pooled_size must be in [1, patch_size]")
    if not prior.is_floating_point():
        raise TypeError("PCAF inputs must be floating-point tensors")
    return batch, time, height, width


def _patchify(value: torch.Tensor, patch_size: int) -> torch.Tensor:
    """Convert ``[N,T,1,H,W]`` into ``[N,R,T,P]`` patches."""

    batch, time, _, height, width = value.shape
    grid_h, grid_w = height // patch_size, width // patch_size
    patches = value.reshape(batch, time, 1, grid_h, patch_size, grid_w, patch_size)
    patches = patches.permute(0, 3, 5, 1, 2, 4, 6).contiguous()
    return patches.reshape(batch, grid_h * grid_w, time, patch_size * patch_size)


def _unpatchify(patches: torch.Tensor, patch_size: int, height: int, width: int) -> torch.Tensor:
    """Convert ``[N,R,T,P]`` patches back to ``[N,T,1,H,W]``."""

    batch, regions, time, pixels = patches.shape
    grid_h, grid_w = height // patch_size, width // patch_size
    if regions != grid_h * grid_w or pixels != patch_size * patch_size:
        raise ValueError("Patch tensor is incompatible with the requested image shape")
    value = patches.reshape(batch, grid_h, grid_w, time, 1, patch_size, patch_size)
    value = value.permute(0, 3, 4, 1, 5, 2, 6).contiguous()
    return value.reshape(batch, time, 1, height, width)


def _descriptors(prior: torch.Tensor, *, patch_size: int, pooled_size: int, eps: float) -> torch.Tensor:
    """Return robust-standardised descriptors with shape ``[N,R,T,D]``."""

    patches = _patchify(prior, patch_size)
    batch, regions, time, _ = patches.shape
    images = patches.reshape(batch * regions * time, 1, patch_size, patch_size)
    pooled = F.adaptive_avg_pool2d(images, (pooled_size, pooled_size)).flatten(1)
    pooled = pooled.reshape(batch, regions, time, pooled_size * pooled_size)
    mean = patches.mean(dim=-1, keepdim=True)
    std = patches.std(dim=-1, unbiased=False, keepdim=True)
    descriptor = torch.cat((pooled, mean, std), dim=-1)

    # Standardise every descriptor coordinate independently within the W-frame
    # patch. Constant static structure therefore contributes zero distance.
    median = descriptor.median(dim=2, keepdim=True).values
    q25 = torch.quantile(descriptor.float(), 0.25, dim=2, keepdim=True).to(descriptor.dtype)
    q75 = torch.quantile(descriptor.float(), 0.75, dim=2, keepdim=True).to(descriptor.dtype)
    scale = (q75 - q25).clamp_min(float(eps))
    return (descriptor - median) / scale


@lru_cache(maxsize=16)
def _medoid_combinations(time: int, clusters: int) -> torch.Tensor:
    values = list(combinations(range(time), clusters))
    if not values:
        raise ValueError("No valid medoid combinations")
    return torch.tensor(values, dtype=torch.long)


def _exact_k_medoids(descriptor: torch.Tensor, clusters: int) -> tuple[torch.Tensor, torch.Tensor, torch.Tensor]:
    """Solve the small W-frame K-medoids problem exactly and deterministically."""

    batch, regions, time, _ = descriptor.shape
    distances = torch.cdist(descriptor.float(), descriptor.float(), p=2).square()
    choices = _medoid_combinations(time, clusters).to(device=descriptor.device)
    # [N,R,T,C,K] -> objective [N,R,C]. torch.argmin selects the first
    # lexicographic medoid tuple on exact ties.
    candidate_distances = distances[..., choices]
    objectives = candidate_distances.amin(dim=-1).sum(dim=2)
    selected = objectives.argmin(dim=-1)
    medoids = choices[selected]
    assigned_distances = distances.gather(
        dim=-1,
        index=medoids.unsqueeze(2).expand(batch, regions, time, clusters),
    )
    labels = assigned_distances.argmin(dim=-1)
    return labels, medoids, distances


def _source_orders(distances: torch.Tensor) -> torch.Tensor:
    """Order sources by target-first, distance, temporal offset, frame index."""

    batch, regions, time, _ = distances.shape
    source = torch.arange(time, device=distances.device)
    orders = []
    for target in range(time):
        # Stable sorts implement the exact lexicographic tie-break without
        # perturbing genuinely close floating-point distances.
        temporal_order = torch.argsort((source - target).abs(), stable=True)
        temporal_distances = distances[:, :, target, temporal_order]
        distance_order = torch.argsort(temporal_distances, dim=-1, stable=True)
        order = temporal_order[distance_order]
        is_target = order == target
        order = torch.cat((order[is_target].reshape(batch, regions, 1), order[~is_target].reshape(batch, regions, time - 1)), dim=-1)
        orders.append(order)
    return torch.stack(orders, dim=2)


def fuse_pcaf(
    prior: torch.Tensor,
    observed_rss: torch.Tensor,
    sampling_mask: torch.Tensor,
    valid_mask: torch.Tensor,
    *,
    patch_size: int = 16,
    clusters: int = 3,
    pooled_size: int = 4,
    standardize_eps: float = 1.0e-6,
    max_distance: float | None = None,
) -> PCAFResult:
    """Fuse W-frame sparse observations using local prior-state clusters.

    The clustering is shared by all targets in a window. For each target and
    pixel, the first observed source in the target's cluster is selected under
    the deterministic target-first patch-distance ordering.
    """

    batch, time, height, width = _validate_inputs(
        prior,
        observed_rss,
        sampling_mask,
        valid_mask,
        patch_size=patch_size,
        clusters=clusters,
        pooled_size=pooled_size,
    )
    if max_distance is not None and max_distance < 0:
        raise ValueError("max_distance must be non-negative or None")
    descriptors = _descriptors(prior, patch_size=patch_size, pooled_size=pooled_size, eps=standardize_eps)
    labels, medoids, distances = _exact_k_medoids(descriptors, clusters)
    source_orders = _source_orders(distances)

    rss_patches = _patchify(observed_rss, patch_size)
    mask_patches = _patchify(sampling_mask, patch_size) > 0.5
    valid_patches = _patchify(valid_mask, patch_size) > 0.5
    _, regions, _, pixels = rss_patches.shape
    fused_rss_targets = []
    fused_mask_targets = []
    offset_histograms = []
    cluster_size_histograms = []
    singleton_counts = []
    for target in range(time):
        order = source_orders[:, :, target]
        ordered_masks = mask_patches.gather(2, order.unsqueeze(-1).expand(batch, regions, time, pixels))
        ordered_rss = rss_patches.gather(2, order.unsqueeze(-1).expand(batch, regions, time, pixels))
        target_label = labels[:, :, target : target + 1]
        same_cluster = (labels == target_label).gather(2, order)
        available = ordered_masks & same_cluster.unsqueeze(-1)
        if max_distance is not None:
            ordered_distances = distances[:, :, target].gather(2, order)
            available = available & (ordered_distances <= float(max_distance)).unsqueeze(-1)
        has_source = available.any(dim=2)
        first_rank = available.to(torch.int64).argmax(dim=2)
        selected_rss = ordered_rss.gather(2, first_rank.unsqueeze(2)).squeeze(2)
        selected_source = order.gather(2, first_rank).squeeze(2)

        target_valid = valid_patches[:, :, target]
        fused_mask = has_source & target_valid
        fused_rss = selected_rss * fused_mask.to(selected_rss.dtype)
        fused_mask_targets.append(fused_mask)
        fused_rss_targets.append(fused_rss)

        offsets = selected_source - target + (time - 1)
        offset_one_hot = F.one_hot(offsets.clamp(0, 2 * time - 2), num_classes=2 * time - 1)
        offset_histograms.append((offset_one_hot * fused_mask.unsqueeze(-1)).sum(dim=(1, 2)))

        cluster_sizes = (labels == target_label).sum(dim=2)
        cluster_size_histograms.append(F.one_hot(cluster_sizes, num_classes=time + 1).sum(dim=1))
        singleton_counts.append((cluster_sizes == 1).sum(dim=1))

    fused_rss_patches = torch.stack(fused_rss_targets, dim=2)
    fused_mask_patches = torch.stack(fused_mask_targets, dim=2)
    fused_rss = _unpatchify(fused_rss_patches, patch_size, height, width)
    fused_mask = _unpatchify(fused_mask_patches, patch_size, height, width).to(sampling_mask.dtype)
    fused_rss = fused_rss.to(observed_rss.dtype)

    diagnostics = PCAFDiagnostics(
        original_observed_pixels=sampling_mask.sum(dim=(2, 3, 4), dtype=torch.float64),
        fused_observed_pixels=fused_mask.sum(dim=(2, 3, 4), dtype=torch.float64),
        valid_pixels=valid_mask.sum(dim=(2, 3, 4), dtype=torch.float64),
        singleton_patches=torch.stack(singleton_counts, dim=1).to(torch.float64),
        cluster_size_histogram=torch.stack(cluster_size_histograms, dim=1).to(torch.float64),
        source_offset_histogram=torch.stack(offset_histograms, dim=1).to(torch.float64),
    )
    return PCAFResult(
        fused_rss=fused_rss,
        fused_mask=fused_mask,
        labels=labels,
        medoids=medoids,
        diagnostics=diagnostics,
    )
