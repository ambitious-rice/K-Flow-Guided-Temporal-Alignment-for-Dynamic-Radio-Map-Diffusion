"""Distributed collection and evaluation for cross-fitted W16 priors."""

from __future__ import annotations

import json
import os
from pathlib import Path
from typing import Any

import torch
from torch.utils.data import DataLoader

from rmdm.data import WindowDataset
from rmdm.evaluation.fixed_sparse_protocol import (
    add_fixed_observation_noise,
    apply_fixed_sparse_observations,
    frame_names_by_sample,
)
from rmdm_hvdit_v4_joint.training.engine import append_jsonl, write_json_atomic

from .calibration import VarianceCalibration
from .ensemble import CrossfitDDIMEnsemble
from .folds import assign_observation_folds, hide_fold
from .statistics import estimate_window_noise


def balanced_manifest_videos(path: str | Path, stage: str, videos_per_scene: int) -> list[str]:
    with Path(path).open("r", encoding="utf-8") as handle:
        payload = json.load(handle)
    videos = payload[stage]["videos"]
    selected: list[str] = []
    counts: dict[str, int] = {}
    for item in videos:
        scene = str(item["scene_id"])
        if counts.get(scene, 0) >= videos_per_scene:
            continue
        selected.append(str(item["video_id"]))
        counts[scene] = counts.get(scene, 0) + 1
    if not counts or any(value != videos_per_scene for value in counts.values()):
        raise ValueError(f"manifest cannot provide {videos_per_scene} videos for every scene")
    return selected


def make_loader(accelerator: Any, config: Any, video_ids: list[str], starts: list[int]) -> Any:
    dataset = WindowDataset(
        root=config.data.root,
        split="val",
        split_file=config.data.split_file,
        window_size=16,
        seed=config.sampling.seed,
        cache_size=config.data.cache_size,
        tx_heatmap_sigma_px=config.data.tx_heatmap_sigma_px,
        fixed_starts=starts,
        video_ids=video_ids,
    )
    loader = DataLoader(
        dataset,
        batch_size=1,
        shuffle=False,
        num_workers=min(config.data.workers, 2),
        pin_memory=True,
        persistent_workers=False,
        drop_last=False,
    )
    return accelerator.prepare_data_loader(loader)


def unit_name(video_id: str, start: int, rate: float, sigma: float) -> str:
    video = video_id.replace("/", "__")
    return f"{video}__s{start:02d}__p{rate:g}__sigma{sigma:.3f}".replace(".", "p")


def _atomic_torch_save(path: Path, payload: dict[str, Any]) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    temporary = path.with_suffix(path.suffix + ".tmp")
    torch.save(payload, temporary)
    os.replace(temporary, path)


def crossfit_prior(
    accelerator: Any,
    model: Any,
    sampler: CrossfitDDIMEnsemble,
    sparse_batch: dict[str, Any],
    *,
    rate: float,
    folds: int,
    members: int,
    member_batch_size: int,
    steps: int,
    seed: int,
    namespace: str,
) -> tuple[torch.Tensor, torch.Tensor, torch.Tensor]:
    target = sparse_batch["target"]
    batch_size = target.shape[0]
    names = frame_names_by_sample(sparse_batch, batch_size=batch_size, window_size=16)
    assignment = assign_observation_folds(
        sparse_batch["sampling_mask"], names, folds=folds, seed=seed
    )
    prior_mean = torch.zeros_like(target)
    prior_variance = torch.zeros_like(target)
    for fold in range(folds):
        hidden = hide_fold(sparse_batch, assignment, fold)
        mean, variance = sampler.moments(
            model,
            hidden,
            frame_names=names,
            rate=rate,
            fold=fold,
            members=members,
            member_batch_size=member_batch_size,
            steps=steps,
            seed=seed,
            namespace=namespace,
            accelerator=accelerator,
        )
        held = assignment == fold
        prior_mean[held] = mean[held]
        prior_variance[held] = variance[held]
    return prior_mean, prior_variance, assignment


def _stack_sparse_batches(batches: list[dict[str, Any]]) -> dict[str, Any]:
    if not batches:
        raise ValueError("cannot stack an empty sparse batch list")
    result: dict[str, Any] = {}
    for name, value in batches[0].items():
        if torch.is_tensor(value):
            result[name] = torch.cat([batch[name] for batch in batches], dim=0)
        elif name == "frame_names":
            # Default collation stores frame names transposed as T lists of B strings.
            result[name] = [
                tuple(batch[name][frame_index][0] for batch in batches)
                for frame_index in range(len(value))
            ]
        elif isinstance(value, list):
            result[name] = [item for batch in batches for item in batch[name]]
        else:
            result[name] = value
    return result


def _slice_sparse_batch(batch: dict[str, Any], index: int) -> dict[str, Any]:
    result: dict[str, Any] = {}
    for name, value in batch.items():
        if torch.is_tensor(value):
            result[name] = value[index : index + 1]
        elif name == "frame_names":
            result[name] = [tuple([value[frame][index]]) for frame in range(len(value))]
        elif isinstance(value, list):
            result[name] = [value[index]]
        else:
            result[name] = value
    return result


def _selected_vectors(
    sparse_batch: dict[str, Any], prior_mean: torch.Tensor, prior_variance: torch.Tensor
) -> dict[str, torch.Tensor]:
    selected = sparse_batch["sampling_mask"] > 0.5
    return {
        "flat_indices": torch.nonzero(selected.reshape(-1), as_tuple=False)
        .flatten()
        .detach()
        .cpu(),
        "observed": sparse_batch["observed_rss"][selected].detach().float().cpu(),
        "target": sparse_batch["target"][selected].detach().float().cpu(),
        "prior_mean": prior_mean[selected].detach().float().cpu(),
        "raw_variance": prior_variance[selected].detach().float().cpu(),
    }


def _estimate_payload(vectors: dict[str, torch.Tensor], calibration: dict[str, Any]) -> dict[str, Any]:
    observed = vectors["observed"].numpy()
    target = vectors["target"].numpy()
    mean = vectors["prior_mean"].numpy()
    raw_variance = vectors["raw_variance"].numpy()
    calibrated = VarianceCalibration(**calibration["variance_calibration"]).apply(raw_variance)
    constant = torch.full_like(
        vectors["raw_variance"], float(calibration["constant_variance"])
    ).numpy()
    raw = estimate_window_noise(observed, mean, raw_variance)
    constant_result = estimate_window_noise(observed, mean, constant)
    primary = estimate_window_noise(observed, mean, calibrated)
    residual = target - mean
    standardized = residual / calibrated**0.5
    methods = {
        "naive": {"sigma": float(((observed - mean) ** 2).mean() ** 0.5)},
        "raw_ensemble_mle": raw.__dict__,
        "constant_variance_mle": constant_result.__dict__,
        "calibrated_ensemble_mle": primary.__dict__,
        "finite_sample_oracle": {"sigma": float(((observed - target) ** 2).mean() ** 0.5)},
    }
    return {
        "methods": methods,
        "prior_diagnostics": {
            "mean_bias": float(residual.mean()),
            "standardized_mean": float(standardized.mean()),
            "standardized_second_moment": float((standardized**2).mean()),
            "coverage_68": float((abs(standardized) <= 1.0).mean()),
            "coverage_95": float((abs(standardized) <= 1.96).mean()),
            "mean_raw_variance": float(raw_variance.mean()),
            "mean_calibrated_variance": float(calibrated.mean()),
        },
        "observation_count": int(observed.size),
        "outside_unit_interval_fraction": float(((observed < 0.0) | (observed > 1.0)).mean()),
    }


def run_units(
    accelerator: Any,
    model: Any,
    config: Any,
    loader: Any,
    *,
    output_dir: str | Path,
    rates: list[float],
    noise_stds: list[float],
    folds: int,
    members: int,
    member_batch_size: int,
    steps: int,
    seed: int,
    namespace: str,
    mode: str,
    calibration: dict[str, Any] | None,
    max_units_per_rank: int | None = None,
    sigma_batch_size: int = 1,
) -> None:
    if mode not in {"collect", "evaluate"}:
        raise ValueError("mode must be collect or evaluate")
    if mode == "evaluate" and calibration is None:
        raise ValueError("evaluation requires a variance calibration")
    root = Path(output_dir).resolve()
    rank_dir = root / "units" / f"rank_{accelerator.process_index:02d}"
    progress = root / "progress" / f"rank_{accelerator.process_index:02d}.jsonl"
    rank_dir.mkdir(parents=True, exist_ok=True)
    sampler = CrossfitDDIMEnsemble(config.diffusion)
    model.eval()
    completed = 0
    for rate in rates:
        for dense_batch in loader:
            clean = apply_fixed_sparse_observations(dense_batch, rate=rate, split="val")
            video_id = str(clean["video_id"][0])
            start_value = clean["start"]
            start = int(start_value[0] if torch.is_tensor(start_value) else start_value[0])
            suffix = ".pt" if mode == "collect" else ".json"
            pending = []
            for sigma in noise_stds:
                name = unit_name(video_id, start, rate, sigma)
                complete = (rank_dir / (name + suffix)).exists()
                if mode == "evaluate":
                    complete = complete and (
                        root / "vectors" / f"rank_{accelerator.process_index:02d}" / (name + ".pt")
                    ).exists()
                if not complete:
                    pending.append(sigma)
            for sigma_start in range(0, len(pending), sigma_batch_size):
                sigma_group = pending[sigma_start : sigma_start + sigma_batch_size]
                sparse = _stack_sparse_batches(
                    [
                        add_fixed_observation_noise(
                            clean,
                            standard_deviation=sigma,
                            rate=rate,
                            seed=seed,
                        )
                        for sigma in sigma_group
                    ]
                )
                mean, variance, assignment = crossfit_prior(
                    accelerator,
                    model,
                    sampler,
                    sparse,
                    rate=rate,
                    folds=folds,
                    members=members,
                    member_batch_size=member_batch_size,
                    steps=steps,
                    seed=seed,
                    namespace=namespace,
                )
                for sample_index, sigma in enumerate(sigma_group):
                    destination = rank_dir / (unit_name(video_id, start, rate, sigma) + suffix)
                    sample = _slice_sparse_batch(sparse, sample_index)
                    vectors = _selected_vectors(
                        sample,
                        mean[sample_index : sample_index + 1],
                        variance[sample_index : sample_index + 1],
                    )
                    metadata = {
                        "video_id": video_id,
                        "scene_id": video_id.split("/", 1)[0],
                        "start": start,
                        "rate": float(rate),
                        "true_sigma": float(sigma),
                        "folds": folds,
                        "members": members,
                        "ddim_steps": steps,
                        "rank": accelerator.process_index,
                    }
                    if mode == "collect":
                        _atomic_torch_save(
                            destination,
                            {
                                "schema": "w16_noise_calibration_unit_v1",
                                **metadata,
                                **vectors,
                                "fold_counts": torch.bincount(
                                    assignment[sample_index][assignment[sample_index] >= 0].long(),
                                    minlength=folds,
                                ).cpu(),
                            },
                        )
                    else:
                        _atomic_torch_save(
                            root
                            / "vectors"
                            / f"rank_{accelerator.process_index:02d}"
                            / (unit_name(video_id, start, rate, sigma) + ".pt"),
                            {
                                "schema": "w16_noise_estimation_vectors_v1",
                                **metadata,
                                **vectors,
                            },
                        )
                        write_json_atomic(
                            destination,
                            {
                                "schema": "w16_noise_estimation_unit_v1",
                                **metadata,
                                **_estimate_payload(vectors, calibration or {}),
                            },
                        )
                    append_jsonl(progress, {**metadata, "output": str(destination)})
                    completed += 1
                    print(
                        f"[noise-estimation] rank={accelerator.process_index} "
                        f"video={video_id} start={start} p={rate:g} sigma={sigma:g}",
                        flush=True,
                    )
                    if max_units_per_rank is not None and completed >= max_units_per_rank:
                        write_json_atomic(
                            root / "progress" / f"rank_{accelerator.process_index:02d}.complete.json",
                            {"status": "limited_complete", "rank": accelerator.process_index, "units": completed},
                        )
                        return
    write_json_atomic(
        root / "progress" / f"rank_{accelerator.process_index:02d}.complete.json",
        {"status": "complete", "rank": accelerator.process_index},
    )


__all__ = [
    "balanced_manifest_videos",
    "crossfit_prior",
    "make_loader",
    "run_units",
    "unit_name",
]
