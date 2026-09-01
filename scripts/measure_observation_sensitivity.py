#!/usr/bin/env python3
"""Measure finite-difference prediction sensitivity to one RSS observation.

This is an evaluation-only diagnostic.  For every selected validation frame it
keeps the sparse mask and DDIM initial noise fixed, perturbs one visible RSS
value by ``delta``, and records ``(x_hat(y + delta e_i) - x_hat(y)) / delta``.
The same frames, masks, selected pixels, and initial noise are used for the
RMDM-SF U-Net, the original V4 x0 DiT, and the observation-alignment DiT.

The legacy V4 modules are intentionally imported at runtime: this script does
not define, train, or alter any model architecture.
"""

from __future__ import annotations

import argparse
import csv
import json
import math
from dataclasses import asdict, dataclass
from pathlib import Path
from typing import Any

import torch
from diffusers import DDIMScheduler
from torch.utils.data import DataLoader

from rmdm.data import WindowDataset
from rmdm.evaluation.fixed_sparse_protocol import (
    apply_fixed_sparse_observations,
    deterministic_frame_noise_like,
    frame_names_by_sample,
)
from rmdm_hvdit_v4_joint.evaluation.evaluator import manifest_video_ids
from rmdm_hvdit_v4_joint.model import build_t1_system
from rmdm_hvdit_v4_joint.evaluation.legacy_rmdm import LegacyRMDMT1ProtocolAdapter
from rmdm_hvdit_v4_x0_continue import ARCHITECTURE_ID
from rmdm_hvdit_v4_x0_continue.checkpoint import CONTINUATION_SCHEMA
from rmdm_hvdit_v4_x0_continue.config import load_config
from train_sparse_dynamic_rmdm import build_model_config
from utils import build_unet_from_config


RING_BOUNDS = (0.0, 1.0, 2.0, 4.0, 8.0, 16.0, 32.0, float("inf"))


@dataclass(frozen=True)
class SelectedPoint:
    sample_index: int
    row: int
    column: int
    video_id: str
    start: str


def _to_device(value: Any, device: torch.device) -> Any:
    if isinstance(value, torch.Tensor):
        return value.to(device, non_blocking=True)
    if isinstance(value, dict):
        return {name: _to_device(item, device) for name, item in value.items()}
    if isinstance(value, tuple):
        return tuple(_to_device(item, device) for item in value)
    if isinstance(value, list):
        return [_to_device(item, device) for item in value]
    return value


def _clone_batch(batch: dict[str, Any]) -> dict[str, Any]:
    return {
        name: value.clone() if isinstance(value, torch.Tensor) else value
        for name, value in batch.items()
    }


def _load_dit(config: Any, checkpoint_path: Path, device: torch.device) -> torch.nn.Module:
    payload = torch.load(checkpoint_path, map_location="cpu", weights_only=False, mmap=True)
    if (
        payload.get("schema") != CONTINUATION_SCHEMA
        or payload.get("architecture_id") != ARCHITECTURE_ID
        or "model" not in payload
    ):
        raise ValueError(f"not a V4 T1 checkpoint: {checkpoint_path}")
    model = build_t1_system(config)
    model.load_state_dict(payload["model"], strict=True)
    del payload
    return model.to(device).eval().requires_grad_(False)


def _load_unet(checkpoint_path: Path, device: torch.device) -> torch.nn.Module:
    payload = torch.load(checkpoint_path, map_location="cpu", weights_only=False, mmap=True)
    if payload.get("schema") != "rmdm_sf_sparse_checkpoint_v1" or "model" not in payload:
        raise ValueError(f"not an RMDM-SF checkpoint: {checkpoint_path}")
    train_args = argparse.Namespace(**payload["args"])
    model = build_unet_from_config(build_model_config(train_args))
    model.load_state_dict(payload["model"], strict=True)
    without_tx = bool(getattr(train_args, "without_tx", False))
    del payload
    return LegacyRMDMT1ProtocolAdapter(model, without_tx=without_tx).to(device).eval().requires_grad_(False)


@torch.no_grad()
def _ddim_sample(
    model: torch.nn.Module,
    scheduler: DDIMScheduler,
    sparse_batch: dict[str, Any],
    initial_noise: torch.Tensor,
    *,
    steps: int,
) -> torch.Tensor:
    condition_cache = model.encode_conditions(sparse_batch)
    sample = initial_noise.clone()
    scheduler.set_timesteps(steps, device=sample.device)
    for timestep in scheduler.timesteps:
        times = torch.full(
            (sample.shape[0],), int(timestep), device=sample.device, dtype=torch.long
        )
        predicted_noise = model.denoise(scheduler.scale_model_input(sample, timestep), times, condition_cache)
        sample = scheduler.step(
            predicted_noise,
            timestep,
            sample,
            eta=0.0,
            use_clipped_model_output=False,
            return_dict=False,
        )[0]
    return sample.clamp(0.0, 1.0)


def _value_at(tensor: torch.Tensor, sample_index: int, row: int, column: int) -> float:
    return float(tensor[sample_index, ..., row, column].reshape(-1)[0].item())


def _select_points(sparse_batch: dict[str, Any], seed: int) -> list[SelectedPoint]:
    mask = sparse_batch["sampling_mask"] > 0.5
    height, width = mask.shape[-2:]
    mask_2d = mask.reshape(mask.shape[0], -1, height, width)[:, 0]
    points: list[SelectedPoint] = []
    for sample_index in range(mask.shape[0]):
        candidates = torch.nonzero(mask_2d[sample_index], as_tuple=False)
        if len(candidates) == 0:
            raise RuntimeError(f"sample {sample_index} has no visible observations")
        margin = torch.minimum(
            torch.minimum(candidates[:, 0], candidates[:, 1]),
            torch.minimum(height - 1 - candidates[:, 0], width - 1 - candidates[:, 1]),
        )
        interior = candidates[margin >= 8]
        pool = interior if len(interior) else candidates
        generator = torch.Generator(device="cpu").manual_seed(seed + sample_index)
        selected = pool[torch.randint(len(pool), (1,), generator=generator).item()]
        video_id = str(sparse_batch.get("video_id", [""] * mask.shape[0])[sample_index])
        start_value = sparse_batch.get("start", [""] * mask.shape[0])[sample_index]
        if isinstance(start_value, torch.Tensor):
            start_value = start_value.item()
        points.append(
            SelectedPoint(
                sample_index=sample_index,
                row=int(selected[0]),
                column=int(selected[1]),
                video_id=video_id,
                start=str(start_value),
            )
        )
    return points


def _perturb(sparse_batch: dict[str, Any], points: list[SelectedPoint], delta: float) -> dict[str, Any]:
    perturbed = _clone_batch(sparse_batch)
    for point in points:
        perturbed["observed_rss"][point.sample_index, ..., point.row, point.column] += delta
    return perturbed


def _ring_index(distance: torch.Tensor) -> torch.Tensor:
    bounds = torch.tensor(RING_BOUNDS[1:-1], device=distance.device)
    return torch.bucketize(distance, bounds, right=False)


def _response_metrics(
    derivative: torch.Tensor,
    sparse_batch: dict[str, Any],
    points: list[SelectedPoint],
) -> tuple[dict[str, float], list[dict[str, float]]]:
    mask = (sparse_batch["sampling_mask"] > 0.5).reshape(
        derivative.shape[0], -1, derivative.shape[-2], derivative.shape[-1]
    )[:, 0]
    response = derivative.abs().reshape(
        derivative.shape[0], -1, derivative.shape[-2], derivative.shape[-1]
    )[:, 0]
    height, width = response.shape[-2:]
    rows = torch.arange(height, device=response.device).reshape(height, 1)
    columns = torch.arange(width, device=response.device).reshape(1, width)
    records: list[dict[str, float]] = []
    ring_sum = torch.zeros(len(RING_BOUNDS) - 1, device=response.device, dtype=torch.float64)
    ring_count = torch.zeros_like(ring_sum)
    self_values: list[float] = []
    unobserved_means: list[float] = []
    unobserved_l1_fractions: list[float] = []
    for point in points:
        sample = point.sample_index
        self_value = _value_at(response, sample, point.row, point.column)
        unobserved = ~mask[sample]
        unobserved_abs = response[sample][unobserved]
        total_l1 = float(response[sample].sum().item())
        unobserved_l1 = float(unobserved_abs.sum().item())
        self_values.append(self_value)
        unobserved_means.append(float(unobserved_abs.mean().item()))
        unobserved_l1_fractions.append(unobserved_l1 / max(total_l1, 1.0e-12))
        distance = torch.sqrt((rows - point.row).square() + (columns - point.column).square())
        ring = _ring_index(distance)
        for ring_id in range(len(ring_sum)):
            selected = (ring == ring_id) & unobserved
            ring_sum[ring_id] += response[sample][selected].double().sum()
            ring_count[ring_id] += selected.sum()
        records.append(
            {
                **asdict(point),
                "self_abs_derivative": self_value,
                "self_signed_derivative": _value_at(derivative, sample, point.row, point.column),
                "unobserved_mean_abs_derivative": unobserved_means[-1],
                "unobserved_l1_fraction": unobserved_l1_fractions[-1],
                "unobserved_peak_abs_derivative": float(unobserved_abs.max().item()),
            }
        )
    summary: dict[str, float] = {
        "observed_self_abs_derivative_mean": float(torch.tensor(self_values).mean().item()),
        "observed_self_abs_derivative_median": float(torch.tensor(self_values).median().item()),
        "unobserved_mean_abs_derivative_mean": float(torch.tensor(unobserved_means).mean().item()),
        "unobserved_l1_fraction_mean": float(torch.tensor(unobserved_l1_fractions).mean().item()),
    }
    for ring_id in range(len(ring_sum)):
        low, high = RING_BOUNDS[ring_id], RING_BOUNDS[ring_id + 1]
        label = f"ring_{int(low)}_{'inf' if math.isinf(high) else int(high)}_unobserved_mean_abs_derivative"
        summary[label] = float((ring_sum[ring_id] / ring_count[ring_id].clamp_min(1)).item())
    return summary, records


def _run_model(
    name: str,
    model: torch.nn.Module,
    scheduler: DDIMScheduler,
    sparse_batch: dict[str, Any],
    perturbed_batch: dict[str, Any],
    initial_noise: torch.Tensor,
    points: list[SelectedPoint],
    delta: float,
    steps: int,
) -> tuple[dict[str, Any], torch.Tensor, torch.Tensor, torch.Tensor]:
    baseline = _ddim_sample(model, scheduler, sparse_batch, initial_noise, steps=steps)
    perturbed = _ddim_sample(model, scheduler, perturbed_batch, initial_noise, steps=steps)
    derivative = (perturbed - baseline) / delta
    summary, records = _response_metrics(derivative, sparse_batch, points)
    return {"model": name, "summary": summary, "per_sample": records}, baseline.cpu(), perturbed.cpu(), derivative.cpu()


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--config", required=True, help="V4 x0 config used to create the common validation batch")
    parser.add_argument("--unet-checkpoint", required=True)
    parser.add_argument("--original-dit-checkpoint", required=True)
    parser.add_argument("--alignment-dit-checkpoint", required=True)
    parser.add_argument("--output-dir", required=True)
    parser.add_argument("--rate", type=float, default=1.0)
    parser.add_argument("--delta", type=float, default=0.01)
    parser.add_argument("--video-count", type=int, default=4)
    parser.add_argument("--frames-per-video", type=int, default=8)
    parser.add_argument("--frame-stride", type=int, default=12)
    parser.add_argument("--batch-size", type=int, default=8)
    parser.add_argument("--ddim-steps", type=int, default=20)
    parser.add_argument("--seed", type=int, default=20260901)
    parser.add_argument("--gpu", default="0")
    args = parser.parse_args()
    if args.delta <= 0 or args.rate <= 0:
        raise ValueError("rate and delta must be positive")
    if args.frames_per_video <= 0 or args.video_count <= 0:
        raise ValueError("video-count and frames-per-video must be positive")

    output_dir = Path(args.output_dir).expanduser().resolve()
    output_dir.mkdir(parents=True, exist_ok=False)
    device = torch.device(f"cuda:{args.gpu}" if torch.cuda.is_available() else "cpu")
    if device.type != "cuda":
        raise RuntimeError("this diagnostic requires CUDA")
    torch.manual_seed(args.seed)
    config_path = Path(args.config).expanduser().resolve()
    config = load_config(config_path)
    video_ids = manifest_video_ids(Path(config.evaluation.subset_manifest), "stage_a")[: args.video_count]
    starts = tuple(index * args.frame_stride for index in range(args.frames_per_video))
    dataset = WindowDataset(
        root=config.data.root,
        split="val",
        split_file=config.data.split_file,
        window_size=1,
        seed=config.sampling.seed,
        cache_size=config.data.cache_size,
        tx_heatmap_sigma_px=config.data.tx_heatmap_sigma_px,
        fixed_starts=starts,
        video_ids=video_ids,
    )
    loader = DataLoader(dataset, batch_size=args.batch_size, shuffle=False, num_workers=2, pin_memory=True)
    dense_batches = list(loader)
    if sum(int(batch["target"].shape[0]) for batch in dense_batches) != len(dataset):
        raise RuntimeError("validation loader did not cover the selected dataset")

    common_batches: list[tuple[dict[str, Any], dict[str, Any], torch.Tensor, list[SelectedPoint]]] = []
    offset = 0
    for dense_batch in dense_batches:
        dense_batch = _to_device(dense_batch, device)
        sparse_batch = apply_fixed_sparse_observations(dense_batch, rate=args.rate, split="val")
        points = _select_points(sparse_batch, args.seed + offset)
        perturbed_batch = _perturb(sparse_batch, points, args.delta)
        initial_noise = deterministic_frame_noise_like(
            sparse_batch["target"],
            frame_names_by_sample(sparse_batch, batch_size=sparse_batch["target"].shape[0], window_size=1),
            rate=args.rate,
            seed=config.sampling.seed,
        )
        common_batches.append((sparse_batch, perturbed_batch, initial_noise, points))
        offset += len(points)

    scheduler = DDIMScheduler(
        num_train_timesteps=config.diffusion.train_timesteps,
        beta_schedule=config.diffusion.beta_schedule,
        prediction_type=config.diffusion.prediction_type,
        clip_sample=True,
        set_alpha_to_one=True,
        steps_offset=0,
    )
    model_loaders = (
        ("unet", lambda: _load_unet(Path(args.unet_checkpoint), device)),
        ("original_dit", lambda: _load_dit(config, Path(args.original_dit_checkpoint), device)),
        ("alignment_dit", lambda: _load_dit(config, Path(args.alignment_dit_checkpoint), device)),
    )
    results: dict[str, Any] = {}
    response_tensors: dict[str, list[torch.Tensor]] = {}
    for name, loader_fn in model_loaders:
        model = loader_fn()
        model_results: list[dict[str, Any]] = []
        baselines: list[torch.Tensor] = []
        perturbeds: list[torch.Tensor] = []
        derivatives: list[torch.Tensor] = []
        for sparse_batch, perturbed_batch, initial_noise, points in common_batches:
            result, baseline, perturbed, derivative = _run_model(
                name, model, scheduler, sparse_batch, perturbed_batch, initial_noise, points, args.delta, args.ddim_steps
            )
            model_results.append(result)
            baselines.append(baseline)
            perturbeds.append(perturbed)
            derivatives.append(derivative)
        summary_keys = model_results[0]["summary"].keys()
        results[name] = {
            "summary": {
                key: sum(result["summary"][key] for result in model_results) / len(model_results)
                for key in summary_keys
            },
            "per_sample": [record for result in model_results for record in result["per_sample"]],
        }
        response_tensors[name] = [torch.cat(values, dim=0) for values in (baselines, perturbeds, derivatives)]
        del model
        torch.cuda.empty_cache()

    metadata = {
        "schema": "rmdm_observation_sensitivity_v1",
        "diagnostic": "finite difference (x_hat(y + delta e_i) - x_hat(y)) / delta",
        "config": str(config_path),
        "checkpoints": {
            "unet": str(Path(args.unet_checkpoint).resolve()),
            "original_dit": str(Path(args.original_dit_checkpoint).resolve()),
            "alignment_dit": str(Path(args.alignment_dit_checkpoint).resolve()),
        },
        "rate": args.rate,
        "delta": args.delta,
        "ddim_steps": args.ddim_steps,
        "seed": args.seed,
        "device": str(device),
        "video_ids": video_ids,
        "fixed_starts": starts,
        "sample_count": len(dataset),
        "selection": "one deterministic visible point per frame; prefer positions at least 8 pixels from an image boundary",
        "same_masks_initial_noise_and_selected_points_for_all_models": True,
        "models": results,
    }
    (output_dir / "summary.json").write_text(json.dumps(metadata, indent=2, ensure_ascii=False) + "\n")
    with (output_dir / "per_sample.csv").open("w", newline="") as handle:
        fieldnames = ["model", *asdict(SelectedPoint(0, 0, 0, "", "")).keys(), "self_abs_derivative", "self_signed_derivative", "unobserved_mean_abs_derivative", "unobserved_l1_fraction", "unobserved_peak_abs_derivative"]
        writer = csv.DictWriter(handle, fieldnames=fieldnames)
        writer.writeheader()
        for name, result in results.items():
            for record in result["per_sample"]:
                writer.writerow({"model": name, **record})
    torch.save(
        {
            "baseline_predictions": {name: tensors[0] for name, tensors in response_tensors.items()},
            "perturbed_predictions": {name: tensors[1] for name, tensors in response_tensors.items()},
            "finite_difference_derivatives": {name: tensors[2] for name, tensors in response_tensors.items()},
        },
        output_dir / "responses.pt",
    )
    print(json.dumps({"output_dir": str(output_dir), "models": {name: value["summary"] for name, value in results.items()}}, indent=2), flush=True)


if __name__ == "__main__":
    main()
