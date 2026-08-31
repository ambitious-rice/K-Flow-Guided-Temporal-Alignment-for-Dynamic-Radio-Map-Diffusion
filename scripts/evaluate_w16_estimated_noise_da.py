#!/usr/bin/env python3
# ruff: noqa: E402
"""Evaluate late-step W16 data assimilation with estimated noise floors."""

from __future__ import annotations

import argparse
import json
import os
from pathlib import Path
import sys

REPOSITORY_ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(REPOSITORY_ROOT))
sys.path.insert(0, str(REPOSITORY_ROOT / "src"))

import torch
from accelerate import Accelerator, DataLoaderConfiguration

from rmdm.evaluation.fixed_sparse_protocol import (
    add_fixed_observation_noise,
    apply_fixed_sparse_observations,
    deterministic_frame_noise_like,
    frame_names_by_sample,
)
from rmdm.evaluation.metrics import MetricAccumulator
from rmdm_hvdit_v4_joint.training.engine import append_jsonl, write_json_atomic
from rmdm_hvdit_v4_x0_w16_ratebalanced import CANDIDATE_SCHEMA, CHECKPOINT_SCHEMA
from rmdm_hvdit_v4_x0_w16_ratebalanced.config import load_config
from rmdm_hvdit_v4_x0_w16_ratebalanced.model import build_w16_system
from rmdm_noise_estimation.assimilation import NoiseAwareDDIMSampler
from rmdm_noise_estimation.runner import balanced_manifest_videos, make_loader, unit_name


def _numbers(value: str, cast) -> list:
    return [cast(item) for item in value.split(",") if item]


def _index_estimates(root: Path) -> dict[str, Path]:
    return {path.name: path for path in root.glob("rank_*/*.json")}


def _method_result(prediction: torch.Tensor, sparse: dict) -> dict:
    accumulator = MetricAccumulator()
    accumulator.update(
        prediction.cpu(),
        sparse["target"].cpu(),
        sparse["building"].cpu(),
        sparse["vehicle"].cpu(),
        sparse["sampling_mask"].cpu(),
    )
    return {"metrics": accumulator.compute(), "raw": accumulator.raw()}


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--config", required=True)
    parser.add_argument("--checkpoint", required=True)
    parser.add_argument("--manifest", required=True)
    parser.add_argument("--subset-stage", choices=("stage_a", "stage_b_extra"), default="stage_b_extra")
    parser.add_argument("--estimation-dir", required=True)
    parser.add_argument("--output-dir", required=True)
    parser.add_argument("--rates", required=True)
    parser.add_argument("--noise-stds", required=True)
    parser.add_argument("--ddim-steps", type=int, default=50)
    parser.add_argument("--guided-steps", type=int, default=15)
    parser.add_argument("--strength", type=float, default=0.5)
    parser.add_argument("--max-update", type=float, default=0.25)
    parser.add_argument("--videos-per-scene", type=int, default=10)
    parser.add_argument("--noise-seed", type=int, default=20260805)
    parser.add_argument("--expected-visible-gpus", required=True)
    parser.add_argument("--max-units-per-rank", type=int)
    parser.add_argument(
        "--estimated-noise-scales",
        default="1",
        help="Comma-separated multipliers for the estimated DA noise variance.",
    )
    args = parser.parse_args()
    if not 0 < args.guided_steps <= args.ddim_steps:
        raise ValueError("guided-steps must be within [1, ddim-steps]")

    expected = _numbers(args.expected_visible_gpus, int)
    estimated_scales = _numbers(args.estimated_noise_scales, float)
    if not estimated_scales or any(value <= 0.0 for value in estimated_scales):
        raise ValueError("estimated-noise-scales must contain positive values")
    if len(set(estimated_scales)) != len(estimated_scales):
        raise ValueError("estimated-noise-scales must not contain duplicates")
    if _numbers(os.environ.get("CUDA_VISIBLE_DEVICES", ""), int) != expected:
        raise RuntimeError("CUDA_VISIBLE_DEVICES does not match the requested GPUs")
    accelerator = Accelerator(
        mixed_precision="bf16",
        dataloader_config=DataLoaderConfiguration(even_batches=False),
    )
    config = load_config(args.config)
    videos = balanced_manifest_videos(args.manifest, args.subset_stage, args.videos_per_scene)
    loader = make_loader(accelerator, config, videos, [0, 16, 32, 48, 64, 80])

    model = build_w16_system(config)
    checkpoint = torch.load(args.checkpoint, map_location="cpu", weights_only=False)
    if checkpoint.get("schema") not in {CHECKPOINT_SCHEMA, CANDIDATE_SCHEMA}:
        raise ValueError("checkpoint is not a rate-balanced W16 artifact")
    model.load_state_dict(checkpoint["model"], strict=True)
    model.requires_grad_(False).eval().to(accelerator.device)
    sampler = NoiseAwareDDIMSampler(config.diffusion)

    estimates = _index_estimates(Path(args.estimation_dir) / "units")
    output_root = Path(args.output_dir)
    rank_dir = output_root / "units" / f"rank_{accelerator.process_index:02d}"
    progress = output_root / "progress" / f"rank_{accelerator.process_index:02d}.jsonl"
    completed = 0

    for rate in _numbers(args.rates, float):
        for dense in loader:
            clean = apply_fixed_sparse_observations(dense, rate=rate, split="val")
            video_id = str(clean["video_id"][0])
            start = int(clean["start"][0])
            for sigma in _numbers(args.noise_stds, float):
                name = unit_name(video_id, start, rate, sigma)
                destination = rank_dir / f"{name}.json"
                if destination.exists():
                    continue
                sparse = add_fixed_observation_noise(
                    clean, standard_deviation=sigma, rate=rate, seed=args.noise_seed
                )
                estimate_path = estimates.get(f"{name}.json")
                if estimate_path is None:
                    raise FileNotFoundError(f"missing noise estimate for {name}")
                with estimate_path.open("r", encoding="utf-8") as handle:
                    estimate = json.load(handle)
                estimated_variance = float(
                    estimate["methods"]["calibrated_ensemble_mle"]["variance"]
                )
                initial_noise = deterministic_frame_noise_like(
                    sparse["target"],
                    frame_names_by_sample(sparse, batch_size=1, window_size=16),
                    rate=rate,
                    seed=config.sampling.seed,
                )
                with torch.no_grad(), accelerator.autocast():
                    cache = model.encode_conditions(sparse)
                    no_da = sampler.baseline(
                        model, cache, initial_noise.clone(), steps=args.ddim_steps,
                        accelerator=accelerator,
                    )
                method_specs: list[tuple[str, float]] = [("ordinary_da", 0.0)]
                for scale in estimated_scales:
                    label = "estimated_noise_da" if scale == 1.0 else (
                        f"estimated_noise_da_scale{scale:g}".replace(".", "p")
                    )
                    method_specs.append((label, scale * estimated_variance))
                method_specs.append(("known_noise_da", sigma**2))
                predictions = {"no_da": no_da}
                for method, variance in method_specs:
                    predictions[method] = sampler.guided(
                        model,
                        cache,
                        sparse,
                        initial_noise.clone(),
                        steps=args.ddim_steps,
                        guided_steps=args.guided_steps,
                        strength=args.strength,
                        max_update=args.max_update,
                        noise_variance=variance,
                        accelerator=accelerator,
                    )
                payload = {
                    "schema": "w16_estimated_noise_da_unit_v1",
                    "video_id": video_id,
                    "scene_id": video_id.split("/", 1)[0],
                    "start": start,
                    "rate": rate,
                    "true_sigma": sigma,
                    "subset_stage": args.subset_stage,
                    "estimated_sigma": estimated_variance**0.5,
                    "settings": {
                        "ddim_steps": args.ddim_steps,
                        "guided_steps": args.guided_steps,
                        "strength": args.strength,
                        "max_update": args.max_update,
                        "estimated_noise_variance_scales": estimated_scales,
                    },
                    "methods": {
                        method: _method_result(predictions[method], sparse)
                        for method in predictions
                    },
                }
                write_json_atomic(destination, payload)
                append_jsonl(progress, {key: payload[key] for key in (
                    "video_id", "start", "rate", "true_sigma", "estimated_sigma"
                )})
                completed += 1
                print(f"[estimated-noise-da] rank={accelerator.process_index} {name}", flush=True)
                if args.max_units_per_rank and completed >= args.max_units_per_rank:
                    return
    write_json_atomic(
        output_root / "progress" / f"rank_{accelerator.process_index:02d}.complete.json",
        {"status": "complete", "rank": accelerator.process_index},
    )


if __name__ == "__main__":
    main()
