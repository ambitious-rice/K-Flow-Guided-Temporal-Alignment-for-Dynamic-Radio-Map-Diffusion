#!/usr/bin/env python3
# ruff: noqa: E402
"""Collect or evaluate cross-fitted W16 noise priors."""

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

from rmdm_hvdit_v4_x0_w16_ratebalanced import CANDIDATE_SCHEMA, CHECKPOINT_SCHEMA
from rmdm_hvdit_v4_x0_w16_ratebalanced.config import load_config
from rmdm_hvdit_v4_x0_w16_ratebalanced.model import build_w16_system
from rmdm_hvdit_v4_joint.training.engine import write_json_atomic
from rmdm_noise_estimation.runner import balanced_manifest_videos, make_loader, run_units


def _floats(value: str) -> list[float]:
    return [float(item) for item in value.split(",") if item]


def _ints(value: str) -> list[int]:
    return [int(item) for item in value.split(",") if item]


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--config", required=True)
    parser.add_argument("--checkpoint", required=True)
    parser.add_argument("--manifest", required=True)
    parser.add_argument("--subset-stage", choices=["stage_a", "stage_b_extra"], required=True)
    parser.add_argument("--videos-per-scene", type=int, required=True)
    parser.add_argument("--rates", required=True)
    parser.add_argument("--noise-stds", required=True)
    parser.add_argument("--ddim-steps", type=int, required=True)
    parser.add_argument("--folds", type=int, default=4)
    parser.add_argument("--members", type=int, default=8)
    parser.add_argument("--member-batch-size", type=int, default=2)
    parser.add_argument("--seed", type=int, default=20260805)
    parser.add_argument("--namespace", required=True)
    parser.add_argument("--mode", choices=["collect", "evaluate"], required=True)
    parser.add_argument("--calibration")
    parser.add_argument("--output-dir", required=True)
    parser.add_argument("--expected-visible-gpus", required=True)
    parser.add_argument("--max-units-per-rank", type=int)
    parser.add_argument("--sigma-batch-size", type=int, default=1)
    args = parser.parse_args()

    expected = _ints(args.expected_visible_gpus)
    visible = _ints(os.environ.get("CUDA_VISIBLE_DEVICES", ""))
    if visible != expected:
        raise RuntimeError(f"expected CUDA_VISIBLE_DEVICES={expected}, got {visible}")
    accelerator = Accelerator(
        mixed_precision="bf16",
        dataloader_config=DataLoaderConfiguration(even_batches=False),
    )
    if accelerator.num_processes != len(expected):
        raise RuntimeError("accelerator process count does not match visible GPUs")

    config = load_config(args.config)
    videos = balanced_manifest_videos(
        args.manifest, args.subset_stage, args.videos_per_scene
    )
    loader = make_loader(accelerator, config, videos, [0, 16, 32, 48, 64, 80])
    model = build_w16_system(config)
    payload = torch.load(args.checkpoint, map_location="cpu", weights_only=False)
    if payload.get("schema") not in {CHECKPOINT_SCHEMA, CANDIDATE_SCHEMA}:
        raise ValueError("checkpoint is not a rate-balanced W16 artifact")
    model.load_state_dict(payload["model"], strict=True)
    model.requires_grad_(False).to(accelerator.device)
    calibration = None
    if args.calibration:
        with Path(args.calibration).open("r", encoding="utf-8") as handle:
            calibration = json.load(handle)
    if accelerator.is_main_process:
        write_json_atomic(
            Path(args.output_dir) / "run_config.json",
            {
                "schema": "w16_noise_estimation_run_v1",
                "arguments": vars(args),
                "selected_videos": videos,
                "visible_physical_gpus": expected,
                "checkpoint_epoch": int(payload.get("completed_epoch", payload.get("epoch", -1))),
                "checkpoint_global_step": int(payload.get("global_step", -1)),
            },
        )

    run_units(
        accelerator,
        model,
        config,
        loader,
        output_dir=args.output_dir,
        rates=_floats(args.rates),
        noise_stds=_floats(args.noise_stds),
        folds=args.folds,
        members=args.members,
        member_batch_size=args.member_batch_size,
        steps=args.ddim_steps,
        seed=args.seed,
        namespace=args.namespace,
        mode=args.mode,
        calibration=calibration,
        max_units_per_rank=args.max_units_per_rank,
        sigma_batch_size=args.sigma_batch_size,
    )


if __name__ == "__main__":
    main()
