#!/usr/bin/env python3
"""Paired baseline/candidate full-split evaluation for RMDM clips."""

from __future__ import annotations

import argparse
import csv
import json
import math
from pathlib import Path

import numpy as np
import torch
from diffusers import DDIMScheduler
from torch.utils.data import DataLoader, Subset

from lib.dynamic_clip_loaders import DynamicRadioMapRMDMClip
from sample_temporal_pinn import (
    make_clip_noise,
    psnr_from_mse,
    spatial_grad_mag,
    temporal_smooth_clip,
    temporal_smoothing_mask,
)
from train_temporal_pinn import build_model_config, preprocess_conditions
from utils import build_unet_from_config


FIELDS = [
    "clip_index",
    "start",
    "baseline_mae",
    "baseline_mse",
    "baseline_psnr",
    "candidate_mae",
    "candidate_mse",
    "candidate_psnr",
    "delta_mae",
    "delta_mse",
    "delta_psnr",
]


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser()
    parser.add_argument("--data_dir", required=True)
    parser.add_argument("--split", choices=("train", "val", "test"), default="val")
    parser.add_argument("--split_file", default="split.json")
    parser.add_argument("--init_2d_checkpoint", required=True)
    parser.add_argument("--output_dir", required=True)
    parser.add_argument("--image_size", type=int, default=128)
    parser.add_argument("--clip_length", type=int, default=32)
    parser.add_argument("--frame_stride", type=int, default=32)
    parser.add_argument("--clip_stride", type=int, default=1)
    parser.add_argument("--starts", default="")
    parser.add_argument("--start_index", type=int, default=0)
    parser.add_argument("--end_index", type=int, default=0)
    parser.add_argument("--num_samples", type=int, default=0)
    parser.add_argument("--batch_size", type=int, default=2)
    parser.add_argument("--workers", type=int, default=0)
    parser.add_argument("--cache_size", type=int, default=8)
    parser.add_argument("--tx_heatmap_sigma_px", type=float, default=1.5)
    parser.add_argument("--k2_root", default="")
    parser.add_argument("--path_mask_threshold", type=float, default=0.03)
    parser.add_argument("--path_mask_dilate", type=int, default=5)
    parser.add_argument("--device", default="cuda")
    parser.add_argument("--ddim_steps", type=int, default=20)
    parser.add_argument("--ddim_eta", type=float, default=1.0)
    parser.add_argument("--diffusion_steps", type=int, default=1000)
    parser.add_argument("--noise_schedule", choices=("linear", "cosine"), default="linear")
    parser.add_argument("--seed", type=int, default=1234)
    parser.add_argument("--flush_every", type=int, default=5)
    parser.add_argument("--resume_existing", action="store_true")
    parser.add_argument("--temporal_smoothing_weight", type=float, default=0.05)
    parser.add_argument("--temporal_smoothing_every", type=int, default=1)
    parser.add_argument(
        "--temporal_smoothing_mask_mode",
        choices=(
            "none",
            "inverse_k2",
            "inverse_k2_grad",
            "inverse_rss_grad",
            "inverse_k2_and_rss_grad",
            "inverse_k2_grad_and_rss_grad",
        ),
        default="inverse_rss_grad",
    )
    parser.add_argument("--temporal_smoothing_k2_threshold", type=float, default=0.03)
    parser.add_argument("--temporal_smoothing_k2_grad_threshold", type=float, default=0.2)
    parser.add_argument("--temporal_smoothing_rss_grad_threshold", type=float, default=0.09)
    parser.add_argument("--temporal_smoothing_mask_dilate", type=int, default=3)
    parser.add_argument("--num_channels", type=int, default=96)
    parser.add_argument("--num_res_blocks", type=int, default=2)
    parser.add_argument("--attention_resolutions", default="16")
    parser.add_argument("--channel_mult", default="")
    parser.add_argument("--dropout", type=float, default=0.0)
    parser.add_argument("--use_checkpoint", type=bool, default=False)
    parser.add_argument("--use_scale_shift_norm", type=bool, default=True)
    parser.add_argument("--resblock_updown", type=bool, default=False)
    parser.add_argument("--use_fp16", type=bool, default=False)
    parser.add_argument("--num_heads", type=int, default=4)
    parser.add_argument("--num_head_channels", type=int, default=-1)
    parser.add_argument("--num_heads_upsample", type=int, default=-1)
    parser.add_argument("--use_new_attention_order", type=bool, default=False)
    return parser.parse_args()


def load_existing_rows(path: Path) -> list[dict[str, float]]:
    if not path.exists():
        return []
    rows: list[dict[str, float]] = []
    with path.open("r", newline="", encoding="utf-8") as handle:
        reader = csv.DictReader(handle)
        for row in reader:
            parsed: dict[str, float] = {}
            for key in FIELDS:
                if key in ("clip_index", "start"):
                    parsed[key] = int(row[key])
                else:
                    parsed[key] = float(row[key])
            rows.append(parsed)
    return rows


def summarize(rows: list[dict[str, float]]) -> dict[str, dict[str, float]]:
    keys = FIELDS[2:]
    out = {}
    for key in keys:
        arr = np.asarray([float(row[key]) for row in rows], dtype=np.float64)
        out[key] = {
            "mean": float(arr.mean()) if arr.size else float("nan"),
            "std": float(arr.std()) if arr.size else float("nan"),
            "count": int(arr.size),
        }
    return out


def write_outputs(out_dir: Path, args: argparse.Namespace, rows: list[dict[str, float]], partial: bool) -> None:
    csv_tmp = out_dir / "per_clip_paired.csv.tmp"
    with csv_tmp.open("w", newline="", encoding="utf-8") as handle:
        writer = csv.DictWriter(handle, fieldnames=FIELDS)
        writer.writeheader()
        writer.writerows(rows)
    csv_tmp.replace(out_dir / "per_clip_paired.csv")
    summary = {
        "split": args.split,
        "start_index": args.start_index,
        "end_index": args.end_index,
        "num_clips": len(rows),
        "partial": bool(partial),
        "init_2d_checkpoint": args.init_2d_checkpoint,
        "seed": args.seed,
        "paired": True,
        "fast_2d_forward": True,
        "deterministic_clip_noise": True,
        "primary_metrics_only": True,
        "candidate": {
            "temporal_smoothing_weight": args.temporal_smoothing_weight,
            "temporal_smoothing_every": args.temporal_smoothing_every,
            "temporal_smoothing_mask_mode": args.temporal_smoothing_mask_mode,
            "temporal_smoothing_k2_threshold": args.temporal_smoothing_k2_threshold,
            "temporal_smoothing_k2_grad_threshold": args.temporal_smoothing_k2_grad_threshold,
            "temporal_smoothing_rss_grad_threshold": args.temporal_smoothing_rss_grad_threshold,
            "temporal_smoothing_mask_dilate": args.temporal_smoothing_mask_dilate,
        },
        "metrics": summarize(rows),
    }
    summary_tmp = out_dir / "summary_paired.json.tmp"
    summary_tmp.write_text(json.dumps(summary, indent=2), encoding="utf-8")
    summary_tmp.replace(out_dir / "summary_paired.json")


@torch.no_grad()
def sample_paired(
    model: torch.nn.Module,
    scheduler: DDIMScheduler,
    conditions: torch.Tensor,
    initial_noise: torch.Tensor,
    k2: torch.Tensor | None,
    args: argparse.Namespace,
    device: torch.device,
) -> tuple[torch.Tensor, torch.Tensor]:
    batch, frames, _, height, width = conditions.shape
    baseline = initial_noise.to(device=device, dtype=conditions.dtype).clone()
    candidate = baseline.clone()
    scheduler.set_timesteps(args.ddim_steps, device=device)
    every = max(1, int(args.temporal_smoothing_every))
    cond_pair = torch.cat([conditions, conditions], dim=0)
    for step_idx, timestep in enumerate(scheduler.timesteps):
        image_pair = torch.cat([baseline, candidate], dim=0)
        scaled = scheduler.scale_model_input(
            image_pair.reshape(2 * batch * frames, 1, height, width),
            timestep,
        ).reshape_as(image_pair)
        model_input = torch.cat([cond_pair, scaled], dim=2)
        flat_input = model_input.reshape(2 * batch * frames, model_input.shape[2], height, width)
        flat_t = torch.full((2 * batch * frames,), int(timestep), device=device, dtype=torch.long)
        flat_noise_pred, _ = model(flat_input, flat_t)
        noise_pred_pair = flat_noise_pred.reshape(2 * batch, frames, 1, height, width)
        flat_next = scheduler.step(
            noise_pred_pair.reshape(2 * batch * frames, 1, height, width),
            timestep,
            image_pair.reshape(2 * batch * frames, 1, height, width),
            eta=args.ddim_eta,
            use_clipped_model_output=False,
            return_dict=False,
        )[0]
        next_pair = flat_next.reshape_as(image_pair)
        baseline = next_pair[:batch]
        candidate = next_pair[batch:]
        if args.temporal_smoothing_weight > 0 and (step_idx + 1) % every == 0:
            mask = temporal_smoothing_mask(
                candidate,
                k2,
                args.temporal_smoothing_mask_mode,
                args.temporal_smoothing_k2_threshold,
                args.temporal_smoothing_k2_grad_threshold,
                args.temporal_smoothing_rss_grad_threshold,
                args.temporal_smoothing_mask_dilate,
            )
            candidate = temporal_smooth_clip(candidate, args.temporal_smoothing_weight, mask)
    return baseline.clamp(0.0, 1.0), candidate.clamp(0.0, 1.0)


def main() -> None:
    args = parse_args()
    torch.manual_seed(args.seed)
    np.random.seed(args.seed)
    device = torch.device(args.device if torch.cuda.is_available() and args.device.startswith("cuda") else "cpu")
    out_dir = Path(args.output_dir)
    out_dir.mkdir(parents=True, exist_ok=True)
    rows = load_existing_rows(out_dir / "per_clip_paired.csv") if args.resume_existing else []
    resume_count = len(rows)
    if resume_count:
        print(f"Resuming from {resume_count} existing clips in {out_dir / 'per_clip_paired.csv'}", flush=True)

    starts = [int(item) for item in args.starts.split(",") if item.strip()] or None
    include_k2 = "k2" in args.temporal_smoothing_mask_mode
    dataset = DynamicRadioMapRMDMClip(
        root=args.data_dir,
        split=args.split,
        split_file=args.split_file,
        clip_length=args.clip_length,
        frame_stride=args.frame_stride,
        clip_stride=args.clip_stride,
        cache_size=args.cache_size,
        tx_heatmap_sigma_px=args.tx_heatmap_sigma_px,
        starts=starts,
        include_k2=include_k2,
        k2_root=args.k2_root or None,
        path_mask_threshold=args.path_mask_threshold,
        path_mask_dilate=args.path_mask_dilate,
    )
    dataset_len = len(dataset)
    start_index = max(0, int(args.start_index))
    end_index = int(args.end_index) if int(args.end_index) > 0 else dataset_len
    end_index = min(dataset_len, max(start_index, end_index))
    effective_start = min(end_index, start_index + resume_count)
    indices = list(range(effective_start, end_index))
    if args.num_samples > 0:
        indices = indices[: min(args.num_samples, len(indices))]
    if effective_start > 0 or end_index < dataset_len or args.num_samples > 0:
        dataset = Subset(dataset, indices)
    loader = DataLoader(dataset, batch_size=args.batch_size, shuffle=False, num_workers=args.workers, pin_memory=True)

    model = build_unet_from_config(build_model_config(args))
    model.load_state_dict(torch.load(args.init_2d_checkpoint, map_location="cpu"), strict=True)
    model.to(device).eval()
    beta_schedule = "linear" if args.noise_schedule == "linear" else "squaredcos_cap_v2"
    scheduler = DDIMScheduler(
        num_train_timesteps=args.diffusion_steps,
        beta_schedule=beta_schedule,
        prediction_type="epsilon",
        clip_sample=True,
        set_alpha_to_one=True,
        steps_offset=0,
    )

    flush_every = max(0, int(args.flush_every))
    for batch in loader:
        raw_inputs = batch["inputs"].to(device, non_blocking=True)
        conditions = preprocess_conditions(raw_inputs)
        gt = batch["image"].to(device, non_blocking=True)
        k2 = batch["k2"].to(device, non_blocking=True) if include_k2 else None
        batch_size, frames = gt.shape[:2]
        global_start_index = start_index + len(rows)
        initial_noise = make_clip_noise(
            batch_size,
            frames,
            gt.shape[-2],
            gt.shape[-1],
            args.seed,
            global_start_index,
            device,
        )
        baseline, candidate = sample_paired(model, scheduler, conditions, initial_noise, k2, args, device)
        for item in range(batch_size):
            base_err = baseline[item : item + 1] - gt[item : item + 1]
            cand_err = candidate[item : item + 1] - gt[item : item + 1]
            base_mse = float(base_err.square().mean().detach().cpu())
            cand_mse = float(cand_err.square().mean().detach().cpu())
            row = {
                "clip_index": len(rows),
                "start": int(batch["start"][item]),
                "baseline_mae": float(base_err.abs().mean().detach().cpu()),
                "baseline_mse": base_mse,
                "baseline_psnr": psnr_from_mse(base_mse),
                "candidate_mae": float(cand_err.abs().mean().detach().cpu()),
                "candidate_mse": cand_mse,
                "candidate_psnr": psnr_from_mse(cand_mse),
            }
            row["delta_mae"] = row["candidate_mae"] - row["baseline_mae"]
            row["delta_mse"] = row["candidate_mse"] - row["baseline_mse"]
            row["delta_psnr"] = row["candidate_psnr"] - row["baseline_psnr"]
            rows.append(row)
            if flush_every > 0 and len(rows) % flush_every == 0:
                write_outputs(out_dir, args, rows, partial=True)
                metrics = summarize(rows)
                print(
                    json.dumps(
                        {
                            "partial": True,
                            "num_clips": len(rows),
                            "baseline_mae": metrics["baseline_mae"]["mean"],
                            "baseline_mse": metrics["baseline_mse"]["mean"],
                            "baseline_psnr": metrics["baseline_psnr"]["mean"],
                            "candidate_mae": metrics["candidate_mae"]["mean"],
                            "candidate_mse": metrics["candidate_mse"]["mean"],
                            "candidate_psnr": metrics["candidate_psnr"]["mean"],
                            "delta_psnr": metrics["delta_psnr"]["mean"],
                        },
                        indent=2,
                    ),
                    flush=True,
                )
    write_outputs(out_dir, args, rows, partial=False)
    print((out_dir / "summary_paired.json").read_text(encoding="utf-8"), flush=True)


if __name__ == "__main__":
    main()
