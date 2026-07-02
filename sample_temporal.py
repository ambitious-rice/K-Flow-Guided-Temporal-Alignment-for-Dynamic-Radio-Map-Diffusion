#!/usr/bin/env python3
"""Sample/evaluate temporal RMDM clips without touching the single-frame baseline."""

from __future__ import annotations

import argparse
import csv
import json
import math
from pathlib import Path

import numpy as np
import torch
import torch.nn.functional as F
from diffusers import DDIMScheduler
from torch.utils.data import DataLoader, Subset

from lib.dynamic_clip_loaders import DynamicRadioMapRMDMClip
from temporal_unet import RMDMTemporalWrapper
from train_temporal import build_model_config, preprocess_conditions
from utils import build_unet_from_config


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser()
    parser.add_argument("--data_dir", required=True)
    parser.add_argument("--split", choices=("train", "val", "test"), default="val")
    parser.add_argument("--split_file", default="split.json")
    parser.add_argument("--checkpoint_path", default="")
    parser.add_argument("--init_2d_checkpoint", default="")
    parser.add_argument("--output_dir", default="./eval_dynamic_rmdm_temporal")
    parser.add_argument("--image_size", type=int, default=128)
    parser.add_argument("--clip_length", type=int, default=32)
    parser.add_argument("--frame_stride", type=int, default=16)
    parser.add_argument("--clip_stride", type=int, default=1)
    parser.add_argument("--starts", default="")
    parser.add_argument("--batch_size", type=int, default=1)
    parser.add_argument("--workers", type=int, default=2)
    parser.add_argument("--cache_size", type=int, default=8)
    parser.add_argument("--tx_heatmap_sigma_px", type=float, default=1.5)
    parser.add_argument("--num_samples", type=int, default=20)
    parser.add_argument("--device", default="cuda")
    parser.add_argument("--ddim_steps", type=int, default=50)
    parser.add_argument("--ddim_eta", type=float, default=1.0)
    parser.add_argument("--diffusion_steps", type=int, default=1000)
    parser.add_argument("--noise_schedule", choices=("linear", "cosine"), default="linear")
    parser.add_argument("--seed", type=int, default=1234)
    parser.add_argument("--save_arrays", action="store_true")
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
    parser.add_argument("--temporal_hidden_channels", type=int, default=16)
    return parser.parse_args()


def psnr_from_mse(mse: float) -> float:
    return float(-10.0 * math.log10(max(float(mse), 1e-12)))


@torch.no_grad()
def sample_ddim_video(
    model: RMDMTemporalWrapper,
    scheduler: DDIMScheduler,
    conditions: torch.Tensor,
    steps: int,
    eta: float,
    device: torch.device,
) -> torch.Tensor:
    batch, frames, _, height, width = conditions.shape
    image = torch.randn((batch, frames, 1, height, width), device=device)
    scheduler.set_timesteps(steps, device=device)
    for timestep in scheduler.timesteps:
        scaled = scheduler.scale_model_input(image.reshape(batch * frames, 1, height, width), timestep)
        scaled = scaled.reshape_as(image)
        model_input = torch.cat([conditions, scaled], dim=2)
        t = torch.full((batch,), int(timestep), device=device, dtype=torch.long)
        noise_pred, _ = model(model_input, t)
        flat = scheduler.step(
            noise_pred.reshape(batch * frames, 1, height, width),
            timestep,
            image.reshape(batch * frames, 1, height, width),
            eta=eta,
            use_clipped_model_output=False,
            return_dict=False,
        )[0]
        image = flat.reshape_as(image)
    return image.clamp(0.0, 1.0)


def summarize(rows: list[dict[str, float]]) -> dict[str, dict[str, float]]:
    keys = ["mae", "mse", "psnr", "delta_l1", "delta_mse"]
    out = {}
    for key in keys:
        arr = np.asarray([float(row[key]) for row in rows], dtype=np.float64)
        out[key] = {
            "mean": float(arr.mean()) if arr.size else float("nan"),
            "std": float(arr.std()) if arr.size else float("nan"),
            "count": int(arr.size),
        }
    return out


def main() -> None:
    args = parse_args()
    torch.manual_seed(args.seed)
    np.random.seed(args.seed)
    device = torch.device(args.device if torch.cuda.is_available() and args.device.startswith("cuda") else "cpu")

    starts = [int(item) for item in args.starts.split(",") if item.strip()] or None
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
    )
    if args.num_samples > 0:
        dataset = Subset(dataset, list(range(min(args.num_samples, len(dataset)))))
    loader = DataLoader(dataset, batch_size=args.batch_size, shuffle=False, num_workers=args.workers, pin_memory=True)

    base_model = build_unet_from_config(build_model_config(args))
    model = RMDMTemporalWrapper(
        base_model,
        input_channels=4,
        out_channels=1,
        temporal_hidden_channels=args.temporal_hidden_channels,
    )
    if args.checkpoint_path:
        state = torch.load(args.checkpoint_path, map_location="cpu")
        model.load_state_dict(state, strict=True)
    elif args.init_2d_checkpoint:
        state = torch.load(args.init_2d_checkpoint, map_location="cpu")
        model.load_2d_state_dict(state, strict=True)
    else:
        raise ValueError("Provide --checkpoint_path or --init_2d_checkpoint")
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

    out_dir = Path(args.output_dir)
    out_dir.mkdir(parents=True, exist_ok=True)
    rows = []
    for batch_idx, batch in enumerate(loader):
        conditions = preprocess_conditions(batch["inputs"].to(device, non_blocking=True))
        gt = batch["image"].to(device, non_blocking=True)
        pred = sample_ddim_video(model, scheduler, conditions, args.ddim_steps, args.ddim_eta, device)
        err = pred - gt
        batch_size = pred.shape[0]
        for item in range(batch_size):
            mse = float(err[item].square().mean().detach().cpu())
            mae = float(err[item].abs().mean().detach().cpu())
            pred_delta = pred[item, 1:] - pred[item, :-1]
            gt_delta = gt[item, 1:] - gt[item, :-1]
            delta = pred_delta - gt_delta
            row = {
                "clip_index": len(rows),
                "start": int(batch["start"][item]),
                "mae": mae,
                "mse": mse,
                "psnr": psnr_from_mse(mse),
                "delta_l1": float(delta.abs().mean().detach().cpu()),
                "delta_mse": float(delta.square().mean().detach().cpu()),
            }
            rows.append(row)
            if args.save_arrays:
                np.savez_compressed(
                    out_dir / f"clip_{row['clip_index']:06d}.npz",
                    pred=pred[item].detach().cpu().numpy(),
                    gt=gt[item].detach().cpu().numpy(),
                    names=np.asarray(batch["names"][item]),
                )

    with (out_dir / "per_clip.csv").open("w", newline="", encoding="utf-8") as handle:
        writer = csv.DictWriter(handle, fieldnames=["clip_index", "start", "mae", "mse", "psnr", "delta_l1", "delta_mse"])
        writer.writeheader()
        writer.writerows(rows)
    summary = {
        "checkpoint_path": args.checkpoint_path,
        "init_2d_checkpoint": args.init_2d_checkpoint,
        "split": args.split,
        "num_clips": len(rows),
        "metrics": summarize(rows),
    }
    (out_dir / "summary.json").write_text(json.dumps(summary, indent=2), encoding="utf-8")
    print(json.dumps(summary, indent=2), flush=True)


if __name__ == "__main__":
    main()
