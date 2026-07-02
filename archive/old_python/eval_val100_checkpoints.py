#!/usr/bin/env python3
"""Evaluate RMDM checkpoints on a fixed 100-sample DynamicRadio val subset."""

import argparse
import csv
import json
import os
import re
from pathlib import Path
from typing import Dict, List

import numpy as np
import torch
from skimage.metrics import structural_similarity as ssim
from torch.utils.data import DataLoader, Subset

from lib import loaders as radio_loaders
from sample_test import (
    build_model_config,
    calculate_metrics,
    create_scheduler,
    preprocess_conditions,
    sample_ddim,
    sample_ddpm,
    sample_dpm,
)
from utils import build_unet_from_config


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description="Evaluate every saved RMDM checkpoint on 100 DynamicRadio val samples."
    )
    parser.add_argument("--checkpoint_dir", type=str, required=True)
    parser.add_argument("--data_dir", type=str, required=True)
    parser.add_argument("--output_dir", type=str, default="./eval_val100_checkpoints")
    parser.add_argument("--num_samples", type=int, default=100)
    parser.add_argument("--seed", type=int, default=1234)
    parser.add_argument("--batch_size", type=int, default=20)
    parser.add_argument("--workers", type=int, default=4)
    parser.add_argument("--image_size", type=int, default=128)
    parser.add_argument("--split_file", type=str, default="split.json")
    parser.add_argument("--frame_stride", type=int, default=1)
    parser.add_argument("--cache_size", type=int, default=8)
    parser.add_argument("--tx_heatmap_sigma_px", type=float, default=1.5)
    parser.add_argument(
        "--prediction_mode",
        type=str,
        default="cal",
        choices=["cal", "ddim", "ddpm", "dpm"],
        help="cal is the direct supervised branch; ddim/ddpm/dpm run full diffusion sampling.",
    )
    parser.add_argument("--diffusion_steps", type=int, default=1000)
    parser.add_argument("--noise_schedule", type=str, default="linear", choices=["linear", "cosine"])
    parser.add_argument("--ddim_steps", type=int, default=50)
    parser.add_argument("--ddpm_steps", type=int, default=1000)
    parser.add_argument("--dpm_steps", type=int, default=50)
    parser.add_argument("--ddim_eta", type=float, default=1.0)
    parser.add_argument(
        "--checkpoint_names",
        type=str,
        default="",
        help="Optional comma-separated checkpoint filenames to evaluate.",
    )

    parser.add_argument("--num_channels", type=int, default=96)
    parser.add_argument("--num_res_blocks", type=int, default=2)
    parser.add_argument("--attention_resolutions", type=str, default="16")
    parser.add_argument("--channel_mult", type=str, default="")
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


def checkpoint_step(path: Path) -> int:
    if path.name == "model_phy.pth":
        return 75000
    match = re.search(r"model_phy_step(\d+)\.pth$", path.name)
    if match:
        return int(match.group(1))
    return -1


def list_checkpoints(checkpoint_dir: str, checkpoint_names: str = "") -> List[Path]:
    paths = [
        p for p in Path(checkpoint_dir).glob("model_phy*.pth")
        if p.name == "model_phy.pth" or re.search(r"model_phy_step\d+\.pth$", p.name)
    ]
    if checkpoint_names:
        requested = {name.strip() for name in checkpoint_names.split(",") if name.strip()}
        paths = [p for p in paths if p.name in requested]
    return sorted(paths, key=lambda p: (checkpoint_step(p), p.name))


def summarize(values: List[float]) -> Dict[str, float]:
    valid = np.array([v for v in values if np.isfinite(v)], dtype=np.float64)
    if valid.size == 0:
        return {"mean": float("nan"), "std": float("nan")}
    return {"mean": float(valid.mean()), "std": float(valid.std())}


def main() -> None:
    args = parse_args()
    if args.image_size != 128:
        raise ValueError("DynamicRadio val samples are 128x128; use --image_size 128.")

    os.makedirs(args.output_dir, exist_ok=True)
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

    dataset = radio_loaders.DynamicRadioMapRMDM(
        root=args.data_dir,
        split="val",
        split_file=args.split_file,
        frame_stride=args.frame_stride,
        cache_size=args.cache_size,
        tx_heatmap_sigma_px=args.tx_heatmap_sigma_px,
    )
    rng = np.random.default_rng(args.seed)
    sample_count = min(args.num_samples, len(dataset))
    indices = rng.choice(len(dataset), size=sample_count, replace=False).tolist()
    subset = Subset(dataset, indices)
    loader = DataLoader(
        subset,
        batch_size=args.batch_size,
        shuffle=False,
        num_workers=args.workers,
        pin_memory=True,
    )

    config = build_model_config(args, in_ch=3, out_ch=1)
    model = build_unet_from_config(config).to(device)
    model.eval()

    checkpoints = list_checkpoints(args.checkpoint_dir, args.checkpoint_names)
    if not checkpoints:
        raise FileNotFoundError(f"No model_phy*.pth checkpoints found in {args.checkpoint_dir}")

    rows = []
    detail: Dict[str, Dict[str, List[float]]] = {}
    scheduler = None
    if args.prediction_mode in {"ddim", "ddpm", "dpm"}:
        scheduler = create_scheduler(
            args.prediction_mode,
            num_train_timesteps=args.diffusion_steps,
            noise_schedule=args.noise_schedule,
        )

    print(
        f"Evaluating {len(checkpoints)} checkpoints on {sample_count} val samples "
        f"using {device}, mode={args.prediction_mode}."
    )

    for ckpt_path in checkpoints:
        torch.manual_seed(args.seed)
        if torch.cuda.is_available():
            torch.cuda.manual_seed_all(args.seed)
        state = torch.load(ckpt_path, map_location=device)
        model.load_state_dict(state)
        model.eval()

        metrics_all = {"MSE": [], "NMSE": [], "PSNR": [], "SSIM": []}
        with torch.no_grad():
            for inputs, image_gain, _ in loader:
                if image_gain.dim() == 3:
                    image_gain = image_gain.unsqueeze(1)

                conditions = preprocess_conditions(inputs.to(device))
                target = image_gain.to(device)
                if args.prediction_mode == "cal":
                    noisy_placeholder = torch.zeros_like(target)
                    timestep = torch.zeros(target.shape[0], device=device, dtype=torch.long)
                    model_input = torch.cat([conditions, noisy_placeholder], dim=1)
                    out = model(model_input, timestep)
                    pred = out[1] if isinstance(out, tuple) else out
                elif args.prediction_mode == "ddim":
                    pred, _ = sample_ddim(
                        model,
                        scheduler,
                        conditions,
                        num_inference_steps=args.ddim_steps,
                        device=str(device),
                        ddim_eta=args.ddim_eta,
                    )
                elif args.prediction_mode == "ddpm":
                    pred, _ = sample_ddpm(
                        model,
                        scheduler,
                        conditions,
                        num_inference_steps=args.ddpm_steps,
                        device=str(device),
                    )
                elif args.prediction_mode == "dpm":
                    pred, _ = sample_dpm(
                        model,
                        scheduler,
                        conditions,
                        num_inference_steps=args.dpm_steps,
                        device=str(device),
                    )
                else:
                    raise ValueError(f"Unsupported prediction_mode: {args.prediction_mode}")

                batch_metrics = calculate_metrics(pred, target)
                for metric_name, values in batch_metrics.items():
                    metrics_all[metric_name].extend(values)

        step = checkpoint_step(ckpt_path)
        row = {
            "checkpoint": ckpt_path.name,
            "step": step,
            "prediction_mode": args.prediction_mode,
            "num_samples": sample_count,
        }
        for metric_name in ("MSE", "NMSE", "PSNR", "SSIM"):
            stats = summarize(metrics_all[metric_name])
            row[f"{metric_name}_mean"] = stats["mean"]
            row[f"{metric_name}_std"] = stats["std"]
        rows.append(row)
        detail[ckpt_path.name] = metrics_all
        print(
            f"{ckpt_path.name:24s} "
            f"MSE={row['MSE_mean']:.6f} NMSE={row['NMSE_mean']:.6f} "
            f"PSNR={row['PSNR_mean']:.3f} SSIM={row['SSIM_mean']:.4f}"
        )

    csv_path = Path(args.output_dir) / f"val100_checkpoint_metrics_{args.prediction_mode}.csv"
    with csv_path.open("w", newline="") as f:
        writer = csv.DictWriter(f, fieldnames=list(rows[0].keys()))
        writer.writeheader()
        writer.writerows(rows)

    json_path = Path(args.output_dir) / f"val100_checkpoint_metrics_{args.prediction_mode}.json"
    with json_path.open("w") as f:
        json.dump({"sample_indices": indices, "summary": rows, "detail": detail}, f, indent=2)

    print(f"Saved summary to {csv_path}")
    print(f"Saved details to {json_path}")


if __name__ == "__main__":
    main()
