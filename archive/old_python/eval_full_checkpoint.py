#!/usr/bin/env python3
"""Full-split evaluation for one RMDM checkpoint with detailed per-sample records."""

import argparse
import csv
import json
import os
import time
from pathlib import Path
from typing import Dict, List

import numpy as np
import torch
import torch.nn.functional as F
from skimage.metrics import structural_similarity as ssim
from torch.utils.data import DataLoader, Subset

from lib.loaders import DynamicRadioMapRMDM
from sample_test import build_model_config, create_scheduler, preprocess_conditions, sample_ddim, sample_ddpm, sample_dpm
from utils import build_unet_from_config


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Evaluate one RMDM checkpoint on a full dataset split.")
    parser.add_argument("--checkpoint_path", required=True)
    parser.add_argument("--data_dir", required=True)
    parser.add_argument("--output_dir", required=True)
    parser.add_argument("--split", default="test", choices=["train", "val", "test"])
    parser.add_argument("--num_samples", type=int, default=-1, help="<=0 means all samples after sharding.")
    parser.add_argument("--seed", type=int, default=1234)
    parser.add_argument("--num_shards", type=int, default=1)
    parser.add_argument("--shard_rank", type=int, default=0)
    parser.add_argument("--batch_size", type=int, default=20)
    parser.add_argument("--workers", type=int, default=4)
    parser.add_argument("--image_size", type=int, default=128)
    parser.add_argument("--split_file", default="split.json")
    parser.add_argument("--frame_stride", type=int, default=1)
    parser.add_argument("--cache_size", type=int, default=8)
    parser.add_argument("--tx_heatmap_sigma_px", type=float, default=1.5)
    parser.add_argument("--sampler", default="ddim", choices=["ddim", "ddpm", "dpm"])
    parser.add_argument("--ddim_steps", type=int, default=50)
    parser.add_argument("--ddpm_steps", type=int, default=1000)
    parser.add_argument("--dpm_steps", type=int, default=50)
    parser.add_argument("--ddim_eta", type=float, default=1.0)
    parser.add_argument("--diffusion_steps", type=int, default=1000)
    parser.add_argument("--noise_schedule", default="linear", choices=["linear", "cosine"])
    parser.add_argument("--log_interval", type=int, default=20)

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


def calculate_batch_metrics(pred: torch.Tensor, gt: torch.Tensor) -> Dict[str, List[float]]:
    pred = torch.nan_to_num(pred, nan=0.0, posinf=0.0, neginf=0.0)
    gt = torch.nan_to_num(gt, nan=0.0, posinf=0.0, neginf=0.0)
    out = {"MSE": [], "MAE": [], "RMSE": [], "NMSE": [], "PSNR": [], "SSIM": []}
    pred_np = pred.detach().cpu().numpy()
    gt_np = gt.detach().cpu().numpy()
    for i in range(pred.shape[0]):
        pi = pred[i:i + 1]
        gi = gt[i:i + 1]
        mse = F.mse_loss(pi, gi).item()
        mae = F.l1_loss(pi, gi).item()
        rmse = float(np.sqrt(mse))
        gt_power = torch.mean(gi ** 2).item()
        nmse = mse / gt_power if gt_power > 0 else float("inf")
        data_range = (gi.max() - gi.min()).item()
        if data_range <= 1e-12:
            data_range = 1.0
        psnr = 20.0 * np.log10(data_range) - 10.0 * np.log10(mse) if mse > 0 else float("inf")
        gt_img = gt_np[i, 0]
        pred_img = pred_np[i, 0]
        ssim_range = gt_img.max() - gt_img.min()
        if ssim_range <= 1e-12:
            ssim_range = 1.0
        out["MSE"].append(mse)
        out["MAE"].append(mae)
        out["RMSE"].append(rmse)
        out["NMSE"].append(nmse)
        out["PSNR"].append(float(psnr))
        out["SSIM"].append(float(ssim(gt_img, pred_img, data_range=ssim_range)))
    return out


def summarize(values: List[float]) -> Dict[str, float]:
    arr = np.asarray([v for v in values if np.isfinite(v)], dtype=np.float64)
    if arr.size == 0:
        return {"mean": float("nan"), "std": float("nan"), "min": float("nan"), "max": float("nan"), "count": 0}
    return {
        "mean": float(arr.mean()),
        "std": float(arr.std()),
        "min": float(arr.min()),
        "max": float(arr.max()),
        "count": int(arr.size),
    }


def main() -> None:
    args = parse_args()
    if args.image_size != 128:
        raise ValueError("DynamicRadio data is 128x128; use --image_size 128.")
    if not (0 <= args.shard_rank < args.num_shards):
        raise ValueError("--shard_rank must satisfy 0 <= shard_rank < num_shards")

    output_dir = Path(args.output_dir)
    output_dir.mkdir(parents=True, exist_ok=True)
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

    dataset = DynamicRadioMapRMDM(
        root=args.data_dir,
        split=args.split,
        split_file=args.split_file,
        frame_stride=args.frame_stride,
        cache_size=args.cache_size,
        tx_heatmap_sigma_px=args.tx_heatmap_sigma_px,
    )
    all_indices = np.arange(len(dataset), dtype=np.int64)
    shard_indices = all_indices[args.shard_rank::args.num_shards]
    if args.num_samples > 0:
        shard_indices = shard_indices[:args.num_samples]
    subset = Subset(dataset, shard_indices.tolist())
    loader = DataLoader(
        subset,
        batch_size=args.batch_size,
        shuffle=False,
        num_workers=args.workers,
        pin_memory=True,
    )

    cfg = build_model_config(args, in_ch=3, out_ch=1)
    model = build_unet_from_config(cfg).to(device)
    model.load_state_dict(torch.load(args.checkpoint_path, map_location=device))
    model.eval()
    scheduler = create_scheduler(args.sampler, args.diffusion_steps, args.noise_schedule)

    torch.manual_seed(args.seed)
    if torch.cuda.is_available():
        torch.cuda.manual_seed_all(args.seed)

    detail_path = output_dir / f"details_{args.split}_shard{args.shard_rank}of{args.num_shards}.csv"
    summary_path = output_dir / f"summary_{args.split}_shard{args.shard_rank}of{args.num_shards}.json"
    config_path = output_dir / f"config_{args.split}_shard{args.shard_rank}of{args.num_shards}.json"
    metrics_all = {"MSE": [], "MAE": [], "RMSE": [], "NMSE": [], "PSNR": [], "SSIM": []}

    config = vars(args).copy()
    config.update({
        "dataset_len": len(dataset),
        "shard_samples": len(subset),
        "device": str(device),
        "detail_path": str(detail_path),
        "summary_path": str(summary_path),
    })
    config_path.write_text(json.dumps(config, indent=2))

    start = time.time()
    processed = 0
    with detail_path.open("w", newline="") as f:
        writer = csv.writer(f)
        writer.writerow(["global_index", "name", "MSE", "MAE", "RMSE", "NMSE", "PSNR", "SSIM"])
        with torch.no_grad():
            for batch_idx, (inputs, image_gain, names) in enumerate(loader):
                if image_gain.dim() == 3:
                    image_gain = image_gain.unsqueeze(1)
                conditions = preprocess_conditions(inputs.to(device))
                gt = image_gain.to(device)
                if args.sampler == "ddim":
                    pred, _ = sample_ddim(model, scheduler, conditions, args.ddim_steps, str(device), args.ddim_eta)
                elif args.sampler == "ddpm":
                    pred, _ = sample_ddpm(model, scheduler, conditions, args.ddpm_steps, str(device))
                elif args.sampler == "dpm":
                    pred, _ = sample_dpm(model, scheduler, conditions, args.dpm_steps, str(device))
                else:
                    raise ValueError(args.sampler)

                batch_metrics = calculate_batch_metrics(pred, gt)
                batch_size = gt.shape[0]
                base = processed
                for i in range(batch_size):
                    global_index = int(shard_indices[base + i])
                    row = [
                        global_index,
                        str(names[i]),
                        batch_metrics["MSE"][i],
                        batch_metrics["MAE"][i],
                        batch_metrics["RMSE"][i],
                        batch_metrics["NMSE"][i],
                        batch_metrics["PSNR"][i],
                        batch_metrics["SSIM"][i],
                    ]
                    writer.writerow(row)
                for key, values in batch_metrics.items():
                    metrics_all[key].extend(values)
                processed += batch_size

                if batch_idx % args.log_interval == 0:
                    elapsed = time.time() - start
                    rate = processed / elapsed if elapsed > 0 else 0.0
                    remaining = (len(subset) - processed) / rate if rate > 0 else float("inf")
                    print(
                        f"shard {args.shard_rank}/{args.num_shards} "
                        f"batch {batch_idx + 1}/{len(loader)} processed {processed}/{len(subset)} "
                        f"rate {rate:.3f} samples/s eta {remaining/3600:.2f}h",
                        flush=True,
                    )

    total_time = time.time() - start
    summary = {
        "config": config,
        "timing": {
            "total_time_seconds": total_time,
            "samples_per_second": processed / total_time if total_time > 0 else 0.0,
            "processed_samples": processed,
        },
        "metrics": {key: summarize(values) for key, values in metrics_all.items()},
    }
    summary_path.write_text(json.dumps(summary, indent=2))
    print(json.dumps(summary["timing"], indent=2), flush=True)
    print(json.dumps(summary["metrics"], indent=2), flush=True)


if __name__ == "__main__":
    main()
