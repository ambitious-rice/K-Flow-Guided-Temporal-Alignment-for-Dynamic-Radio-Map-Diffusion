#!/usr/bin/env python3
"""Render GT / frozen RMDM / residual student comparison clips."""

from __future__ import annotations

import argparse
import json
from pathlib import Path

import cv2
import numpy as np
import torch
from diffusers import DDIMScheduler
from PIL import Image
from torch.utils.data import DataLoader, Subset

from lib.dynamic_clip_loaders import DynamicRadioMapRMDMClip
from residual_kflow_student import FrozenRMDMResidualStudent
from sample_residual_kflow_paired import (
    add_temporal_metrics,
    sample_pair,
)
from sample_temporal_pinn import make_clip_noise, psnr_from_mse
from train_temporal_pinn import build_model_config, preprocess_conditions
from utils import build_unet_from_config


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser()
    parser.add_argument("--data_dir", required=True)
    parser.add_argument("--split", choices=("train", "val", "test"), default="test")
    parser.add_argument("--split_file", default="split.json")
    parser.add_argument("--init_2d_checkpoint", required=True)
    parser.add_argument("--residual_checkpoint", required=True)
    parser.add_argument("--output_dir", required=True)
    parser.add_argument("--clip_indices", required=True, help="Comma-separated dataset indices in the chosen split/stride config")
    parser.add_argument("--image_size", type=int, default=128)
    parser.add_argument("--clip_length", type=int, default=32)
    parser.add_argument("--frame_stride", type=int, default=32)
    parser.add_argument("--clip_stride", type=int, default=1)
    parser.add_argument("--batch_size", type=int, default=1)
    parser.add_argument("--workers", type=int, default=0)
    parser.add_argument("--cache_size", type=int, default=8)
    parser.add_argument("--tx_heatmap_sigma_px", type=float, default=1.5)
    parser.add_argument("--device", default="cuda")
    parser.add_argument("--ddim_steps", type=int, default=20)
    parser.add_argument("--ddim_eta", type=float, default=1.0)
    parser.add_argument("--diffusion_steps", type=int, default=1000)
    parser.add_argument("--noise_schedule", choices=("linear", "cosine"), default="linear")
    parser.add_argument("--seed", type=int, default=1234)
    parser.add_argument("--head_hidden_channels", type=int, default=32)
    parser.add_argument("--head_num_layers", type=int, default=2)
    parser.add_argument("--residual_alpha", type=float, default=0.05)
    parser.add_argument("--override_residual_alpha", type=float, default=0.03)
    parser.add_argument("--fps", type=float, default=8.0)
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


def to_u8(frame: torch.Tensor) -> np.ndarray:
    arr = frame.detach().cpu().numpy()
    if arr.ndim == 3:
        arr = arr[0]
    return np.clip(np.rint(arr * 255.0), 0, 255).astype(np.uint8)


def save_video(frames: list[np.ndarray], path: Path, fps: float) -> None:
    if not frames:
        return
    height, width = frames[0].shape[:2]
    writer = cv2.VideoWriter(str(path), cv2.VideoWriter_fourcc(*"mp4v"), fps, (width, height))
    for frame in frames:
        if frame.ndim == 2:
            frame = cv2.cvtColor(frame, cv2.COLOR_GRAY2BGR)
        writer.write(frame)
    writer.release()


def panel_frame(gt: np.ndarray, baseline: np.ndarray, candidate: np.ndarray) -> np.ndarray:
    gap = np.full((gt.shape[0], 4), 255, dtype=np.uint8)
    return np.concatenate([gt, gap, baseline, gap, candidate], axis=1)


def save_clip(root: Path, gt: torch.Tensor, baseline: torch.Tensor, candidate: torch.Tensor, fps: float) -> None:
    root.mkdir(parents=True, exist_ok=True)
    panel_frames = []
    for idx in range(gt.shape[0]):
        gt_u8 = to_u8(gt[idx])
        base_u8 = to_u8(baseline[idx])
        cand_u8 = to_u8(candidate[idx])
        panel = panel_frame(gt_u8, base_u8, cand_u8)
        panel_frames.append(panel)
        for name, image in (("gt", gt_u8), ("baseline", base_u8), ("candidate", cand_u8), ("panel", panel)):
            frame_dir = root / f"{name}_frames"
            frame_dir.mkdir(exist_ok=True)
            Image.fromarray(image, mode="L").save(frame_dir / f"frame_{idx:06d}.png")
    save_video(panel_frames, root / "gt_baseline_candidate.mp4", fps)


def main() -> None:
    args = parse_args()
    device = torch.device(args.device if torch.cuda.is_available() else "cpu")
    indices = [int(item.strip()) for item in args.clip_indices.split(",") if item.strip()]
    dataset = DynamicRadioMapRMDMClip(
        root=args.data_dir,
        split=args.split,
        split_file=args.split_file,
        clip_length=args.clip_length,
        frame_stride=args.frame_stride,
        clip_stride=args.clip_stride,
        cache_size=args.cache_size,
        tx_heatmap_sigma_px=args.tx_heatmap_sigma_px,
        include_k2=False,
    )
    loader = DataLoader(Subset(dataset, indices), batch_size=1, shuffle=False, num_workers=args.workers, pin_memory=True)

    base_model = build_unet_from_config(build_model_config(args)).to(device)
    base_model.load_state_dict(torch.load(args.init_2d_checkpoint, map_location="cpu"), strict=True)
    base_model.eval()

    student = FrozenRMDMResidualStudent(
        base_model,
        condition_channels=3,
        hidden_channels=args.head_hidden_channels,
        head_num_layers=args.head_num_layers,
        alpha=args.residual_alpha,
        diffusion_steps=args.diffusion_steps,
    ).to(device)
    payload = torch.load(args.residual_checkpoint, map_location="cpu")
    student.residual_head.load_state_dict(payload["residual_head"], strict=True)
    student.alpha = float(payload.get("residual_alpha", args.residual_alpha))
    if args.override_residual_alpha >= 0:
        student.alpha = float(args.override_residual_alpha)
    student.eval()

    beta_schedule = "linear" if args.noise_schedule == "linear" else "squaredcos_cap_v2"
    scheduler = DDIMScheduler(
        num_train_timesteps=args.diffusion_steps,
        beta_schedule=beta_schedule,
        prediction_type="epsilon",
        clip_sample=False,
    )

    out_dir = Path(args.output_dir)
    summaries = []
    for local_idx, batch in enumerate(loader):
        clip_index = indices[local_idx]
        raw_inputs = batch["inputs"].to(device, non_blocking=True)
        conditions = preprocess_conditions(raw_inputs)
        target = batch["image"].to(device, non_blocking=True)
        noise = make_clip_noise(
            target.shape[0],
            target.shape[1],
            target.shape[-2],
            target.shape[-1],
            seed=args.seed,
            global_start_index=clip_index,
            device=device,
        ).to(dtype=target.dtype)
        with torch.no_grad():
            baseline, candidate = sample_pair(base_model, student, scheduler, conditions, noise, args, device)
        gt = target[0]
        base = baseline[0]
        cand = candidate[0]
        base_mse = torch.mean((base - gt).square()).item()
        cand_mse = torch.mean((cand - gt).square()).item()
        row = {
            "clip_index": clip_index,
            "start": int(batch["start"][0]),
            "baseline_mae": torch.mean((base - gt).abs()).item(),
            "baseline_mse": base_mse,
            "baseline_psnr": psnr_from_mse(base_mse),
            "candidate_mae": torch.mean((cand - gt).abs()).item(),
            "candidate_mse": cand_mse,
            "candidate_psnr": psnr_from_mse(cand_mse),
        }
        row["delta_mae"] = row["candidate_mae"] - row["baseline_mae"]
        row["delta_mse"] = row["candidate_mse"] - row["baseline_mse"]
        row["delta_psnr"] = row["candidate_psnr"] - row["baseline_psnr"]
        add_temporal_metrics(row, gt, base, cand)
        save_clip(out_dir / f"clip_{clip_index:06d}", gt, base, cand, args.fps)
        summaries.append(row)

    out_dir.mkdir(parents=True, exist_ok=True)
    (out_dir / "render_summary.json").write_text(json.dumps(summaries, indent=2), encoding="utf-8")
    print(json.dumps(summaries, indent=2), flush=True)


if __name__ == "__main__":
    main()
