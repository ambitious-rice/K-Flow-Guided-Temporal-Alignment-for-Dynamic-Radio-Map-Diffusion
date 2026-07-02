#!/usr/bin/env python3
"""Paired DDIM evaluation for dual decoder temporal adapter variants."""

from __future__ import annotations

import argparse
from pathlib import Path

import numpy as np
import torch
from diffusers import DDIMScheduler
from torch.utils.data import DataLoader, Subset

from dual_decoder_adapter_variants import DualDecoderAdapterVariantStudent
from sample_hwm_decoder_adapter_paired import add_temporal_metrics, load_existing_rows, psnr_from_mse, summarize, write_outputs
from sample_temporal_pinn import make_clip_noise
from train_temporal_pinn import build_model_config, preprocess_conditions
from utils import build_unet_from_config
from lib.dynamic_clip_loaders import DynamicRadioMapRMDMClip


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser()
    parser.add_argument("--data_dir", required=True)
    parser.add_argument("--split", choices=("train", "val", "test"), default="test")
    parser.add_argument("--split_file", default="split.json")
    parser.add_argument("--scene_ids", default="")
    parser.add_argument("--init_2d_checkpoint", required=True)
    parser.add_argument("--dual_checkpoint", required=True)
    parser.add_argument("--output_dir", required=True)
    parser.add_argument("--image_size", type=int, default=128)
    parser.add_argument("--clip_length", type=int, default=32)
    parser.add_argument("--frame_stride", type=int, default=32)
    parser.add_argument("--clip_stride", type=int, default=1)
    parser.add_argument("--start_index", type=int, default=0)
    parser.add_argument("--end_index", type=int, default=0)
    parser.add_argument("--num_samples", type=int, default=0)
    parser.add_argument("--index_file", default="")
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
    parser.add_argument("--flush_every", type=int, default=5)
    parser.add_argument("--resume_existing", action="store_true")
    parser.add_argument("--save_npz", action="store_true")
    parser.add_argument("--save_frames_dir", default="")
    parser.add_argument("--adapter_variant", choices=("multiscale", "keypath", "keypath_attn", "adaptive_alpha"), required=True)
    parser.add_argument("--hwm_adapter_indices", default="-2,-1")
    parser.add_argument("--stage2_adapter_indices", default="-2,-1")
    parser.add_argument("--adapter_reduction", type=int, default=1)
    parser.add_argument("--adapter_min_hidden", type=int, default=16)
    parser.add_argument("--keypath_gate_quantile", type=float, default=0.70)
    parser.add_argument("--keypath_gate_floor", type=float, default=0.05)
    parser.add_argument("--keypath_gate_dilate", type=int, default=2)
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


def _parse_indices(text: str) -> tuple[int, ...]:
    return tuple(int(item.strip()) for item in text.split(",") if item.strip())


def save_clip_npz(
    out_dir: Path,
    clip_index: int,
    start: int,
    gt: torch.Tensor,
    baseline: torch.Tensor,
    candidate: torch.Tensor,
) -> None:
    out_dir.mkdir(parents=True, exist_ok=True)
    gt_np = gt.detach().cpu().numpy().astype(np.float32)
    base_np = baseline.detach().cpu().numpy().astype(np.float32)
    cand_np = candidate.detach().cpu().numpy().astype(np.float32)
    np.savez_compressed(
        out_dir / f"clip_{clip_index:06d}_start_{start:06d}.npz",
        clip_index=np.int64(clip_index),
        start=np.int64(start),
        gt=gt_np,
        baseline=base_np,
        candidate=cand_np,
        gt_uint8=np.clip(np.rint(gt_np * 255.0), 0, 255).astype(np.uint8),
        baseline_uint8=np.clip(np.rint(base_np * 255.0), 0, 255).astype(np.uint8),
        candidate_uint8=np.clip(np.rint(cand_np * 255.0), 0, 255).astype(np.uint8),
    )


@torch.no_grad()
def sample_pair(base_model, student, scheduler, conditions, k2, initial_noise, args, device):
    batch, frames, _, height, width = conditions.shape
    baseline = initial_noise.to(device=device, dtype=conditions.dtype).clone()
    candidate = baseline.clone()
    scheduler.set_timesteps(args.ddim_steps, device=device)
    for timestep in scheduler.timesteps:
        t_clip = torch.full((batch,), int(timestep), device=device, dtype=torch.long)
        flat_t = t_clip[:, None].repeat(1, frames).reshape(batch * frames)
        scaled_base = scheduler.scale_model_input(baseline.reshape(batch * frames, 1, height, width), timestep).reshape_as(baseline)
        flat_input = torch.cat([conditions, scaled_base], dim=2).reshape(batch * frames, conditions.shape[2] + 1, height, width)
        eps_base, _ = base_model(flat_input, flat_t)
        baseline = scheduler.step(eps_base, timestep, baseline.reshape(batch * frames, 1, height, width), eta=args.ddim_eta, use_clipped_model_output=False, return_dict=False)[0].reshape_as(baseline)

        scaled_candidate = scheduler.scale_model_input(candidate.reshape(batch * frames, 1, height, width), timestep).reshape_as(candidate)
        eps_candidate, _, _ = student(conditions, scaled_candidate, t_clip, k2=k2)
        candidate = scheduler.step(eps_candidate.reshape(batch * frames, 1, height, width), timestep, candidate.reshape(batch * frames, 1, height, width), eta=args.ddim_eta, use_clipped_model_output=False, return_dict=False)[0].reshape_as(candidate)
    return baseline.clamp(0.0, 1.0), candidate.clamp(0.0, 1.0)


def main() -> None:
    args = parse_args()
    args.adapter_checkpoint = args.dual_checkpoint
    device = torch.device(args.device if torch.cuda.is_available() else "cpu")
    scene_ids = [item.strip() for item in args.scene_ids.split(",") if item.strip()] or None
    dataset = DynamicRadioMapRMDMClip(
        root=args.data_dir,
        split=args.split,
        split_file=args.split_file,
        clip_length=args.clip_length,
        frame_stride=args.frame_stride,
        clip_stride=args.clip_stride,
        cache_size=args.cache_size,
        tx_heatmap_sigma_px=args.tx_heatmap_sigma_px,
        include_k2=True,
        scene_ids=scene_ids,
    )
    if args.index_file:
        indices = [int(line.strip()) for line in Path(args.index_file).read_text(encoding="utf-8").splitlines() if line.strip() and not line.lstrip().startswith("#")]
        indices = [idx for idx in indices if 0 <= idx < len(dataset)]
    else:
        end_index = args.end_index if args.end_index > 0 else len(dataset)
        if args.num_samples > 0:
            end_index = min(end_index, args.start_index + args.num_samples)
        indices = list(range(args.start_index, min(end_index, len(dataset))))
    loader = DataLoader(Subset(dataset, indices), batch_size=args.batch_size, shuffle=False, num_workers=args.workers, pin_memory=True)

    base_model = build_unet_from_config(build_model_config(args)).to(device)
    base_model.load_state_dict(torch.load(args.init_2d_checkpoint, map_location="cpu"), strict=True)
    base_model.eval()
    student = DualDecoderAdapterVariantStudent(
        base_model,
        variant=args.adapter_variant,
        hwm_adapter_indices=_parse_indices(args.hwm_adapter_indices),
        stage2_adapter_indices=_parse_indices(args.stage2_adapter_indices),
        adapter_reduction=args.adapter_reduction,
        adapter_min_hidden=args.adapter_min_hidden,
        detach_anchor_gate=False,
        keypath_quantile=args.keypath_gate_quantile,
        keypath_gate_floor=args.keypath_gate_floor,
        keypath_gate_dilate=args.keypath_gate_dilate,
    ).to(device)
    payload = torch.load(args.dual_checkpoint, map_location="cpu")
    if payload.get("base_model") is not None:
        student.base_model.load_state_dict(payload["base_model"], strict=True)
    student.hwm_adapter.adapters.load_state_dict(payload["hwm_adapters"], strict=True)
    student.stage2_adapters.load_state_dict(payload["stage2_adapters"], strict=True)
    student.eval()

    beta_schedule = "linear" if args.noise_schedule == "linear" else "squaredcos_cap_v2"
    scheduler = DDIMScheduler(num_train_timesteps=args.diffusion_steps, beta_schedule=beta_schedule, prediction_type="epsilon", clip_sample=False)
    out_dir = Path(args.output_dir)
    frames_dir = Path(args.save_frames_dir) if args.save_frames_dir else out_dir / "saved_clips"
    rows = load_existing_rows(out_dir / "per_clip_paired.csv") if args.resume_existing else []
    done = {int(row["clip_index"]) for row in rows}
    for local_idx, batch in enumerate(loader):
        clip_index = indices[local_idx * args.batch_size]
        if clip_index in done:
            continue
        raw_inputs = batch["inputs"].to(device, non_blocking=True)
        conditions = preprocess_conditions(raw_inputs)
        k2 = batch["k2"].to(device, non_blocking=True)
        target = batch["image"].to(device, non_blocking=True)
        batch_indices = indices[local_idx * args.batch_size : local_idx * args.batch_size + target.shape[0]]
        noise = torch.empty_like(target)
        for noise_item, global_index in enumerate(batch_indices):
            noise[noise_item : noise_item + 1] = make_clip_noise(1, target.shape[1], target.shape[-2], target.shape[-1], seed=args.seed, global_start_index=int(global_index), device=device).to(dtype=target.dtype)
        baseline, candidate = sample_pair(base_model, student, scheduler, conditions, k2, noise, args, device)
        for b in range(target.shape[0]):
            idx = indices[local_idx * args.batch_size + b]
            gt = target[b]
            base = baseline[b]
            cand = candidate[b]
            base_mae = torch.mean((base - gt).abs()).item()
            base_mse = torch.mean((base - gt).square()).item()
            cand_mae = torch.mean((cand - gt).abs()).item()
            cand_mse = torch.mean((cand - gt).square()).item()
            row = {
                "clip_index": idx,
                "start": int(batch["start"][b]),
                "baseline_mae": base_mae,
                "baseline_mse": base_mse,
                "baseline_psnr": psnr_from_mse(base_mse),
                "candidate_mae": cand_mae,
                "candidate_mse": cand_mse,
                "candidate_psnr": psnr_from_mse(cand_mse),
            }
            row["delta_mae"] = row["candidate_mae"] - row["baseline_mae"]
            row["delta_mse"] = row["candidate_mse"] - row["baseline_mse"]
            row["delta_psnr"] = row["candidate_psnr"] - row["baseline_psnr"]
            add_temporal_metrics(row, gt, base, cand)
            if args.save_npz:
                save_clip_npz(frames_dir, idx, int(batch["start"][b]), gt, base, cand)
            rows.append(row)
        if len(rows) % max(1, args.flush_every) == 0:
            write_outputs(out_dir, args, rows, partial=True)
            print({k: summarize(rows)[k]["mean"] for k in ("delta_mae", "delta_mse", "delta_psnr", "delta_tof", "delta_twe")}, flush=True)
    write_outputs(out_dir, args, rows, partial=False)


if __name__ == "__main__":
    main()
