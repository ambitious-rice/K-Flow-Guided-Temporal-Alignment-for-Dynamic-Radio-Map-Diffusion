#!/usr/bin/env python3
"""Paired DDIM evaluation for frozen RMDM vs Stage-1 HWM decoder adapter."""

from __future__ import annotations

import argparse
import csv
import json
from pathlib import Path

import cv2
import numpy as np
import torch
import torch.nn.functional as F
from diffusers import DDIMScheduler
from torch.utils.data import DataLoader, Subset

from hwm_decoder_adapter_student import HWMDecoderAdapterStudent
from lib.dynamic_clip_loaders import DynamicRadioMapRMDMClip
from lib.kflow_loss import compute_kflow
from sample_temporal_pinn import make_clip_noise, psnr_from_mse
from train_temporal_pinn import build_model_config, preprocess_conditions
from utils import build_unet_from_config
from utils.nn import timestep_embedding


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
    "baseline_tof",
    "candidate_tof",
    "delta_tof",
    "baseline_twe",
    "candidate_twe",
    "delta_twe",
    "baseline_kflow_epe",
    "candidate_kflow_epe",
    "delta_kflow_epe",
    "baseline_kflow_l1",
    "candidate_kflow_l1",
    "delta_kflow_l1",
    "baseline_tdelta_l1",
    "candidate_tdelta_l1",
    "delta_tdelta_l1",
    "baseline_tdelta_mse",
    "candidate_tdelta_mse",
    "delta_tdelta_mse",
]


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser()
    parser.add_argument("--data_dir", required=True)
    parser.add_argument("--split", choices=("train", "val", "test"), default="test")
    parser.add_argument("--split_file", default="split.json")
    parser.add_argument("--scene_ids", default="")
    parser.add_argument("--init_2d_checkpoint", required=True)
    parser.add_argument("--adapter_checkpoint", required=True)
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
    parser.add_argument("--adapter_indices", default="-2,-1")
    parser.add_argument("--adapter_reduction", type=int, default=1)
    parser.add_argument("--adapter_min_hidden", type=int, default=16)
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
    rows = []
    with path.open("r", newline="", encoding="utf-8") as handle:
        reader = csv.DictReader(handle)
        for row in reader:
            rows.append({key: int(row[key]) if key in ("clip_index", "start") else float(row[key]) for key in FIELDS})
    return rows


def summarize(rows: list[dict[str, float]]) -> dict[str, dict[str, float]]:
    out = {}
    for key in FIELDS[2:]:
        arr = np.asarray([float(row[key]) for row in rows], dtype=np.float64)
        out[key] = {"mean": float(arr.mean()) if arr.size else float("nan"), "std": float(arr.std()) if arr.size else float("nan"), "count": int(arr.size)}
    return out


def write_outputs(out_dir: Path, args: argparse.Namespace, rows: list[dict[str, float]], partial: bool) -> None:
    out_dir.mkdir(parents=True, exist_ok=True)
    csv_tmp = out_dir / "per_clip_paired.csv.tmp"
    with csv_tmp.open("w", newline="", encoding="utf-8") as handle:
        writer = csv.DictWriter(handle, fieldnames=FIELDS)
        writer.writeheader()
        writer.writerows(rows)
    csv_tmp.replace(out_dir / "per_clip_paired.csv")
    summary = {
        "split": args.split,
        "scene_ids": args.scene_ids,
        "start_index": args.start_index,
        "end_index": args.end_index,
        "num_clips": len(rows),
        "partial": bool(partial),
        "init_2d_checkpoint": args.init_2d_checkpoint,
        "adapter_checkpoint": args.adapter_checkpoint,
        "seed": args.seed,
        "paired": True,
        "metrics": summarize(rows),
    }
    tmp = out_dir / "summary_paired.json.tmp"
    tmp.write_text(json.dumps(summary, indent=2), encoding="utf-8")
    tmp.replace(out_dir / "summary_paired.json")


def _optical_flow(prev_gray: np.ndarray, next_gray: np.ndarray) -> np.ndarray:
    prev_u8 = np.clip(np.rint(prev_gray * 255.0), 0, 255).astype(np.uint8)
    next_u8 = np.clip(np.rint(next_gray * 255.0), 0, 255).astype(np.uint8)
    return cv2.calcOpticalFlowFarneback(prev_u8, next_u8, None, 0.5, 3, 15, 3, 5, 1.2, 0)


def _warp_with_backward_flow(source: np.ndarray, backward_flow: np.ndarray) -> tuple[np.ndarray, np.ndarray]:
    height, width = source.shape[:2]
    grid_x, grid_y = np.meshgrid(np.arange(width, dtype=np.float32), np.arange(height, dtype=np.float32))
    map_x = grid_x + backward_flow[..., 0].astype(np.float32)
    map_y = grid_y + backward_flow[..., 1].astype(np.float32)
    valid = (map_x >= 0) & (map_x <= width - 1) & (map_y >= 0) & (map_y <= height - 1)
    warped = cv2.remap(source.astype(np.float32), map_x, map_y, interpolation=cv2.INTER_LINEAR, borderMode=cv2.BORDER_CONSTANT, borderValue=0)
    return warped, valid


def temporal_flow_metrics(gt: torch.Tensor, pred: torch.Tensor) -> dict[str, float]:
    gt_np = gt[:, 0].detach().cpu().numpy().astype(np.float32)
    pred_np = pred[:, 0].detach().cpu().numpy().astype(np.float32)
    tof = []
    twe = []
    for idx in range(1, gt_np.shape[0]):
        flow_gt = _optical_flow(gt_np[idx - 1], gt_np[idx])
        flow_pred = _optical_flow(pred_np[idx - 1], pred_np[idx])
        tof.append(float(np.linalg.norm(flow_pred - flow_gt, axis=-1).mean()))
        flow_gt_backward = _optical_flow(gt_np[idx], gt_np[idx - 1])
        warped_pred_prev, valid = _warp_with_backward_flow(pred_np[idx - 1], flow_gt_backward)
        twe.append(float(np.abs(pred_np[idx] - warped_pred_prev)[valid].mean()) if np.any(valid) else float(np.abs(pred_np[idx] - warped_pred_prev).mean()))
    return {"tof": float(np.mean(tof)) if tof else 0.0, "twe": float(np.mean(twe)) if twe else 0.0}


@torch.no_grad()
def temporal_kflow_metrics(gt: torch.Tensor, pred: torch.Tensor) -> dict[str, float]:
    gt_flow = compute_kflow(gt[:-1, 0], gt[1:, 0], delta=1e-6)
    pred_flow = compute_kflow(pred[:-1, 0], pred[1:, 0], delta=1e-6)
    diff = pred_flow - gt_flow
    return {
        "kflow_epe": float(torch.linalg.vector_norm(diff, dim=-3).mean().detach().cpu()),
        "kflow_l1": float(diff.abs().mean().detach().cpu()),
    }


def temporal_delta_metrics(gt: torch.Tensor, pred: torch.Tensor) -> dict[str, float]:
    diff = (pred[1:] - pred[:-1]) - (gt[1:] - gt[:-1])
    return {"tdelta_l1": float(diff.abs().mean().detach().cpu()), "tdelta_mse": float(diff.square().mean().detach().cpu())}


def add_temporal_metrics(row: dict[str, float], gt: torch.Tensor, baseline: torch.Tensor, candidate: torch.Tensor) -> None:
    base_flow = temporal_flow_metrics(gt, baseline)
    cand_flow = temporal_flow_metrics(gt, candidate)
    base_kflow = temporal_kflow_metrics(gt, baseline)
    cand_kflow = temporal_kflow_metrics(gt, candidate)
    base_delta = temporal_delta_metrics(gt, baseline)
    cand_delta = temporal_delta_metrics(gt, candidate)
    for prefix, metrics in (("baseline", base_flow), ("candidate", cand_flow)):
        row[f"{prefix}_tof"] = metrics["tof"]
        row[f"{prefix}_twe"] = metrics["twe"]
    row["delta_tof"] = row["candidate_tof"] - row["baseline_tof"]
    row["delta_twe"] = row["candidate_twe"] - row["baseline_twe"]
    for prefix, metrics in (("baseline", base_kflow), ("candidate", cand_kflow)):
        row[f"{prefix}_kflow_epe"] = metrics["kflow_epe"]
        row[f"{prefix}_kflow_l1"] = metrics["kflow_l1"]
    row["delta_kflow_epe"] = row["candidate_kflow_epe"] - row["baseline_kflow_epe"]
    row["delta_kflow_l1"] = row["candidate_kflow_l1"] - row["baseline_kflow_l1"]
    for prefix, metrics in (("baseline", base_delta), ("candidate", cand_delta)):
        row[f"{prefix}_tdelta_l1"] = metrics["tdelta_l1"]
        row[f"{prefix}_tdelta_mse"] = metrics["tdelta_mse"]
    row["delta_tdelta_l1"] = row["candidate_tdelta_l1"] - row["baseline_tdelta_l1"]
    row["delta_tdelta_mse"] = row["candidate_tdelta_mse"] - row["baseline_tdelta_mse"]


@torch.no_grad()
def forward_with_adapted_hwm(adapter: HWMDecoderAdapterStudent, conditions: torch.Tensor, noisy: torch.Tensor, timesteps: torch.Tensor) -> torch.Tensor:
    batch, frames, _, height, width = conditions.shape
    model = adapter.unet
    flat_t = timesteps[:, None].repeat(1, frames).reshape(batch * frames)
    flat_input = torch.cat([conditions, noisy], dim=2).reshape(batch * frames, conditions.shape[2] + 1, height, width)
    anch_seq, _, _ = adapter.forward_hwm(conditions)
    anch = tuple(a.reshape(batch * frames, *a.shape[2:]).contiguous() for a in anch_seq)

    hs = []
    emb = model.time_embed(timestep_embedding(flat_t, model.model_channels))
    h = flat_input.type(model.dtype)
    for ind, module in enumerate(model.input_blocks):
        if len(emb.size()) > 2:
            emb = emb.squeeze()
        if ind == 0:
            h = module(h, emb)
            gate_map = None
            if isinstance(anch, (list, tuple)) and len(anch) > 0:
                for a in anch[:2]:
                    a_resized = F.interpolate(a, size=h.shape[-2:], mode="bilinear", align_corners=False)
                    a_mean = a_resized.mean(dim=1, keepdim=True)
                    gate_map = a_mean if gate_map is None else gate_map + a_mean
            if gate_map is not None:
                h = h * (1 + torch.sigmoid(gate_map).detach())
        else:
            h = module(h, emb)
        hs.append(h)
    h = model.middle_block(h, emb)
    for module in model.output_blocks:
        h = torch.cat([h, hs.pop()], dim=1)
        h = module(h, emb)
    return model.out(h.type(flat_input.dtype)).reshape(batch, frames, 1, height, width).contiguous()


@torch.no_grad()
def sample_pair(base_model, adapter, scheduler, conditions, initial_noise, args, device):
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
        eps_candidate = forward_with_adapted_hwm(adapter, conditions, scaled_candidate, t_clip)
        candidate = scheduler.step(eps_candidate.reshape(batch * frames, 1, height, width), timestep, candidate.reshape(batch * frames, 1, height, width), eta=args.ddim_eta, use_clipped_model_output=False, return_dict=False)[0].reshape_as(candidate)
    return baseline.clamp(0.0, 1.0), candidate.clamp(0.0, 1.0)


def main() -> None:
    args = parse_args()
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
        include_k2=False,
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
    dataset = Subset(dataset, indices)
    loader = DataLoader(dataset, batch_size=args.batch_size, shuffle=False, num_workers=args.workers, pin_memory=True)

    base_model = build_unet_from_config(build_model_config(args)).to(device)
    base_model.load_state_dict(torch.load(args.init_2d_checkpoint, map_location="cpu"), strict=True)
    base_model.eval()
    adapter_indices = tuple(int(item.strip()) for item in args.adapter_indices.split(",") if item.strip())
    adapter = HWMDecoderAdapterStudent(base_model, adapter_indices=adapter_indices, reduction=args.adapter_reduction, min_hidden=args.adapter_min_hidden).to(device)
    payload = torch.load(args.adapter_checkpoint, map_location="cpu")
    adapter.adapters.load_state_dict(payload["adapters"], strict=True)
    adapter.eval()

    beta_schedule = "linear" if args.noise_schedule == "linear" else "squaredcos_cap_v2"
    scheduler = DDIMScheduler(num_train_timesteps=args.diffusion_steps, beta_schedule=beta_schedule, prediction_type="epsilon", clip_sample=False)
    out_dir = Path(args.output_dir)
    rows = load_existing_rows(out_dir / "per_clip_paired.csv") if args.resume_existing else []
    done = {int(row["clip_index"]) for row in rows}

    for local_idx, batch in enumerate(loader):
        clip_index = indices[local_idx * args.batch_size]
        if clip_index in done:
            continue
        raw_inputs = batch["inputs"].to(device, non_blocking=True)
        conditions = preprocess_conditions(raw_inputs)
        target = batch["image"].to(device, non_blocking=True)
        batch_indices = indices[local_idx * args.batch_size : local_idx * args.batch_size + target.shape[0]]
        noise = torch.empty_like(target)
        for noise_item, global_index in enumerate(batch_indices):
            noise[noise_item : noise_item + 1] = make_clip_noise(1, target.shape[1], target.shape[-2], target.shape[-1], seed=args.seed, global_start_index=int(global_index), device=device).to(dtype=target.dtype)
        baseline, candidate = sample_pair(base_model, adapter, scheduler, conditions, noise, args, device)
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
            rows.append(row)
        if len(rows) % max(1, args.flush_every) == 0:
            write_outputs(out_dir, args, rows, partial=True)
            print(json.dumps({"partial": True, "num_clips": len(rows), **{k: summarize(rows)[k]["mean"] for k in ("delta_mae", "delta_mse", "delta_psnr", "delta_tof", "delta_twe")}}, indent=2), flush=True)
    write_outputs(out_dir, args, rows, partial=False)


if __name__ == "__main__":
    main()
