#!/usr/bin/env python3
"""Sample and evaluate temporal PINN RMDM clips."""

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
from PIL import Image
from torch.utils.data import DataLoader, Subset

from lib.dynamic_clip_loaders import DynamicRadioMapRMDMClip
from lib.kflow_loss import compute_kflow, masked_flow_l1, refine_mask_by_target_flow, temporal_kflow_loss
from temporal_pinn_unet import TemporalPINNUNet
from train_temporal_pinn import build_model_config, preprocess_conditions
from utils import build_unet_from_config, cal_pinn


FIELDS = ["clip_index", "start", "mae", "mse", "psnr", "cal_mae", "cal_mse", "cal_pinn", "cal_kflow", "pred_kflow_epe", "pred_kflow_l1"]


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser()
    parser.add_argument("--data_dir", required=True)
    parser.add_argument("--split", choices=("train", "val", "test"), default="val")
    parser.add_argument("--split_file", default="split.json")
    parser.add_argument("--checkpoint_path", default="")
    parser.add_argument("--init_2d_checkpoint", default="")
    parser.add_argument("--output_dir", default="./eval_dynamic_rmdm_temporal_pinn")
    parser.add_argument("--image_size", type=int, default=128)
    parser.add_argument("--clip_length", type=int, default=32)
    parser.add_argument("--frame_stride", type=int, default=16)
    parser.add_argument("--clip_stride", type=int, default=1)
    parser.add_argument("--starts", default="")
    parser.add_argument("--start_index", type=int, default=0, help="First dataset clip index to evaluate, inclusive.")
    parser.add_argument("--end_index", type=int, default=0, help="Last dataset clip index to evaluate, exclusive; 0 means no upper bound.")
    parser.add_argument("--batch_size", type=int, default=1)
    parser.add_argument("--workers", type=int, default=2)
    parser.add_argument("--cache_size", type=int, default=8)
    parser.add_argument("--tx_heatmap_sigma_px", type=float, default=1.5)
    parser.add_argument("--k2_root", default="")
    parser.add_argument("--path_mask_threshold", type=float, default=0.03)
    parser.add_argument("--path_mask_dilate", type=int, default=5)
    parser.add_argument("--num_samples", type=int, default=20)
    parser.add_argument("--device", default="cuda")
    parser.add_argument("--ddim_steps", type=int, default=50)
    parser.add_argument("--ddim_eta", type=float, default=1.0)
    parser.add_argument("--temporal_smoothing_weight", type=float, default=0.0)
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
        default="none",
    )
    parser.add_argument("--temporal_smoothing_k2_threshold", type=float, default=0.03)
    parser.add_argument("--temporal_smoothing_k2_grad_threshold", type=float, default=0.2)
    parser.add_argument("--temporal_smoothing_rss_grad_threshold", type=float, default=0.03)
    parser.add_argument("--temporal_smoothing_mask_dilate", type=int, default=3)
    parser.add_argument("--diffusion_steps", type=int, default=1000)
    parser.add_argument("--noise_schedule", choices=("linear", "cosine"), default="linear")
    parser.add_argument("--seed", type=int, default=1234)
    parser.add_argument("--deterministic_clip_noise", action="store_true", help="Seed the initial DDIM noise per global clip index for reproducible sharded comparisons.")
    parser.add_argument("--save_arrays", action="store_true")
    parser.add_argument("--save_frames", action="store_true")
    parser.add_argument("--flush_every", type=int, default=0, help="Write partial per_clip.csv and summary.json every N clips; 0 writes only at the end.")
    parser.add_argument("--resume_existing", action="store_true", help="Resume from an existing per_clip.csv by skipping already written clip indices.")
    parser.add_argument("--fast_2d_forward", action="store_true", help="Use the original 2D RMDM forward on flattened frames; useful for untrained temporal wrappers.")
    parser.add_argument("--primary_metrics_only", action="store_true", help="Compute only MAE/MSE/PSNR; fill auxiliary cal/k-flow metrics with NaN.")
    parser.add_argument("--prior_injection_mode", choices=("none", "uemb"), default="uemb")
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
    parser.add_argument("--temporal_num_heads", type=int, default=4)
    parser.add_argument("--use_frame_positional_encoding", action=argparse.BooleanOptionalAction, default=True)
    parser.add_argument("--max_frames", type=int, default=128)
    parser.add_argument("--kflow_delta", type=float, default=1e-6)
    parser.add_argument("--kflow_clamp", type=float, default=1.0)
    parser.add_argument("--kflow_eval_target_flow_threshold", type=float, default=0.0)
    parser.add_argument("--kflow_eval_target_flow_quantile", type=float, default=0.0)
    parser.add_argument("--pinn_k", type=float, default=0.2)
    return parser.parse_args()


def psnr_from_mse(mse: float) -> float:
    return float(-10.0 * math.log10(max(float(mse), 1e-12)))


def flow_metrics(
    pred: torch.Tensor,
    target: torch.Tensor,
    path_mask: torch.Tensor,
    delta: float,
    clamp: float,
    target_flow_threshold: float = 0.0,
    target_flow_quantile: float = 0.0,
) -> tuple[float, float]:
    pred_flow = compute_kflow(pred[:, :-1, 0], pred[:, 1:, 0], delta=delta)
    target_flow = compute_kflow(target[:, :-1, 0], target[:, 1:, 0], delta=delta)
    if clamp > 0:
        pred_flow = pred_flow.clamp(-clamp, clamp)
        target_flow = target_flow.clamp(-clamp, clamp)
    mask = path_mask.to(device=pred.device, dtype=pred.dtype)
    mask = refine_mask_by_target_flow(
        mask,
        target_flow,
        flow_threshold=target_flow_threshold,
        flow_quantile=target_flow_quantile,
    )
    epe = torch.sqrt((pred_flow - target_flow).square().sum(dim=2) + 1e-12).unsqueeze(2)
    epe_value = (epe * mask).sum() / mask.sum().clamp_min(1.0)
    l1_value = masked_flow_l1(pred_flow, target_flow, mask)
    return float(epe_value.detach().cpu()), float(l1_value.detach().cpu())


def spatial_grad_mag(value: torch.Tensor) -> torch.Tensor:
    field = value[:, :, 0] if value.ndim == 5 else value[:, 0]
    dx = torch.zeros_like(field)
    dy = torch.zeros_like(field)
    if field.shape[-1] > 2:
        dx[..., 1:-1] = 0.5 * (field[..., 2:] - field[..., :-2])
    if field.shape[-1] > 1:
        dx[..., 0] = field[..., 1] - field[..., 0]
        dx[..., -1] = field[..., -1] - field[..., -2]
    if field.shape[-2] > 2:
        dy[..., 1:-1, :] = 0.5 * (field[..., 2:, :] - field[..., :-2, :])
    if field.shape[-2] > 1:
        dy[..., 0, :] = field[..., 1, :] - field[..., 0, :]
        dy[..., -1, :] = field[..., -1, :] - field[..., -2, :]
    return torch.sqrt(dx.square() + dy.square() + 1e-12).unsqueeze(2)


def dilate_video_mask(mask: torch.Tensor, radius: int) -> torch.Tensor:
    if radius <= 0:
        return mask
    batch, frames, channels, height, width = mask.shape
    flat = mask.reshape(batch * frames, channels, height, width)
    flat = F.max_pool2d(flat, kernel_size=2 * int(radius) + 1, stride=1, padding=int(radius))
    return flat.reshape(batch, frames, channels, height, width)


def temporal_smoothing_mask(
    image: torch.Tensor,
    k2: torch.Tensor | None,
    mode: str,
    k2_threshold: float,
    k2_grad_threshold: float,
    rss_grad_threshold: float,
    dilate_radius: int,
) -> torch.Tensor | None:
    if mode == "none":
        return None
    smooth_mask = torch.ones_like(image)
    if "k2" in mode:
        if k2 is None:
            raise ValueError(f"temporal smoothing mode {mode!r} requires k2")
    if "k2_grad" in mode:
        k2_device = k2.to(device=image.device, dtype=image.dtype)
        protected = (spatial_grad_mag(k2_device) > float(k2_grad_threshold)).to(dtype=image.dtype)
        protected = dilate_video_mask(protected, dilate_radius)
        smooth_mask = smooth_mask * (1.0 - protected)
    elif "k2" in mode:
        k2_device = k2.to(device=image.device, dtype=image.dtype)
        protected = (k2_device > float(k2_threshold)).to(dtype=image.dtype)
        protected = dilate_video_mask(protected, dilate_radius)
        smooth_mask = smooth_mask * (1.0 - protected)
    if "rss_grad" in mode:
        protected = (spatial_grad_mag(image) > float(rss_grad_threshold)).to(dtype=image.dtype)
        protected = dilate_video_mask(protected, dilate_radius)
        smooth_mask = smooth_mask * (1.0 - protected)
    return smooth_mask.clamp(0.0, 1.0)


def temporal_smooth_clip(image: torch.Tensor, weight: float, mask: torch.Tensor | None = None) -> torch.Tensor:
    if weight <= 0 or image.ndim != 5 or image.shape[1] <= 2:
        return image
    w = float(weight)
    smoothed = image.clone()
    smoothed[:, 1:-1] = 0.25 * image[:, :-2] + 0.5 * image[:, 1:-1] + 0.25 * image[:, 2:]
    if mask is None:
        return image.lerp(smoothed, w)
    return image + (smoothed - image) * (w * mask.to(device=image.device, dtype=image.dtype))


@torch.no_grad()
def sample_ddim_video(
    model: torch.nn.Module,
    scheduler: DDIMScheduler,
    conditions: torch.Tensor,
    steps: int,
    eta: float,
    device: torch.device,
    temporal_smoothing_weight: float = 0.0,
    temporal_smoothing_every: int = 1,
    temporal_smoothing_mask_mode: str = "none",
    temporal_smoothing_k2: torch.Tensor | None = None,
    temporal_smoothing_k2_threshold: float = 0.03,
    temporal_smoothing_k2_grad_threshold: float = 0.2,
    temporal_smoothing_rss_grad_threshold: float = 0.03,
    temporal_smoothing_mask_dilate: int = 3,
    fast_2d_forward: bool = False,
    initial_noise: torch.Tensor | None = None,
) -> tuple[torch.Tensor, torch.Tensor]:
    batch, frames, _, height, width = conditions.shape
    if initial_noise is None:
        image = torch.randn((batch, frames, 1, height, width), device=device)
    else:
        image = initial_noise.to(device=device, dtype=conditions.dtype)
    last_cal = torch.zeros_like(image)
    scheduler.set_timesteps(steps, device=device)
    every = max(1, int(temporal_smoothing_every))
    for step_idx, timestep in enumerate(scheduler.timesteps):
        scaled = scheduler.scale_model_input(image.reshape(batch * frames, 1, height, width), timestep).reshape_as(image)
        model_input = torch.cat([conditions, scaled], dim=2)
        if fast_2d_forward:
            flat_input = model_input.reshape(batch * frames, model_input.shape[2], height, width)
            flat_t = torch.full((batch * frames,), int(timestep), device=device, dtype=torch.long)
            flat_noise_pred, flat_cal = model(flat_input, flat_t)
            noise_pred = flat_noise_pred.reshape(batch, frames, 1, height, width)
            last_cal = flat_cal.reshape(batch, frames, 1, height, width)
        else:
            t = torch.full((batch,), int(timestep), device=device, dtype=torch.long)
            noise_pred, last_cal = model(model_input, t)
        flat = scheduler.step(
            noise_pred.reshape(batch * frames, 1, height, width),
            timestep,
            image.reshape(batch * frames, 1, height, width),
            eta=eta,
            use_clipped_model_output=False,
            return_dict=False,
        )[0]
        image = flat.reshape_as(image)
        if temporal_smoothing_weight > 0 and (step_idx + 1) % every == 0:
            mask = temporal_smoothing_mask(
                image,
                temporal_smoothing_k2,
                temporal_smoothing_mask_mode,
                temporal_smoothing_k2_threshold,
                temporal_smoothing_k2_grad_threshold,
                temporal_smoothing_rss_grad_threshold,
                temporal_smoothing_mask_dilate,
            )
            image = temporal_smooth_clip(image, temporal_smoothing_weight, mask)
    return image.clamp(0.0, 1.0), last_cal.clamp(0.0, 1.0)


def make_clip_noise(
    batch_size: int,
    frames: int,
    height: int,
    width: int,
    seed: int,
    global_start_index: int,
    device: torch.device,
) -> torch.Tensor:
    noise = torch.empty((batch_size, frames, 1, height, width), device=device)
    for item in range(batch_size):
        generator = torch.Generator(device=device)
        generator.manual_seed(int(seed) + int(global_start_index) + int(item))
        noise[item] = torch.randn((frames, 1, height, width), generator=generator, device=device, dtype=noise.dtype)
    return noise


def summarize(rows: list[dict[str, float]]) -> dict[str, dict[str, float]]:
    keys = ["mae", "mse", "psnr", "cal_mae", "cal_mse", "cal_pinn", "cal_kflow", "pred_kflow_epe", "pred_kflow_l1"]
    out = {}
    for key in keys:
        arr = np.asarray([float(row[key]) for row in rows], dtype=np.float64)
        out[key] = {"mean": float(arr.mean()) if arr.size else float("nan"), "std": float(arr.std()) if arr.size else float("nan"), "count": int(arr.size)}
    return out


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


def write_eval_outputs(out_dir: Path, args: argparse.Namespace, rows: list[dict[str, float]], partial: bool = False) -> None:
    per_clip_tmp = out_dir / "per_clip.csv.tmp"
    with per_clip_tmp.open("w", newline="", encoding="utf-8") as handle:
        writer = csv.DictWriter(handle, fieldnames=FIELDS)
        writer.writeheader()
        writer.writerows(rows)
    per_clip_tmp.replace(out_dir / "per_clip.csv")

    summary = {
        "checkpoint_path": args.checkpoint_path,
        "init_2d_checkpoint": args.init_2d_checkpoint,
        "prior_injection_mode": args.prior_injection_mode,
        "fast_2d_forward": args.fast_2d_forward,
        "deterministic_clip_noise": args.deterministic_clip_noise,
        "primary_metrics_only": args.primary_metrics_only,
        "seed": args.seed,
        "split": args.split,
        "start_index": args.start_index,
        "end_index": args.end_index,
        "num_clips": len(rows),
        "partial": bool(partial),
        "metrics": summarize(rows),
    }
    summary_tmp = out_dir / "summary.json.tmp"
    summary_tmp.write_text(json.dumps(summary, indent=2), encoding="utf-8")
    summary_tmp.replace(out_dir / "summary.json")


def save_clip_frames(root: Path, pred: torch.Tensor, cal: torch.Tensor, gt: torch.Tensor) -> None:
    for name, tensor in (("pred_frames", pred), ("cal_frames", cal), ("gt_frames", gt)):
        frame_dir = root / name
        frame_dir.mkdir(parents=True, exist_ok=True)
        array = tensor.detach().cpu().numpy()
        for idx, frame in enumerate(array):
            image = np.clip(np.rint(frame[0] * 255.0), 0, 255).astype(np.uint8)
            Image.fromarray(image, mode="L").save(frame_dir / f"frame_{idx:06d}.png")


def main() -> None:
    args = parse_args()
    torch.manual_seed(args.seed)
    np.random.seed(args.seed)
    device = torch.device(args.device if torch.cuda.is_available() and args.device.startswith("cuda") else "cpu")
    starts = [int(item) for item in args.starts.split(",") if item.strip()] or None
    out_dir = Path(args.output_dir)
    out_dir.mkdir(parents=True, exist_ok=True)
    rows = load_existing_rows(out_dir / "per_clip.csv") if args.resume_existing else []
    resume_count = len(rows)
    if resume_count:
        print(f"Resuming from {resume_count} existing clips in {out_dir / 'per_clip.csv'}", flush=True)

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
        include_k2=True,
        k2_root=args.k2_root or None,
        path_mask_threshold=args.path_mask_threshold,
        path_mask_dilate=args.path_mask_dilate,
    )
    dataset_len = len(dataset)
    start_index = max(0, int(args.start_index))
    end_index = int(args.end_index) if int(args.end_index) > 0 else dataset_len
    end_index = min(dataset_len, max(start_index, end_index))
    effective_start_index = min(end_index, start_index + resume_count)
    indices = list(range(effective_start_index, end_index))
    if args.num_samples > 0:
        indices = indices[: min(args.num_samples, len(indices))]
    if effective_start_index > 0 or end_index < dataset_len or args.num_samples > 0:
        dataset = Subset(dataset, indices)
    loader = DataLoader(dataset, batch_size=args.batch_size, shuffle=False, num_workers=args.workers, pin_memory=True)

    base_model = build_unet_from_config(build_model_config(args))
    model = TemporalPINNUNet(
        base_model,
        input_channels=4,
        out_channels=1,
        prior_injection_mode=args.prior_injection_mode,
        temporal_num_heads=args.temporal_num_heads,
        use_frame_positional_encoding=args.use_frame_positional_encoding,
        max_frames=args.max_frames,
    )
    if args.checkpoint_path:
        model.load_state_dict(torch.load(args.checkpoint_path, map_location="cpu"), strict=True)
    elif args.init_2d_checkpoint:
        model.load_2d_state_dict(torch.load(args.init_2d_checkpoint, map_location="cpu"), strict=True)
    else:
        raise ValueError("Provide --checkpoint_path or --init_2d_checkpoint")
    eval_model = base_model if args.fast_2d_forward and not args.checkpoint_path else model
    eval_model.to(device).eval()

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
        path_mask = batch["path_mask"].to(device, non_blocking=True)
        current_batch_size, current_frames = gt.shape[:2]
        global_start_index = start_index + len(rows)
        initial_noise = None
        if args.deterministic_clip_noise:
            initial_noise = make_clip_noise(
                current_batch_size,
                current_frames,
                gt.shape[-2],
                gt.shape[-1],
                args.seed,
                global_start_index,
                device,
            )
        pred, cal = sample_ddim_video(
            eval_model,
            scheduler,
            conditions,
            args.ddim_steps,
            args.ddim_eta,
            device,
            temporal_smoothing_weight=args.temporal_smoothing_weight,
            temporal_smoothing_every=args.temporal_smoothing_every,
            temporal_smoothing_mask_mode=args.temporal_smoothing_mask_mode,
            temporal_smoothing_k2=batch.get("k2", None).to(device, non_blocking=True) if "k2" in batch else None,
            temporal_smoothing_k2_threshold=args.temporal_smoothing_k2_threshold,
            temporal_smoothing_k2_grad_threshold=args.temporal_smoothing_k2_grad_threshold,
            temporal_smoothing_rss_grad_threshold=args.temporal_smoothing_rss_grad_threshold,
            temporal_smoothing_mask_dilate=args.temporal_smoothing_mask_dilate,
            fast_2d_forward=args.fast_2d_forward and not args.checkpoint_path,
            initial_noise=initial_noise,
        )
        batch_size, frames = gt.shape[:2]
        if args.primary_metrics_only:
            cal_pinn_values = None
        else:
            buildings = raw_inputs[:, :, 0].reshape(batch_size * frames, *gt.shape[-2:])
            antenna = raw_inputs[:, :, 1].reshape(batch_size * frames, *gt.shape[-2:])
            cal_pinn_values = cal_pinn(cal[:, :, 0].reshape(batch_size * frames, *gt.shape[-2:]), buildings, antenna, k=args.pinn_k)

        for item in range(batch_size):
            err = pred[item : item + 1] - gt[item : item + 1]
            cal_err = cal[item : item + 1] - gt[item : item + 1]
            mse = float(err.square().mean().detach().cpu())
            if args.primary_metrics_only:
                pred_epe, pred_l1 = float("nan"), float("nan")
                cal_mae = float("nan")
                cal_mse = float("nan")
                cal_pinn_value = float("nan")
                cal_kflow = float("nan")
            else:
                pred_epe, pred_l1 = flow_metrics(
                    pred[item : item + 1],
                    gt[item : item + 1],
                    path_mask[item : item + 1],
                    args.kflow_delta,
                    args.kflow_clamp,
                    target_flow_threshold=args.kflow_eval_target_flow_threshold,
                    target_flow_quantile=args.kflow_eval_target_flow_quantile,
                )
                cal_mae = float(cal_err.abs().mean().detach().cpu())
                cal_mse = float(cal_err.square().mean().detach().cpu())
                cal_pinn_value = float(cal_pinn_values[item * frames : (item + 1) * frames].mean().detach().cpu())
                cal_kflow = float(
                    temporal_kflow_loss(
                        cal[item : item + 1],
                        gt[item : item + 1],
                        path_mask[item : item + 1],
                        args.kflow_delta,
                        args.kflow_clamp,
                        target_flow_threshold=args.kflow_eval_target_flow_threshold,
                        target_flow_quantile=args.kflow_eval_target_flow_quantile,
                    ).detach().cpu()
                )
            row = {
                "clip_index": len(rows),
                "start": int(batch["start"][item]),
                "mae": float(err.abs().mean().detach().cpu()),
                "mse": mse,
                "psnr": psnr_from_mse(mse),
                "cal_mae": cal_mae,
                "cal_mse": cal_mse,
                "cal_pinn": cal_pinn_value,
                "cal_kflow": cal_kflow,
                "pred_kflow_epe": pred_epe,
                "pred_kflow_l1": pred_l1,
            }
            rows.append(row)
            if args.save_arrays:
                np.savez_compressed(
                    out_dir / f"clip_{row['clip_index']:06d}.npz",
                    pred=pred[item].detach().cpu().numpy(),
                    cal=cal[item].detach().cpu().numpy(),
                    gt=gt[item].detach().cpu().numpy(),
                    k2=batch["k2"][item].numpy(),
                    path_mask=batch["path_mask"][item].numpy(),
                    names=np.asarray(batch["names"][item]),
                )
            if args.save_frames:
                save_clip_frames(out_dir / f"clip_{row['clip_index']:06d}", pred[item], cal[item], gt[item])
            if flush_every > 0 and len(rows) % flush_every == 0:
                write_eval_outputs(out_dir, args, rows, partial=True)
                main_metrics = summarize(rows)
                print(
                    json.dumps(
                        {
                            "partial": True,
                            "num_clips": len(rows),
                            "mae": main_metrics["mae"]["mean"],
                            "mse": main_metrics["mse"]["mean"],
                            "psnr": main_metrics["psnr"]["mean"],
                        },
                        indent=2,
                    ),
                    flush=True,
                )

    write_eval_outputs(out_dir, args, rows, partial=False)
    print((out_dir / "summary.json").read_text(encoding="utf-8"), flush=True)


if __name__ == "__main__":
    main()
