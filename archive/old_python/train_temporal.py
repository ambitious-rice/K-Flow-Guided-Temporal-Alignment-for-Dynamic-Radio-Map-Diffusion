#!/usr/bin/env python3
"""Train temporal RMDM adapters on DynamicRadio clips.

This script does not modify or replace the original RMDM baseline. It builds the
original single-frame UNet, wraps it with temporal adapters, and optionally
freezes the original 2D model.
"""

from __future__ import annotations

import argparse
import os
import re
from pathlib import Path

import torch
import torch.nn.functional as F
from accelerate import Accelerator
from accelerate.utils import DistributedDataParallelKwargs
from diffusers import DDPMScheduler
from torch.utils.data import DataLoader

from lib.dynamic_clip_loaders import DynamicRadioMapRMDMClip
from temporal_unet import RMDMTemporalWrapper
from utils import build_unet_from_config, cal_pinn


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser()
    parser.add_argument("--data_dir", required=True)
    parser.add_argument("--split_file", default="split.json")
    parser.add_argument("--image_size", type=int, default=128)
    parser.add_argument("--clip_length", type=int, default=32)
    parser.add_argument("--frame_stride", type=int, default=16)
    parser.add_argument("--clip_stride", type=int, default=1)
    parser.add_argument("--batch_size", type=int, default=2)
    parser.add_argument("--workers", type=int, default=4)
    parser.add_argument("--cache_size", type=int, default=8)
    parser.add_argument("--tx_heatmap_sigma_px", type=float, default=1.5)
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
    parser.add_argument("--diffusion_steps", type=int, default=1000)
    parser.add_argument("--noise_schedule", choices=("linear", "cosine"), default="linear")
    parser.add_argument("--lr", type=float, default=5e-5)
    parser.add_argument("--base_lr", type=float, default=2.5e-6)
    parser.add_argument("--mixed_precision", choices=("no", "fp16", "bf16"), default="no")
    parser.add_argument("--max_steps", type=int, default=5000)
    parser.add_argument("--gradient_accumulation_steps", type=int, default=1)
    parser.add_argument("--log_interval", type=int, default=50)
    parser.add_argument("--save_interval", type=int, default=1000)
    parser.add_argument("--save_dir", default="./checkpoints_dynamic_rmdm_temporal")
    parser.add_argument("--init_2d_checkpoint", required=True)
    parser.add_argument("--resume_from", default="")
    parser.set_defaults(train_temporal_only=True)
    parser.add_argument("--train_temporal_only", dest="train_temporal_only", action="store_true")
    parser.add_argument("--train_base", dest="train_temporal_only", action="store_false")
    parser.add_argument("--train_base_keywords", default="")
    parser.add_argument("--temporal_hidden_channels", type=int, default=16)
    return parser.parse_args()


def build_model_config(args: argparse.Namespace) -> dict:
    return {
        "image_size": args.image_size,
        "in_ch": 4,
        "out_ch": 1,
        "num_channels": args.num_channels,
        "num_res_blocks": args.num_res_blocks,
        "channel_mult": args.channel_mult,
        "num_heads": args.num_heads,
        "num_head_channels": args.num_head_channels,
        "num_heads_upsample": args.num_heads_upsample,
        "attention_resolutions": args.attention_resolutions,
        "dropout": args.dropout,
        "class_cond": False,
        "use_checkpoint": args.use_checkpoint,
        "use_scale_shift_norm": args.use_scale_shift_norm,
        "resblock_updown": args.resblock_updown,
        "use_fp16": args.use_fp16,
        "use_new_attention_order": args.use_new_attention_order,
        "learn_sigma": False,
    }


def preprocess_conditions(conditions: torch.Tensor) -> torch.Tensor:
    conditions = conditions.clone()
    if conditions.size(2) >= 2:
        conditions[:, :, 0, ...] = conditions[:, :, 0, ...] + 10.0 * conditions[:, :, 1, ...]
    return conditions


def configure_trainable(model: RMDMTemporalWrapper, args: argparse.Namespace):
    if args.train_temporal_only:
        model.freeze_base_model()
    keywords = [item.strip() for item in args.train_base_keywords.split(",") if item.strip()]
    if keywords:
        for name, param in model.base_model.named_parameters():
            if any(keyword in name for keyword in keywords):
                param.requires_grad = True

    temporal_params = []
    base_params = []
    for name, param in model.named_parameters():
        if not param.requires_grad:
            continue
        if name.startswith("base_model."):
            base_params.append(param)
        else:
            temporal_params.append(param)
    groups = []
    if temporal_params:
        groups.append({"params": temporal_params, "lr": args.lr, "name": "temporal"})
    if base_params:
        groups.append({"params": base_params, "lr": args.base_lr, "name": "base"})
    return groups


def infer_step(path: str) -> int:
    match = re.search(r"step(\d+)", os.path.basename(path))
    return int(match.group(1)) if match else 0


def main() -> None:
    args = parse_args()
    ddp_kwargs = DistributedDataParallelKwargs(find_unused_parameters=True)
    accelerator = Accelerator(
        mixed_precision=args.mixed_precision,
        gradient_accumulation_steps=args.gradient_accumulation_steps,
        kwargs_handlers=[ddp_kwargs],
    )
    device = accelerator.device

    dataset = DynamicRadioMapRMDMClip(
        root=args.data_dir,
        split="train",
        split_file=args.split_file,
        clip_length=args.clip_length,
        frame_stride=args.frame_stride,
        clip_stride=args.clip_stride,
        cache_size=args.cache_size,
        tx_heatmap_sigma_px=args.tx_heatmap_sigma_px,
    )
    loader = DataLoader(
        dataset,
        batch_size=args.batch_size,
        shuffle=True,
        num_workers=args.workers,
        pin_memory=True,
        drop_last=True,
    )

    base_model = build_unet_from_config(build_model_config(args))
    model = RMDMTemporalWrapper(
        base_model,
        input_channels=4,
        out_channels=1,
        temporal_hidden_channels=args.temporal_hidden_channels,
    )
    if args.resume_from:
        state = torch.load(args.resume_from, map_location="cpu")
        model.load_state_dict(state, strict=True)
        start_step = infer_step(args.resume_from)
    else:
        state = torch.load(args.init_2d_checkpoint, map_location="cpu")
        model.load_2d_state_dict(state, strict=True)
        start_step = 0

    param_groups = configure_trainable(model, args)
    if not param_groups:
        raise RuntimeError("No trainable parameters selected.")
    optimizer = torch.optim.AdamW(param_groups, weight_decay=1e-4)

    beta_schedule = "linear" if args.noise_schedule == "linear" else "squaredcos_cap_v2"
    scheduler = DDPMScheduler(
        num_train_timesteps=args.diffusion_steps,
        beta_schedule=beta_schedule,
        prediction_type="epsilon",
    )

    model, optimizer, loader = accelerator.prepare(model, optimizer, loader)
    model.train()

    if accelerator.is_main_process:
        trainable = sum(p.numel() for p in model.parameters() if p.requires_grad)
        total = sum(p.numel() for p in model.parameters())
        print(f"dataset clips: {len(dataset)}")
        print(f"trainable params: {trainable} / {total}")
        print(f"optimizer groups: {[{k: v for k, v in g.items() if k != 'params'} for g in param_groups]}")

    step = start_step
    while step < args.max_steps:
        for batch in loader:
            with accelerator.accumulate(model):
                conditions = preprocess_conditions(batch["inputs"].to(device, non_blocking=True))
                target_clean = batch["image"].to(device, non_blocking=True)
                batch_size, frames = target_clean.shape[:2]
                t = torch.randint(0, scheduler.config.num_train_timesteps, (batch_size,), device=device).long()
                noise = torch.randn_like(target_clean)
                flat_target = target_clean.reshape(batch_size * frames, *target_clean.shape[2:])
                flat_noise = noise.reshape(batch_size * frames, *noise.shape[2:])
                flat_t = t[:, None].repeat(1, frames).reshape(batch_size * frames)
                noisy = scheduler.add_noise(flat_target, flat_noise, flat_t).reshape_as(target_clean)

                model_input = torch.cat([conditions, noisy], dim=2)
                pred_noise, cal = model(model_input, t)

                loss_diff = F.mse_loss(pred_noise, noise)
                loss_cal = F.mse_loss(cal, target_clean)
                buildings = batch["inputs"][:, :, 0].reshape(batch_size * frames, *target_clean.shape[-2:]).to(device)
                antenna = batch["inputs"][:, :, 1].reshape(batch_size * frames, *target_clean.shape[-2:]).to(device)
                flat_cal = cal[:, :, 0].reshape(batch_size * frames, *target_clean.shape[-2:])
                loss_pinn = cal_pinn(flat_cal, buildings, antenna, k=0.2).mean()
                loss = loss_diff + loss_cal + loss_pinn

                optimizer.zero_grad()
                accelerator.backward(loss)
                optimizer.step()

            if accelerator.is_main_process and step % args.log_interval == 0:
                print(
                    f"step {step} loss {loss.item():.5f} diff {loss_diff.item():.5f} "
                    f"cal {loss_cal.item():.5f} pinn {loss_pinn.item():.5f}",
                    flush=True,
                )
            if accelerator.is_main_process and args.save_interval > 0 and step > start_step:
                if step % args.save_interval == 0:
                    Path(args.save_dir).mkdir(parents=True, exist_ok=True)
                    unwrapped = accelerator.unwrap_model(model)
                    torch.save(unwrapped.state_dict(), Path(args.save_dir) / f"model_temporal_step{step}.pth")

            step += 1
            if step >= args.max_steps:
                break

    if accelerator.is_main_process:
        Path(args.save_dir).mkdir(parents=True, exist_ok=True)
        unwrapped = accelerator.unwrap_model(model)
        torch.save(unwrapped.state_dict(), Path(args.save_dir) / "model_temporal.pth")


if __name__ == "__main__":
    main()
