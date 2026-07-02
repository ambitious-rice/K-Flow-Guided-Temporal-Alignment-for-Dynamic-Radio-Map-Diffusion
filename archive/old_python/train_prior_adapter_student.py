#!/usr/bin/env python3
"""Train temporal adapters on the frozen RMDM highway/PINN prior branch."""

from __future__ import annotations

import argparse
import json
from pathlib import Path

import torch
import torch.nn.functional as F
from accelerate import Accelerator
from accelerate.utils import DistributedDataParallelKwargs
from diffusers import DDPMScheduler
from torch.utils.data import DataLoader

from lib.dynamic_clip_loaders import DynamicRadioMapRMDMClip
from prior_adapter_student import FrozenRMDMPriorAdapterStudent
from residual_kflow_student import (
    KPathDeltaConfig,
    StaticKeyPathConfig,
    kpath_delta_target_loss,
    predict_x0_from_eps,
    static_keypath_consistency_loss,
)
from train_temporal_pinn import build_model_config, preprocess_conditions
from utils import build_unet_from_config


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser()
    parser.add_argument("--data_dir", required=True)
    parser.add_argument("--split_file", default="split.json")
    parser.add_argument("--scene_ids", default="")
    parser.add_argument("--image_size", type=int, default=128)
    parser.add_argument("--clip_length", type=int, default=32)
    parser.add_argument("--frame_stride", type=int, default=1)
    parser.add_argument("--clip_stride", type=int, default=1)
    parser.add_argument("--batch_size", type=int, default=1)
    parser.add_argument("--workers", type=int, default=2)
    parser.add_argument("--cache_size", type=int, default=8)
    parser.add_argument("--tx_heatmap_sigma_px", type=float, default=1.5)
    parser.add_argument("--k2_root", default="")
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
    parser.add_argument("--init_2d_checkpoint", required=True)
    parser.add_argument("--resume_adapter_checkpoint", default="")
    parser.add_argument("--diffusion_steps", type=int, default=1000)
    parser.add_argument("--noise_schedule", choices=("linear", "cosine"), default="linear")
    parser.add_argument("--adapter_reduction", type=int, default=4)
    parser.add_argument("--adapter_min_hidden", type=int, default=8)
    parser.add_argument("--no_cal_adapter", action="store_true")
    parser.add_argument("--lr", type=float, default=2e-4)
    parser.add_argument("--weight_decay", type=float, default=1e-4)
    parser.add_argument("--mixed_precision", choices=("no", "fp16", "bf16"), default="no")
    parser.add_argument("--max_steps", type=int, default=500)
    parser.add_argument("--gradient_accumulation_steps", type=int, default=4)
    parser.add_argument("--log_interval", type=int, default=10)
    parser.add_argument("--save_interval", type=int, default=250)
    parser.add_argument("--save_dir", default="./checkpoints_prior_adapter_student")
    parser.add_argument("--diff_loss_weight", type=float, default=1.0)
    parser.add_argument("--teacher_anchor_weight", type=float, default=0.0)
    parser.add_argument("--low_timestep_prob", type=float, default=0.5)
    parser.add_argument("--max_loss_timestep", type=int, default=300)
    parser.add_argument("--static_keypath_weight", type=float, default=0.001)
    parser.add_argument("--static_keypath_margin", type=float, default=0.005)
    parser.add_argument("--static_keypath_k2_quantile", type=float, default=0.70)
    parser.add_argument("--static_keypath_flow_quantile", type=float, default=0.70)
    parser.add_argument("--kpath_delta_weight", type=float, default=0.001)
    parser.add_argument("--kpath_delta_k2_quantile", type=float, default=0.70)
    parser.add_argument("--kpath_delta_static_flow_quantile", type=float, default=0.70)
    parser.add_argument("--kpath_delta_dynamic_flow_quantile", type=float, default=0.85)
    parser.add_argument("--kpath_delta_margin", type=float, default=0.005)
    parser.add_argument("--kpath_delta_dynamic_lambda", type=float, default=0.1)
    return parser.parse_args()


def save_checkpoint(model: FrozenRMDMPriorAdapterStudent, args: argparse.Namespace, step: int, path: Path) -> None:
    raw = model.module if hasattr(model, "module") else model
    path.parent.mkdir(parents=True, exist_ok=True)
    payload = {
        "step": int(step),
        "args": vars(args),
        "anchor_adapters": raw.anchor_adapters.state_dict(),
        "adapt_cal": raw.adapt_cal,
    }
    if raw.cal_adapter is not None:
        payload["cal_adapter"] = raw.cal_adapter.state_dict()
    torch.save(payload, path)


def main() -> None:
    args = parse_args()
    scene_ids = [item.strip() for item in args.scene_ids.split(",") if item.strip()] or None
    accelerator = Accelerator(
        mixed_precision=args.mixed_precision,
        gradient_accumulation_steps=args.gradient_accumulation_steps,
        kwargs_handlers=[DistributedDataParallelKwargs(find_unused_parameters=False)],
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
        include_k2=True,
        k2_root=args.k2_root or None,
        scene_ids=scene_ids,
    )
    loader = DataLoader(dataset, batch_size=args.batch_size, shuffle=True, num_workers=args.workers, pin_memory=True, drop_last=True)

    base_model = build_unet_from_config(build_model_config(args))
    base_model.load_state_dict(torch.load(args.init_2d_checkpoint, map_location="cpu"), strict=True)
    model = FrozenRMDMPriorAdapterStudent(
        base_model,
        input_channels=4,
        out_channels=1,
        reduction=args.adapter_reduction,
        min_hidden=args.adapter_min_hidden,
        adapt_cal=not args.no_cal_adapter,
    )
    if args.resume_adapter_checkpoint:
        payload = torch.load(args.resume_adapter_checkpoint, map_location="cpu")
        model.anchor_adapters.load_state_dict(payload["anchor_adapters"], strict=True)
        if model.cal_adapter is not None and "cal_adapter" in payload:
            model.cal_adapter.load_state_dict(payload["cal_adapter"], strict=True)

    optimizer = torch.optim.AdamW(model.adapter_parameters(), lr=args.lr, weight_decay=args.weight_decay)
    beta_schedule = "linear" if args.noise_schedule == "linear" else "squaredcos_cap_v2"
    scheduler = DDPMScheduler(num_train_timesteps=args.diffusion_steps, beta_schedule=beta_schedule, prediction_type="epsilon")
    model, optimizer, loader = accelerator.prepare(model, optimizer, loader)
    model.train()

    if accelerator.is_main_process:
        save_dir = Path(args.save_dir)
        save_dir.mkdir(parents=True, exist_ok=True)
        (save_dir / "run_config.json").write_text(json.dumps(vars(args), indent=2), encoding="utf-8")
        print(f"dataset clips: {len(dataset)}")
        print(f"scene_ids: {scene_ids or 'all train scenes'}")
        print(f"trainable params: {sum(p.numel() for p in model.parameters() if p.requires_grad)} / {sum(p.numel() for p in model.parameters())}")
        print(f"teacher_anchor_weight: {args.teacher_anchor_weight}")

    static_cfg = StaticKeyPathConfig(
        keypath_k2_quantile=args.static_keypath_k2_quantile,
        static_flow_quantile=args.static_keypath_flow_quantile,
        margin=args.static_keypath_margin,
    )
    kpd_cfg = KPathDeltaConfig(
        keypath_k2_quantile=args.kpath_delta_k2_quantile,
        static_flow_quantile=args.kpath_delta_static_flow_quantile,
        dynamic_flow_quantile=args.kpath_delta_dynamic_flow_quantile,
        margin=args.kpath_delta_margin,
        dynamic_lambda=args.kpath_delta_dynamic_lambda,
    )

    step = 0
    while step < args.max_steps:
        for batch in loader:
            with accelerator.accumulate(model):
                raw_inputs = batch["inputs"].to(device, non_blocking=True)
                conditions = preprocess_conditions(raw_inputs)
                target = batch["image"].to(device, non_blocking=True)
                k2 = batch["k2"].to(device, non_blocking=True)
                batch_size, frames = target.shape[:2]
                t = torch.randint(0, scheduler.config.num_train_timesteps, (batch_size,), device=device).long()
                if args.low_timestep_prob > 0:
                    choose_low = torch.rand(batch_size, device=device) < float(args.low_timestep_prob)
                    low_t = torch.randint(0, min(args.max_loss_timestep + 1, scheduler.config.num_train_timesteps), (batch_size,), device=device).long()
                    t = torch.where(choose_low, low_t, t)
                noise = torch.randn_like(target)
                flat_t = t[:, None].repeat(1, frames).reshape(batch_size * frames)
                noisy = scheduler.add_noise(
                    target.reshape(batch_size * frames, *target.shape[2:]),
                    noise.reshape(batch_size * frames, *noise.shape[2:]),
                    flat_t,
                ).reshape_as(target)

                eps, _, extra = model(conditions, noisy, t)
                eps_base = extra["eps_base"].detach()
                loss_diff = F.mse_loss(eps, noise)
                loss_anchor = F.mse_loss(eps, eps_base)
                x0 = predict_x0_from_eps(noisy, eps, t, scheduler.alphas_cumprod)
                low_mask = t <= int(args.max_loss_timestep)
                loss_static, static_stats = static_keypath_consistency_loss(x0, target, k2, batch_mask=low_mask, cfg=static_cfg)
                loss_kpd, kpd_stats = kpath_delta_target_loss(x0, target, k2, batch_mask=low_mask, cfg=kpd_cfg)
                loss = (
                    args.diff_loss_weight * loss_diff
                    + args.teacher_anchor_weight * loss_anchor
                    + args.static_keypath_weight * loss_static
                    + args.kpath_delta_weight * loss_kpd
                )
                optimizer.zero_grad()
                accelerator.backward(loss)
                optimizer.step()

            if accelerator.is_main_process and step % args.log_interval == 0:
                print(
                    f"step {step} loss {loss.item():.6f} diff {loss_diff.item():.6f} "
                    f"anchor {loss_anchor.item():.9f} static {loss_static.item():.6f} "
                    f"kpd {loss_kpd.item():.6f} kpd_static {kpd_stats['static_loss'].item():.6f} "
                    f"kpd_dynamic {kpd_stats['dynamic_loss'].item():.6f} "
                    f"kpd_pred {kpd_stats['pred_dynamic'].item():.6f} "
                    f"kpd_gt {kpd_stats['gt_dynamic'].item():.6f} "
                    f"eps_delta {(eps - eps_base).detach().abs().mean().item():.9f} "
                    f"anchor_delta {extra['anchor_delta'].item():.9f} "
                    f"cal_delta {extra['cal_delta'].item():.9f} "
                    f"gate {extra['gate_mean'].item():.6f}",
                    flush=True,
                )
            if accelerator.is_main_process and args.save_interval > 0 and step > 0 and step % args.save_interval == 0:
                save_checkpoint(accelerator.unwrap_model(model), args, step, Path(args.save_dir) / f"prior_adapter_step{step}.pth")
            step += 1
            if step >= args.max_steps:
                break

    if accelerator.is_main_process:
        save_checkpoint(accelerator.unwrap_model(model), args, step, Path(args.save_dir) / "prior_adapter_final.pth")


if __name__ == "__main__":
    main()
