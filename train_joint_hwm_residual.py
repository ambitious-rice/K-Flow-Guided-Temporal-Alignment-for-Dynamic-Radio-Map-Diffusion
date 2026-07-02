#!/usr/bin/env python3
"""Jointly train Stage-1 HWM adapters and Stage-2 residual head."""

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

from joint_hwm_residual_student import JointHWMResidualStudent
from lib.dynamic_clip_loaders import DynamicRadioMapRMDMClip
from residual_kflow_student import KPathDeltaConfig, kpath_delta_target_loss, predict_x0_from_eps
from train_temporal_pinn import build_model_config, preprocess_conditions
from utils import build_unet_from_config, cal_pinn


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser()
    parser.add_argument("--data_dir", required=True)
    parser.add_argument("--split_file", default="split.json")
    parser.add_argument("--scene_ids", default="")
    parser.add_argument("--image_size", type=int, default=128)
    parser.add_argument("--clip_length", type=int, default=32)
    parser.add_argument("--frame_stride", type=int, default=1)
    parser.add_argument("--clip_stride", type=int, default=1)
    parser.add_argument("--batch_size", type=int, default=2)
    parser.add_argument("--workers", type=int, default=4)
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
    parser.add_argument("--init_hwm_adapter_checkpoint", default="")
    parser.add_argument("--resume_joint_checkpoint", default="")
    parser.add_argument("--adapter_indices", default="-2,-1")
    parser.add_argument("--adapter_reduction", type=int, default=1)
    parser.add_argument("--adapter_min_hidden", type=int, default=16)
    parser.add_argument("--head_hidden_channels", type=int, default=32)
    parser.add_argument("--head_num_layers", type=int, default=2)
    parser.add_argument("--residual_alpha", type=float, default=0.03)
    parser.add_argument("--diffusion_steps", type=int, default=1000)
    parser.add_argument("--noise_schedule", choices=("linear", "cosine"), default="linear")
    parser.add_argument("--low_timestep_prob", type=float, default=0.5)
    parser.add_argument("--temporal_max_timestep", type=int, default=300)
    parser.add_argument("--lr", type=float, default=1e-4)
    parser.add_argument("--weight_decay", type=float, default=1e-4)
    parser.add_argument("--mixed_precision", choices=("no", "fp16", "bf16"), default="no")
    parser.add_argument("--max_steps", type=int, default=500)
    parser.add_argument("--gradient_accumulation_steps", type=int, default=1)
    parser.add_argument("--log_interval", type=int, default=20)
    parser.add_argument("--save_interval", type=int, default=250)
    parser.add_argument("--save_dir", default="./checkpoints_joint_hwm_residual")
    parser.add_argument("--diff_loss_weight", type=float, default=1.0)
    parser.add_argument("--x0_recon_weight", type=float, default=0.05)
    parser.add_argument("--stage1_cal_weight", type=float, default=0.3)
    parser.add_argument("--stage1_pinn_weight", type=float, default=0.3)
    parser.add_argument("--stage1_kpath_weight", type=float, default=1.0)
    parser.add_argument("--final_kpath_weight", type=float, default=0.001)
    parser.add_argument("--pinn_k", type=float, default=0.2)
    parser.add_argument("--kpath_delta_k2_quantile", type=float, default=0.70)
    parser.add_argument("--kpath_delta_static_flow_quantile", type=float, default=0.70)
    parser.add_argument("--kpath_delta_dynamic_flow_quantile", type=float, default=0.85)
    parser.add_argument("--kpath_delta_margin", type=float, default=0.005)
    parser.add_argument("--kpath_delta_dynamic_lambda", type=float, default=1.0)
    return parser.parse_args()


def save_checkpoint(model: JointHWMResidualStudent, args: argparse.Namespace, step: int, path: Path) -> None:
    raw = model.module if hasattr(model, "module") else model
    path.parent.mkdir(parents=True, exist_ok=True)
    torch.save(
        {
            "step": int(step),
            "args": vars(args),
            "hwm_adapters": raw.hwm_adapter.adapters.state_dict(),
            "residual_head": raw.residual_head.state_dict(),
            "residual_alpha": raw.alpha,
            "adapter_indices": raw.hwm_adapter.adapter_indices,
        },
        path,
    )


def main() -> None:
    args = parse_args()
    accelerator = Accelerator(
        mixed_precision=args.mixed_precision,
        gradient_accumulation_steps=args.gradient_accumulation_steps,
        kwargs_handlers=[DistributedDataParallelKwargs(find_unused_parameters=True)],
    )
    device = accelerator.device
    scene_ids = [item.strip() for item in args.scene_ids.split(",") if item.strip()] or None
    adapter_indices = tuple(int(item.strip()) for item in args.adapter_indices.split(",") if item.strip())
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
    model = JointHWMResidualStudent(
        base_model,
        adapter_indices=adapter_indices,
        adapter_reduction=args.adapter_reduction,
        adapter_min_hidden=args.adapter_min_hidden,
        head_hidden_channels=args.head_hidden_channels,
        head_num_layers=args.head_num_layers,
        residual_alpha=args.residual_alpha,
        diffusion_steps=args.diffusion_steps,
        detach_anchor_gate=False,
    )
    if args.init_hwm_adapter_checkpoint:
        model.load_hwm_adapter_checkpoint(args.init_hwm_adapter_checkpoint)
    if args.resume_joint_checkpoint:
        payload = torch.load(args.resume_joint_checkpoint, map_location="cpu")
        model.hwm_adapter.adapters.load_state_dict(payload["hwm_adapters"], strict=True)
        model.residual_head.load_state_dict(payload["residual_head"], strict=True)
        model.alpha = float(payload.get("residual_alpha", args.residual_alpha))

    optimizer = torch.optim.AdamW(model.trainable_parameters(), lr=args.lr, weight_decay=args.weight_decay)
    model, optimizer, loader = accelerator.prepare(model, optimizer, loader)
    model.train()

    beta_schedule = "linear" if args.noise_schedule == "linear" else "squaredcos_cap_v2"
    scheduler = DDPMScheduler(num_train_timesteps=args.diffusion_steps, beta_schedule=beta_schedule, prediction_type="epsilon", clip_sample=False)
    scheduler.alphas_cumprod = scheduler.alphas_cumprod.to(device)
    kpd_cfg = KPathDeltaConfig(
        keypath_k2_quantile=args.kpath_delta_k2_quantile,
        static_flow_quantile=args.kpath_delta_static_flow_quantile,
        dynamic_flow_quantile=args.kpath_delta_dynamic_flow_quantile,
        margin=args.kpath_delta_margin,
        dynamic_lambda=args.kpath_delta_dynamic_lambda,
    )

    if accelerator.is_main_process:
        save_dir = Path(args.save_dir)
        save_dir.mkdir(parents=True, exist_ok=True)
        (save_dir / "run_config.json").write_text(json.dumps(vars(args), indent=2), encoding="utf-8")
        print(f"dataset clips: {len(dataset)}")
        print(f"scene_ids: {scene_ids or 'all train scenes'}")
        print(f"trainable params: {sum(p.numel() for p in model.parameters() if p.requires_grad)} / {sum(p.numel() for p in model.parameters())}")

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
                    low_mask = torch.rand(batch_size, device=device) < float(args.low_timestep_prob)
                    low_t = torch.randint(0, min(args.temporal_max_timestep + 1, args.diffusion_steps), (batch_size,), device=device).long()
                    t = torch.where(low_mask, low_t, t)
                noise = torch.randn_like(target)
                flat_target = target.reshape(batch_size * frames, *target.shape[2:])
                flat_noise = noise.reshape(batch_size * frames, *noise.shape[2:])
                flat_t = t[:, None].repeat(1, frames).reshape(batch_size * frames)
                noisy = scheduler.add_noise(flat_target, flat_noise, flat_t).reshape_as(target)

                eps_final, cal, extra = model(conditions, noisy, t)
                loss_diff = F.mse_loss(eps_final, noise)
                x0_final = predict_x0_from_eps(noisy, eps_final, t, scheduler.alphas_cumprod).clamp(0.0, 1.0)
                low_batch = t <= int(args.temporal_max_timestep)
                if args.x0_recon_weight > 0 and torch.any(low_batch):
                    mask = low_batch.to(dtype=target.dtype).view(batch_size, 1, 1, 1, 1)
                    denom = mask.sum().clamp_min(1.0) * frames * target.shape[2] * target.shape[3] * target.shape[4]
                    loss_x0 = ((x0_final - target).square() * mask).sum() / denom
                else:
                    loss_x0 = target.new_tensor(0.0)

                buildings = raw_inputs[:, :, 0].reshape(batch_size * frames, *target.shape[-2:])
                antenna = raw_inputs[:, :, 1].reshape(batch_size * frames, *target.shape[-2:])
                flat_cal = cal[:, :, 0].reshape(batch_size * frames, *target.shape[-2:])
                loss_cal = F.mse_loss(cal, target)
                loss_pinn = cal_pinn(flat_cal, buildings, antenna, k=args.pinn_k).mean()
                loss_stage1_kpd, stage1_stats = kpath_delta_target_loss(cal, target, k2, cfg=kpd_cfg)
                if args.final_kpath_weight > 0 and torch.any(low_batch):
                    loss_final_kpd, final_stats = kpath_delta_target_loss(x0_final, target, k2, batch_mask=low_batch, cfg=kpd_cfg)
                else:
                    loss_final_kpd = target.new_tensor(0.0)
                    final_stats = {
                        "static_loss": target.new_tensor(0.0),
                        "dynamic_loss": target.new_tensor(0.0),
                        "pred_dynamic": target.new_tensor(0.0),
                        "gt_dynamic": target.new_tensor(0.0),
                    }
                loss = (
                    args.diff_loss_weight * loss_diff
                    + args.x0_recon_weight * loss_x0
                    + args.stage1_cal_weight * loss_cal
                    + args.stage1_pinn_weight * loss_pinn
                    + args.stage1_kpath_weight * loss_stage1_kpd
                    + args.final_kpath_weight * loss_final_kpd
                )
                optimizer.zero_grad()
                accelerator.backward(loss)
                optimizer.step()

            if accelerator.is_main_process and step % args.log_interval == 0:
                print(
                    f"step {step} loss {loss.item():.6f} diff {loss_diff.item():.6f} "
                    f"x0 {loss_x0.item():.6f} cal {loss_cal.item():.6f} pinn {loss_pinn.item():.6f} "
                    f"s1kpd {loss_stage1_kpd.item():.6f} s1_static {stage1_stats['static_loss'].item():.6f} "
                    f"s1_dyn {stage1_stats['dynamic_loss'].item():.6f} s1_pred {stage1_stats['pred_dynamic'].item():.6f} "
                    f"s1_gt {stage1_stats['gt_dynamic'].item():.6f} fkpd {loss_final_kpd.item():.6f} "
                    f"f_static {final_stats['static_loss'].item():.6f} f_dyn {final_stats['dynamic_loss'].item():.6f} "
                    f"f_pred {final_stats['pred_dynamic'].item():.6f} f_gt {final_stats['gt_dynamic'].item():.6f} "
                    f"eps_delta {extra['delta_eps'].detach().abs().mean().item():.9f} "
                    f"adapter_delta {extra['adapter_delta_mean'].detach().item():.9f}",
                    flush=True,
                )
            if accelerator.is_main_process and args.save_interval > 0 and step > 0 and step % args.save_interval == 0:
                save_checkpoint(accelerator.unwrap_model(model), args, step, Path(args.save_dir) / f"joint_hwm_residual_step{step}.pth")
            step += 1
            if step >= args.max_steps:
                break

    if accelerator.is_main_process:
        save_checkpoint(accelerator.unwrap_model(model), args, step, Path(args.save_dir) / "joint_hwm_residual_final.pth")


if __name__ == "__main__":
    main()
