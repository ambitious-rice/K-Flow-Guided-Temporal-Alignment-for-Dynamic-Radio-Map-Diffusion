#!/usr/bin/env python3
"""Train a zero-init residual head on top of frozen RMDM teacher/base."""

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
from residual_kflow_student import (
    DynamicKeyPathConfig,
    FrozenRMDMResidualStudent,
    GlobalDynamicRangeConfig,
    GlobalTemporalResidualConfig,
    KPathDeltaConfig,
    KFlowStructureConfig,
    SafeKFlowConfig,
    StaticKeyPathConfig,
    dynamic_keypath_range_loss,
    global_dynamic_range_loss,
    global_temporal_residual_loss,
    kpath_delta_target_loss,
    kflow_structure_loss,
    predict_x0_from_eps,
    safe_kflow_hinge_loss,
    static_keypath_consistency_loss,
)
from train_temporal_pinn import build_model_config, preprocess_conditions
from utils import build_unet_from_config


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser()
    parser.add_argument("--data_dir", required=True)
    parser.add_argument("--split_file", default="split.json")
    parser.add_argument("--scene_ids", default="", help="Comma-separated train scene ids for small probes")
    parser.add_argument("--image_size", type=int, default=128)
    parser.add_argument("--clip_length", type=int, default=32)
    parser.add_argument("--frame_stride", type=int, default=16)
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
    parser.add_argument("--resume_residual_checkpoint", default="")
    parser.add_argument("--diffusion_steps", type=int, default=1000)
    parser.add_argument("--noise_schedule", choices=("linear", "cosine"), default="linear")
    parser.add_argument("--head_hidden_channels", type=int, default=32)
    parser.add_argument("--head_num_layers", type=int, default=2)
    parser.add_argument("--residual_alpha", type=float, default=0.05)
    parser.add_argument("--disable_output_head", action="store_true")
    parser.add_argument("--use_middle_adapter", action="store_true")
    parser.add_argument("--middle_adapter_reduction", type=int, default=4)
    parser.add_argument("--middle_adapter_min_hidden", type=int, default=32)
    parser.add_argument("--middle_adapter_align_timestep", type=int, default=300)
    parser.add_argument("--middle_adapter_timestep_gate", choices=("linear", "hard", "none"), default="linear")
    parser.add_argument("--no_frame_pos", action="store_true")
    parser.add_argument("--no_timestep_channel", action="store_true")
    parser.add_argument("--lr", type=float, default=1e-4)
    parser.add_argument("--weight_decay", type=float, default=1e-4)
    parser.add_argument("--mixed_precision", choices=("no", "fp16", "bf16"), default="no")
    parser.add_argument("--max_steps", type=int, default=1000)
    parser.add_argument("--gradient_accumulation_steps", type=int, default=1)
    parser.add_argument("--log_interval", type=int, default=20)
    parser.add_argument("--save_interval", type=int, default=250)
    parser.add_argument("--save_dir", default="./checkpoints_dynamic_rmdm_residual_kflow")
    parser.add_argument("--diff_loss_weight", type=float, default=1.0)
    parser.add_argument("--teacher_anchor_weight", type=float, default=1.0)
    parser.add_argument("--kflow_weight", type=float, default=0.05)
    parser.add_argument("--kflow_warmup_steps", type=int, default=200)
    parser.add_argument("--kflow_max_timestep", type=int, default=300)
    parser.add_argument(
        "--low_timestep_prob",
        type=float,
        default=0.5,
        help="Probability of sampling clip timesteps from [0, kflow_max_timestep] so safe k-flow sees low-noise x0.",
    )
    parser.add_argument("--x0_recon_weight", type=float, default=0.0)
    parser.add_argument("--x0_recon_max_timestep", type=int, default=300)
    parser.add_argument("--static_keypath_weight", type=float, default=0.0)
    parser.add_argument("--static_keypath_warmup_steps", type=int, default=200)
    parser.add_argument("--static_keypath_max_timestep", type=int, default=300)
    parser.add_argument("--static_keypath_margin", type=float, default=0.005)
    parser.add_argument("--static_keypath_k2_quantile", type=float, default=0.70)
    parser.add_argument("--static_keypath_flow_quantile", type=float, default=0.70)
    parser.add_argument("--static_keypath_flow_threshold", type=float, default=0.0)
    parser.add_argument("--static_keypath_pair_stride", type=int, default=1)
    parser.add_argument("--dynamic_keypath_weight", type=float, default=0.0)
    parser.add_argument("--dynamic_keypath_warmup_steps", type=int, default=100)
    parser.add_argument("--dynamic_keypath_max_timestep", type=int, default=300)
    parser.add_argument("--dynamic_keypath_k2_quantile", type=float, default=0.70)
    parser.add_argument("--dynamic_keypath_flow_quantile", type=float, default=0.85)
    parser.add_argument("--dynamic_keypath_flow_threshold", type=float, default=0.0)
    parser.add_argument("--dynamic_keypath_lower_scale", type=float, default=0.5)
    parser.add_argument("--dynamic_keypath_upper_scale", type=float, default=2.0)
    parser.add_argument("--dynamic_keypath_upper_weight", type=float, default=0.2)
    parser.add_argument("--dynamic_keypath_pair_stride", type=int, default=1)
    parser.add_argument("--kflow_structure_weight", type=float, default=0.0)
    parser.add_argument("--kflow_structure_warmup_steps", type=int, default=100)
    parser.add_argument("--kflow_structure_max_timestep", type=int, default=300)
    parser.add_argument("--kflow_structure_k2_quantile", type=float, default=0.70)
    parser.add_argument("--kflow_structure_flow_quantile", type=float, default=0.85)
    parser.add_argument("--kflow_structure_flow_threshold", type=float, default=0.0)
    parser.add_argument("--kflow_structure_pair_stride", type=int, default=1)
    parser.add_argument("--kflow_structure_smooth_l1_beta", type=float, default=0.1)
    parser.add_argument("--global_temporal_weight", type=float, default=0.0)
    parser.add_argument("--global_temporal_warmup_steps", type=int, default=100)
    parser.add_argument("--global_temporal_max_timestep", type=int, default=300)
    parser.add_argument("--global_temporal_margin", type=float, default=0.005)
    parser.add_argument("--global_temporal_pair_stride", type=int, default=1)
    parser.add_argument("--global_dynamic_weight", type=float, default=0.0)
    parser.add_argument("--global_dynamic_warmup_steps", type=int, default=100)
    parser.add_argument("--global_dynamic_max_timestep", type=int, default=300)
    parser.add_argument("--global_dynamic_lower_scale", type=float, default=0.5)
    parser.add_argument("--global_dynamic_upper_scale", type=float, default=2.0)
    parser.add_argument("--global_dynamic_upper_weight", type=float, default=0.2)
    parser.add_argument("--global_dynamic_pair_stride", type=int, default=1)
    parser.add_argument("--kpath_delta_weight", type=float, default=0.0)
    parser.add_argument("--kpath_delta_warmup_steps", type=int, default=100)
    parser.add_argument("--kpath_delta_max_timestep", type=int, default=300)
    parser.add_argument("--kpath_delta_k2_quantile", type=float, default=0.70)
    parser.add_argument("--kpath_delta_static_flow_quantile", type=float, default=0.70)
    parser.add_argument("--kpath_delta_dynamic_flow_quantile", type=float, default=0.85)
    parser.add_argument("--kpath_delta_static_flow_threshold", type=float, default=0.0)
    parser.add_argument("--kpath_delta_dynamic_flow_threshold", type=float, default=0.0)
    parser.add_argument("--kpath_delta_margin", type=float, default=0.005)
    parser.add_argument("--kpath_delta_dynamic_lambda", type=float, default=0.1)
    parser.add_argument("--kpath_delta_pair_stride", type=int, default=1)
    parser.add_argument("--kflow_delta", type=float, default=1e-6)
    parser.add_argument("--kflow_clamp", type=float, default=1.0)
    parser.add_argument("--kflow_hinge_margin", type=float, default=0.0)
    parser.add_argument("--kflow_warp_error_threshold", type=float, default=0.08)
    parser.add_argument("--kflow_rss_grad_max", type=float, default=0.12)
    parser.add_argument("--kflow_rss_delta_max", type=float, default=0.25)
    parser.add_argument("--kflow_k2_strong_threshold", type=float, default=0.95)
    parser.add_argument("--kflow_k2_exclude_dilate", type=int, default=2)
    return parser.parse_args()


def warmup(step: int, target: float, warmup_steps: int) -> float:
    if target <= 0:
        return 0.0
    if warmup_steps <= 0:
        return float(target)
    return float(target) * min(1.0, max(0.0, float(step) / float(warmup_steps)))


def save_checkpoint(model: FrozenRMDMResidualStudent, args: argparse.Namespace, step: int, path: Path) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    raw = model.module if hasattr(model, "module") else model
    payload = {
        "step": int(step),
        "args": vars(args),
        "residual_head": raw.residual_head.state_dict(),
        "residual_alpha": raw.alpha,
        "use_output_head": raw.use_output_head,
        "use_middle_adapter": raw.use_middle_adapter,
    }
    if raw.middle_adapter is not None:
        payload["middle_adapter"] = raw.middle_adapter.state_dict()
    torch.save(payload, path)


def main() -> None:
    args = parse_args()
    scene_ids = [item.strip() for item in args.scene_ids.split(",") if item.strip()] or None

    ddp_kwargs = DistributedDataParallelKwargs(find_unused_parameters=False)
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
        include_k2=True,
        k2_root=args.k2_root or None,
        scene_ids=scene_ids,
    )
    loader = DataLoader(dataset, batch_size=args.batch_size, shuffle=True, num_workers=args.workers, pin_memory=True, drop_last=True)

    base_model = build_unet_from_config(build_model_config(args))
    state = torch.load(args.init_2d_checkpoint, map_location="cpu")
    base_model.load_state_dict(state, strict=True)
    model = FrozenRMDMResidualStudent(
        base_model,
        condition_channels=3,
        hidden_channels=args.head_hidden_channels,
        head_num_layers=args.head_num_layers,
        alpha=args.residual_alpha,
        use_frame_pos=not args.no_frame_pos,
        use_timestep_channel=not args.no_timestep_channel,
        diffusion_steps=args.diffusion_steps,
        use_output_head=not args.disable_output_head,
        use_middle_adapter=args.use_middle_adapter,
        middle_adapter_reduction=args.middle_adapter_reduction,
        middle_adapter_min_hidden=args.middle_adapter_min_hidden,
        middle_adapter_align_timestep=args.middle_adapter_align_timestep,
        middle_adapter_timestep_gate=args.middle_adapter_timestep_gate,
    )
    if args.resume_residual_checkpoint:
        payload = torch.load(args.resume_residual_checkpoint, map_location="cpu")
        if "residual_head" in payload:
            model.residual_head.load_state_dict(payload["residual_head"], strict=True)
        if model.middle_adapter is not None and "middle_adapter" in payload:
            model.middle_adapter.load_state_dict(payload["middle_adapter"], strict=True)
        model.alpha = float(payload.get("residual_alpha", args.residual_alpha))

    optimizer = torch.optim.AdamW(model.trainable_parameters(), lr=args.lr, weight_decay=args.weight_decay)
    beta_schedule = "linear" if args.noise_schedule == "linear" else "squaredcos_cap_v2"
    scheduler = DDPMScheduler(num_train_timesteps=args.diffusion_steps, beta_schedule=beta_schedule, prediction_type="epsilon")

    model, optimizer, loader = accelerator.prepare(model, optimizer, loader)
    model.train()

    if accelerator.is_main_process:
        save_dir = Path(args.save_dir)
        save_dir.mkdir(parents=True, exist_ok=True)
        (save_dir / "run_config.json").write_text(json.dumps(vars(args), indent=2), encoding="utf-8")
        trainable = sum(p.numel() for p in model.parameters() if p.requires_grad)
        total = sum(p.numel() for p in model.parameters())
        print(f"dataset clips: {len(dataset)}")
        print(f"scene_ids: {scene_ids or 'all train scenes'}")
        print(f"trainable params: {trainable} / {total}")
        print(f"residual alpha: {args.residual_alpha}")
        if args.resume_residual_checkpoint:
            print(f"resume residual checkpoint: {args.resume_residual_checkpoint}")

    kflow_cfg = SafeKFlowConfig(
        delta=args.kflow_delta,
        clamp=args.kflow_clamp,
        hinge_margin=args.kflow_hinge_margin,
        warp_error_threshold=args.kflow_warp_error_threshold,
        rss_grad_max=args.kflow_rss_grad_max,
        rss_delta_max=args.kflow_rss_delta_max,
        k2_strong_threshold=args.kflow_k2_strong_threshold,
        k2_exclude_dilate=args.kflow_k2_exclude_dilate,
    )
    static_cfg = StaticKeyPathConfig(
        delta=args.kflow_delta,
        clamp=args.kflow_clamp,
        keypath_k2_quantile=args.static_keypath_k2_quantile,
        static_flow_quantile=args.static_keypath_flow_quantile,
        static_flow_threshold=args.static_keypath_flow_threshold,
        margin=args.static_keypath_margin,
        pair_stride=args.static_keypath_pair_stride,
    )
    dynamic_cfg = DynamicKeyPathConfig(
        delta=args.kflow_delta,
        clamp=args.kflow_clamp,
        keypath_k2_quantile=args.dynamic_keypath_k2_quantile,
        dynamic_flow_quantile=args.dynamic_keypath_flow_quantile,
        dynamic_flow_threshold=args.dynamic_keypath_flow_threshold,
        margin_low_scale=args.dynamic_keypath_lower_scale,
        margin_high_scale=args.dynamic_keypath_upper_scale,
        high_weight=args.dynamic_keypath_upper_weight,
        pair_stride=args.dynamic_keypath_pair_stride,
    )
    structure_cfg = KFlowStructureConfig(
        delta=args.kflow_delta,
        clamp=args.kflow_clamp,
        keypath_k2_quantile=args.kflow_structure_k2_quantile,
        dynamic_flow_quantile=args.kflow_structure_flow_quantile,
        dynamic_flow_threshold=args.kflow_structure_flow_threshold,
        pair_stride=args.kflow_structure_pair_stride,
        smooth_l1_beta=args.kflow_structure_smooth_l1_beta,
    )
    global_temporal_cfg = GlobalTemporalResidualConfig(
        margin=args.global_temporal_margin,
        pair_stride=args.global_temporal_pair_stride,
    )
    global_dynamic_cfg = GlobalDynamicRangeConfig(
        margin_low_scale=args.global_dynamic_lower_scale,
        margin_high_scale=args.global_dynamic_upper_scale,
        high_weight=args.global_dynamic_upper_weight,
        pair_stride=args.global_dynamic_pair_stride,
    )
    kpath_delta_cfg = KPathDeltaConfig(
        delta=args.kflow_delta,
        clamp=args.kflow_clamp,
        keypath_k2_quantile=args.kpath_delta_k2_quantile,
        static_flow_quantile=args.kpath_delta_static_flow_quantile,
        dynamic_flow_quantile=args.kpath_delta_dynamic_flow_quantile,
        static_flow_threshold=args.kpath_delta_static_flow_threshold,
        dynamic_flow_threshold=args.kpath_delta_dynamic_flow_threshold,
        margin=args.kpath_delta_margin,
        dynamic_lambda=args.kpath_delta_dynamic_lambda,
        pair_stride=args.kpath_delta_pair_stride,
    )

    step = 0
    while step < args.max_steps:
        for batch in loader:
            with accelerator.accumulate(model):
                raw_inputs = batch["inputs"].to(device, non_blocking=True)
                conditions = preprocess_conditions(raw_inputs)
                target = batch["image"].to(device, non_blocking=True)
                k2 = batch.get("k2")
                k2 = k2.to(device, non_blocking=True) if k2 is not None else None

                batch_size, frames = target.shape[:2]
                t = torch.randint(0, scheduler.config.num_train_timesteps, (batch_size,), device=device).long()
                low_prob = min(max(float(args.low_timestep_prob), 0.0), 1.0)
                if low_prob > 0:
                    choose_low = torch.rand(batch_size, device=device) < low_prob
                    low_t = torch.randint(
                        0,
                        min(int(args.kflow_max_timestep) + 1, scheduler.config.num_train_timesteps),
                        (batch_size,),
                        device=device,
                    ).long()
                    t = torch.where(choose_low, low_t, t)
                noise = torch.randn_like(target)
                flat_target = target.reshape(batch_size * frames, *target.shape[2:])
                flat_noise = noise.reshape(batch_size * frames, *noise.shape[2:])
                flat_t = t[:, None].repeat(1, frames).reshape(batch_size * frames)
                noisy = scheduler.add_noise(flat_target, flat_noise, flat_t).reshape_as(target)

                eps_final, _, extra = model(conditions, noisy, t)
                eps_base = extra["eps_base"].detach()
                loss_diff = F.mse_loss(eps_final, noise)
                loss_anchor = F.mse_loss(eps_final, eps_base)

                x0_student = predict_x0_from_eps(noisy, eps_final, t, scheduler.alphas_cumprod)
                with torch.no_grad():
                    x0_teacher = predict_x0_from_eps(noisy, eps_base, t, scheduler.alphas_cumprod)

                low_noise_batch = t <= int(args.kflow_max_timestep)
                current_kflow_weight = warmup(step, args.kflow_weight, args.kflow_warmup_steps)
                if current_kflow_weight > 0 and torch.any(low_noise_batch):
                    loss_kflow, kflow_stats = safe_kflow_hinge_loss(
                        x0_student,
                        x0_teacher,
                        target,
                        k2=k2,
                        batch_mask=low_noise_batch,
                        cfg=kflow_cfg,
                    )
                else:
                    loss_kflow = target.new_tensor(0.0)
                    kflow_stats = {
                        "mask_mean": target.new_tensor(0.0),
                        "teacher_error": target.new_tensor(0.0),
                        "student_error": target.new_tensor(0.0),
                    }

                if args.x0_recon_weight > 0:
                    recon_batch = (t <= int(args.x0_recon_max_timestep)).to(dtype=target.dtype)
                    while recon_batch.ndim < target.ndim:
                        recon_batch = recon_batch.view(-1, *([1] * (target.ndim - 1)))
                    denom = recon_batch.sum().clamp_min(1.0) * target.shape[1] * target.shape[2] * target.shape[3] * target.shape[4]
                    loss_x0 = ((x0_student.clamp(0.0, 1.0) - target).square() * recon_batch).sum() / denom
                else:
                    loss_x0 = target.new_tensor(0.0)

                static_batch = t <= int(args.static_keypath_max_timestep)
                current_static_weight = warmup(step, args.static_keypath_weight, args.static_keypath_warmup_steps)
                if current_static_weight > 0 and torch.any(static_batch):
                    loss_static, static_stats = static_keypath_consistency_loss(
                        x0_student,
                        target,
                        k2,
                        batch_mask=static_batch,
                        cfg=static_cfg,
                    )
                else:
                    loss_static = target.new_tensor(0.0)
                    static_stats = {
                        "mask_mean": target.new_tensor(0.0),
                        "raw_diff": target.new_tensor(0.0),
                        "active_ratio": target.new_tensor(0.0),
                    }

                dynamic_batch = t <= int(args.dynamic_keypath_max_timestep)
                current_dynamic_weight = warmup(step, args.dynamic_keypath_weight, args.dynamic_keypath_warmup_steps)
                if current_dynamic_weight > 0 and torch.any(dynamic_batch):
                    loss_dynamic, dynamic_stats = dynamic_keypath_range_loss(
                        x0_student,
                        target,
                        k2,
                        batch_mask=dynamic_batch,
                        cfg=dynamic_cfg,
                    )
                else:
                    loss_dynamic = target.new_tensor(0.0)
                    dynamic_stats = {
                        "mask_mean": target.new_tensor(0.0),
                        "gt_median": target.new_tensor(0.0),
                        "pred_diff": target.new_tensor(0.0),
                        "low_active_ratio": target.new_tensor(0.0),
                        "high_active_ratio": target.new_tensor(0.0),
                    }

                structure_batch = t <= int(args.kflow_structure_max_timestep)
                current_structure_weight = warmup(step, args.kflow_structure_weight, args.kflow_structure_warmup_steps)
                if current_structure_weight > 0 and torch.any(structure_batch):
                    loss_structure, structure_stats = kflow_structure_loss(
                        x0_student,
                        target,
                        k2,
                        batch_mask=structure_batch,
                        cfg=structure_cfg,
                    )
                else:
                    loss_structure = target.new_tensor(0.0)
                    structure_stats = {
                        "mask_mean": target.new_tensor(0.0),
                        "pred_mean": target.new_tensor(0.0),
                        "flow_mean": target.new_tensor(0.0),
                        "structure_l1": target.new_tensor(0.0),
                    }

                global_batch = t <= int(args.global_temporal_max_timestep)
                current_global_weight = warmup(step, args.global_temporal_weight, args.global_temporal_warmup_steps)
                if current_global_weight > 0 and torch.any(global_batch):
                    loss_global, global_stats = global_temporal_residual_loss(
                        x0_student,
                        batch_mask=global_batch,
                        cfg=global_temporal_cfg,
                    )
                else:
                    loss_global = target.new_tensor(0.0)
                    global_stats = {
                        "raw_diff": target.new_tensor(0.0),
                        "active_ratio": target.new_tensor(0.0),
                    }

                global_dynamic_batch = t <= int(args.global_dynamic_max_timestep)
                current_global_dynamic_weight = warmup(step, args.global_dynamic_weight, args.global_dynamic_warmup_steps)
                if current_global_dynamic_weight > 0 and torch.any(global_dynamic_batch):
                    loss_global_dynamic, global_dynamic_stats = global_dynamic_range_loss(
                        x0_student,
                        target,
                        batch_mask=global_dynamic_batch,
                        cfg=global_dynamic_cfg,
                    )
                else:
                    loss_global_dynamic = target.new_tensor(0.0)
                    global_dynamic_stats = {
                        "gt_median": target.new_tensor(0.0),
                        "pred_diff": target.new_tensor(0.0),
                        "low_active_ratio": target.new_tensor(0.0),
                        "high_active_ratio": target.new_tensor(0.0),
                    }

                kpath_delta_batch = t <= int(args.kpath_delta_max_timestep)
                current_kpath_delta_weight = warmup(step, args.kpath_delta_weight, args.kpath_delta_warmup_steps)
                if current_kpath_delta_weight > 0 and torch.any(kpath_delta_batch):
                    loss_kpath_delta, kpath_delta_stats = kpath_delta_target_loss(
                        x0_student,
                        target,
                        k2,
                        batch_mask=kpath_delta_batch,
                        cfg=kpath_delta_cfg,
                    )
                else:
                    loss_kpath_delta = target.new_tensor(0.0)
                    kpath_delta_stats = {
                        "static_mask_mean": target.new_tensor(0.0),
                        "dynamic_mask_mean": target.new_tensor(0.0),
                        "static_loss": target.new_tensor(0.0),
                        "dynamic_loss": target.new_tensor(0.0),
                        "pred_dynamic": target.new_tensor(0.0),
                        "gt_dynamic": target.new_tensor(0.0),
                    }

                loss = (
                    args.diff_loss_weight * loss_diff
                    + args.teacher_anchor_weight * loss_anchor
                    + current_kflow_weight * loss_kflow
                    + args.x0_recon_weight * loss_x0
                    + current_static_weight * loss_static
                    + current_dynamic_weight * loss_dynamic
                    + current_structure_weight * loss_structure
                    + current_global_weight * loss_global
                    + current_global_dynamic_weight * loss_global_dynamic
                    + current_kpath_delta_weight * loss_kpath_delta
                )

                optimizer.zero_grad()
                accelerator.backward(loss)
                optimizer.step()

            if accelerator.is_main_process and step % args.log_interval == 0:
                print(
                    f"step {step} loss {loss.item():.6f} diff {loss_diff.item():.6f} "
                    f"anchor {loss_anchor.item():.9f} kflow {loss_kflow.item():.6f} "
                    f"kw {current_kflow_weight:.6f} x0 {loss_x0.item():.6f} "
                    f"mask {kflow_stats['mask_mean'].item():.4f} "
                    f"teacher_e {kflow_stats['teacher_error'].item():.6f} "
                    f"student_e {kflow_stats['student_error'].item():.6f} "
                    f"static {loss_static.item():.6f} sw {current_static_weight:.6f} "
                    f"smask {static_stats['mask_mean'].item():.4f} "
                    f"sactive {static_stats['active_ratio'].item():.4f} "
                    f"sdiff {static_stats['raw_diff'].item():.6f} "
                    f"dynamic {loss_dynamic.item():.6f} dw {current_dynamic_weight:.6f} "
                    f"dmask {dynamic_stats['mask_mean'].item():.4f} "
                    f"dgt {dynamic_stats['gt_median'].item():.6f} "
                    f"dpred {dynamic_stats['pred_diff'].item():.6f} "
                    f"dlow {dynamic_stats['low_active_ratio'].item():.4f} "
                    f"dhigh {dynamic_stats['high_active_ratio'].item():.4f} "
                    f"struct {loss_structure.item():.6f} stw {current_structure_weight:.6f} "
                    f"stmask {structure_stats['mask_mean'].item():.4f} "
                    f"stpred {structure_stats['pred_mean'].item():.6f} "
                    f"stflow {structure_stats['flow_mean'].item():.6f} "
                    f"stl1 {structure_stats['structure_l1'].item():.6f} "
                    f"global {loss_global.item():.6f} gw {current_global_weight:.6f} "
                    f"gactive {global_stats['active_ratio'].item():.4f} "
                    f"gdiff {global_stats['raw_diff'].item():.6f} "
                    f"gdyn {loss_global_dynamic.item():.6f} gdw {current_global_dynamic_weight:.6f} "
                    f"gdgt {global_dynamic_stats['gt_median'].item():.6f} "
                    f"gdpred {global_dynamic_stats['pred_diff'].item():.6f} "
                    f"gdlow {global_dynamic_stats['low_active_ratio'].item():.4f} "
                    f"gdhigh {global_dynamic_stats['high_active_ratio'].item():.4f} "
                    f"kpd {loss_kpath_delta.item():.6f} kpdw {current_kpath_delta_weight:.6f} "
                    f"kpd_smask {kpath_delta_stats['static_mask_mean'].item():.4f} "
                    f"kpd_dmask {kpath_delta_stats['dynamic_mask_mean'].item():.4f} "
                    f"kpd_static {kpath_delta_stats['static_loss'].item():.6f} "
                    f"kpd_dynamic {kpath_delta_stats['dynamic_loss'].item():.6f} "
                    f"kpd_pred {kpath_delta_stats['pred_dynamic'].item():.6f} "
                    f"kpd_gt {kpath_delta_stats['gt_dynamic'].item():.6f} "
                    f"eps_delta {(eps_final - eps_base).detach().abs().mean().item():.9f} "
                    f"mid_gate {extra.get('middle_gate', target.new_tensor(0.0)).item():.6f} "
                    f"mid_lam {extra.get('middle_lambda', target.new_tensor(0.0)).item():.6f} "
                    f"mid_delta {extra.get('middle_delta', target.new_tensor(0.0)).item():.9f}",
                    flush=True,
                )

            if accelerator.is_main_process and args.save_interval > 0 and step > 0 and step % args.save_interval == 0:
                unwrapped = accelerator.unwrap_model(model)
                save_checkpoint(unwrapped, args, step, Path(args.save_dir) / f"residual_student_step{step}.pth")

            step += 1
            if step >= args.max_steps:
                break

    if accelerator.is_main_process:
        unwrapped = accelerator.unwrap_model(model)
        save_checkpoint(unwrapped, args, step, Path(args.save_dir) / "residual_student_final.pth")


if __name__ == "__main__":
    main()
