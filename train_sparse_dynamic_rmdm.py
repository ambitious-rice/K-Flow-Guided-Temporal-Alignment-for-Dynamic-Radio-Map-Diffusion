#!/usr/bin/env python3
"""Train the single-frame sparse-observation RMDM baseline (RMDM-SF).

This is intentionally separate from the legacy ``train.py`` entrypoint.  It
uses the new dynamic dataset protocol:

  [building, Tx-or-zero, vehicle_t, sparse_RSS_t, sampling_mask_t] -> RSS_t.

The full radio-map image contributes to diffusion and reconstruction loss.
For the PINN, buildings and the current-frame vehicles are both known
zero-RSS obstacles.  The model remains an ordinary 2-D, single-frame RMDM; no
temporal input, data assimilation or hard observation overwrite is used here.
"""

from __future__ import annotations

import argparse
import json
import os
import random
import sys
import time
from pathlib import Path

import numpy as np
import torch
import torch.nn.functional as F
from accelerate import Accelerator
from accelerate.utils import DistributedDataParallelKwargs, set_seed
from diffusers import DDPMScheduler
from torch.utils.data import DataLoader

from lib.loaders import DynamicSparseRadioMapRMDM
from utils import build_unet_from_config, cal_pinn, cal_pinn_masked, masked_mean


DEFAULT_DATA_DIR = "/data/fzj/CARLA_0.9.15/datasets/DynamicRadioMap/M20_Formal075_RadioMapSeerPack"
DEFAULT_SCENE_SPLIT = "/data/fzj/CARLA_0.9.15/configs/dynamic_radio/multi20_formal_scene_split.json"


def write_json_atomic(path: Path, payload: dict) -> None:
    """Publish a run-state transition without exposing a partial JSON file."""
    path.parent.mkdir(parents=True, exist_ok=True)
    temporary = path.with_suffix(path.suffix + ".tmp")
    temporary.write_text(json.dumps(payload, ensure_ascii=False, indent=2) + "\n", encoding="utf-8")
    os.replace(temporary, path)


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--data_dir", default=DEFAULT_DATA_DIR)
    parser.add_argument("--scene_split_file", default=DEFAULT_SCENE_SPLIT)
    parser.add_argument("--image_size", type=int, default=128)
    parser.add_argument("--frame_stride", type=int, default=1)
    parser.add_argument("--cache_size", type=int, default=8)
    parser.add_argument("--tx_heatmap_sigma_px", type=float, default=1.5)
    parser.add_argument(
        "--without_tx",
        action="store_true",
        help="Zero the Tx condition presented to the model.",
    )
    parser.add_argument(
        "--use_tx_source_supervision",
        action="store_true",
        help="Keep the ground-truth Tx heatmap only for the PINN source-anchor loss.",
    )
    parser.add_argument("--sample_rates", default="1,2,3,4,5,6,7,8,9,10")
    parser.add_argument("--mask_seed", type=int, default=20260714)

    parser.add_argument("--num_channels", type=int, default=96)
    parser.add_argument("--num_res_blocks", type=int, default=2)
    parser.add_argument("--attention_resolutions", default="16")
    parser.add_argument("--channel_mult", default="")
    parser.add_argument("--dropout", type=float, default=0.0)
    parser.add_argument("--num_heads", type=int, default=4)
    parser.add_argument("--num_head_channels", type=int, default=-1)
    parser.add_argument("--num_heads_upsample", type=int, default=-1)
    parser.add_argument("--use_checkpoint", action=argparse.BooleanOptionalAction, default=False)
    parser.add_argument("--use_scale_shift_norm", action=argparse.BooleanOptionalAction, default=True)
    parser.add_argument("--resblock_updown", action=argparse.BooleanOptionalAction, default=False)
    parser.add_argument("--use_fp16", action=argparse.BooleanOptionalAction, default=False)
    parser.add_argument("--use_new_attention_order", action=argparse.BooleanOptionalAction, default=False)

    parser.add_argument("--diffusion_steps", type=int, default=1000)
    parser.add_argument("--noise_schedule", choices=("linear", "cosine"), default="linear")
    parser.add_argument(
        "--loss_domain",
        choices=("full_image", "free_space"),
        default="full_image",
        help="Ablation switch: RMDM-style full-image loss or dynamic free-space-only loss.",
    )
    parser.add_argument("--pinn_k", type=float, default=0.2)
    parser.add_argument("--pinn_weight", type=float, default=1.0)
    parser.add_argument("--lr", type=float, default=1e-4)
    parser.add_argument("--weight_decay", type=float, default=1e-4)
    parser.add_argument("--batch_size", type=int, default=32, help="Per-process batch size")
    parser.add_argument("--workers", type=int, default=6, help="Workers per process")
    parser.add_argument("--epochs", type=int, default=10)
    parser.add_argument("--max_steps", type=int, default=0, help="Optional global optimizer-step cap; 0 means all epochs")
    parser.add_argument("--gradient_accumulation_steps", type=int, default=1)
    parser.add_argument("--mixed_precision", choices=("no", "fp16", "bf16"), default="bf16")
    parser.add_argument("--seed", type=int, default=20260714)
    parser.add_argument("--log_interval", type=int, default=100)
    parser.add_argument("--save_every_epochs", type=int, default=1)
    parser.add_argument("--save_dir", default="./runs/rmdm_sf_sparse")
    parser.add_argument("--resume_from", default="")
    return parser.parse_args()


def parse_sample_rates(value: str) -> tuple[int, ...]:
    rates = tuple(int(item.strip()) for item in value.split(",") if item.strip())
    if not rates or any(rate < 1 or rate > 100 for rate in rates):
        raise ValueError("--sample_rates must be comma-separated integer percentages in [1, 100]")
    return rates


def build_model_config(args: argparse.Namespace) -> dict:
    return {
        "image_size": args.image_size,
        # Five sparse conditions plus the noisy target.  The HWM/cal branch
        # automatically receives the five condition channels.
        "in_ch": 6,
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


def preprocess_conditions(conditions: torch.Tensor, *, without_tx: bool = False) -> torch.Tensor:
    """Keep the original RMDM building-plus-Tx conditioning convention."""
    conditions = conditions.clone()
    if without_tx:
        conditions[:, 1].zero_()
    conditions[:, 0] = conditions[:, 0] + 10.0 * conditions[:, 1]
    return conditions


def pinn_tx_heatmap(
    conditions: torch.Tensor,
    *,
    without_tx: bool,
    use_tx_source_supervision: bool,
) -> torch.Tensor:
    """Select Tx supervision independently from the model's Tx condition."""
    tx_heatmap = conditions[:, 1].clone()
    if without_tx and not use_tx_source_supervision:
        tx_heatmap.zero_()
    return tx_heatmap


def load_checkpoint(model, optimizer, checkpoint_path: str) -> tuple[int, int]:
    state = torch.load(checkpoint_path, map_location="cpu")
    if isinstance(state, dict) and "model" in state:
        model.load_state_dict(state["model"], strict=True)
        if "optimizer" in state:
            optimizer.load_state_dict(state["optimizer"])
        return int(state.get("epoch", 0)), int(state.get("global_step", 0))
    model.load_state_dict(state, strict=True)
    return 0, 0


def save_checkpoint(accelerator, model, optimizer, args, epoch: int, global_step: int, path: Path) -> None:
    accelerator.wait_for_everyone()
    if accelerator.is_main_process:
        path.parent.mkdir(parents=True, exist_ok=True)
        torch.save(
            {
                "schema": "rmdm_sf_sparse_checkpoint_v1",
                "model": accelerator.unwrap_model(model).state_dict(),
                "optimizer": optimizer.state_dict(),
                "epoch": epoch,
                "global_step": global_step,
                "args": vars(args),
            },
            path,
        )
    accelerator.wait_for_everyone()


def main() -> None:
    args = parse_args()
    if args.image_size != 128:
        raise ValueError("The new DynamicRadioMap data contract is 128x128")
    if args.epochs <= 0 or args.batch_size <= 0:
        raise ValueError("epochs and batch_size must be positive")
    sample_rates = parse_sample_rates(args.sample_rates)

    # The original RMDM has two legacy branch parameters which are not on the
    # active loss path for every forward call. Keep DDP behaviour consistent
    # with the legacy trainer instead of treating this architectural detail as
    # an error on the second distributed iteration.
    ddp_kwargs = DistributedDataParallelKwargs(find_unused_parameters=True)
    accelerator = Accelerator(
        mixed_precision=args.mixed_precision,
        gradient_accumulation_steps=args.gradient_accumulation_steps,
        kwargs_handlers=[ddp_kwargs],
    )
    set_seed(args.seed, device_specific=True)
    random.seed(args.seed + accelerator.process_index)
    np.random.seed(args.seed + accelerator.process_index)
    # Publish the running state before any expensive dataset/model setup so an
    # early construction failure is recorded as failed rather than leaving a
    # watcher waiting forever.
    save_dir = Path(args.save_dir).expanduser().resolve()
    status_file = save_dir / "training_status.json"
    os.environ["RMDM_TRAIN_STATUS_PATH"] = str(status_file)
    if accelerator.is_main_process:
        write_json_atomic(
            status_file,
            {
                "schema": "rmdm_sparse_training_status_v1",
                "state": "running",
                "expected_epochs": args.epochs,
                "resume_start_epoch": 0,
                "global_step": 0,
            },
        )

    dataset = DynamicSparseRadioMapRMDM(
        root=args.data_dir,
        split="train",
        split_file=args.scene_split_file,
        frame_stride=args.frame_stride,
        cache_size=args.cache_size,
        tx_heatmap_sigma_px=args.tx_heatmap_sigma_px,
        sampling_mode="train",
        sample_rates=sample_rates,
        manifest_seed=args.mask_seed,
    )
    loader = DataLoader(
        dataset,
        batch_size=args.batch_size,
        shuffle=True,
        num_workers=args.workers,
        pin_memory=True,
        persistent_workers=False,
        drop_last=True,
    )
    model = build_unet_from_config(build_model_config(args))
    optimizer = torch.optim.AdamW(model.parameters(), lr=args.lr, weight_decay=args.weight_decay)
    beta_schedule = "linear" if args.noise_schedule == "linear" else "squaredcos_cap_v2"
    scheduler = DDPMScheduler(
        num_train_timesteps=args.diffusion_steps,
        beta_schedule=beta_schedule,
        prediction_type="epsilon",
    )

    start_epoch = 0
    global_step = 0
    if args.resume_from:
        start_epoch, global_step = load_checkpoint(model, optimizer, args.resume_from)
        if accelerator.is_main_process:
            print(f"[resume] {args.resume_from}: epoch={start_epoch}, global_step={global_step}", flush=True)

    model, optimizer, loader = accelerator.prepare(model, optimizer, loader)
    if accelerator.is_main_process:
        save_dir.mkdir(parents=True, exist_ok=True)
        write_json_atomic(
            status_file,
            {
                "schema": "rmdm_sparse_training_status_v1",
                "state": "running",
                "expected_epochs": args.epochs,
                "resume_start_epoch": start_epoch,
                "global_step": global_step,
            },
        )
        (save_dir / "train_config.json").write_text(
            json.dumps(
                {
                    "args": vars(args),
                    "model_config": build_model_config(args),
                    "dataset_length": len(dataset),
                    "world_size": accelerator.num_processes,
                    "global_batch_size": args.batch_size * accelerator.num_processes * args.gradient_accumulation_steps,
                    "condition_layout": [
                        "building",
                        "zero_tx_channel" if args.without_tx else "tx_heatmap",
                        "vehicle",
                        "sparse_rss",
                        "sampling_mask",
                    ],
                    "without_tx": bool(args.without_tx),
                    "use_tx_source_supervision": bool(args.use_tx_source_supervision),
                    "sampling_domain": "not_building_and_not_vehicle",
                    "training_loss_domain": args.loss_domain,
                    "pinn_domain": (
                        "full_image_obstacle_aware: building_or_current_frame_vehicle"
                        if args.loss_domain == "full_image"
                        else "free_space_only: building_and_vehicle_excluded"
                    ),
                    "evaluation_domains": [
                        "unobserved_free_space",
                        "unobserved_full_image",
                        "full_image",
                    ],
                    "sampler_version": dataset.SAMPLER_VERSION,
                },
                ensure_ascii=False,
                indent=2,
            ),
            encoding="utf-8",
        )
        print(
            f"[setup] frames={len(dataset):,} world_size={accelerator.num_processes} "
            f"per_gpu_batch={args.batch_size} global_batch="
            f"{args.batch_size * accelerator.num_processes * args.gradient_accumulation_steps} "
            f"rates={sample_rates}",
            flush=True,
        )

    log_file = save_dir / "train_metrics.jsonl"
    completed_epochs = start_epoch
    stopped_early = False
    for epoch in range(start_epoch, args.epochs):
        model.train()
        epoch_start = time.monotonic()
        epoch_sums = torch.zeros(4, device=accelerator.device)
        epoch_batches = torch.zeros(1, device=accelerator.device)
        optimizer.zero_grad(set_to_none=True)
        reached_step_limit = False
        for local_step, batch in enumerate(loader):
            conditions = batch["inputs"].to(accelerator.device, non_blocking=True)
            target_clean = batch["target"].to(accelerator.device, non_blocking=True)
            valid_mask = batch["valid_mask"].to(accelerator.device, non_blocking=True)
            # Keep the sparse-observation domain unchanged: samples are drawn
            # only in free space.  The full-image ablation retains obstacles in
            # diffusion/reconstruction and uses them in the physics loss;
            # free_space reproduces the alternative dynamic-only objective.
            raw_buildings = conditions[:, 0].clone()
            tx_heatmap = pinn_tx_heatmap(
                conditions,
                without_tx=args.without_tx,
                use_tx_source_supervision=args.use_tx_source_supervision,
            )
            raw_vehicles = conditions[:, 2].clone()
            obstacle_mask = ((raw_buildings > 0.5) | (raw_vehicles > 0.5)).to(dtype=conditions.dtype)
            conditions = preprocess_conditions(conditions, without_tx=args.without_tx)

            with accelerator.accumulate(model):
                timesteps = torch.randint(
                    0, scheduler.config.num_train_timesteps, (target_clean.shape[0],), device=accelerator.device
                ).long()
                noise = torch.randn_like(target_clean)
                target_noisy = scheduler.add_noise(target_clean, noise, timesteps)
                model_input = torch.cat([conditions, target_noisy], dim=1)
                with accelerator.autocast():
                    prediction = model(model_input, timesteps)
                    if not isinstance(prediction, tuple) or len(prediction) < 2:
                        raise RuntimeError("RMDM must return (pred_noise, cal)")
                    pred_noise, cal = prediction[:2]
                    if args.loss_domain == "full_image":
                        loss_diff = F.mse_loss(pred_noise, noise)
                        loss_cal = F.mse_loss(cal, target_clean)
                        # Preserve the RMDM obstacle treatment (different k
                        # plus zero-field soft boundary) and extend it to
                        # dynamic vehicles, whose labels are deterministically
                        # zero in this dataset.
                        loss_pinn = cal_pinn(
                            cal[:, 0, :, :],
                            obstacle_mask,
                            tx_heatmap,
                            k=args.pinn_k,
                        ).mean()
                    else:
                        loss_diff = masked_mean((pred_noise - noise).pow(2), valid_mask).mean()
                        loss_cal = masked_mean((cal - target_clean).pow(2), valid_mask).mean()
                        loss_pinn = cal_pinn_masked(
                            cal[:, 0, :, :],
                            valid_mask,
                            tx_heatmap,
                            k=args.pinn_k,
                        ).mean()
                    loss = loss_diff + loss_cal + args.pinn_weight * loss_pinn

                accelerator.backward(loss)
                if accelerator.sync_gradients:
                    optimizer.step()
                    optimizer.zero_grad(set_to_none=True)

            detached = torch.stack([loss.detach(), loss_diff.detach(), loss_cal.detach(), loss_pinn.detach()])
            epoch_sums += detached
            epoch_batches += 1
            if accelerator.is_main_process and accelerator.sync_gradients and global_step % args.log_interval == 0:
                rates = batch["sample_rate"].float()
                print(
                    f"epoch={epoch + 1}/{args.epochs} step={global_step} "
                    f"loss={loss.item():.5f} diff={loss_diff.item():.5f} cal={loss_cal.item():.5f} "
                    f"pinn={loss_pinn.item():.5f} p_mean={rates.mean().item():.2f}%",
                    flush=True,
                )
            if accelerator.sync_gradients:
                global_step += 1
            if args.max_steps and accelerator.sync_gradients and global_step >= args.max_steps:
                reached_step_limit = True
                break

        totals = accelerator.reduce(epoch_sums, reduction="sum")
        num_batches = accelerator.reduce(epoch_batches, reduction="sum").clamp_min(1)
        mean_values = (totals / num_batches).detach().cpu().tolist()
        elapsed_seconds = time.monotonic() - epoch_start
        if accelerator.is_main_process:
            metrics = {
                "epoch": epoch + 1,
                "global_step": global_step,
                "loss": mean_values[0],
                "loss_diff": mean_values[1],
                "loss_cal": mean_values[2],
                "loss_pinn": mean_values[3],
                "elapsed_seconds": elapsed_seconds,
            }
            with log_file.open("a", encoding="utf-8") as handle:
                handle.write(json.dumps(metrics) + "\n")
            print("[epoch] " + json.dumps(metrics), flush=True)

        if (epoch + 1) % args.save_every_epochs == 0 or epoch + 1 == args.epochs:
            save_checkpoint(accelerator, model, optimizer, args, epoch + 1, global_step, save_dir / f"epoch_{epoch + 1:03d}.pth")
            save_checkpoint(accelerator, model, optimizer, args, epoch + 1, global_step, save_dir / "last.pth")
        completed_epochs = epoch + 1
        if reached_step_limit:
            stopped_early = True
            if accelerator.is_main_process:
                print(f"[done] reached --max_steps={args.max_steps}", flush=True)
            break
    accelerator.end_training()
    if accelerator.is_main_process:
        write_json_atomic(
            status_file,
            {
                "schema": "rmdm_sparse_training_status_v1",
                "state": "completed" if completed_epochs == args.epochs and not stopped_early else "partial",
                "expected_epochs": args.epochs,
                "completed_epochs": completed_epochs,
                "global_step": global_step,
                "last_checkpoint": str(save_dir / "last.pth"),
            },
        )


if __name__ == "__main__":
    try:
        main()
    except BaseException as error:
        status_path = os.environ.get("RMDM_TRAIN_STATUS_PATH")
        if status_path and int(os.environ.get("RANK", "0")) == 0:
            write_json_atomic(
                Path(status_path),
                {
                    "schema": "rmdm_sparse_training_status_v1",
                    "state": "failed",
                    "error_type": type(error).__name__,
                    "error": str(error),
                },
            )
        raise
