"""Three-step real-data DDP smoke for the isolated x0 training path."""

from __future__ import annotations

import argparse
import json
import os
from pathlib import Path

import torch
from torch.utils.data import DataLoader

from rmdm.data import SamplingPolicy, WindowDataset
from rmdm.diffusion import DiffusionProcess
from rmdm_hvdit_v4_joint.training.engine import (
    make_accelerator,
    make_optimizer,
    prepare_model_optimizer_loader,
    seed_everything,
    write_json_atomic,
)
from rmdm_hvdit_v4_x0.model import build_t1_system
from rmdm_hvdit_v4_x0.training.step import training_step

from .common import config_argument, load_arguments


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    config_argument(parser)
    parser.add_argument("--batch-size", type=int, default=2)
    parser.add_argument("--steps", type=int, default=3)
    parser.add_argument("--output", required=True)
    args = parser.parse_args()
    config, _, _ = load_arguments(args)
    physical_gpus = config.pipeline.allowed_physical_gpus
    visible = [int(value) for value in os.environ.get("CUDA_VISIBLE_DEVICES", "").split(",") if value]
    if visible != physical_gpus:
        raise RuntimeError(f"Expected CUDA_VISIBLE_DEVICES={physical_gpus}, got {visible}")
    if args.batch_size <= 0 or args.steps != 3:
        raise ValueError("The fixed x0 DDP smoke requires a positive batch and exactly three steps")

    accelerator = make_accelerator(
        mixed_precision="bf16",
        gradient_accumulation_steps=1,
        data_seed=config.t1_train.seed,
    )
    if accelerator.num_processes != 4:
        raise RuntimeError(f"Expected four smoke processes, got {accelerator.num_processes}")
    seed_everything(config.t1_train.seed)
    model = build_t1_system(config)
    dataset = WindowDataset(
        root=config.data.root,
        split="train",
        split_file=config.data.split_file,
        window_size=1,
        seed=config.sampling.seed,
        cache_size=2,
        tx_heatmap_sigma_px=config.data.tx_heatmap_sigma_px,
        fixed_starts=tuple(range(config.data.frames_per_video)),
    )
    loader = DataLoader(
        dataset,
        batch_size=args.batch_size,
        shuffle=True,
        num_workers=0,
        pin_memory=True,
        drop_last=True,
    )
    optimizer = make_optimizer(
        model,
        learning_rate=config.t1_train.learning_rate,
        betas=config.t1_train.betas,
        epsilon=config.t1_train.epsilon,
        weight_decay=config.t1_train.weight_decay,
    )
    model, optimizer, loader = prepare_model_optimizer_loader(accelerator, model, optimizer, loader)
    policy = SamplingPolicy(config.sampling, split="train")
    policy.set_epoch(0)
    diffusion = DiffusionProcess(config.diffusion)
    records: list[dict[str, float | int]] = []
    iterator = iter(loader)
    required = (
        "denoiser.input_stem.dense_projection.weight",
        "denoiser.condition_stem.dense_projection.weight",
        "denoiser.local_encoder.0.attention.qkv.weight",
        "denoiser.global_bottleneck.0.attention.qkv.weight",
        "denoiser.output_head.token_projection.weight",
        "denoiser.output_head.final_projection.weight",
    )
    final_gradient_l1: dict[str, float] = {}
    for step in range(1, args.steps + 1):
        batch = next(iterator)
        optimizer.zero_grad(set_to_none=True)
        with accelerator.autocast():
            result = training_step(
                model,
                batch,
                policy,
                diffusion,
                training_seed=config.t1_train.seed,
                epoch=0,
                pinn_k=config.stage1.pinn_k,
                pinn_weight=config.stage1.pinn_weight,
            )
        if not torch.isfinite(result.loss):
            raise RuntimeError(f"Non-finite x0 smoke loss at step {step}: {result.loss}")
        accelerator.backward(result.loss)
        accelerator.clip_grad_norm_(model.parameters(), config.t1_train.gradient_clip_norm)
        core = accelerator.unwrap_model(model)
        parameters = dict(core.named_parameters())
        if step == args.steps:
            for name in required:
                gradient = parameters[name].grad
                if gradient is None:
                    raise RuntimeError(f"x0 smoke missed gradient {name}")
                final_gradient_l1[name] = float(gradient.detach().float().abs().sum())
            if any(value <= 0.0 for value in final_gradient_l1.values()):
                raise RuntimeError(f"x0 smoke retained a zero key gradient: {final_gradient_l1}")
        optimizer.step()
        values = accelerator.reduce(
            torch.stack(
                (
                    result.loss.detach().float(),
                    result.clean_data_loss.detach().float(),
                    result.calibration_loss.detach().float(),
                    result.pinn_loss.detach().float(),
                )
            ),
            reduction="mean",
        )
        records.append(
            {
                "step": step,
                "loss": float(values[0]),
                "clean_data_loss": float(values[1]),
                "calibration_loss": float(values[2]),
                "pinn_loss": float(values[3]),
            }
        )
    report = {
        "schema": "rmdm_hvdit_v4_x0_ddp_smoke_v1",
        "physical_gpus": physical_gpus,
        "world_size": accelerator.num_processes,
        "per_gpu_batch_size": args.batch_size,
        "steps": records,
        "required_gradient_l1": final_gradient_l1,
        "prediction_type": diffusion.scheduler.config.prediction_type,
        "passed": True,
    }
    if accelerator.is_main_process:
        write_json_atomic(Path(args.output), report)
        print(json.dumps(report, ensure_ascii=False, sort_keys=True), flush=True)
    accelerator.wait_for_everyone()
    if torch.distributed.is_initialized():
        torch.distributed.destroy_process_group()


if __name__ == "__main__":
    main()
