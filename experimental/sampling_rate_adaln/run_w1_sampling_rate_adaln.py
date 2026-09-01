"""Freeze W1 and train only a continuous sampling-rate AdaLN branch.

This is intentionally a bounded validation experiment, not a replacement W1
training recipe.  It evaluates the same frozen checkpoint before and after
training the new zero-initialized branch on a three-scene held-out subset.
"""

from __future__ import annotations

import argparse
import json
from datetime import datetime
from pathlib import Path
from typing import Any

import torch
from torch.utils.data import DataLoader

from experimental.sampling_rate_adaln.adapter import install_sampling_rate_conditioning
from rmdm.data import SamplingPolicy, WindowDataset
from rmdm.diffusion import DiffusionProcess
from rmdm_hvdit_v4_joint.evaluation.evaluator import evaluate_stage_a
from rmdm_hvdit_v4_joint.training.engine import (
    append_jsonl,
    cosine_scheduler,
    make_accelerator,
    prepare_model_optimizer_loader,
    seed_everything,
    step_scheduler_on_global_update,
    write_json_atomic,
)
from rmdm_hvdit_v4_x0.model import build_t1_system
from rmdm_hvdit_v4_x0.training.step import training_step
from rmdm_hvdit_v4_x0_continue.config import load_config


def _rates(raw: str) -> list[float]:
    values = [float(item) for item in raw.split(",") if item.strip()]
    if not values or any(value <= 0.0 for value in values):
        raise ValueError("--rates must be a non-empty list of positive percentages")
    return values


def _small_manifest(source: Path, output: Path) -> Path:
    payload = json.loads(source.read_text(encoding="utf-8"))
    stage_a = payload["stage_a"]
    videos = stage_a["videos"]
    selected: list[dict[str, Any]] = []
    seen_scenes: set[str] = set()
    for video in videos:
        scene = str(video["scene_id"])
        if scene not in seen_scenes:
            selected.append(video)
            seen_scenes.add(scene)
    if len(selected) != 3:
        raise ValueError("expected one Stage-A video from each of three validation scenes")
    result = {"schema": "w1_sampling_rate_adaln_subset_v1", "stage_a": {"videos": selected}}
    write_json_atomic(output, result)
    return output


def _save_checkpoint(path: Path, model: torch.nn.Module, *, config_path: Path, source_checkpoint: Path, steps: int) -> None:
    payload = {
        "schema": "w1_sampling_rate_adaln_v1",
        "source_checkpoint": str(source_checkpoint),
        "config": str(config_path),
        "steps": int(steps),
        "model": model.state_dict(),
        "trainable_parameter_names": [name for name, parameter in model.named_parameters() if parameter.requires_grad],
    }
    temporary = path.with_suffix(".tmp")
    torch.save(payload, temporary)
    temporary.replace(path)


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--config", required=True)
    parser.add_argument("--source-checkpoint", required=True)
    parser.add_argument("--manifest", required=True)
    parser.add_argument("--output-dir", required=True)
    parser.add_argument("--steps", type=int, default=3000)
    parser.add_argument("--learning-rate", type=float, default=2.0e-4)
    parser.add_argument("--rates", default="1,2,3,5,8,10")
    parser.add_argument("--ddim-steps", type=int, default=20)
    args = parser.parse_args()
    if args.steps <= 0 or args.learning_rate <= 0.0 or args.ddim_steps <= 0:
        raise ValueError("steps, learning rate, and DDIM steps must be positive")

    config_path = Path(args.config).resolve()
    source_checkpoint = Path(args.source_checkpoint).resolve()
    source_manifest = Path(args.manifest).resolve()
    output = Path(args.output_dir).resolve()
    if output.exists():
        raise FileExistsError(f"refusing to overwrite existing output: {output}")
    output.mkdir(parents=True)
    config = load_config(config_path)
    train = config.t1_train
    accelerator = make_accelerator(
        mixed_precision=train.mixed_precision,
        gradient_accumulation_steps=train.gradient_accumulation_steps,
        data_seed=train.seed,
    )
    expected_processes = len(config.pipeline.allowed_physical_gpus)
    if accelerator.num_processes != expected_processes:
        raise RuntimeError(f"expected {expected_processes} processes, got {accelerator.num_processes}")
    seed_everything(train.seed)

    model = build_t1_system(config)
    source = torch.load(source_checkpoint, map_location="cpu", weights_only=False)
    model.load_state_dict(source["model"], strict=True)
    install_sampling_rate_conditioning(model, freeze_backbone=True)
    trainable = [(name, parameter) for name, parameter in model.named_parameters() if parameter.requires_grad]
    if not trainable or any(not name.startswith("denoiser.sampling_rate_conditioner.") for name, _ in trainable):
        raise RuntimeError("only the sampling-rate conditioner may be trainable")
    trainable_count = sum(parameter.numel() for _, parameter in trainable)

    dataset = WindowDataset(
        root=config.data.root,
        split="train",
        split_file=config.data.split_file,
        window_size=1,
        seed=config.sampling.seed,
        cache_size=config.data.cache_size,
        tx_heatmap_sigma_px=config.data.tx_heatmap_sigma_px,
        fixed_starts=tuple(range(config.data.frames_per_video)),
    )
    loader = DataLoader(
        dataset,
        batch_size=train.per_gpu_batch_size,
        shuffle=True,
        num_workers=config.data.workers,
        pin_memory=True,
        persistent_workers=config.data.workers > 0,
        drop_last=True,
    )
    optimizer = torch.optim.AdamW([parameter for _, parameter in trainable], lr=args.learning_rate, betas=tuple(train.betas), eps=train.epsilon, weight_decay=0.0)
    scheduler = cosine_scheduler(
        optimizer,
        total_steps=args.steps,
        warmup_steps=min(100, max(args.steps // 10, 1)),
        base_learning_rate=args.learning_rate,
        min_learning_rate=args.learning_rate * 0.1,
    )
    model, optimizer, loader = prepare_model_optimizer_loader(accelerator, model, optimizer, loader)
    policy = SamplingPolicy(config.sampling, split="train")
    diffusion = DiffusionProcess(config.diffusion)
    if diffusion.scheduler.config.prediction_type != "sample":
        raise RuntimeError("this experiment requires x0 prediction")

    rates = _rates(args.rates)
    config.evaluation.rates = rates
    config.evaluation.ddim_steps = args.ddim_steps
    small_manifest = output / "validation_subset.json"
    if accelerator.is_main_process:
        _small_manifest(source_manifest, small_manifest)
        write_json_atomic(output / "run_config.json", {
            "schema": "w1_sampling_rate_adaln_run_v1",
            "source_checkpoint": str(source_checkpoint),
            "train_steps": args.steps,
            "learning_rate": args.learning_rate,
            "freeze_backbone": True,
            "trainable_parameter_count": trainable_count,
            "rates": rates,
            "ddim_steps": args.ddim_steps,
            "started_at": datetime.now().astimezone().isoformat(timespec="seconds"),
        })
    accelerator.wait_for_everyone()

    # The branch is exactly zero at this point, so this is the W1 baseline under
    # the identical seeds, masks, videos, and DDIM protocol used after training.
    baseline = evaluate_stage_a(
        accelerator, model, config, variant="t1", subset_stage="stage_a", split="val",
        manifest_path=small_manifest, rates=rates, log_interval=100,
    )
    if accelerator.is_main_process:
        write_json_atomic(output / "baseline.json", baseline)
    accelerator.wait_for_everyone()

    model.train()
    optimizer.zero_grad(set_to_none=True)
    global_step = 0
    epoch = 0
    while global_step < args.steps:
        dataset.set_epoch(epoch)
        policy.set_epoch(epoch)
        if hasattr(loader, "set_epoch"):
            loader.set_epoch(epoch)
        for dense_batch in loader:
            with accelerator.accumulate(model):
                with accelerator.autocast():
                    result = training_step(
                        model, dense_batch, policy, diffusion, training_seed=train.seed, epoch=epoch,
                        pinn_k=config.stage1.pinn_k, pinn_weight=config.stage1.pinn_weight,
                        use_tx_source_supervision=config.model.use_tx_source_supervision,
                    )
                accelerator.backward(result.loss)
                if accelerator.sync_gradients:
                    accelerator.clip_grad_norm_([parameter for _, parameter in trainable], train.gradient_clip_norm)
                optimizer.step()
                if accelerator.sync_gradients:
                    step_scheduler_on_global_update(accelerator, scheduler)
                optimizer.zero_grad(set_to_none=True)
            if not accelerator.sync_gradients:
                continue
            global_step += 1
            if global_step % 100 == 0 or global_step == args.steps:
                values = accelerator.reduce(torch.stack((result.loss.detach().float(), result.clean_data_loss.detach().float(), result.sampling_rate_mean.detach().float())), reduction="mean")
                if accelerator.is_main_process:
                    append_jsonl(output / "train.jsonl", {
                        "step": global_step, "epoch": epoch, "loss": float(values[0]),
                        "clean_data_loss": float(values[1]), "sampling_rate_mean": float(values[2]),
                        "learning_rate": float(optimizer.param_groups[0]["lr"]),
                    })
            if global_step >= args.steps:
                break
        epoch += 1

    candidate = evaluate_stage_a(
        accelerator, model, config, variant="t1", subset_stage="stage_a", split="val",
        manifest_path=small_manifest, rates=rates, log_interval=100,
    )
    if accelerator.is_main_process:
        core = accelerator.unwrap_model(model)
        _save_checkpoint(output / "sampling_rate_adaln.pth", core, config_path=config_path, source_checkpoint=source_checkpoint, steps=args.steps)
        write_json_atomic(output / "candidate.json", candidate)
        deltas = {
            key: candidate["rates"][key]["metrics"]["full_image"]["nmse"] - baseline["rates"][key]["metrics"]["full_image"]["nmse"]
            for key in baseline["rates"]
        }
        write_json_atomic(output / "summary.json", {
            "schema": "w1_sampling_rate_adaln_summary_v1",
            "baseline_macro_nmse": baseline["macro_full_image_nmse_p1_p2_p3"],
            "candidate_macro_nmse": candidate["macro_full_image_nmse_p1_p2_p3"],
            "delta_full_image_nmse": deltas,
            "improved_rate_count": sum(delta < 0.0 for delta in deltas.values()),
            "rate_count": len(deltas),
            "completed_at": datetime.now().astimezone().isoformat(timespec="seconds"),
        })


if __name__ == "__main__":
    main()
