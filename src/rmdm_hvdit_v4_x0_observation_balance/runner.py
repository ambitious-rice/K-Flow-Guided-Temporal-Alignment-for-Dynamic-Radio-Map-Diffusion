"""Run one fixed-length segment of W1 observation-balance fine-tuning."""

from __future__ import annotations

from dataclasses import asdict
from datetime import datetime
from pathlib import Path
from typing import Any

import torch
from torch.utils.data import DataLoader

from rmdm.data import SamplingPolicy, WindowDataset
from rmdm.diffusion import DiffusionProcess
from rmdm_hvdit_v4_joint.training.engine import (
    append_jsonl,
    cosine_scheduler,
    make_accelerator,
    make_optimizer,
    prepare_model_optimizer_loader,
    require_scheduler_global_step,
    seed_everything,
    step_scheduler_on_global_update,
    write_json_atomic,
)
from rmdm_hvdit_v4_x0.model import build_t1_system

from .augmentation import observation_alignment_weight
from .checkpoint import load_source_model, load_training_checkpoint, save_training_checkpoint
from .config import ObservationBalanceConfig
from .step import training_step


def _configure_determinism(seed: int) -> None:
    seed_everything(seed)
    torch.backends.cudnn.benchmark = False
    torch.backends.cudnn.deterministic = True
    torch.backends.cuda.matmul.allow_tf32 = False
    torch.backends.cudnn.allow_tf32 = False
    torch.use_deterministic_algorithms(True)


def _resolve(root: Path, value: str | Path) -> Path:
    path = Path(value).expanduser()
    return path.resolve() if path.is_absolute() else (root / path).resolve()


def run_training_segment(
    model_config: Any,
    training_config: ObservationBalanceConfig,
    *,
    repository_root: str | Path,
    resume_from: str | Path | None = None,
) -> dict[str, Any]:
    root = Path(repository_root).expanduser().resolve()
    output = _resolve(root, training_config.output_dir)
    source = _resolve(root, training_config.source_checkpoint)
    train = model_config.t1_train
    physical_gpus = list(training_config.evaluation.gpus)
    accelerator = make_accelerator(
        mixed_precision=training_config.mixed_precision,
        gradient_accumulation_steps=train.gradient_accumulation_steps,
        data_seed=training_config.seed,
    )
    if accelerator.num_processes != len(physical_gpus):
        raise RuntimeError(
            f"expected {len(physical_gpus)} training processes, got {accelerator.num_processes}"
        )
    _configure_determinism(training_config.seed)

    model = build_t1_system(model_config)
    dataset = WindowDataset(
        root=model_config.data.root,
        split="train",
        split_file=model_config.data.split_file,
        window_size=1,
        seed=model_config.sampling.seed,
        cache_size=model_config.data.cache_size,
        tx_heatmap_sigma_px=model_config.data.tx_heatmap_sigma_px,
        fixed_starts=tuple(range(model_config.data.frames_per_video)),
    )
    loader = DataLoader(
        dataset,
        batch_size=train.per_gpu_batch_size,
        shuffle=True,
        num_workers=model_config.data.workers,
        pin_memory=True,
        persistent_workers=model_config.data.workers > 0,
        drop_last=True,
    )
    optimizer_config = training_config.optimizer
    optimizer = make_optimizer(
        model,
        learning_rate=optimizer_config.learning_rate,
        betas=optimizer_config.betas,
        epsilon=optimizer_config.epsilon,
        weight_decay=optimizer_config.weight_decay,
    )
    scheduler = cosine_scheduler(
        optimizer,
        total_steps=training_config.max_steps,
        warmup_steps=optimizer_config.warmup_steps,
        base_learning_rate=optimizer_config.learning_rate,
        min_learning_rate=optimizer_config.min_learning_rate,
    )

    if resume_from:
        payload = load_training_checkpoint(resume_from, model, optimizer, scheduler)
        branch_step = int(payload["branch_step"])
        source_global_step = int(payload["source_global_step"])
        epoch = int(payload["epoch"])
        microbatch_offset = int(payload["microbatches_consumed_in_epoch"])
    else:
        source_global_step = load_source_model(source, model)
        branch_step = 0
        epoch = 0
        microbatch_offset = 0
    if branch_step >= training_config.max_steps:
        raise ValueError("training branch has already reached max_steps")
    require_scheduler_global_step(scheduler, branch_step)
    segment_end = min(
        ((branch_step // training_config.segment_steps) + 1) * training_config.segment_steps,
        training_config.max_steps,
    )

    model, optimizer, loader = prepare_model_optimizer_loader(
        accelerator, model, optimizer, loader
    )
    policy = SamplingPolicy(model_config.sampling, split="train")
    diffusion = DiffusionProcess(model_config.diffusion)
    if diffusion.scheduler.config.prediction_type != "sample":
        raise RuntimeError("observation-balance training requires x0 prediction")
    regularizer = getattr(model_config, "regularizer", None)
    regularizer_type = getattr(regularizer, "type", "pinn")
    regularizer_weight = getattr(regularizer, "weight", None)
    hessian_epsilon = getattr(regularizer, "epsilon", 1.0e-3)

    if accelerator.is_main_process:
        write_json_atomic(output / "resolved_training_config.json", asdict(training_config))
        write_json_atomic(
            output / "status.json",
            {
                "state": "training",
                "branch_step": branch_step,
                "segment_end": segment_end,
                "source_global_step": source_global_step,
                "source_checkpoint": str(source),
            },
        )
        append_jsonl(
            output / "execution_history.jsonl",
            {
                "event": "segment_started",
                "at": datetime.now().astimezone().isoformat(timespec="seconds"),
                "branch_step": branch_step,
                "segment_end": segment_end,
                "resume_from": str(resume_from or ""),
            },
        )
    accelerator.wait_for_everyone()

    metric_names = (
        "loss", "clean_data_loss", "clean_condition_data_loss", "noisy_condition_data_loss",
        "observation_alignment_loss", "heldout_loss", "calibration_loss", "spatial_regularizer_loss",
        "sampling_rate_mean", "observation_noise_std_mean", "noisy_condition_fraction",
        "condition_dropout_fraction", "derived_epsilon_mse",
    )
    optimizer.zero_grad(set_to_none=True)
    last_microbatches_consumed = microbatch_offset
    while branch_step < segment_end:
        dataset.set_epoch(epoch)
        policy.set_epoch(epoch)
        loader.set_epoch(epoch)
        epoch_loader = accelerator.skip_first_batches(loader, microbatch_offset) if microbatch_offset else loader
        epoch_loader.set_epoch(epoch)
        for batch_index, dense_batch in enumerate(epoch_loader, start=microbatch_offset):
            last_microbatches_consumed = batch_index + 1
            weight = observation_alignment_weight(
                branch_step + 1,
                max_steps=training_config.max_steps,
                config=training_config.alignment,
            )
            with accelerator.accumulate(model):
                with accelerator.autocast():
                    result = training_step(
                        model,
                        dense_batch,
                        policy,
                        diffusion,
                        training_seed=training_config.seed,
                        epoch=epoch,
                        branch_step=branch_step,
                        observation_alignment_weight=weight,
                        heldout_weight=training_config.alignment.heldout_weight,
                        noise_config=training_config.noise,
                        alignment_config=training_config.alignment,
                        pinn_k=model_config.stage1.pinn_k,
                        pinn_weight=model_config.stage1.pinn_weight,
                        regularizer_type=regularizer_type,
                        regularizer_weight=regularizer_weight,
                        hessian_epsilon=hessian_epsilon,
                        use_tx_source_supervision=model_config.model.use_tx_source_supervision,
                    )
                accelerator.backward(result.loss)
                if accelerator.sync_gradients:
                    accelerator.clip_grad_norm_(
                        model.parameters(), optimizer_config.gradient_clip_norm
                    )
                optimizer.step()
                step_scheduler_on_global_update(accelerator, scheduler)
                optimizer.zero_grad(set_to_none=True)
            if not accelerator.sync_gradients:
                continue
            branch_step += 1
            require_scheduler_global_step(scheduler, branch_step)

            if branch_step % training_config.log_every_steps == 0:
                values = accelerator.reduce(
                    torch.stack([getattr(result, name).detach().float() for name in metric_names]),
                    reduction="mean",
                )
                if accelerator.is_main_process:
                    append_jsonl(
                        output / "train.jsonl",
                        {
                            "branch_step": branch_step,
                            "global_step": source_global_step + branch_step,
                            "epoch": epoch,
                            **dict(zip(metric_names, values.cpu().tolist())),
                            "observation_alignment_weight": weight,
                            "heldout_weight": training_config.alignment.heldout_weight,
                            "learning_rate": float(optimizer.param_groups[0]["lr"]),
                        },
                    )
            if branch_step >= segment_end:
                break
        if branch_step >= segment_end:
            break
        epoch += 1
        microbatch_offset = 0
        last_microbatches_consumed = 0

    checkpoint = output / "checkpoints" / f"step_{branch_step:06d}.pth"
    save_training_checkpoint(
        accelerator,
        checkpoint,
        model,
        optimizer,
        scheduler,
        branch_step=branch_step,
        source_global_step=source_global_step,
        epoch=epoch,
        microbatches_consumed_in_epoch=last_microbatches_consumed,
        source_checkpoint=str(source),
    )
    result_payload = {
        "state": "awaiting_fast_w1" if branch_step < training_config.max_steps else "max_steps_reached",
        "branch_step": branch_step,
        "global_step": source_global_step + branch_step,
        "checkpoint": str(checkpoint),
    }
    if accelerator.is_main_process:
        write_json_atomic(output / "status.json", result_payload)
        append_jsonl(
            output / "execution_history.jsonl",
            {
                "event": "segment_completed",
                "at": datetime.now().astimezone().isoformat(timespec="seconds"),
                **result_payload,
            },
        )
        print(result_payload, flush=True)
    accelerator.wait_for_everyone()
    accelerator.end_training()
    return result_payload


__all__ = ["run_training_segment"]
