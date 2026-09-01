"""Resumable 10k W1 clean-x0 pilot followed by one aligned Stage-A evaluation."""

from __future__ import annotations

import json
from datetime import datetime
from pathlib import Path
from typing import Any

import torch
from torch.utils.data import DataLoader

from rmdm.data import SamplingPolicy, WindowDataset
from rmdm.diffusion import DiffusionProcess
from rmdm_hvdit_v4_joint.evaluation.evaluator import evaluate_stage_a
from rmdm_hvdit_v4_joint.training.engine import (
    append_jsonl,
    cosine_scheduler,
    make_accelerator,
    make_optimizer,
    prepare_model_optimizer_loader,
    require_scheduler_global_step,
    require_visible_physical_gpus,
    seed_everything,
    step_scheduler_on_global_update,
    validate_parameter_contract,
    write_json_atomic,
)
from rmdm_hvdit_v4_x0.evaluation import build_step10k_comparison
from rmdm_hvdit_v4_x0.model import build_t1_system
from rmdm_hvdit_v4_x0.provenance import build_dependency_manifest

from .checkpoint import load_checkpoint, save_checkpoint
from .step import training_step


def _output_path(config: Any, repository_root: Path, output_dir: str | Path | None) -> Path:
    allowed = (repository_root / config.pipeline.output_root).resolve()
    output = Path(output_dir).expanduser().resolve() if output_dir else allowed / "t1_pilot_10k"
    if not output.is_relative_to(allowed):
        raise ValueError(f"x0 pilot output must stay below {allowed}, got {output}")
    return output


def run_t1_pilot(
    config: Any,
    *,
    config_path: str | Path,
    repository_root: str | Path,
    output_dir: str | Path | None = None,
    resume_from: str | Path | None = None,
) -> dict[str, Any]:
    root = Path(repository_root).expanduser().resolve()
    output = _output_path(config, root, output_dir)
    train = config.t1_train
    regularizer = getattr(config, "regularizer", None)
    regularizer_type = getattr(regularizer, "type", "pinn")
    regularizer_weight = getattr(regularizer, "weight", None)
    hessian_epsilon = getattr(regularizer, "epsilon", 1.0e-3)
    physical_gpus = list(config.pipeline.allowed_physical_gpus)
    require_visible_physical_gpus(physical_gpus)
    accelerator = make_accelerator(
        mixed_precision=train.mixed_precision,
        gradient_accumulation_steps=train.gradient_accumulation_steps,
        data_seed=train.seed,
    )
    if accelerator.num_processes != len(physical_gpus):
        raise RuntimeError(
            f"x0 pilot requires {len(physical_gpus)} DDP processes, "
            f"got {accelerator.num_processes}"
        )

    seed_everything(train.seed)
    dependency_manifest = build_dependency_manifest(
        config,
        config_path=config_path,
        repository_root=root,
    )
    model = build_t1_system(config)
    trainable, total = validate_parameter_contract(
        model,
        config.model.expected_trainable_parameters_min,
        config.model.expected_trainable_parameters_max,
    )
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
    if len(dataset) != 1_050_000:
        raise RuntimeError(f"W1 pilot must expose 1,050,000 train frames, got {len(dataset):,}")
    loader = DataLoader(
        dataset,
        batch_size=train.per_gpu_batch_size,
        shuffle=True,
        num_workers=config.data.workers,
        pin_memory=True,
        persistent_workers=config.data.workers > 0,
        drop_last=True,
    )
    optimizer = make_optimizer(
        model,
        learning_rate=train.learning_rate,
        betas=train.betas,
        epsilon=train.epsilon,
        weight_decay=train.weight_decay,
    )
    scheduler = cosine_scheduler(
        optimizer,
        total_steps=train.lr_schedule_steps,
        warmup_steps=train.warmup_steps,
        base_learning_rate=train.learning_rate,
        min_learning_rate=train.min_learning_rate,
    )

    checkpoint_path = output / "checkpoints" / "last.pth"
    validation_path = output / "validation" / "stage_a_step_010000.json"
    comparison_path = output / "validation" / "comparison_vs_v4_epsilon_step_010000.json"
    global_step = 0
    epoch = 0
    resume_microbatch_offset = 0
    validation_pending = False
    if resume_from:
        payload = load_checkpoint(
            resume_from,
            model,
            optimizer,
            scheduler,
            dependency_manifest,
        )
        global_step = int(payload["global_step"])
        epoch = int(payload["epoch"])
        resume_microbatch_offset = int(payload["microbatches_consumed_in_epoch"])
        validation_pending = bool(payload.get("validation_pending", False))
    elif checkpoint_path.exists():
        raise FileExistsError(
            f"Pilot checkpoint already exists at {checkpoint_path}; pass --resume-from explicitly"
        )

    model, optimizer, loader = prepare_model_optimizer_loader(accelerator, model, optimizer, loader)
    require_scheduler_global_step(scheduler, global_step)
    sampling_policy = SamplingPolicy(config.sampling, split="train")
    diffusion = DiffusionProcess(config.diffusion)
    if diffusion.scheduler.config.prediction_type != "sample":
        raise RuntimeError("Training diffusion scheduler does not interpret the network output as x0")

    if accelerator.is_main_process:
        write_json_atomic(output / "dependency_manifest.json", dependency_manifest)
        write_json_atomic(
            output / "status.json",
            {
                "schema": "rmdm_hvdit_v4_x0_t1_status_v1",
                "state": "training" if global_step < train.max_steps else "validation_pending",
                "global_step": global_step,
                "epoch": epoch,
                "world_size": accelerator.num_processes,
                "physical_gpus": physical_gpus,
                "per_gpu_batch_size": train.per_gpu_batch_size,
                "gradient_accumulation_steps": train.gradient_accumulation_steps,
                "effective_global_batch_size": train.effective_global_batch_size,
                "prediction_target": "x0",
                "prediction_type": config.diffusion.prediction_type,
                "spatial_regularizer": regularizer_type,
                "spatial_regularizer_weight": (
                    float(config.stage1.pinn_weight)
                    if regularizer_weight is None
                    else float(regularizer_weight)
                ),
                "trainable_parameters": trainable,
                "total_parameters": total,
                "dependency_manifest_sha256": dependency_manifest["manifest_sha256"],
            },
        )
        append_jsonl(
            output / "execution_history.jsonl",
            {
                "event": "start_or_resume",
                "at": datetime.now().astimezone().isoformat(timespec="seconds"),
                "global_step": global_step,
                "epoch": epoch,
                "microbatch_offset": resume_microbatch_offset,
                "resume_from": str(resume_from) if resume_from else "",
                "physical_gpus": physical_gpus,
            },
        )
    accelerator.wait_for_everyone()

    optimizer.zero_grad(set_to_none=True)
    last_microbatches_consumed = resume_microbatch_offset
    while global_step < train.max_steps:
        dataset.set_epoch(epoch)
        sampling_policy.set_epoch(epoch)
        if hasattr(loader, "set_epoch"):
            loader.set_epoch(epoch)
        for batch_index, dense_batch in enumerate(loader):
            if batch_index < resume_microbatch_offset:
                continue
            last_microbatches_consumed = batch_index + 1
            with accelerator.accumulate(model):
                with accelerator.autocast():
                    result = training_step(
                        model,
                        dense_batch,
                        sampling_policy,
                        diffusion,
                        training_seed=train.seed,
                        epoch=epoch,
                        pinn_k=config.stage1.pinn_k,
                        pinn_weight=config.stage1.pinn_weight,
                        regularizer_type=regularizer_type,
                        regularizer_weight=regularizer_weight,
                        hessian_epsilon=hessian_epsilon,
                        use_tx_source_supervision=config.model.use_tx_source_supervision,
                    )
                accelerator.backward(result.loss)
                if accelerator.sync_gradients:
                    accelerator.clip_grad_norm_(model.parameters(), train.gradient_clip_norm)
                optimizer.step()
                step_scheduler_on_global_update(accelerator, scheduler)
                optimizer.zero_grad(set_to_none=True)
            if not accelerator.sync_gradients:
                continue

            global_step += 1
            require_scheduler_global_step(scheduler, global_step)
            if global_step % train.log_every_steps == 0:
                reduced = accelerator.reduce(
                    torch.stack(
                        (
                            result.loss.detach().float(),
                            result.clean_data_loss.detach().float(),
                            result.calibration_loss.detach().float(),
                            result.pinn_loss.detach().float(),
                            result.equation_regularizer_loss.detach().float(),
                            result.semantic_anchor_loss.detach().float(),
                            result.sampling_rate_mean.detach().float(),
                            result.x0_mse_per_sample.mean().detach().float(),
                            result.derived_epsilon_mse_per_sample.mean().detach().float(),
                        )
                    ),
                    reduction="mean",
                )
                if accelerator.is_main_process:
                    append_jsonl(
                        output / "train.jsonl",
                        {
                            "global_step": global_step,
                            "epoch": epoch,
                            "loss": float(reduced[0]),
                            "clean_data_loss": float(reduced[1]),
                            "calibration_loss": float(reduced[2]),
                            "spatial_regularizer_type": regularizer_type,
                            "spatial_regularizer_loss": float(reduced[3]),
                            "equation_regularizer_loss": float(reduced[4]),
                            "semantic_anchor_loss": float(reduced[5]),
                            "sampling_rate_mean": float(reduced[6]),
                            "x0_mse": float(reduced[7]),
                            "derived_epsilon_mse": float(reduced[8]),
                            "learning_rate": float(optimizer.param_groups[0]["lr"]),
                            "scheduler_step": int(scheduler.last_epoch),
                        },
                    )

            should_checkpoint = (
                global_step % train.checkpoint_every_steps == 0
                or global_step == train.max_steps
            )
            if should_checkpoint:
                validation_pending = global_step == train.max_steps
                save_checkpoint(
                    accelerator,
                    checkpoint_path,
                    model,
                    optimizer,
                    scheduler,
                    config,
                    dependency_manifest,
                    epoch=epoch,
                    global_step=global_step,
                    microbatches_consumed_in_epoch=last_microbatches_consumed,
                    validation_pending=validation_pending,
                )
            if global_step >= train.max_steps:
                break
        if global_step >= train.max_steps:
            break
        epoch += 1
        resume_microbatch_offset = 0

    if global_step != train.max_steps:
        raise RuntimeError(f"x0 pilot stopped at {global_step}, expected {train.max_steps}")
    if accelerator.is_main_process:
        write_json_atomic(
            output / "status.json",
            {
                "schema": "rmdm_hvdit_v4_x0_t1_status_v1",
                "state": "validating",
                "global_step": global_step,
                "epoch": epoch,
                "checkpoint": str(checkpoint_path),
                "prediction_target": "x0",
            },
        )
    accelerator.wait_for_everyone()

    stage_a = evaluate_stage_a(accelerator, model, config, variant="t1")
    stage_a["schema"] = "rmdm_hvdit_v4_x0_evaluation_v1"
    stage_a["prediction_target"] = "x0"
    comparison = build_step10k_comparison(stage_a, repository_root=root)
    if accelerator.is_main_process:
        write_json_atomic(validation_path, stage_a)
        write_json_atomic(comparison_path, comparison)
    save_checkpoint(
        accelerator,
        checkpoint_path,
        model,
        optimizer,
        scheduler,
        config,
        dependency_manifest,
        epoch=epoch,
        global_step=global_step,
        microbatches_consumed_in_epoch=last_microbatches_consumed,
        validation_pending=False,
        validation_path=str(validation_path),
    )
    if accelerator.is_main_process:
        write_json_atomic(
            output / "status.json",
            {
                "schema": "rmdm_hvdit_v4_x0_t1_status_v1",
                "state": "complete",
                "global_step": global_step,
                "epoch": epoch,
                "checkpoint": str(checkpoint_path),
                "validation": str(validation_path),
                "comparison": str(comparison_path),
                "macro_full_image_nmse_p1_p2_p3": stage_a["macro_full_image_nmse_p1_p2_p3"],
                "prediction_target": "x0",
            },
        )
        append_jsonl(
            output / "execution_history.jsonl",
            {
                "event": "pilot_complete",
                "at": datetime.now().astimezone().isoformat(timespec="seconds"),
                "global_step": global_step,
                "validation": str(validation_path),
                "comparison": str(comparison_path),
            },
        )
    accelerator.wait_for_everyone()
    result_payload = {
        "status": "complete",
        "global_step": global_step,
        "checkpoint": str(checkpoint_path),
        "validation": str(validation_path),
        "comparison": str(comparison_path),
        "macro": stage_a["macro_full_image_nmse_p1_p2_p3"],
    }
    if accelerator.is_main_process:
        print(json.dumps(result_payload, ensure_ascii=False, sort_keys=True), flush=True)
    accelerator.end_training()
    return result_payload
