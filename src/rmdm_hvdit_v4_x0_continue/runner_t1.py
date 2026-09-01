"""Resume exact W1 x0 training from 10k with periodic aligned validation."""

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
from rmdm_hvdit_v4_x0.model import build_t1_system
from rmdm_hvdit_v4_x0.training.step import training_step

from .checkpoint import (
    load_continuation_checkpoint,
    load_objective_finetune_checkpoint,
    load_source_checkpoint,
    save_checkpoint,
)
from .provenance import build_dependency_manifest


def _validation_due(config: Any, step: int) -> bool:
    start = config.t1_train.validation_first_step
    interval = config.t1_train.validation_every_steps
    return step == config.t1_train.max_steps or (
        step >= start and (step - start) % interval == 0
    )


def _output_path(config: Any, root: Path, output_dir: str | Path | None) -> Path:
    allowed = (root / config.pipeline.output_root).resolve()
    output = Path(output_dir).expanduser().resolve() if output_dir else allowed / "t1_from10k_to50k"
    if not output.is_relative_to(allowed):
        raise ValueError(f"Continuation output must stay below {allowed}, got {output}")
    return output


def _read_json(path: str | Path) -> dict[str, Any]:
    with Path(path).expanduser().resolve().open("r", encoding="utf-8") as handle:
        payload = json.load(handle)
    if not isinstance(payload, dict):
        raise TypeError(f"Expected JSON object: {path}")
    return payload


def run_continuation(
    config: Any,
    *,
    config_path: str | Path,
    repository_root: str | Path,
    source_checkpoint: str | Path | None,
    output_dir: str | Path | None = None,
    resume_from: str | Path | None = None,
    finetune_from: str | Path | None = None,
    observation_alignment_weight: float = 0.0,
    from_scratch: bool = False,
    required_source_config_differences: set[str] | None = None,
) -> dict[str, Any]:
    root = Path(repository_root).expanduser().resolve()
    source = Path(source_checkpoint).expanduser().resolve() if source_checkpoint else None
    output = _output_path(config, root, output_dir)
    train = config.t1_train
    if (
        not from_scratch
        and required_source_config_differences is None
        and not config.model.use_explicit_tx_condition
    ):
        required_source_config_differences = {
            "t1_train.max_steps",
            "t1_train.validation_first_step",
            "t1_train.patience_validations",
            "pipeline.output_root",
            "pipeline.free_memory_mib",
            "pipeline.lock_file",
        }
        if len(config.pipeline.allowed_physical_gpus) == 8:
            required_source_config_differences.update(
                {"t1_train.per_gpu_batch_size", "pipeline.allowed_physical_gpus"}
            )
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
    expected_world_size = len(physical_gpus)
    if expected_world_size not in {4, 8}:
        raise RuntimeError(
            f"x0 continuation supports four or eight DDP processes, configured {expected_world_size}"
        )
    if accelerator.num_processes != expected_world_size:
        raise RuntimeError(
            "x0 continuation process count must match allowed physical GPUs: "
            f"expected {expected_world_size}, got {accelerator.num_processes}"
        )

    seed_everything(train.seed)
    dependency_manifest = build_dependency_manifest(
        config,
        config_path=config_path,
        repository_root=root,
        source_checkpoint=source,
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
        raise RuntimeError(f"W1 continuation must expose 1,050,000 frames, got {len(dataset):,}")
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

    last_path = output / "checkpoints" / "last.pth"
    best_path = output / "checkpoints" / "best.pth"
    global_step: int
    epoch: int
    resume_microbatch_offset: int
    validation_pending: bool
    best_score: float
    best_step: int
    best_checkpoint: str
    validations_without_improvement: int
    if finetune_from and (resume_from or from_scratch):
        raise ValueError("finetune_from cannot be combined with resume_from or from_scratch")
    if observation_alignment_weight < 0:
        raise ValueError("observation_alignment_weight must be non-negative")
    if resume_from:
        payload = load_continuation_checkpoint(
            resume_from,
            model,
            optimizer,
            scheduler,
            dependency_manifest,
        )
        extra = payload.get("extra", {})
        global_step = int(payload["global_step"])
        epoch = int(payload["epoch"])
        resume_microbatch_offset = int(payload["microbatches_consumed_in_epoch"])
        validation_pending = bool(payload.get("validation_pending", False))
        best_score = float(extra["best_score"])
        best_step = int(extra["best_step"])
        best_checkpoint = str(extra["best_checkpoint"])
        validations_without_improvement = int(extra["validations_without_improvement"])
    elif from_scratch:
        if last_path.exists():
            raise FileExistsError(f"Scratch-training output already exists at {last_path}")
        global_step = 0
        epoch = 0
        resume_microbatch_offset = 0
        validation_pending = False
        best_score = float("inf")
        best_step = 0
        best_checkpoint = ""
        validations_without_improvement = 0
    elif finetune_from:
        if last_path.exists():
            raise FileExistsError(f"Fine-tuning output already exists; pass --resume-from {last_path}")
        payload = load_objective_finetune_checkpoint(
            finetune_from,
            model,
            optimizer,
            scheduler,
        )
        extra = payload.get("extra", {})
        global_step = int(payload["global_step"])
        epoch = int(payload["epoch"])
        resume_microbatch_offset = int(payload["microbatches_consumed_in_epoch"])
        validation_pending = _validation_due(config, global_step)
        best_score = float(extra.get("best_score", float("inf")))
        best_step = global_step
        best_checkpoint = str(Path(finetune_from).expanduser().resolve())
        validations_without_improvement = 0
    else:
        if source is None:
            raise ValueError("source_checkpoint is required unless --from-scratch is used")
        if last_path.exists():
            raise FileExistsError(f"Continuation already exists; pass --resume-from {last_path}")
        payload = load_source_checkpoint(
            source,
            model,
            optimizer,
            scheduler,
            config,
            required_config_differences=required_source_config_differences,
        )
        source_validation = _read_json(payload["validation_path"])
        global_step = int(payload["global_step"])
        epoch = int(payload["epoch"])
        resume_microbatch_offset = int(payload["microbatches_consumed_in_epoch"])
        validation_pending = False
        best_score = float(source_validation["macro_full_image_nmse_p1_p2_p3"])
        best_step = global_step
        best_checkpoint = str(source)
        validations_without_improvement = 0

    model, optimizer, loader = prepare_model_optimizer_loader(accelerator, model, optimizer, loader)
    require_scheduler_global_step(scheduler, global_step)
    policy = SamplingPolicy(config.sampling, split="train")
    diffusion = DiffusionProcess(config.diffusion)
    if diffusion.scheduler.config.prediction_type != "sample":
        raise RuntimeError("Continuation scheduler no longer interprets output as x0")

    def checkpoint_extra() -> dict[str, Any]:
        return {
            "best_score": best_score,
            "best_step": best_step,
            "best_checkpoint": best_checkpoint,
            "validations_without_improvement": validations_without_improvement,
            "source_checkpoint": str(source),
            "initialization": "random" if from_scratch else "checkpoint",
            "physical_gpus": physical_gpus,
        }

    if accelerator.is_main_process:
        write_json_atomic(output / "dependency_manifest.json", dependency_manifest)
        write_json_atomic(
            output / "status.json",
            {
                "schema": "rmdm_hvdit_v4_x0_continue_status_v1",
                "state": "training",
                "global_step": global_step,
                "epoch": epoch,
                "source_checkpoint": str(source),
                "initialization": "random" if from_scratch else "checkpoint",
                "world_size": accelerator.num_processes,
                "physical_gpus": physical_gpus,
                "per_gpu_batch_size": train.per_gpu_batch_size,
                "gradient_accumulation_steps": train.gradient_accumulation_steps,
                "effective_global_batch_size": train.effective_global_batch_size,
                "prediction_target": "x0",
                "observation_alignment_weight": float(observation_alignment_weight),
                "spatial_regularizer": regularizer_type,
                "spatial_regularizer_weight": (
                    float(config.stage1.pinn_weight)
                    if regularizer_weight is None
                    else float(regularizer_weight)
                ),
                "trainable_parameters": trainable,
                "total_parameters": total,
                "best_step": best_step,
                "best_score": best_score,
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
                "finetune_from": str(finetune_from) if finetune_from else "",
                "observation_alignment_weight": float(observation_alignment_weight),
                "source_checkpoint": str(source),
                "initialization": "random" if from_scratch else "checkpoint",
                "physical_gpus": physical_gpus,
            },
        )
    accelerator.wait_for_everyone()

    last_microbatches_consumed = resume_microbatch_offset

    def complete_validation() -> None:
        nonlocal best_score, best_step, best_checkpoint, validations_without_improvement
        if accelerator.is_main_process:
            write_json_atomic(
                output / "status.json",
                {
                    "schema": "rmdm_hvdit_v4_x0_continue_status_v1",
                    "state": "validating",
                    "global_step": global_step,
                    "epoch": epoch,
                    "best_step": best_step,
                    "best_score": best_score,
                },
            )
        stage_a = evaluate_stage_a(accelerator, model, config, variant="t1")
        stage_a["schema"] = "rmdm_hvdit_v4_x0_continue_evaluation_v1"
        stage_a["prediction_target"] = "x0"
        score = float(stage_a["macro_full_image_nmse_p1_p2_p3"])
        result_path = output / "validation" / f"stage_a_step_{global_step:06d}.json"
        if accelerator.is_main_process:
            write_json_atomic(result_path, stage_a)
        improved = score < best_score
        if improved:
            best_score = score
            best_step = global_step
            best_checkpoint = str(best_path)
            validations_without_improvement = 0
            save_checkpoint(
                accelerator,
                best_path,
                model,
                optimizer,
                scheduler,
                config,
                dependency_manifest,
                epoch=epoch,
                global_step=global_step,
                microbatches_consumed_in_epoch=last_microbatches_consumed,
                validation_pending=False,
                extra=checkpoint_extra(),
            )
        else:
            validations_without_improvement += 1
        save_checkpoint(
            accelerator,
            last_path,
            model,
            optimizer,
            scheduler,
            config,
            dependency_manifest,
            epoch=epoch,
            global_step=global_step,
            microbatches_consumed_in_epoch=last_microbatches_consumed,
            validation_pending=False,
            extra=checkpoint_extra(),
        )
        if accelerator.is_main_process:
            write_json_atomic(
                output / "status.json",
                {
                    "schema": "rmdm_hvdit_v4_x0_continue_status_v1",
                    "state": "training",
                    "global_step": global_step,
                    "epoch": epoch,
                    "latest_validation": str(result_path),
                    "latest_score": score,
                    "best_step": best_step,
                    "best_score": best_score,
                    "best_checkpoint": best_checkpoint,
                    "validations_without_improvement": validations_without_improvement,
                },
            )

    if validation_pending:
        complete_validation()
        validation_pending = False

    optimizer.zero_grad(set_to_none=True)
    stop_early = (
        global_step >= train.early_stop_min_step
        and validations_without_improvement >= train.patience_validations
    )
    while global_step < train.max_steps and not stop_early:
        dataset.set_epoch(epoch)
        policy.set_epoch(epoch)
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
                        policy,
                        diffusion,
                        training_seed=train.seed,
                        epoch=epoch,
                        pinn_k=config.stage1.pinn_k,
                        pinn_weight=config.stage1.pinn_weight,
                        regularizer_type=regularizer_type,
                        regularizer_weight=regularizer_weight,
                        hessian_epsilon=hessian_epsilon,
                        use_tx_source_supervision=config.model.use_tx_source_supervision,
                        observation_alignment_weight=observation_alignment_weight,
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
                            result.observation_alignment_loss.detach().float(),
                            result.calibration_loss.detach().float(),
                            result.pinn_loss.detach().float(),
                            result.equation_regularizer_loss.detach().float(),
                            result.semantic_anchor_loss.detach().float(),
                            result.sampling_rate_mean.detach().float(),
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
                            "observation_alignment_loss": float(reduced[2]),
                            "calibration_loss": float(reduced[3]),
                            "spatial_regularizer_type": regularizer_type,
                            "spatial_regularizer_loss": float(reduced[4]),
                            "equation_regularizer_loss": float(reduced[5]),
                            "semantic_anchor_loss": float(reduced[6]),
                            "sampling_rate_mean": float(reduced[7]),
                            "derived_epsilon_mse": float(reduced[8]),
                            "learning_rate": float(optimizer.param_groups[0]["lr"]),
                            "scheduler_step": int(scheduler.last_epoch),
                        },
                    )

            if _validation_due(config, global_step):
                save_checkpoint(
                    accelerator,
                    last_path,
                    model,
                    optimizer,
                    scheduler,
                    config,
                    dependency_manifest,
                    epoch=epoch,
                    global_step=global_step,
                    microbatches_consumed_in_epoch=last_microbatches_consumed,
                    validation_pending=True,
                    extra=checkpoint_extra(),
                )
                complete_validation()
                stop_early = (
                    global_step >= train.early_stop_min_step
                    and validations_without_improvement >= train.patience_validations
                )
            elif global_step % train.checkpoint_every_steps == 0:
                save_checkpoint(
                    accelerator,
                    last_path,
                    model,
                    optimizer,
                    scheduler,
                    config,
                    dependency_manifest,
                    epoch=epoch,
                    global_step=global_step,
                    microbatches_consumed_in_epoch=last_microbatches_consumed,
                    validation_pending=False,
                    extra=checkpoint_extra(),
                )
            if global_step >= train.max_steps or stop_early:
                break
        if global_step >= train.max_steps or stop_early:
            break
        epoch += 1
        resume_microbatch_offset = 0

    state = "early_stopped" if stop_early and global_step < train.max_steps else "complete"
    if accelerator.is_main_process:
        write_json_atomic(
            output / "status.json",
            {
                "schema": "rmdm_hvdit_v4_x0_continue_status_v1",
                "state": state,
                "global_step": global_step,
                "epoch": epoch,
                "best_step": best_step,
                "best_score": best_score,
                "best_checkpoint": best_checkpoint,
                "validations_without_improvement": validations_without_improvement,
                "prediction_target": "x0",
            },
        )
        append_jsonl(
            output / "execution_history.jsonl",
            {
                "event": state,
                "at": datetime.now().astimezone().isoformat(timespec="seconds"),
                "global_step": global_step,
                "best_step": best_step,
                "best_score": best_score,
            },
        )
    accelerator.wait_for_everyone()
    result_payload = {
        "status": state,
        "global_step": global_step,
        "best_step": best_step,
        "best_score": best_score,
        "best_checkpoint": best_checkpoint,
    }
    if accelerator.is_main_process:
        print(json.dumps(result_payload, ensure_ascii=False, sort_keys=True), flush=True)
    accelerator.end_training()
    return result_payload
