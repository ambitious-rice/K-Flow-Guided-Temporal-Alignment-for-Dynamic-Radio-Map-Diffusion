"""T1 pretraining runner with Stage-A-only selection and fail-closed gate."""

from __future__ import annotations

from datetime import datetime
from pathlib import Path
from typing import Any

import torch
from torch.utils.data import DataLoader

from rmdm.data import SamplingPolicy, WindowDataset
from rmdm.diffusion import DiffusionProcess
from rmdm_hvdit_v4_joint.config import ExperimentConfig
from rmdm_hvdit_v4_joint.evaluation import evaluate_stage_a, evaluate_t1_gate
from rmdm_hvdit_v4_joint.model import build_t1_system
from rmdm_hvdit_v4_joint.provenance import build_dependency_manifest

from .checkpoint import load_training_checkpoint, save_training_checkpoint
from .engine import (
    append_jsonl,
    cosine_scheduler,
    make_accelerator,
    make_optimizer,
    prepare_model_optimizer_loader,
    require_visible_physical_gpus,
    require_scheduler_global_step,
    seed_everything,
    step_scheduler_on_global_update,
    validate_parameter_contract,
    write_json_atomic,
)
from .execution import (
    AUTHORIZED_T1_GPU_PROFILES,
    build_execution_profile,
    convert_resume_microbatch_offset,
)
from .step import training_step


def _validation_due(config: ExperimentConfig, step: int) -> bool:
    start = config.t1_train.validation_first_step
    interval = config.t1_train.validation_every_steps
    return step >= start and (step - start) % interval == 0


def _require_committed_best(
    payload: dict[str, Any],
    *,
    best_step: int,
    dependency_manifest_sha256: str,
) -> None:
    if (
        int(payload.get("global_step", -1)) != best_step
        or int(payload.get("extra", {}).get("best_step", -1)) != best_step
        or payload.get("dependency_manifest_sha256") != dependency_manifest_sha256
    ):
        raise RuntimeError("T1 best checkpoint metadata does not match the committed Stage-A selection")


def _synchronized_pause_code(accelerator: Any, local_code: int) -> int:
    code = torch.tensor([int(local_code)], device=accelerator.device, dtype=torch.int64)
    gathered = accelerator.gather(code)
    return int(gathered.max().item())


def run_t1_training(
    config: ExperimentConfig,
    *,
    config_path: str | Path,
    repository_root: str | Path,
    output_dir: str | Path | None = None,
    resume_from: str | Path | None = None,
    execution_gpus: list[int] | None = None,
    gradient_accumulation_steps: int | None = None,
    stop_at: datetime | None = None,
    pause_before_validation_seconds: int = 600,
) -> dict[str, Any]:
    train = config.t1_train
    physical_gpus = list(execution_gpus or config.pipeline.allowed_physical_gpus)
    accumulation = int(gradient_accumulation_steps or train.gradient_accumulation_steps)
    require_visible_physical_gpus(physical_gpus)
    output = Path(output_dir or (Path(config.pipeline.output_root) / "t1_pretrain")).expanduser().resolve()
    accelerator = make_accelerator(
        mixed_precision=train.mixed_precision,
        gradient_accumulation_steps=accumulation,
        data_seed=train.seed,
    )
    execution = build_execution_profile(
        physical_gpus=physical_gpus,
        actual_world_size=accelerator.num_processes,
        per_gpu_batch_size=train.per_gpu_batch_size,
        gradient_accumulation_steps=accumulation,
        effective_global_batch_size=train.effective_global_batch_size,
        authorized_profiles=AUTHORIZED_T1_GPU_PROFILES,
    )
    if pause_before_validation_seconds < 0:
        raise ValueError("pause_before_validation_seconds cannot be negative")
    seed_everything(train.seed)
    dependency_manifest = build_dependency_manifest(
        config,
        config_path=config_path,
        repository_root=repository_root,
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
        raise RuntimeError(f"T1 must expose all 1,050,000 train frames, got {len(dataset):,}")
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
        # Preserve the original 50k cosine schedule while allowing the
        # continuation to train at the configured LR floor through 80k.
        total_steps=train.lr_schedule_steps,
        warmup_steps=train.warmup_steps,
        base_learning_rate=train.learning_rate,
        min_learning_rate=train.min_learning_rate,
    )
    global_step = 0
    completed_epoch = 0
    best_score = float("inf")
    best_step = 0
    validations_without_improvement = 0
    resume_microbatch_offset = 0
    resume_samples_consumed = 0
    resume_source_execution: dict[str, Any] | None = None
    resume_validation_pending = False
    resume = str(resume_from) if resume_from is not None else train.resume_from
    if resume:
        payload = load_training_checkpoint(
            resume,
            model,
            phase="t1",
            dependency_manifest=dependency_manifest,
            optimizer=optimizer,
            scheduler=scheduler,
        )
        global_step = int(payload["global_step"])
        completed_epoch = int(payload["completed_epoch"])
        extra = payload.get("extra", {})
        best_score = float(extra.get("best_score", best_score))
        best_step = int(extra.get("best_step", 0))
        validations_without_improvement = int(extra.get("validations_without_improvement", 0))
        resume_validation_pending = bool(extra.get("validation_pending", False))
        (
            resume_microbatch_offset,
            resume_samples_consumed,
            resume_source_execution,
        ) = convert_resume_microbatch_offset(payload, execution)

    model, optimizer, loader = prepare_model_optimizer_loader(accelerator, model, optimizer, loader)
    require_scheduler_global_step(scheduler, global_step)
    policy = SamplingPolicy(config.sampling, split="train")
    diffusion = DiffusionProcess(config.diffusion)
    checkpoint_dir = output / "checkpoints"
    validation_dir = output / "validation"
    best_path = checkpoint_dir / "best.pth"
    last_path = checkpoint_dir / "last.pth"
    if accelerator.is_main_process:
        write_json_atomic(output / "dependency_manifest.json", dependency_manifest)
        write_json_atomic(
            output / "status.json",
            {
                "schema": "rmdm_hvdit_v4_joint_t1_status_v1",
                "state": "running",
                "global_step": global_step,
                "dataset_frames": len(dataset),
                "world_size": accelerator.num_processes,
                "execution_profile": execution.to_dict(),
                "effective_global_batch_size": train.effective_global_batch_size,
                "trainable_parameters": trainable,
                "total_parameters": total,
                "dependency_manifest_sha256": dependency_manifest["manifest_sha256"],
            },
        )
        append_jsonl(
            output / "execution_history.jsonl",
            {
                "event": "training_start_or_resume",
                "at": datetime.now().astimezone().isoformat(timespec="seconds"),
                "global_step": global_step,
                "completed_epoch": completed_epoch,
                "resume_microbatch_offset": resume_microbatch_offset,
                "resume_samples_consumed_in_epoch": resume_samples_consumed,
                "source_execution_profile": resume_source_execution,
                "target_execution_profile": execution.to_dict(),
                "stop_at": stop_at.isoformat(timespec="seconds") if stop_at else None,
            },
        )
    accelerator.wait_for_everyone()

    def checkpoint_extra(
        microbatches_consumed: int,
        *,
        periodic: bool = False,
        planned_pause: str = "",
        validation_pending: bool = False,
    ) -> dict[str, Any]:
        return {
            "best_score": best_score,
            "best_step": best_step,
            "best_validation_path": (
                str(validation_dir / f"stage_a_step_{best_step:06d}.json") if best_step else ""
            ),
            "validations_without_improvement": validations_without_improvement,
            "microbatches_consumed_in_epoch": int(microbatches_consumed),
            "samples_consumed_in_epoch": int(
                microbatches_consumed * execution.global_microbatch_size
            ),
            "execution_profile": execution.to_dict(),
            "periodic_checkpoint": bool(periodic),
            "planned_pause": planned_pause,
            "validation_pending": bool(validation_pending),
        }

    epoch = completed_epoch

    def complete_stage_a_validation(microbatches_consumed: int) -> None:
        nonlocal best_score, best_step, validations_without_improvement
        stage_a = evaluate_stage_a(accelerator, model, config, variant="t1")
        score = float(stage_a["macro_full_image_nmse_p1_p2_p3"])
        result_path = validation_dir / f"stage_a_step_{global_step:06d}.json"
        if accelerator.is_main_process:
            write_json_atomic(result_path, stage_a)
        improved = score < best_score
        if improved:
            best_score = score
            best_step = global_step
            validations_without_improvement = 0
        else:
            validations_without_improvement += 1
        extra = checkpoint_extra(microbatches_consumed)
        # ``last`` is the validation transaction's commit record.  Persist an
        # improved ``best`` first so an interruption can never leave ``last``
        # pointing at a best_step whose model artifact is still stale.  If the
        # process stops before ``last`` is replaced, its pre-validation pending
        # marker makes resume rerun this exact evaluator safely.
        if improved:
            save_training_checkpoint(
                accelerator,
                best_path,
                model,
                optimizer,
                scheduler,
                config,
                dependency_manifest,
                phase="t1",
                completed_epoch=epoch,
                global_step=global_step,
                extra=extra,
            )
        save_training_checkpoint(
            accelerator,
            last_path,
            model,
            optimizer,
            scheduler,
            config,
            dependency_manifest,
            phase="t1",
            completed_epoch=epoch,
            global_step=global_step,
            extra=extra,
        )

    if resume_validation_pending:
        complete_stage_a_validation(resume_microbatch_offset)
        resume_validation_pending = False

    stop_early = (
        global_step >= train.early_stop_min_step
        and validations_without_improvement >= train.patience_validations
    )
    optimizer.zero_grad(set_to_none=True)
    planned_pause = ""
    while global_step < train.max_steps and not stop_early:
        dataset.set_epoch(epoch)
        policy.set_epoch(epoch)
        if hasattr(loader, "set_epoch"):
            loader.set_epoch(epoch)
        microbatches_consumed = 0
        for batch_index, dense_batch in enumerate(loader):
            if epoch == completed_epoch and batch_index < resume_microbatch_offset:
                continue
            microbatches_consumed = batch_index + 1
            with accelerator.accumulate(model):
                with accelerator.autocast():
                    result = training_step(
                        model,
                        dense_batch,
                        policy,
                        diffusion,
                        training_seed=train.seed,
                        epoch=epoch,
                        variant="t1",
                        pinn_k=config.stage1.pinn_k,
                        pinn_weight=config.stage1.pinn_weight,
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
                            result.diffusion_loss.detach().float(),
                            result.calibration_loss.detach().float(),
                            result.pinn_loss.detach().float(),
                            result.sampling_rate_mean.detach().float(),
                            result.epsilon_mse_per_sample.mean().detach().float(),
                            result.x0_mse_per_sample.mean().detach().float(),
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
                            "diffusion_loss": float(reduced[1]),
                            "calibration_loss": float(reduced[2]),
                            "pinn_loss": float(reduced[3]),
                            "sampling_rate_mean": float(reduced[4]),
                            "epsilon_mse": float(reduced[5]),
                            "x0_mse": float(reduced[6]),
                            "learning_rate": float(optimizer.param_groups[0]["lr"]),
                            "scheduler_step": int(scheduler.last_epoch),
                        },
                    )
            now = datetime.now().astimezone()
            seconds_to_stop = (stop_at - now).total_seconds() if stop_at else None
            local_pause_code = 0
            if stop_at is not None and now >= stop_at:
                local_pause_code = 2
            elif (
                stop_at is not None
                and (
                    _validation_due(config, global_step)
                    or _validation_due(config, global_step + 1)
                )
                and seconds_to_stop is not None
                and seconds_to_stop <= pause_before_validation_seconds
            ):
                local_pause_code = 1
            if stop_at is not None:
                pause_code = _synchronized_pause_code(accelerator, local_pause_code)
                if pause_code == 2:
                    planned_pause = "wall_clock_deadline"
                elif pause_code == 1:
                    planned_pause = "before_validation_near_wall_clock_deadline"

            if planned_pause:
                save_training_checkpoint(
                    accelerator,
                    last_path,
                    model,
                    optimizer,
                    scheduler,
                    config,
                    dependency_manifest,
                    phase="t1",
                    completed_epoch=epoch,
                    global_step=global_step,
                    extra=checkpoint_extra(
                        microbatches_consumed,
                        planned_pause=planned_pause,
                        # If the wall-clock boundary lands exactly on a
                        # validation step, resume must run that validation
                        # before consuming another training sample.
                        validation_pending=_validation_due(config, global_step),
                    ),
                )
            elif _validation_due(config, global_step):
                # Persist optimizer progress before a long evaluator can fail or
                # be interrupted. A pending marker makes resume retry this exact
                # validation before consuming another training sample.
                save_training_checkpoint(
                    accelerator,
                    last_path,
                    model,
                    optimizer,
                    scheduler,
                    config,
                    dependency_manifest,
                    phase="t1",
                    completed_epoch=epoch,
                    global_step=global_step,
                    extra=checkpoint_extra(
                        microbatches_consumed,
                        validation_pending=True,
                    ),
                )
                complete_stage_a_validation(microbatches_consumed)
                stop_early = (
                    global_step >= train.early_stop_min_step
                    and validations_without_improvement >= train.patience_validations
                )
            elif global_step % train.checkpoint_every_steps == 0:
                extra = checkpoint_extra(microbatches_consumed, periodic=True)
                save_training_checkpoint(
                    accelerator,
                    last_path,
                    model,
                    optimizer,
                    scheduler,
                    config,
                    dependency_manifest,
                    phase="t1",
                    completed_epoch=epoch,
                    global_step=global_step,
                    extra=extra,
                )
            if global_step >= train.max_steps or stop_early or planned_pause:
                break
        if planned_pause:
            break
        epoch += 1
        completed_epoch = epoch
        resume_microbatch_offset = 0

    if planned_pause:
        if accelerator.is_main_process:
            write_json_atomic(
                output / "status.json",
                {
                    "schema": "rmdm_hvdit_v4_joint_t1_status_v1",
                    "state": "paused",
                    "global_step": global_step,
                    "completed_epoch": epoch,
                    "pause_reason": planned_pause,
                    "last_checkpoint": str(last_path),
                    "execution_profile": execution.to_dict(),
                    "effective_global_batch_size": train.effective_global_batch_size,
                },
            )
            append_jsonl(
                output / "execution_history.jsonl",
                {
                    "event": "planned_pause",
                    "at": datetime.now().astimezone().isoformat(timespec="seconds"),
                    "global_step": global_step,
                    "completed_epoch": epoch,
                    "reason": planned_pause,
                    "checkpoint": str(last_path),
                    "execution_profile": execution.to_dict(),
                },
            )
        accelerator.wait_for_everyone()
        accelerator.end_training()
        return {
            "status": "paused",
            "paused": True,
            "passed": False,
            "global_step": global_step,
            "last_checkpoint": str(last_path),
            "pause_reason": planned_pause,
        }

    local_gate_pause = bool(
        stop_at is not None
        and best_step > 0
        and (stop_at - datetime.now().astimezone()).total_seconds()
        <= pause_before_validation_seconds
    )
    gate_pause = (
        bool(_synchronized_pause_code(accelerator, int(local_gate_pause)))
        if stop_at is not None
        else False
    )
    if gate_pause:
        if accelerator.is_main_process:
            write_json_atomic(
                output / "status.json",
                {
                    "schema": "rmdm_hvdit_v4_joint_t1_status_v1",
                    "state": "paused",
                    "global_step": global_step,
                    "completed_epoch": completed_epoch,
                    "pause_reason": "before_gate_near_wall_clock_deadline",
                    "last_checkpoint": str(last_path),
                    "execution_profile": execution.to_dict(),
                    "effective_global_batch_size": train.effective_global_batch_size,
                },
            )
        accelerator.wait_for_everyone()
        accelerator.end_training()
        return {
            "status": "paused",
            "paused": True,
            "passed": False,
            "global_step": global_step,
            "last_checkpoint": str(last_path),
            "pause_reason": "before_gate_near_wall_clock_deadline",
        }

    if best_step <= 0 or not best_path.is_file():
        raise RuntimeError("T1 completed without a valid Stage-A best checkpoint")
    accelerator.wait_for_everyone()
    best_payload = torch.load(best_path, map_location="cpu", weights_only=False)
    _require_committed_best(
        best_payload,
        best_step=best_step,
        dependency_manifest_sha256=dependency_manifest["manifest_sha256"],
    )
    accelerator.unwrap_model(model).load_state_dict(best_payload["model"], strict=True)
    best_validation_path = validation_dir / f"stage_a_step_{best_step:06d}.json"
    stage_a_best = __import__("json").loads(best_validation_path.read_text(encoding="utf-8"))
    ablated = evaluate_stage_a(
        accelerator,
        model,
        config,
        variant="t1",
        ablate_raw_observations=True,
    )
    gate = evaluate_t1_gate(config, full_result=stage_a_best, ablated_result=ablated)
    if accelerator.is_main_process:
        write_json_atomic(validation_dir / "best_raw_observation_ablation.json", ablated)
        write_json_atomic(output / "gate.json", gate)
        write_json_atomic(
            output / "status.json",
            {
                "schema": "rmdm_hvdit_v4_joint_t1_status_v1",
                "state": "passed" if gate["passed"] else "gate_failed",
                "global_step": global_step,
                "completed_epoch": completed_epoch,
                "best_step": best_step,
                "best_score": best_score,
                "best_checkpoint": str(best_path),
                "gate": gate,
            },
        )
    accelerator.wait_for_everyone()
    accelerator.end_training()
    return {
        "status": "completed",
        "passed": bool(gate["passed"]),
        "best_checkpoint": str(best_path),
        "best_step": best_step,
        "best_score": best_score,
        "gate": gate,
    }
