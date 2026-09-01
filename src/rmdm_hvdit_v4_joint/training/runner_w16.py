"""Configured multi-GPU W16 runner with first96 Stage-A and top-3 Stage-B."""

from __future__ import annotations

import json
import math
from pathlib import Path
from typing import Any

import torch
from torch.utils.data import DataLoader

from rmdm.data import SamplingPolicy, WindowDataset
from rmdm.diffusion import DiffusionProcess
from rmdm_hvdit_v4_joint import ARCHITECTURE_ID
from rmdm_hvdit_v4_joint.config import ExperimentConfig
from rmdm_hvdit_v4_joint.evaluation import combine_results, evaluate_stage_a
from rmdm_hvdit_v4_joint.model import build_w16_system
from rmdm_hvdit_v4_joint.provenance import build_dependency_manifest, sha256_file
from rmdm_hvdit_v4_joint.transfer.inflate_t1_to_w16 import load_w16_initialization

from .checkpoint import (
    W16_SELECTION_CANDIDATE_SCHEMA,
    load_training_checkpoint,
    save_training_checkpoint,
    write_torch_atomic,
)
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
from .execution import AUTHORIZED_PIPELINE_GPU_PROFILES, build_execution_profile
from .step import training_step


def _stage_a_due(config: ExperimentConfig, completed_epoch: int) -> bool:
    start = config.w16_train.validation_first_epoch
    interval = config.w16_train.validation_every_epochs
    return completed_epoch >= start and (completed_epoch - start) % interval == 0


def _candidate_payload(
    accelerator: Any,
    model: torch.nn.Module,
    *,
    epoch: int,
    global_step: int,
    stage_a: dict[str, Any],
    dependency_manifest: dict[str, Any],
) -> dict[str, Any] | None:
    if not accelerator.is_main_process:
        return None
    state = {key: value.detach().cpu() for key, value in accelerator.unwrap_model(model).state_dict().items()}
    return {
        "schema": W16_SELECTION_CANDIDATE_SCHEMA,
        "architecture_id": ARCHITECTURE_ID,
        "phase": "w16",
        "epoch": int(epoch),
        "global_step": int(global_step),
        "score": float(stage_a["macro_full_image_nmse_p1_p2_p3"]),
        "stage_a": stage_a,
        "dependency_manifest_sha256": dependency_manifest["manifest_sha256"],
        "model": state,
    }


def run_w16_training(
    config: ExperimentConfig,
    *,
    config_path: str | Path,
    repository_root: str | Path,
    initialization_path: str | Path,
    output_dir: str | Path | None = None,
    per_gpu_batch_size: int | None = None,
    resume_from: str | Path | None = None,
    execution_gpus: list[int] | None = None,
) -> dict[str, Any]:
    train = config.w16_train
    physical_gpus = list(execution_gpus or config.pipeline.allowed_physical_gpus)
    require_visible_physical_gpus(physical_gpus)
    microbatch = int(per_gpu_batch_size or train.default_per_gpu_batch_size)
    if microbatch not in train.microbatch_candidates:
        raise ValueError(f"W16 microbatch {microbatch} is not an audited candidate")
    denominator = len(physical_gpus) * microbatch
    if train.effective_global_batch_size % denominator:
        raise ValueError("W16 global batch cannot be represented by this distributed microbatch")
    accumulation = train.effective_global_batch_size // denominator
    output = Path(output_dir or (Path(config.pipeline.output_root) / "w16_train")).expanduser().resolve()
    accelerator = make_accelerator(
        mixed_precision=train.mixed_precision,
        gradient_accumulation_steps=accumulation,
        even_batches=False,
        data_seed=train.seed,
    )
    execution = build_execution_profile(
        physical_gpus=physical_gpus,
        actual_world_size=accelerator.num_processes,
        per_gpu_batch_size=microbatch,
        gradient_accumulation_steps=accumulation,
        effective_global_batch_size=train.effective_global_batch_size,
        authorized_profiles=AUTHORIZED_PIPELINE_GPU_PROFILES,
    )
    seed_everything(train.seed)
    dependency_manifest = build_dependency_manifest(
        config,
        config_path=config_path,
        repository_root=repository_root,
    )
    model = build_w16_system(config)
    trainable, total = validate_parameter_contract(
        model,
        config.model.expected_trainable_parameters_min,
        config.model.expected_trainable_parameters_max,
    )
    resume = str(resume_from) if resume_from is not None else train.resume_from
    if not resume:
        load_w16_initialization(
            initialization_path,
            model,
            expected_dependency_manifest_sha256=dependency_manifest["manifest_sha256"],
        )
    dataset = WindowDataset(
        root=config.data.root,
        split="train",
        split_file=config.data.split_file,
        window_size=16,
        seed=config.sampling.seed,
        cache_size=config.data.cache_size,
        tx_heatmap_sigma_px=config.data.tx_heatmap_sigma_px,
    )
    expected_windows = train.updates_per_epoch * train.effective_global_batch_size
    if len(dataset) != 10_500 or len(dataset) - expected_windows != 4:
        raise RuntimeError(
            "W16 fairness budget requires 10,500 available windows and exactly four deterministic "
            f"drop-last samples; got available={len(dataset):,}, processed={expected_windows:,}"
        )
    loader = DataLoader(
        dataset,
        batch_size=microbatch,
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
    warmup_steps = int(math.ceil(train.warmup_epochs * train.updates_per_epoch))
    scheduler = cosine_scheduler(
        optimizer,
        total_steps=train.max_steps,
        warmup_steps=warmup_steps,
        base_learning_rate=train.learning_rate,
        min_learning_rate=train.min_learning_rate,
    )
    completed_epoch = 0
    global_step = 0
    best_score = float("inf")
    best_epoch = 0
    validations_without_improvement = 0
    candidates: list[dict[str, Any]] = []
    resume_validation_pending = False
    if resume:
        payload = load_training_checkpoint(
            resume,
            model,
            phase="w16",
            dependency_manifest=dependency_manifest,
            optimizer=optimizer,
            scheduler=scheduler,
        )
        completed_epoch = int(payload["completed_epoch"])
        global_step = int(payload["global_step"])
        extra = payload.get("extra", {})
        if int(extra.get("per_gpu_batch_size", microbatch)) != microbatch:
            raise ValueError("W16 resume microbatch differs from the audited original run")
        if int(extra.get("gradient_accumulation_steps", accumulation)) != accumulation:
            raise ValueError("W16 resume accumulation differs from the audited original run")
        best_score = float(extra.get("best_score", best_score))
        best_epoch = int(extra.get("best_epoch", 0))
        validations_without_improvement = int(extra.get("validations_without_improvement", 0))
        candidates = list(extra.get("candidates", []))
        resume_validation_pending = bool(extra.get("validation_pending", False))

    model, optimizer, loader = prepare_model_optimizer_loader(accelerator, model, optimizer, loader)
    require_scheduler_global_step(scheduler, global_step)
    policy = SamplingPolicy(config.sampling, split="train")
    diffusion = DiffusionProcess(config.diffusion)
    checkpoint_dir = output / "checkpoints"
    candidate_dir = output / "selection_candidates"
    validation_dir = output / "validation"
    last_path = checkpoint_dir / "last.pth"
    best_path = checkpoint_dir / "best.pth"
    if accelerator.is_main_process:
        write_json_atomic(output / "dependency_manifest.json", dependency_manifest)
        write_json_atomic(
            output / "batch_resolution.json",
            {
                "schema": "rmdm_hvdit_v4_joint_w16_batch_resolution_v1",
                "physical_gpus": list(execution.physical_gpus),
                "world_size": execution.world_size,
                "per_gpu_batch_size": microbatch,
                "gradient_accumulation_steps": accumulation,
                "effective_global_batch_size": train.effective_global_batch_size,
            },
        )
        write_json_atomic(
            output / "status.json",
            {
                "schema": "rmdm_hvdit_v4_joint_w16_status_v1",
                "state": "running",
                "completed_epoch": completed_epoch,
                "global_step": global_step,
                "trainable_parameters": trainable,
                "total_parameters": total,
                "dependency_manifest_sha256": dependency_manifest["manifest_sha256"],
                "execution_profile": execution.to_dict(),
            },
        )
    accelerator.wait_for_everyone()

    def checkpoint_extra(*, validation_pending: bool = False) -> dict[str, Any]:
        return {
            "best_score": best_score,
            "best_epoch": best_epoch,
            "validations_without_improvement": validations_without_improvement,
            "candidates": candidates,
            "per_gpu_batch_size": microbatch,
            "gradient_accumulation_steps": accumulation,
            "execution_profile": execution.to_dict(),
            "validation_pending": bool(validation_pending),
        }

    def complete_stage_a_validation() -> None:
        nonlocal best_score, best_epoch, validations_without_improvement
        if not _stage_a_due(config, completed_epoch):
            raise RuntimeError(
                f"W16 checkpoint requests validation at non-validation epoch {completed_epoch}"
            )
        stage_a = evaluate_stage_a(accelerator, model, config, variant="w16")
        score = float(stage_a["macro_full_image_nmse_p1_p2_p3"])
        stage_a_path = validation_dir / f"stage_a_epoch_{completed_epoch:03d}.json"
        candidate_path = candidate_dir / f"epoch_{completed_epoch:03d}.pth"
        candidate_payload = _candidate_payload(
            accelerator,
            model,
            epoch=completed_epoch,
            global_step=global_step,
            stage_a=stage_a,
            dependency_manifest=dependency_manifest,
        )
        if accelerator.is_main_process:
            write_json_atomic(stage_a_path, stage_a)
            assert candidate_payload is not None
            write_torch_atomic(candidate_path, candidate_payload)
        candidates.append(
            {
                "epoch": completed_epoch,
                "global_step": global_step,
                "score": score,
                "checkpoint": str(candidate_path),
                "stage_a": str(stage_a_path),
            }
        )
        improved = score < best_score
        if improved:
            best_score = score
            best_epoch = completed_epoch
            validations_without_improvement = 0
        elif completed_epoch > train.early_stop_min_epoch:
            validations_without_improvement += 1

        extra = checkpoint_extra()
        # As in T1, ``last`` is the validation commit record.  Saving an
        # improved best artifact before it prevents resume from observing new
        # best metadata paired with stale best weights.
        if improved:
            save_training_checkpoint(
                accelerator,
                best_path,
                model,
                optimizer,
                scheduler,
                config,
                dependency_manifest,
                phase="w16",
                completed_epoch=completed_epoch,
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
            phase="w16",
            completed_epoch=completed_epoch,
            global_step=global_step,
            extra=extra,
        )

    if resume_validation_pending:
        complete_stage_a_validation()

    stop_early = (
        completed_epoch > train.early_stop_min_epoch
        and validations_without_improvement >= train.patience_validations
    )
    optimizer.zero_grad(set_to_none=True)
    for epoch in range(completed_epoch, train.epochs):
        if stop_early:
            break
        dataset.set_epoch(epoch)
        policy.set_epoch(epoch)
        if hasattr(loader, "set_epoch"):
            loader.set_epoch(epoch)
        epoch_start_step = global_step
        running_loss = torch.zeros((), device=accelerator.device)
        microbatches = 0
        for dense_batch in loader:
            if microbatches >= train.updates_per_epoch * accumulation:
                break
            with accelerator.accumulate(model):
                with accelerator.autocast():
                    result = training_step(
                        model,
                        dense_batch,
                        policy,
                        diffusion,
                        training_seed=train.seed,
                        epoch=epoch,
                        variant="w16",
                        pinn_k=config.stage1.pinn_k,
                        pinn_weight=config.stage1.pinn_weight,
                    )
                accelerator.backward(result.loss)
                if accelerator.sync_gradients:
                    accelerator.clip_grad_norm_(model.parameters(), train.gradient_clip_norm)
                optimizer.step()
                step_scheduler_on_global_update(accelerator, scheduler)
                optimizer.zero_grad(set_to_none=True)
            running_loss += result.loss.detach().float()
            microbatches += 1
            if accelerator.sync_gradients:
                global_step += 1
                require_scheduler_global_step(scheduler, global_step)
                if global_step % train.log_every_steps == 0:
                    reduced = accelerator.reduce(result.loss.detach().float(), reduction="mean")
                    if accelerator.is_main_process:
                        append_jsonl(
                            output / "train_steps.jsonl",
                            {
                                "epoch": epoch + 1,
                                "global_step": global_step,
                                "loss": float(reduced),
                                "learning_rate": float(optimizer.param_groups[0]["lr"]),
                                "scheduler_step": int(scheduler.last_epoch),
                            },
                        )
                if global_step >= train.max_steps:
                    break
        completed_epoch = epoch + 1
        updates = global_step - epoch_start_step
        if global_step < train.max_steps and updates != train.updates_per_epoch:
            raise RuntimeError(f"Epoch {completed_epoch} produced {updates} updates, expected {train.updates_per_epoch}")
        epoch_loss = accelerator.reduce(running_loss / max(microbatches, 1), reduction="mean")
        if accelerator.is_main_process:
            append_jsonl(
                output / "train_epochs.jsonl",
                {
                    "epoch": completed_epoch,
                    "global_step": global_step,
                    "loss": float(epoch_loss),
                    "updates": updates,
                    "learning_rate": float(optimizer.param_groups[0]["lr"]),
                },
            )

        if _stage_a_due(config, completed_epoch):
            # Commit the optimizer state before entering a long evaluator.  A
            # failed/interrupted validation then resumes here without
            # retraining the just-completed epoch or silently skipping it.
            save_training_checkpoint(
                accelerator,
                last_path,
                model,
                optimizer,
                scheduler,
                config,
                dependency_manifest,
                phase="w16",
                completed_epoch=completed_epoch,
                global_step=global_step,
                extra=checkpoint_extra(validation_pending=True),
            )
            complete_stage_a_validation()
            stop_early = (
                completed_epoch > train.early_stop_min_epoch
                and validations_without_improvement >= train.patience_validations
            )
        else:
            save_training_checkpoint(
                accelerator,
                last_path,
                model,
                optimizer,
                scheduler,
                config,
                dependency_manifest,
                phase="w16",
                completed_epoch=completed_epoch,
                global_step=global_step,
                extra=checkpoint_extra(),
            )
        if global_step >= train.max_steps or stop_early:
            break

    if not candidates:
        raise RuntimeError("W16 training produced no Stage-A selection candidates")
    top = sorted(candidates, key=lambda item: (float(item["score"]), int(item["epoch"])))[: train.stage_b_top_k]
    stage_b_records: list[dict[str, Any]] = []
    core = accelerator.unwrap_model(model)
    for record in top:
        payload = torch.load(record["checkpoint"], map_location="cpu", weights_only=False)
        if (
            payload.get("schema") != W16_SELECTION_CANDIDATE_SCHEMA
            or payload.get("architecture_id") != ARCHITECTURE_ID
            or payload.get("dependency_manifest_sha256") != dependency_manifest["manifest_sha256"]
            or int(payload.get("epoch", -1)) != int(record["epoch"])
        ):
            raise ValueError(f"Invalid selection candidate: {record['checkpoint']}")
        core.load_state_dict(payload["model"], strict=True)
        stage_b = evaluate_stage_a(
            accelerator,
            model,
            config,
            variant="w16",
            subset_stage="stage_b_extra",
        )
        stage_a = json.loads(Path(record["stage_a"]).read_text(encoding="utf-8"))
        combined = combine_results(stage_a, stage_b)
        combined_path = validation_dir / f"combined_epoch_{int(record['epoch']):03d}.json"
        if accelerator.is_main_process:
            write_json_atomic(validation_dir / f"stage_b_epoch_{int(record['epoch']):03d}.json", stage_b)
            write_json_atomic(combined_path, combined)
        stage_b_records.append(
            {
                **record,
                "combined_score": float(combined["macro_full_image_nmse_p1_p2_p3"]),
                "combined": str(combined_path),
            }
        )
    selected = min(stage_b_records, key=lambda item: (float(item["combined_score"]), int(item["epoch"])))
    if accelerator.is_main_process:
        selected = {**selected, "checkpoint_sha256": sha256_file(selected["checkpoint"])}
        write_json_atomic(
            output / "selection.json",
            {
                "schema": "rmdm_hvdit_v4_joint_w16_selection_v1",
                "stage_a_top_k": top,
                "combined_records": stage_b_records,
                "selected": selected,
            },
        )
        write_json_atomic(
            output / "status.json",
            {
                "schema": "rmdm_hvdit_v4_joint_w16_status_v1",
                "state": "validation_complete",
                "completed_epoch": completed_epoch,
                "global_step": global_step,
                "stopped_early": stop_early,
                "selected": selected,
            },
        )
    accelerator.wait_for_everyone()
    accelerator.end_training()
    return {"selected": selected, "top_k": stage_b_records}
