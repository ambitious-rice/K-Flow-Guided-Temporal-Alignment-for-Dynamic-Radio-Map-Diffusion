"""Training runner for the isolated unified-input JointRMDM variant."""

from __future__ import annotations

import json
import math
import random
from pathlib import Path
from typing import Any

import numpy as np
import torch
from accelerate import Accelerator, DataLoaderConfiguration
from accelerate.utils import DistributedDataParallelKwargs, set_seed
from torch.utils.data import DataLoader

from rmdm.config import ExperimentConfig
from rmdm.data import SamplingPolicy, WindowDataset
from rmdm.diffusion import DiffusionProcess
from rmdm.evaluation import combine_evaluation_results, evaluate_rates
from rmdm.training.checkpoint import load_checkpoint, save_checkpoint, write_json_atomic
from rmdm.training.engine import train_one_epoch
from rmdm.training.runner import (
    _load_model_state_on_all_ranks,
    _lr_scheduler,
    _trim_top_records,
    _write_jsonl,
)
from rmdm.training.validation import run_stage_a, run_stage_b, should_run_stage_a

from .joint_rmdm import build_unified_joint_rmdm
from .unified_denoiser import UNIFIED_INPUT_CHANNELS


ARCHITECTURE = "rmdm_joint_w16_unified_pixel_input_v1"


def _parameter_diagnostics(model: torch.nn.Module) -> dict[str, float]:
    denoiser = model.joint_denoiser
    blocks = list(denoiser.high_encoder) + list(denoiser.bottleneck) + list(denoiser.high_decoder)

    def rms(tensors: list[torch.Tensor]) -> float:
        squared_sum = 0.0
        count = 0
        for tensor in tensors:
            value = tensor.detach().float()
            squared_sum += float(value.square().sum())
            count += value.numel()
        return math.sqrt(squared_sum / max(count, 1))

    spatial_gate_weights = []
    temporal_gate_weights = []
    mlp_gate_weights = []
    for block in blocks:
        weight = block.modulation[-1].weight
        dim = block.dim
        spatial_gate_weights.append(weight[2 * dim : 3 * dim])
        temporal_gate_weights.append(weight[5 * dim : 6 * dim])
        mlp_gate_weights.append(weight[8 * dim : 9 * dim])
    diagnostics = {
        "output_head_weight_rms": rms([denoiser.output.weight]),
        "spatial_gate_weight_rms": rms(spatial_gate_weights),
        "temporal_gate_weight_rms": rms(temporal_gate_weights),
        "mlp_gate_weight_rms": rms(mlp_gate_weights),
    }
    for index, name in enumerate(UNIFIED_INPUT_CHANNELS):
        diagnostics[f"patch_embed_{name}_weight_rms"] = rms([denoiser.patch_embed.weight[:, index]])
    return diagnostics


def run_training(config: ExperimentConfig) -> dict[str, Any]:
    config.validate()
    ddp_kwargs = DistributedDataParallelKwargs(find_unused_parameters=False)
    accelerator = Accelerator(
        mixed_precision=config.train.mixed_precision,
        gradient_accumulation_steps=config.train.gradient_accumulation_steps,
        dataloader_config=DataLoaderConfiguration(even_batches=False),
        kwargs_handlers=[ddp_kwargs],
    )
    set_seed(config.train.seed, device_specific=True)
    random.seed(config.train.seed + accelerator.process_index)
    np.random.seed(config.train.seed + accelerator.process_index)

    output_dir = Path(config.train.output_dir).expanduser().resolve()
    status_path = output_dir / "training_status.json"
    log_path = output_dir / "train_metrics.jsonl"
    checkpoint_dir = output_dir / "checkpoints"
    if accelerator.is_main_process:
        output_dir.mkdir(parents=True, exist_ok=True)
        checkpoint_dir.mkdir(parents=True, exist_ok=True)
        resolved = config.to_dict()
        resolved["architecture"] = ARCHITECTURE
        resolved["unified_input_channels"] = list(UNIFIED_INPUT_CHANNELS)
        write_json_atomic(output_dir / "resolved_config.json", resolved)
        write_json_atomic(
            status_path,
            {
                "schema": "rmdm_joint_w16_unified_training_status_v1",
                "architecture": ARCHITECTURE,
                "state": "initializing",
                "world_size": accelerator.num_processes,
                "expected_epochs": config.train.epochs,
            },
        )

    dataset = WindowDataset(
        root=config.data.root,
        split="train",
        split_file=config.data.split_file,
        window_size=config.data.window_size,
        seed=config.sampling.seed,
        cache_size=config.data.cache_size,
        tx_heatmap_sigma_px=config.data.tx_heatmap_sigma_px,
    )
    loader = DataLoader(
        dataset,
        batch_size=config.train.per_gpu_batch_size,
        shuffle=True,
        num_workers=config.data.workers,
        pin_memory=True,
        persistent_workers=False,
        drop_last=True,
    )
    model = build_unified_joint_rmdm(config)
    trainable_parameters = [parameter for parameter in model.parameters() if parameter.requires_grad]
    trainable_parameter_count = sum(parameter.numel() for parameter in trainable_parameters)
    total_parameter_count = sum(parameter.numel() for parameter in model.parameters())
    if config.train.expected_trainable_parameters and (
        trainable_parameter_count != config.train.expected_trainable_parameters
    ):
        raise ValueError(
            "Trainable parameter count mismatch: "
            f"expected {config.train.expected_trainable_parameters:,}, got {trainable_parameter_count:,}"
        )
    if config.train.expected_total_parameters and total_parameter_count != config.train.expected_total_parameters:
        raise ValueError(
            "Total parameter count mismatch: "
            f"expected {config.train.expected_total_parameters:,}, got {total_parameter_count:,}"
        )
    optimizer = torch.optim.AdamW(
        trainable_parameters,
        lr=config.train.learning_rate,
        betas=tuple(config.train.betas),
        weight_decay=config.train.weight_decay,
    )
    batches_per_rank = len(dataset) // (accelerator.num_processes * config.train.per_gpu_batch_size)
    if batches_per_rank <= 0:
        raise ValueError("Dataset is too small for the distributed per-GPU batch with drop_last=True")
    updates_per_epoch = math.ceil(batches_per_rank / config.train.gradient_accumulation_steps)
    total_steps = max(updates_per_epoch * config.train.epochs, 1)
    warmup_steps = int(config.train.warmup_steps)
    if config.train.warmup_epochs:
        warmup_steps = int(math.ceil(config.train.warmup_epochs * updates_per_epoch))
    lr_scheduler = _lr_scheduler(optimizer, config, total_steps, warmup_steps)

    start_epoch = 0
    global_step = 0
    best_score = float("inf")
    validations_without_improvement = 0
    top_records: list[dict[str, Any]] = []
    if config.train.resume_from:
        resume_payload = load_checkpoint(config.train.resume_from, model, optimizer)
        lr_scheduler.load_state_dict(resume_payload["lr_scheduler"])
        start_epoch = int(resume_payload["completed_epoch"])
        global_step = int(resume_payload["global_step"])
        extra = resume_payload.get("extra", {})
        if extra.get("architecture", ARCHITECTURE) != ARCHITECTURE:
            raise ValueError(f"Checkpoint is not a {ARCHITECTURE} run")
        best_score = float(extra.get("best_score", best_score))
        validations_without_improvement = int(extra.get("validations_without_improvement", 0))
        top_records = list(extra.get("top_records", []))

    model, optimizer, loader = accelerator.prepare(model, optimizer, loader)
    sampling_policy = SamplingPolicy(config.sampling, split="train")
    diffusion = DiffusionProcess(config.diffusion)
    if accelerator.is_main_process:
        write_json_atomic(
            status_path,
            {
                "schema": "rmdm_joint_w16_unified_training_status_v1",
                "architecture": ARCHITECTURE,
                "state": "running",
                "world_size": accelerator.num_processes,
                "dataset_videos": len(dataset),
                "trainable_parameters": trainable_parameter_count,
                "total_parameters": total_parameter_count,
                "per_gpu_batch_size": config.train.per_gpu_batch_size,
                "effective_global_batch_size": (
                    accelerator.num_processes
                    * config.train.per_gpu_batch_size
                    * config.train.gradient_accumulation_steps
                ),
                "updates_per_epoch": updates_per_epoch,
                "warmup_steps": warmup_steps,
                "start_epoch": start_epoch,
                "global_step": global_step,
                "expected_epochs": config.train.epochs,
            },
        )
        print(
            f"[unified-setup] videos={len(dataset)} world_size={accelerator.num_processes} "
            f"per_gpu_batch={config.train.per_gpu_batch_size} "
            f"effective_global_batch={accelerator.num_processes * config.train.per_gpu_batch_size * config.train.gradient_accumulation_steps} "
            f"updates_per_epoch={updates_per_epoch} total_steps={total_steps} warmup_steps={warmup_steps} "
            f"trainable_params={trainable_parameter_count:,} total_params={total_parameter_count:,}",
            flush=True,
        )

    reached_max_steps = False
    stopped_early = False
    completed_epoch = start_epoch
    for epoch in range(start_epoch, config.train.epochs):
        dataset.set_epoch(epoch)
        if hasattr(loader, "set_epoch"):
            loader.set_epoch(epoch)
        epoch_result = train_one_epoch(
            accelerator,
            model,
            loader,
            optimizer,
            lr_scheduler,
            sampling_policy,
            diffusion,
            config,
            epoch=epoch,
            global_step=global_step,
        )
        global_step = epoch_result.global_step
        completed_epoch = epoch + 1
        epoch_payload = {
            "epoch": completed_epoch,
            "global_step": global_step,
            "loss": epoch_result.mean_loss,
            "p_mean": epoch_result.mean_sampling_rate,
            "batches": epoch_result.batches,
            "elapsed_seconds": epoch_result.elapsed_seconds,
            "lr": float(optimizer.param_groups[0]["lr"]),
            "timestep_buckets": epoch_result.timestep_buckets,
            "parameter_diagnostics": _parameter_diagnostics(accelerator.unwrap_model(model)),
        }
        if accelerator.is_main_process:
            _write_jsonl(log_path, epoch_payload)
            print("[unified-epoch] " + json.dumps(epoch_payload), flush=True)
        reached_max_steps = epoch_result.reached_max_steps

        if not reached_max_steps and should_run_stage_a(config, completed_epoch):
            stage_a = run_stage_a(accelerator, model, config)
            score = float(stage_a["macro_full_image_nmse_p1_p2_p3"])
            result_path = output_dir / "validation" / f"stage_a_epoch_{completed_epoch:03d}.json"
            if accelerator.is_main_process:
                write_json_atomic(result_path, stage_a)
            patience_eligible = completed_epoch > config.evaluation.early_stop_min_epoch
            if score < best_score:
                best_score = score
                validations_without_improvement = 0
            elif patience_eligible:
                validations_without_improvement += 1
            candidate_path = checkpoint_dir / f"top_epoch_{completed_epoch:03d}.pth"
            record = {
                "epoch": completed_epoch,
                "stage_a_score": score,
                "checkpoint": str(candidate_path),
                "stage_a_result": str(result_path),
            }
            candidate_records = top_records + [record]
            qualifies = record in sorted(
                candidate_records,
                key=lambda item: (float(item["stage_a_score"]), int(item["epoch"])),
            )[: config.evaluation.stage_a_top_k]
            if qualifies:
                save_checkpoint(
                    accelerator,
                    candidate_path,
                    model,
                    optimizer,
                    lr_scheduler,
                    config,
                    completed_epoch=completed_epoch,
                    global_step=global_step,
                    extra={"architecture": ARCHITECTURE},
                )
            top_records = _trim_top_records(
                accelerator,
                candidate_records,
                config.evaluation.stage_a_top_k,
            )
            if accelerator.is_main_process:
                print(
                    f"[unified-stage-a] epoch={completed_epoch} score={score:.8g} "
                    f"best={best_score:.8g} patience={validations_without_improvement}/"
                    f"{config.evaluation.patience_validations} eligible={patience_eligible}",
                    flush=True,
                )
            if patience_eligible and validations_without_improvement >= config.evaluation.patience_validations:
                stopped_early = True

        checkpoint_extra = {
            "architecture": ARCHITECTURE,
            "best_score": best_score,
            "validations_without_improvement": validations_without_improvement,
            "top_records": top_records,
        }
        save_checkpoint(
            accelerator,
            output_dir / "last.pth",
            model,
            optimizer,
            lr_scheduler,
            config,
            completed_epoch=completed_epoch,
            global_step=global_step,
            extra=checkpoint_extra,
        )
        if accelerator.is_main_process:
            write_json_atomic(
                status_path,
                {
                    "schema": "rmdm_joint_w16_unified_training_status_v1",
                    "architecture": ARCHITECTURE,
                    "state": "running",
                    "completed_epoch": completed_epoch,
                    "global_step": global_step,
                    "best_score": best_score,
                    "validations_without_improvement": validations_without_improvement,
                    "stopped_early": stopped_early,
                    "reached_max_steps": reached_max_steps,
                },
            )
        if stopped_early or reached_max_steps:
            break

    final_selection = None
    if not reached_max_steps and top_records:
        combined_candidates = []
        for record in top_records:
            _load_model_state_on_all_ranks(accelerator, model, record["checkpoint"])
            stage_b = run_stage_b(accelerator, model, config)
            with Path(record["stage_a_result"]).open("r", encoding="utf-8") as handle:
                stage_a = json.load(handle)
            combined = combine_evaluation_results(stage_a, stage_b)
            combined_path = output_dir / "validation" / f"combined_epoch_{int(record['epoch']):03d}.json"
            if accelerator.is_main_process:
                write_json_atomic(combined_path, combined)
            combined_candidates.append(
                {
                    **record,
                    "stage_b_score": stage_b["macro_full_image_nmse_p1_p2_p3"],
                    "combined_score": combined["macro_full_image_nmse_p1_p2_p3"],
                    "combined_result": str(combined_path),
                }
            )
        selected = min(combined_candidates, key=lambda item: (float(item["combined_score"]), int(item["epoch"])))
        _load_model_state_on_all_ranks(accelerator, model, selected["checkpoint"])
        full_a = evaluate_rates(
            accelerator,
            model,
            config,
            subset_stage="stage_a",
            rates=config.evaluation.full_rates,
            ddim_steps=config.diffusion.ddim_steps,
        )
        full_b = evaluate_rates(
            accelerator,
            model,
            config,
            subset_stage="stage_b_extra",
            rates=config.evaluation.full_rates,
            ddim_steps=config.diffusion.ddim_steps,
        )
        full_curve = combine_evaluation_results(full_a, full_b)
        final_selection = {
            "selected": selected,
            "candidates": combined_candidates,
            "full_validation_curve": full_curve,
        }
        if accelerator.is_main_process:
            write_json_atomic(output_dir / "final_selection.json", final_selection)

    final_state = "smoke_complete" if reached_max_steps else "complete"
    summary = {
        "state": final_state,
        "completed_epoch": completed_epoch,
        "global_step": global_step,
        "stopped_early": stopped_early,
        "reached_max_steps": reached_max_steps,
        "top_records": top_records,
        "final_selection": final_selection,
    }
    if accelerator.is_main_process:
        write_json_atomic(
            status_path,
            {
                "schema": "rmdm_joint_w16_unified_training_status_v1",
                "architecture": ARCHITECTURE,
                **summary,
            },
        )
    accelerator.wait_for_everyone()
    accelerator.end_training()
    return summary
