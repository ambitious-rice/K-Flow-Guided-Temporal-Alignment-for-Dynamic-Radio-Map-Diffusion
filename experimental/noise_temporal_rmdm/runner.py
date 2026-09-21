"""Resumable local T1 trainer; it never opens a remote connection."""

from __future__ import annotations

from datetime import datetime
from pathlib import Path
import subprocess
from typing import Any

import torch
from torch.utils.data import DataLoader, Subset

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

from .checkpoint import load, save
from .config import ExperimentConfig
from .model import build_model, parameter_counts
from .step import training_step
from .validation import validation_video_ids


def output_directory(repository_root: str | Path, *, smoke: bool) -> Path:
    root = Path(repository_root).expanduser().resolve()
    return root / "runs" / "noise_temporal_rmdm" / ("smoke" if smoke else "t1")


def source_metadata(repository_root: Path) -> dict[str, Any]:
    """Record Git state for provenance without enforcing project policy."""

    def git(*args: str) -> str:
        result = subprocess.run(
            ["git", *args], cwd=repository_root, text=True, capture_output=True, check=False
        )
        return result.stdout.strip() if result.returncode == 0 else ""

    return {
        "commit": git("rev-parse", "HEAD") or None,
        "branch": git("branch", "--show-current") or None,
        "dirty": bool(git("status", "--porcelain")),
    }


def run(
    config: ExperimentConfig,
    *,
    config_path: str | Path,
    repository_root: str | Path,
    smoke: bool = False,
    smoke_data_limit: int = 0,
    resume_from: str = "",
) -> None:
    """Run training only; formal DDIM validation is intentionally a separate job."""

    repository_root = Path(repository_root).expanduser().resolve()
    source = source_metadata(repository_root)
    output = output_directory(repository_root, smoke=smoke)
    if not resume_from and output.exists() and any(output.iterdir()):
        raise FileExistsError(f"refusing to overwrite non-empty stage directory: {output}")
    accelerator = make_accelerator(
        mixed_precision=config.train.mixed_precision,
        gradient_accumulation_steps=config.train.gradient_accumulation_steps,
        data_seed=config.train.seed,
    )
    seed_everything(config.train.seed)
    model = build_model(config)
    trainable, total = parameter_counts(model)
    if config.model.expected_trainable_parameters_min and not (
        config.model.expected_trainable_parameters_min <= trainable
        <= config.model.expected_trainable_parameters_max
    ):
        raise ValueError(f"trainable parameter count {trainable:,} violates configured contract")
    # Fail closed before training if the manifest does not provide exactly the
    # requested two-scene validation domain.
    validation_ids = validation_video_ids(
        (repository_root / config.validation.subset_manifest
         if not Path(config.validation.subset_manifest).is_absolute()
         else config.validation.subset_manifest),
        included_scenes=config.validation.included_scenes,
        excluded_scenes=config.validation.excluded_scenes,
    )
    split_file = Path(config.data.split_file)
    if not split_file.is_absolute():
        split_file = repository_root / split_file
    dataset: Any = WindowDataset(
        root=config.data.root,
        split="train",
        split_file=str(split_file),
        window_size=1,
        seed=config.sampling.seed,
        cache_size=config.data.cache_size,
        include_tx=False,
        fixed_starts=tuple(range(config.data.frames_per_video)),
    )
    if smoke:
        if smoke_data_limit <= 0:
            raise ValueError("smoke training requires a positive smoke_data_limit")
        dataset = Subset(dataset, range(min(len(dataset), smoke_data_limit)))
    loader = DataLoader(
        dataset,
        batch_size=config.train.per_gpu_batch_size,
        shuffle=True,
        num_workers=config.data.workers,
        pin_memory=True,
        persistent_workers=config.data.workers > 0,
        drop_last=True,
    )
    optimizer = make_optimizer(
        model,
        learning_rate=config.train.learning_rate,
        betas=config.train.betas,
        epsilon=config.train.epsilon,
        weight_decay=config.train.weight_decay,
    )
    scheduler = cosine_scheduler(
        optimizer,
        total_steps=config.train.max_steps,
        warmup_steps=config.train.warmup_steps,
        base_learning_rate=config.train.learning_rate,
        min_learning_rate=config.train.min_learning_rate,
    )
    global_step = epoch = offset = 0
    if resume_from:
        payload = load(resume_from, model, optimizer, scheduler)
        global_step = int(payload["global_step"])
        epoch = int(payload["epoch"])
        offset = int(payload["microbatches_consumed_in_epoch"])
    model, optimizer, loader = prepare_model_optimizer_loader(accelerator, model, optimizer, loader)
    require_scheduler_global_step(scheduler, global_step)
    sampling = SamplingPolicy(config.sampling, split="train")
    diffusion = DiffusionProcess(config.diffusion)
    checkpoint_path = output / "checkpoints" / "last.pth"
    if accelerator.is_main_process:
        status = {
            "schema": "noise_temporal_rmdm_t1_status_v1",
            "state": "training",
            "started_at": datetime.now().astimezone().isoformat(timespec="seconds"),
            "global_step": global_step,
            "config": str(Path(config_path).resolve()),
            "output": str(output),
            "trainable_parameters": trainable,
            "total_parameters": total,
            "validation_scenes": config.validation.included_scenes,
            "validation_videos": len(validation_ids),
            "excluded_validation_scenes": config.validation.excluded_scenes,
            "source": source,
        }
        write_json_atomic(output / "status.json", status)
        append_jsonl(output / "history.jsonl", {"event": "start_or_resume", **status})
    optimizer.zero_grad(set_to_none=True)
    last_microbatch = offset
    while global_step < config.train.max_steps:
        base_dataset = dataset.dataset if isinstance(dataset, Subset) else dataset
        base_dataset.set_epoch(epoch)
        sampling.set_epoch(epoch)
        if hasattr(loader, "set_epoch"):
            loader.set_epoch(epoch)
        for batch_index, dense_batch in enumerate(loader):
            if batch_index < offset:
                continue
            last_microbatch = batch_index + 1
            with accelerator.accumulate(model):
                with accelerator.autocast():
                    result = training_step(model, dense_batch, sampling, diffusion, config, epoch=epoch)
                accelerator.backward(result.loss)
                if accelerator.sync_gradients:
                    accelerator.clip_grad_norm_(model.parameters(), config.train.gradient_clip_norm)
                optimizer.step()
                step_scheduler_on_global_update(accelerator, scheduler)
                optimizer.zero_grad(set_to_none=True)
            if not accelerator.sync_gradients:
                continue
            global_step += 1
            require_scheduler_global_step(scheduler, global_step)
            if global_step <= 2 or global_step % config.train.log_every_steps == 0:
                values = accelerator.reduce(torch.stack((
                    result.loss.detach().float(), result.diffusion_loss.detach().float(),
                    result.calibration_loss.detach().float(), result.pinn_loss.detach().float(),
                    result.measurement_sigma_mean.detach().float(), result.sampling_rate_mean.detach().float(),
                )), reduction="mean")
                if accelerator.is_main_process:
                    append_jsonl(output / "train.jsonl", {
                        "global_step": global_step, "epoch": epoch,
                        "loss": float(values[0]), "diffusion_loss": float(values[1]),
                        "calibration_loss": float(values[2]), "pinn_loss": float(values[3]),
                        "measurement_sigma_mean": float(values[4]), "sampling_rate_mean": float(values[5]),
                        "learning_rate": float(optimizer.param_groups[0]["lr"]),
                    })
            if global_step % config.train.checkpoint_every_steps == 0 or global_step == config.train.max_steps:
                save(
                    accelerator, checkpoint_path, model, optimizer, scheduler, config,
                    global_step=global_step, epoch=epoch,
                    microbatches_consumed_in_epoch=last_microbatch,
                    source_provenance=source,
                    extra={"validation_pending": global_step % config.validation.every_steps == 0},
                )
                if global_step % config.validation.every_steps == 0:
                    save(
                        accelerator, output / "checkpoints" / f"step_{global_step:06d}.pth",
                        model, optimizer, scheduler, config,
                        global_step=global_step, epoch=epoch,
                        microbatches_consumed_in_epoch=last_microbatch,
                        source_provenance=source,
                        extra={"validation_pending": True},
                    )
            if global_step >= config.train.max_steps:
                break
        epoch += 1
        offset = 0
    if accelerator.is_main_process:
        completed = {
            "schema": "noise_temporal_rmdm_t1_status_v1", "state": "complete",
            "global_step": global_step,
            "completed_at": datetime.now().astimezone().isoformat(timespec="seconds"),
            "checkpoint": str(checkpoint_path), "trainable_parameters": trainable,
            "total_parameters": total,
            "source": source,
        }
        write_json_atomic(output / "status.json", completed)
        append_jsonl(output / "history.jsonl", {"event": "complete", **completed})
    accelerator.wait_for_everyone()
    accelerator.end_training()
