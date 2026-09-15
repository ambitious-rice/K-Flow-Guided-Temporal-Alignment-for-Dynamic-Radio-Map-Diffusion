"""Real, resumable W1 training loop for the observation-free TX prior."""

from __future__ import annotations

import json
import time
from datetime import datetime
from pathlib import Path
from typing import Any

import torch
import torch.distributed as dist
from torch.utils.data import DataLoader, Subset

from rmdm.data import WindowDataset
from rmdm.diffusion import DDIMSampler, DiffusionProcess
from rmdm.evaluation.fixed_sparse_protocol import frame_names_by_sample
from rmdm.evaluation.metrics import MetricAccumulator
from rmdm_hvdit_v4_joint.evaluation.evaluator import manifest_video_ids
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
    write_json_atomic,
)
from rmdm_hvdit_v4_x0.training.step import training_step

from .adapter import build_scene_prior_system
from .checkpoint import load, save
from .data import deterministic_prior_noise_like, make_prior_training_batch
from .packed import PackedFrameReader


def validation_due(train: Any, step: int) -> bool:
    return step == train.max_steps or (
        step >= train.validation_first_step
        and (step - train.validation_first_step) % train.validation_every_steps == 0
    )


def initial_early_stop_state(payload: dict[str, Any] | None = None) -> dict[str, Any]:
    """Read early-stop progress, safely accepting the original checkpoints."""
    saved = (payload or {}).get("early_stop")
    if not isinstance(saved, dict):
        return {"best_score": float("inf"), "best_step": 0, "stale_validations": 0}
    return {
        "best_score": float(saved.get("best_score", float("inf"))),
        "best_step": int(saved.get("best_step", 0)),
        "stale_validations": int(saved.get("stale_validations", 0)),
    }


def update_early_stop_state(
    state: dict[str, Any], *, score: float, step: int, early_stop_min_step: int
) -> bool:
    improved = score < float(state["best_score"])
    if improved:
        state.update(best_score=float(score), best_step=int(step), stale_validations=0)
    elif step >= early_stop_min_step:
        state["stale_validations"] = int(state["stale_validations"]) + 1
    else:
        state["stale_validations"] = 0
    return improved


def validation_result_path(output: Path, step: int) -> Path:
    return output / "validation" / f"step_{step:06d}.json"


def write_step_validation(path: Path, result: dict[str, Any], step: int) -> None:
    """Write a step result without clobbering a foreign-step record."""
    if path.exists():
        try:
            existing = json.loads(path.read_text(encoding="utf-8"))
        except (OSError, json.JSONDecodeError) as error:
            raise RuntimeError(f"existing validation result is unreadable: {path}") from error
        if existing.get("global_step") != step:
            raise RuntimeError(f"validation result step mismatch at {path}")
    write_json_atomic(path, {**result, "global_step": step})


class _PriorBatchPolicy:
    """Training-step compatibility adapter; it never samples locations."""

    def set_epoch(self, epoch: int) -> None:
        self.epoch = int(epoch)

    def __call__(self, dense_batch: dict[str, Any]) -> dict[str, Any]:
        return make_prior_training_batch(dense_batch)


def _distributed_max(accelerator: Any, value: torch.Tensor) -> torch.Tensor:
    result = value.clone()
    if accelerator.num_processes > 1:
        dist.all_reduce(result, op=dist.ReduceOp.MAX)
    return result


def allow_incomplete_restart(output: Path) -> tuple[bool, str]:
    """Validate the only checkpoint-free in-place restart state."""

    if not output.exists() or not any(output.iterdir()):
        return False, "stage output is empty; start normally"
    if (output / "checkpoints/last.pth").exists():
        return False, "stage has a checkpoint; use --resume-from"
    try:
        status = json.loads((output / "status.json").read_text(encoding="utf-8"))
    except (OSError, json.JSONDecodeError) as error:
        return False, f"status.json is not readable: {error}"
    if status.get("global_step") != 0:
        return False, "checkpoint-free restart requires global_step == 0"
    if status.get("state") not in {"training", "failed"}:
        return False, "checkpoint-free restart requires training or failed state"
    return True, "eligible checkpoint-free step-0 restart"


def _dataset(
    config: Any,
    *,
    split: str,
    fixed_starts: tuple[int, ...] | None = None,
    video_ids: list[str] | None = None,
    packed_root: str = "",
) -> WindowDataset:
    reader = None
    if packed_root:
        reader = PackedFrameReader(
            packed_root,
            source_root=config.data.root,
            split_file=config.data.split_file,
            tx_heatmap_sigma_px=config.data.tx_heatmap_sigma_px,
            split=split,
            video_ids=set(video_ids) if video_ids is not None else None,
        )
    return WindowDataset(
        root=config.data.root if reader is None else None,
        split=split,
        split_file=config.data.split_file,
        window_size=1,
        seed=config.sampling.seed,
        cache_size=config.data.cache_size,
        tx_heatmap_sigma_px=config.data.tx_heatmap_sigma_px,
        fixed_starts=fixed_starts,
        video_ids=video_ids,
        reader=reader,
    )


@torch.no_grad()
def validate_prior(accelerator: Any, model: Any, config: Any) -> dict[str, Any]:
    video_ids = manifest_video_ids(config.evaluation.subset_manifest, "stage_a")
    dataset = _dataset(config, split="val", fixed_starts=tuple(range(config.data.frames_per_video)),
                       video_ids=video_ids)
    loader = DataLoader(dataset, batch_size=config.evaluation.t1_evaluation_batch_size,
                        shuffle=False, num_workers=min(config.data.workers, 2), drop_last=False)
    original_even_batches = bool(accelerator.even_batches)
    try:
        accelerator.even_batches = False
        loader = accelerator.prepare_data_loader(loader)
    finally:
        accelerator.even_batches = original_even_batches
    core = accelerator.unwrap_model(model)
    core.eval()
    sampler = DDIMSampler(config.diffusion)
    metrics = MetricAccumulator(device=accelerator.device)
    scored_frames = torch.zeros((), dtype=torch.long, device=accelerator.device)
    for dense in loader:
        prior = make_prior_training_batch(dense)
        target = prior["target"]
        noise = deterministic_prior_noise_like(
            target, frame_names_by_sample(prior, batch_size=target.shape[0], window_size=1),
            seed=config.sampling.seed,
        )
        prediction = sampler.sample(core, prior, initial_noise=noise, steps=config.evaluation.ddim_steps)
        metrics.update(prediction, target, prior["building"], prior["vehicle"], prior["sampling_mask"])
        scored_frames += target.shape[0] * target.shape[1]
    sums = accelerator.reduce(metrics.sums, reduction="sum")
    scored_frames = accelerator.reduce(scored_frames, reduction="sum")
    expected_frames = len(video_ids) * config.data.frames_per_video
    if int(scored_frames.item()) != expected_frames:
        raise RuntimeError(f"prior validation scored {int(scored_frames.item())}, expected {expected_frames}")
    metrics.sums = sums
    result = {"schema": "tx_prior_validation_v1", "scored_frames": expected_frames,
              "metrics": metrics.compute(), "raw": metrics.raw()}
    core.train()
    return result


def run(config: Any, *, config_path: Path, repository_root: Path, resume_from: str = "",
        smoke: bool = False, smoke_limit: int = 0,
        restart_incomplete: bool = False, packed_root: str = "") -> None:
    root = repository_root.resolve()
    task_root = (root / config.pipeline.output_root).resolve()
    if not task_root.is_relative_to(root) or task_root != (root / "runs/tx_prior").resolve():
        raise ValueError("tx_prior task root must be exactly runs/tx_prior")
    output = task_root / ("smoke" if smoke else "train")
    if restart_incomplete and resume_from:
        raise ValueError("restart_incomplete cannot be combined with resume_from")
    if resume_from:
        resolved_resume = Path(resume_from).expanduser().resolve()
        if resolved_resume.parent != (output / "checkpoints").resolve():
            raise ValueError("resume checkpoint must belong to the selected tx_prior stage")
    require_visible_physical_gpus(list(config.pipeline.allowed_physical_gpus))
    train = config.t1_train
    if restart_incomplete:
        allowed, reason = allow_incomplete_restart(output)
        if not allowed:
            raise RuntimeError(f"cannot restart incomplete tx_prior stage: {reason}")
    elif not resume_from and output.exists() and any(output.iterdir()):
        raise FileExistsError(
            f"refusing to overwrite existing tx_prior output {output}; use --resume-from "
            "or the narrowly scoped --restart-incomplete"
        )
    accelerator = make_accelerator(mixed_precision=train.mixed_precision,
        gradient_accumulation_steps=train.gradient_accumulation_steps, data_seed=train.seed)
    if accelerator.num_processes != len(config.pipeline.allowed_physical_gpus):
        raise RuntimeError("DDP world size does not match configured GPUs")
    seed_everything(train.seed)
    model = build_scene_prior_system(config)
    dataset: Any = _dataset(
        config,
        split="train",
        fixed_starts=tuple(range(config.data.frames_per_video)),
        packed_root=packed_root,
    )
    if smoke_limit:
        dataset = Subset(dataset, range(min(smoke_limit, len(dataset))))
    loader = DataLoader(dataset, batch_size=train.per_gpu_batch_size, shuffle=True,
        num_workers=config.data.workers, pin_memory=True, drop_last=True,
        persistent_workers=config.data.workers > 0)
    optimizer = make_optimizer(model, learning_rate=train.learning_rate, betas=train.betas,
        epsilon=train.epsilon, weight_decay=train.weight_decay)
    scheduler = cosine_scheduler(optimizer, total_steps=train.lr_schedule_steps,
        warmup_steps=train.warmup_steps, base_learning_rate=train.learning_rate,
        min_learning_rate=train.min_learning_rate)
    global_step = epoch = offset = 0
    payload: dict[str, Any] | None = None
    if resume_from:
        payload = load(resume_from, model, optimizer, scheduler)
        global_step, epoch, offset = (int(payload[key]) for key in
                                      ("global_step", "epoch", "microbatches_consumed_in_epoch"))
    early_stop = initial_early_stop_state(payload)
    model, optimizer, loader = prepare_model_optimizer_loader(accelerator, model, optimizer, loader)
    require_scheduler_global_step(scheduler, global_step)
    diffusion = DiffusionProcess(config.diffusion)
    policy = _PriorBatchPolicy()
    stage = "smoke" if smoke else "train"
    checkpoint_path = output / "checkpoints/last.pth"
    best_path = output / "checkpoints/best.pth"
    if accelerator.is_main_process:
        status_context = {
            "stage": stage,
            "physical_gpus": list(config.pipeline.allowed_physical_gpus),
            "world_size": accelerator.num_processes,
            "per_gpu_batch_size": train.per_gpu_batch_size,
            "gradient_accumulation_steps": train.gradient_accumulation_steps,
            "effective_global_batch_size": train.effective_global_batch_size,
            "checkpoint": str(checkpoint_path),
            "data_backend": "packed" if packed_root else "legacy",
            "packed_root": packed_root,
            "validation_metric": "full_image.nmse",
        }
        write_json_atomic(
            output / "status.json",
            {
                "state": "training",
                "global_step": global_step,
                "config": str(config_path),
                "started_at": datetime.now().astimezone().isoformat(),
                "observation_condition": False,
                **status_context,
            },
        )
        append_jsonl(
            output / "history.jsonl",
            {
                "event": "start_or_resume",
                "global_step": global_step,
                "resume_from": resume_from,
                "restart_incomplete": restart_incomplete,
                **status_context,
            },
        )
    optimizer.zero_grad(set_to_none=True)
    if torch.cuda.is_available():
        torch.cuda.synchronize()
    last_time = time.perf_counter()
    last_timed_step = global_step
    data_wait_seconds = 0.0
    fetch_started = time.perf_counter()
    last_microbatches_consumed = offset

    def complete_validation(*, reuse_existing: bool = False) -> None:
        nonlocal last_time, last_timed_step, data_wait_seconds, fetch_started
        result_path = validation_result_path(output, global_step)
        validation = None
        if reuse_existing and result_path.exists():
            validation = json.loads(result_path.read_text(encoding="utf-8"))
            if validation.get("global_step") != global_step:
                raise RuntimeError(f"validation result step mismatch at {result_path}")
        if validation is None:
            validation = validate_prior(accelerator, model, config)
        score = float(validation["metrics"]["full_image"]["nmse"])
        improved = update_early_stop_state(
            early_stop,
            score=score,
            step=global_step,
            early_stop_min_step=train.early_stop_min_step,
        )
        if accelerator.is_main_process:
            write_step_validation(result_path, validation, global_step)
            write_json_atomic(
                output / "validation/prior.json", {**validation, "global_step": global_step}
            )
        if improved:
            save(accelerator, best_path, model, optimizer, scheduler, config,
                global_step=global_step, epoch=epoch,
                microbatches_consumed_in_epoch=last_microbatches_consumed,
                early_stop=early_stop, validation_pending=False)
        save(accelerator, checkpoint_path, model, optimizer, scheduler, config,
            global_step=global_step, epoch=epoch,
            microbatches_consumed_in_epoch=last_microbatches_consumed,
            early_stop=early_stop, validation_pending=False)
        if accelerator.is_main_process:
            append_jsonl(output / "history.jsonl", {
                "event": "validation", "global_step": global_step, "score": score,
                "metric": "full_image.nmse", **early_stop,
            })
        # Validation and checkpoint I/O are not training throughput. Start the
        # next logging window only after every rank has completed both.
        accelerator.wait_for_everyone()
        if torch.cuda.is_available():
            torch.cuda.synchronize()
        last_time = time.perf_counter()
        last_timed_step = global_step
        data_wait_seconds = 0.0
        fetch_started = last_time

    if (
        not smoke
        and validation_due(train, global_step)
        and (
            payload is None
            or "early_stop" not in payload
            or bool(payload.get("validation_pending", False))
        )
    ):
        save(accelerator, checkpoint_path, model, optimizer, scheduler, config,
            global_step=global_step, epoch=epoch,
            microbatches_consumed_in_epoch=last_microbatches_consumed,
            early_stop=early_stop, validation_pending=True)
        complete_validation(reuse_existing=True)

    stop_early = (
        global_step >= train.early_stop_min_step
        and early_stop["stale_validations"] >= train.patience_validations
    )
    while global_step < train.max_steps and not stop_early:
        epoch_dataset = dataset.dataset if isinstance(dataset, Subset) else dataset
        epoch_dataset.set_epoch(epoch)
        if hasattr(loader, "set_epoch"):
            loader.set_epoch(epoch)
        policy.set_epoch(epoch)
        for batch_index, dense in enumerate(loader):
            data_wait_seconds += time.perf_counter() - fetch_started
            if batch_index < offset:
                fetch_started = time.perf_counter()
                continue
            last_microbatches_consumed = batch_index + 1
            with accelerator.accumulate(model):
                with accelerator.autocast():
                    result = training_step(model, dense, policy, diffusion, training_seed=train.seed,
                        epoch=epoch, pinn_k=config.stage1.pinn_k, pinn_weight=config.stage1.pinn_weight,
                        use_tx_source_supervision=True, observation_alignment_weight=0.0)
                accelerator.backward(result.loss)
                if accelerator.sync_gradients:
                    accelerator.clip_grad_norm_(model.parameters(), train.gradient_clip_norm)
                optimizer.step()
                step_scheduler_on_global_update(accelerator, scheduler)
                optimizer.zero_grad(set_to_none=True)
            if not accelerator.sync_gradients:
                fetch_started = time.perf_counter()
                continue
            global_step += 1
            should_log = global_step % train.log_every_steps == 0 or global_step <= 2
            if should_log:
                if torch.cuda.is_available():
                    torch.cuda.synchronize()
                elapsed = time.perf_counter() - last_time
                last_time = time.perf_counter()
                timed_steps = global_step - last_timed_step
                last_timed_step = global_step
                losses = torch.stack(
                    (
                        result.loss.detach().float(),
                        result.clean_data_loss.detach().float(),
                        result.calibration_loss.detach().float(),
                    )
                )
                reduced_losses = accelerator.reduce(losses, reduction="mean")
                slowest = _distributed_max(accelerator, torch.tensor(elapsed, device=accelerator.device))
                max_data_wait = _distributed_max(
                    accelerator,
                    torch.tensor(data_wait_seconds, device=accelerator.device),
                )
                peak_bytes = (
                    float(torch.cuda.max_memory_allocated())
                    if torch.cuda.is_available()
                    else 0.0
                )
                peak_local = torch.tensor(peak_bytes, device=accelerator.device)
                peak = _distributed_max(accelerator, peak_local)
                if accelerator.is_main_process:
                    append_jsonl(output / "train.jsonl", {"global_step": global_step,
                        "loss": float(reduced_losses[0]), "clean_data_loss": float(reduced_losses[1]),
                        "calibration_loss": float(reduced_losses[2]), "step_seconds": float(slowest) / timed_steps,
                        "global_samples_per_second": (
                            train.effective_global_batch_size
                            * timed_steps
                            / max(float(slowest), 1e-9)
                        ),
                        "peak_memory_bytes": int(peak.item()), "learning_rate": optimizer.param_groups[0]["lr"]})
                    append_jsonl(
                        output / "loader.jsonl",
                        {
                            "global_step": global_step,
                            "data_wait_seconds": float(max_data_wait),
                            "data_wait_fraction": float(max_data_wait)
                            / max(float(slowest), 1.0e-9),
                        },
                    )
                data_wait_seconds = 0.0
            if not smoke and validation_due(train, global_step):
                save(accelerator, checkpoint_path, model, optimizer, scheduler, config,
                    global_step=global_step, epoch=epoch,
                    microbatches_consumed_in_epoch=last_microbatches_consumed,
                    early_stop=early_stop, validation_pending=True)
                complete_validation()
                stop_early = (
                    global_step >= train.early_stop_min_step
                    and early_stop["stale_validations"] >= train.patience_validations
                )
            elif global_step % train.checkpoint_every_steps == 0 or global_step == train.max_steps:
                save(accelerator, checkpoint_path, model, optimizer, scheduler, config,
                    global_step=global_step, epoch=epoch,
                    microbatches_consumed_in_epoch=last_microbatches_consumed,
                    early_stop=early_stop, validation_pending=False)
            if global_step >= train.max_steps or stop_early:
                break
            fetch_started = time.perf_counter()
        epoch += 1
        offset = 0
    final_state = "early_stopped" if stop_early and global_step < train.max_steps else "complete"
    if accelerator.is_main_process:
        completed_at = datetime.now().astimezone().isoformat()
        append_jsonl(
            output / "history.jsonl",
            {
                "event": final_state,
                "global_step": global_step,
                "stop_step": global_step,
                **early_stop,
                "completed_at": completed_at,
                **status_context,
            },
        )
        write_json_atomic(
            output / "status.json",
            {
                "state": final_state,
                "global_step": global_step,
                "stop_step": global_step,
                "completed_at": completed_at,
                **early_stop,
                **status_context,
            },
        )
    accelerator.wait_for_everyone()
    accelerator.end_training()
