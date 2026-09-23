"""One trainer for both prediction targets and both window lengths."""
import argparse
from copy import deepcopy
from datetime import datetime
import os
from pathlib import Path
import random
import time

import numpy as np
import torch
from torch.utils.data import DataLoader
from accelerate.utils import gather_object

from experimental.noise_temporal_rmdm.runner import source_metadata
from rmdm.data import SamplingPolicy
from rmdm.diffusion import DiffusionProcess
from rmdm_hvdit_v4_joint.training.engine import (
    make_accelerator, make_optimizer, cosine_scheduler, append_jsonl, write_json_atomic,
)
from .config import load_config
from .data import TrainingDataset
from .evaluate import evaluate, summarize
from .model import NoiseHVDiT, initialize_w16
from .step import training_step


def atomic_save(payload, path):
    path = Path(path)
    path.parent.mkdir(parents=True, exist_ok=True)
    temporary = path.with_suffix(".tmp")
    torch.save(payload, temporary)
    os.replace(temporary, path)


def update_ema(ema, model, decay):
    with torch.no_grad():
        torch._foreach_lerp_(list(ema.parameters()), list(model.parameters()), 1-decay)
        for destination, source in zip(ema.buffers(), model.buffers()):
            destination.copy_(source)


def rng_state():
    return dict(torch=torch.get_rng_state(), cuda=torch.cuda.get_rng_state(),
                numpy=np.random.get_state(), python=random.getstate())


def restore_rng(state):
    torch.set_rng_state(state["torch"])
    torch.cuda.set_rng_state(state["cuda"])
    np.random.set_state(state["numpy"])
    random.setstate(state["python"])


def validation_progress(score, best, bad_count, patience):
    if score < best:
        return score, 0, False
    bad_count += 1
    return best, bad_count, bad_count >= patience


def run(config, *, resume="", initialize="", stop_after=0, skip_validation=False):
    accelerator = make_accelerator(mixed_precision=config.train.mixed_precision,
        gradient_accumulation_steps=config.train.gradient_accumulation_steps, data_seed=config.train.seed)
    # Shared initialization; independent dropout streams begin after DDP setup.
    torch.manual_seed(config.train.seed)
    np.random.seed(config.train.seed)
    random.seed(config.train.seed)
    model = NoiseHVDiT(config)
    if initialize:
        initial = torch.load(initialize, map_location="cpu", weights_only=False)
        initialize_w16(model, initial)
        del initial
    optimizer = make_optimizer(model, learning_rate=config.train.learning_rate, betas=config.train.betas,
                               epsilon=config.train.epsilon, weight_decay=config.train.weight_decay)
    scheduler = cosine_scheduler(optimizer, total_steps=config.train.max_steps,
        warmup_steps=config.train.warmup_steps, base_learning_rate=config.train.learning_rate,
        min_learning_rate=config.train.min_learning_rate)
    dataset = TrainingDataset(config)
    loader = DataLoader(dataset, batch_size=config.train.per_gpu_batch_size, shuffle=True,
        num_workers=config.data.workers, pin_memory=True, drop_last=True,
        persistent_workers=config.data.workers > 0 and config.data.window_size == 1)
    step = epoch = offset = 0
    best_score, bad_validations = float("inf"), 0
    early_stop = False
    payload = None
    if resume:
        payload = torch.load(resume, map_location="cpu", weights_only=False)
        model.load_state_dict(payload["model"])
        optimizer.load_state_dict(payload["optimizer"])
        scheduler.load_state_dict(payload["scheduler"])
        step, epoch, offset = payload["step"], payload["epoch"], payload["offset"]
        best_score = payload.get("best_score", float("inf"))
        bad_validations = payload.get("bad_validations", 0)
    model, optimizer, loader = accelerator.prepare(model, optimizer, loader)
    unwrapped = accelerator.unwrap_model(model)
    ema = deepcopy(unwrapped).eval().requires_grad_(False)
    torch.manual_seed(config.train.seed + accelerator.process_index)
    torch.cuda.manual_seed(config.train.seed + accelerator.process_index)
    saved_rng = None
    if payload:
        ema.load_state_dict(payload["ema"])
        saved_rng = payload["rng"][accelerator.process_index]
        del payload
    output = Path(config.output)
    source = source_metadata(Path.cwd())
    status = dict(state="training", source=source, config=config.to_dict(),
                  started_at=datetime.now().astimezone().isoformat(),
                  parameters=sum(p.numel() for p in unwrapped.parameters()),
                  global_batch=config.train.per_gpu_batch_size*accelerator.num_processes*config.train.gradient_accumulation_steps)
    if accelerator.is_main_process:
        output.mkdir(parents=True, exist_ok=True)
        write_json_atomic(output/"status.json", {**status, "step": step})
    sampling, diffusion = SamplingPolicy(config.sampling, split="train"), DiffusionProcess(config.diffusion)
    optimizer.zero_grad(set_to_none=True)
    sums = None
    started = time.monotonic()
    stop = min(stop_after or config.train.max_steps, config.train.max_steps)
    while step < stop and not early_stop:
        dataset.set_epoch(epoch)
        sampling.set_epoch(epoch)
        loader.set_epoch(epoch)
        active_loader = accelerator.skip_first_batches(loader, offset) if offset else loader
        iterator = iter(active_loader)
        # Creating a DataLoader iterator consumes RNG; resume from the state of
        # the next model forward, not the state before iterator construction.
        if saved_rng:
            restore_rng(saved_rng)
            saved_rng = None
        for batch_index, dense in enumerate(iterator, start=offset):
            with accelerator.accumulate(model):
                with accelerator.autocast():
                    loss, metrics = training_step(model, dense, sampling, diffusion, config, epoch)
                if not torch.isfinite(loss):
                    raise FloatingPointError(f"Nonfinite training loss at step {step}")
                accelerator.backward(loss)
                values = torch.stack(list(metrics.values())).detach()
                sums = values if sums is None else sums + values
                if accelerator.sync_gradients:
                    grad_norm = accelerator.clip_grad_norm_(model.parameters(), config.train.gradient_clip_norm)
                optimizer.step()
                if accelerator.sync_gradients:
                    scheduler.step()
                    update_ema(ema, unwrapped, config.train.ema_decay)
                optimizer.zero_grad(set_to_none=True)
            if not accelerator.sync_gradients:
                continue
            step += 1
            reduced = accelerator.reduce(sums/config.train.gradient_accumulation_steps, reduction="mean")
            sums = None
            if accelerator.is_main_process and (step <= 2 or step % config.train.log_every_steps == 0):
                log = {k: float(v) for k, v in zip(metrics, reduced)}
                log.update(step=step, epoch=epoch, grad_norm=float(grad_norm), lr=scheduler.get_last_lr()[0],
                           elapsed_seconds=time.monotonic()-started,
                           peak_memory_gib=torch.cuda.max_memory_allocated()/1024**3)
                append_jsonl(output/"train.jsonl", log)
                write_json_atomic(output/"status.json", {**status, "step": step, "metrics": log})
                print(log, flush=True)
            checkpoint_due = step % config.train.checkpoint_every_steps == 0 or step == stop
            validation_due = not skip_validation and (step % config.evaluation.every_steps == 0 or step == config.train.max_steps)
            if validation_due:
                # Evaluation must neither advance training RNG nor depend on
                # rank-local BatchNorm running statistics.
                training_rng = rng_state()
                for candidate in (unwrapped, ema):
                    if accelerator.num_processes > 1:
                        for buffer in candidate.buffers():
                            torch.distributed.broadcast(buffer, 0)
                scores = []
                for name, candidate in (("model", unwrapped), ("ema", ema)):
                    rows = evaluate(candidate, config, rank=accelerator.process_index,
                                    world_size=accelerator.num_processes)
                    rows = gather_object(rows)
                    summary = summarize(rows)
                    scores.append(summary["all"]["unobserved_mse"])
                    if accelerator.is_main_process:
                        report = dict(step=step, weights=name, summary=summary, rows=rows)
                        write_json_atomic(output/"validation"/f"step_{step:06d}_{name}.json", report)
                        print({"validation": step, "weights": name, **report["summary"]}, flush=True)
                model.train()
                restore_rng(training_rng)
                score = min(scores)
                best_score, bad_validations, early_stop = validation_progress(
                    score, best_score, bad_validations, config.evaluation.patience)
                if bad_validations == 0:
                    if accelerator.is_main_process:
                        atomic_save(dict(model=unwrapped.state_dict(), ema=ema.state_dict(), step=step,
                                         config=config.to_dict(), source=source), output/"checkpoints"/"best.pth")
                if accelerator.is_main_process:
                    append_jsonl(output/"progress.jsonl", dict(step=step, score=score, best_score=best_score,
                                 bad_validations=bad_validations, early_stop=early_stop))
            if checkpoint_due or validation_due:
                states = gather_object([rng_state()])
                if accelerator.is_main_process:
                    weights = dict(model=unwrapped.state_dict(), ema=ema.state_dict(), step=step,
                                   config=config.to_dict(), source=source)
                    if validation_due:
                        atomic_save(weights, output/"checkpoints"/f"step_{step:06d}.pth")
                    atomic_save({**weights, "optimizer": optimizer.state_dict(), "scheduler": scheduler.state_dict(),
                                 "epoch": epoch, "offset": batch_index+1, "rng": states,
                                 "best_score": best_score, "bad_validations": bad_validations}, output/"checkpoints"/"last.pth")
                accelerator.wait_for_everyone()
            if step >= stop or early_stop:
                break
        epoch += 1
        offset = 0
    if accelerator.is_main_process:
        write_json_atomic(output/"status.json", {**status, "step": step,
            "state": "complete" if step == config.train.max_steps or early_stop else "paused",
            "reason": "validation_patience" if early_stop else "max_steps" if step == config.train.max_steps else "stop_after",
            "best_score": best_score if best_score < float("inf") else None,
            "bad_validations": bad_validations,
            "ended_at": datetime.now().astimezone().isoformat()})
    accelerator.wait_for_everyone()
    accelerator.end_training()


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--config", required=True)
    parser.add_argument("--resume", default="")
    parser.add_argument("--initialize", default="")
    parser.add_argument("--stop-after", type=int, default=0)
    parser.add_argument("--skip-validation", action="store_true")
    parser.add_argument("--output")
    args = parser.parse_args()
    config = load_config(args.config)
    if args.output:
        config.output = args.output
    run(config, resume=args.resume, initialize=args.initialize, stop_after=args.stop_after,
        skip_validation=args.skip_validation)


if __name__ == "__main__":
    main()
