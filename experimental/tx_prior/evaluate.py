"""Read-only single-GPU DDIM20/50 evaluation of one epsilon checkpoint."""

from __future__ import annotations

import argparse
import json
import os
import subprocess
import time
from pathlib import Path
from typing import Any

import torch
from accelerate import Accelerator
from torch.utils.data import DataLoader

from rmdm.diffusion import DDIMSampler
from rmdm.evaluation.fixed_sparse_protocol import frame_names_by_sample
from rmdm.evaluation.metrics import MetricAccumulator
from rmdm_hvdit_v4_joint.evaluation.evaluator import manifest_video_ids
from .adapter import build_scene_prior_system
from .checkpoint import SCHEMA
from .config import load_config
from .data import deterministic_prior_noise_like, make_prior_training_batch
from .runner import _dataset, resolve_task_root


def positive_int(value: str) -> int:
    number = int(value)
    if number <= 0:
        raise argparse.ArgumentTypeError("must be positive")
    return number


def parse_args(argv: list[str] | None = None) -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--config", default="experimental/tx_prior/remote.yaml")
    parser.add_argument("--repository-root", default=".")
    parser.add_argument("--checkpoint", required=True)
    parser.add_argument("--batch-size", type=positive_int, default=4)
    parser.add_argument("--steps", type=int, nargs="+", choices=(20, 50), default=[20, 50])
    parser.add_argument("--output-dir", default="")
    parser.add_argument("--smoke", action="store_true")
    parser.add_argument("--smoke-frames", type=positive_int, default=4)
    parser.add_argument("--expected-frames", type=int, choices=(2000, 3000), default=3000)
    args = parser.parse_args(argv)
    if len(set(args.steps)) != len(args.steps):
        parser.error("--steps must not repeat")
    return args


def load_model_checkpoint(path: Path, model: Any) -> dict[str, Any]:
    # Training checkpoints contain optimizer/RNG objects: deserialize once on
    # CPU, load only model weights, then release all other tensor payloads.
    payload = torch.load(path, map_location="cpu", weights_only=False)
    if not isinstance(payload, dict) or payload.get("schema") != SCHEMA:
        raise ValueError("unexpected tx_prior checkpoint schema (epsilon required)")
    step = payload.get("global_step")
    if isinstance(step, bool) or not isinstance(step, int) or step < 0:
        raise ValueError("checkpoint requires a non-negative integer global_step")
    model.load_state_dict(payload["model"], strict=True)
    return {"path": str(path), "global_step": step, "schema": payload["schema"],
            "resolved_config": payload.get("resolved_config")}


def result_paths(root: Path, checkpoint: dict[str, Any], steps: list[int], *,
                 smoke: bool) -> list[Path]:
    stage = root / f"step_{checkpoint['global_step']:06d}"
    if smoke:
        stage = stage / "smoke"
    paths = [stage / f"ddim{step:02d}.json" for step in steps]
    for path in paths:
        if path.exists():
            raise FileExistsError(f"refusing to overwrite sampling evaluation: {path}")
    return paths


def write_result(path: Path, result: dict[str, Any]) -> None:
    serialized = json.dumps(result, indent=2, allow_nan=False) + "\n"
    path.parent.mkdir(parents=True, exist_ok=True)
    # Exclusive creation also protects against concurrent evaluators.
    with path.open("x", encoding="utf-8") as handle:
        handle.write(serialized)


@torch.no_grad()
def validate_smoke(accelerator: Any, model: Any, config: Any, frames: int) -> dict[str, Any]:
    video_ids = manifest_video_ids(config.evaluation.subset_manifest, "stage_a")[:1]
    frames = min(frames, config.data.frames_per_video)
    dataset = _dataset(config, split="val", fixed_starts=tuple(range(frames)), video_ids=video_ids)
    loader = accelerator.prepare_data_loader(DataLoader(
        dataset, batch_size=config.evaluation.t1_evaluation_batch_size,
        shuffle=False, num_workers=0, drop_last=False))
    core = accelerator.unwrap_model(model)
    core.eval()
    sampler = DDIMSampler(config.diffusion)
    metrics = MetricAccumulator(device=accelerator.device)
    scored = 0
    for dense in loader:
        prior = make_prior_training_batch(dense)
        target = prior["target"]
        noise = deterministic_prior_noise_like(target, frame_names_by_sample(
            prior, batch_size=target.shape[0], window_size=1), seed=config.sampling.seed)
        prediction = sampler.sample(core, prior, initial_noise=noise,
                                    steps=config.evaluation.ddim_steps)
        metrics.update(prediction, target, prior["building"], prior["vehicle"], prior["sampling_mask"])
        scored += target.shape[0]
    if scored != frames or not video_ids:
        raise RuntimeError(f"smoke scored {scored}, expected {frames}")
    return {"schema": "tx_prior_sampling_smoke_v1", "scored_frames": scored,
            "metrics": metrics.compute(), "raw": metrics.raw()}


@torch.no_grad()
def validate_by_scene(accelerator: Any, model: Any, config: Any) -> dict[str, Any]:
    """Same frame-keyed protocol, retaining additive per-scene statistics."""
    ids = manifest_video_ids(config.evaluation.subset_manifest, "stage_a")
    dataset = _dataset(config, split="val", fixed_starts=tuple(range(config.data.frames_per_video)), video_ids=ids)
    loader = accelerator.prepare_data_loader(DataLoader(dataset,
        batch_size=config.evaluation.t1_evaluation_batch_size, shuffle=False,
        num_workers=min(config.data.workers, 2), drop_last=False))
    core = accelerator.unwrap_model(model)
    core.eval()
    sampler = DDIMSampler(config.diffusion)
    total = MetricAccumulator(device=accelerator.device)
    scenes: dict[str, MetricAccumulator] = {}
    counts: dict[str, int] = {}
    scored = 0
    for dense in loader:
        prior = make_prior_training_batch(dense)
        names = frame_names_by_sample(prior, batch_size=prior["target"].shape[0], window_size=1)
        noise = deterministic_prior_noise_like(prior["target"], names, seed=config.sampling.seed)
        prediction = sampler.sample(core, prior, initial_noise=noise, steps=config.evaluation.ddim_steps)
        fields = (prediction, prior["target"], prior["building"], prior["vehicle"], prior["sampling_mask"])
        total.update(*fields)
        for index, name in enumerate(names):
            scene = name[0].split("/")[0]
            if scene not in scenes:
                scenes[scene] = MetricAccumulator(device=accelerator.device)
            scenes[scene].update(*(value[index:index+1] for value in fields))
            counts[scene] = counts.get(scene, 0) + 1
        scored += prediction.shape[0]
    return {"schema": "tx_prior_validation_v1", "scored_frames": scored,
            "metrics": total.compute(), "raw": total.raw(), "scene_frame_counts": counts,
            "scene_metrics": {scene: values.compute() for scene, values in scenes.items()},
            "scene_raw": {scene: values.raw() for scene, values in scenes.items()}}


def evaluate(args: argparse.Namespace) -> None:
    repository = Path(args.repository_root).expanduser().resolve()
    config_path = Path(args.config).expanduser().resolve()
    config = load_config(config_path, smoke=False)
    config.evaluation.t1_evaluation_batch_size = args.batch_size
    task_root = resolve_task_root(repository, config.pipeline.output_root)
    allowed_output = (task_root / "train/sampling_eval").resolve()
    output = Path(args.output_dir).expanduser().resolve() if args.output_dir else allowed_output
    if not output.is_relative_to(allowed_output):
        raise ValueError("--output-dir must remain under runs/tx_prior/train/sampling_eval")
    if not args.smoke:
        expected = len(manifest_video_ids(config.evaluation.subset_manifest, "stage_a")) * config.data.frames_per_video
        if expected != args.expected_frames:
            raise ValueError(f"formal sampling evaluation requires exactly {args.expected_frames} frames, got {expected}")
    # Direct sampler methods bypass the prepared forward autocast wrapper;
    # training validation therefore uses FP32, which we preserve here.
    accelerator = Accelerator(mixed_precision="no")
    if accelerator.num_processes != 1 or accelerator.device.type != "cuda":
        raise RuntimeError("sampling evaluation requires one CUDA process")
    if torch.cuda.device_count() != 1:
        raise RuntimeError("select exactly one GPU with CUDA_VISIBLE_DEVICES")
    model = build_scene_prior_system(config)
    checkpoint = load_model_checkpoint(Path(args.checkpoint).expanduser().resolve(), model)
    paths = result_paths(output, checkpoint, args.steps, smoke=args.smoke)
    source = {"head": subprocess.check_output(["git", "rev-parse", "HEAD"], cwd=repository, text=True).strip(),
              "dirty": subprocess.check_output(["git", "status", "--porcelain"], cwd=repository, text=True)}
    model = accelerator.prepare(model)
    core = accelerator.unwrap_model(model)
    for steps, path in zip(args.steps, paths):
        config.evaluation.ddim_steps = steps
        calls = scored = last_report = 0

        def progress(_module: Any, inputs: Any, _result: Any) -> None:
            nonlocal calls, scored, last_report
            calls += 1
            if calls % steps == 0:
                scored += inputs[0].shape[0]
                if scored - last_report >= 100 or scored == args.expected_frames:
                    print(f"DDIM{steps}: sampled {scored} frames", flush=True)
                    last_report = scored

        hook = core.denoiser.register_forward_hook(progress)
        torch.cuda.synchronize()
        torch.cuda.reset_peak_memory_stats()
        started = time.perf_counter()
        print(f"Starting DDIM{steps}, checkpoint step {checkpoint['global_step']}, batch {args.batch_size}, smoke={args.smoke}", flush=True)
        try:
            result = (validate_smoke(accelerator, model, config, args.smoke_frames)
                      if args.smoke else validate_by_scene(accelerator, model, config))
            if not args.smoke and result["scored_frames"] != args.expected_frames:
                raise RuntimeError("formal evaluation did not score all expected frames")
            torch.cuda.synchronize()
        finally:
            hook.remove()
        elapsed = time.perf_counter() - started
        write_result(path, {**result, "sampling_eval_schema": "tx_prior_sampling_eval_v1",
            "smoke": args.smoke, "checkpoint": checkpoint, "config_path": str(config_path),
            "resolved_config": config.to_dict(), "ddim": {"steps": steps, "eta": 0.0,
            "seed": config.sampling.seed, "noise": "tx-prior-ddim-noise-v1",
            "validation_manifest": config.evaluation.subset_manifest}, "source": source,
            "evaluation_precision": "fp32", "accelerator_mixed_precision": "no", "batch_size": args.batch_size,
            "elapsed_seconds": elapsed, "peak_memory_bytes": torch.cuda.max_memory_allocated(),
            "device": torch.cuda.get_device_name(), "cuda_visible_devices": os.environ.get("CUDA_VISIBLE_DEVICES")})
        print(f"DDIM{steps} finished in {elapsed:.1f}s: {path}", flush=True)


def main() -> None:
    evaluate(parse_args())


if __name__ == "__main__":
    main()
