"""Evaluate the old scene-only UNet on the current full validation protocol."""

from __future__ import annotations

import argparse
import os
from pathlib import Path
import subprocess
import time

import torch
from torch.utils.data import DataLoader

from rmdm.diffusion import DDIMSampler
from rmdm.evaluation.fixed_sparse_protocol import frame_names_by_sample
from rmdm.evaluation.metrics import MetricAccumulator
from rmdm_hvdit_v4_joint.evaluation.evaluator import manifest_video_ids
from .config import load_config
from .data import deterministic_prior_noise_like, make_prior_training_batch
from .diagnose import DiagnosticFrames, LegacySystem
from .evaluate import write_result
from .runner import _dataset, resolve_task_root


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--config", default="experimental/tx_prior/remote.yaml")
    parser.add_argument("--checkpoint", required=True)
    parser.add_argument("--batch-size", type=int, default=4)
    parser.add_argument("--steps", type=int, default=50)
    parser.add_argument("--smoke-frames", type=int, default=0)
    parser.add_argument("--output", required=True)
    args = parser.parse_args()
    config = load_config(args.config, smoke=False)
    output = Path(args.output).resolve()
    allowed = resolve_task_root(Path.cwd(), config.pipeline.output_root) / "train/legacy_eval"
    if not output.is_relative_to(allowed) or output.exists():
        raise ValueError("output must be new and under train/legacy_eval")
    if args.batch_size <= 0 or not 0 < args.steps <= config.diffusion.train_timesteps:
        raise ValueError("invalid batch or sampling steps")
    if torch.cuda.device_count() != 1:
        raise ValueError("select exactly one visible CUDA GPU")
    ids = manifest_video_ids(config.evaluation.subset_manifest, "stage_a")
    dataset = _dataset(config, split="val", fixed_starts=tuple(range(config.data.frames_per_video)), video_ids=ids)
    expected = len(dataset)
    if expected != 3000:
        raise ValueError(f"formal protocol requires 3000 frames, got {expected}")
    if args.smoke_frames:
        if not 2 <= args.smoke_frames <= expected:
            raise ValueError("smoke frames must be between 2 and 3000")
        expected = args.smoke_frames
    dataset = DiagnosticFrames(dataset, expected, include_legacy=True)
    loader = DataLoader(dataset, batch_size=args.batch_size, shuffle=False, num_workers=0)
    model = LegacySystem(args.checkpoint).cuda().eval()
    sampler = DDIMSampler(config.diffusion)
    metrics = MetricAccumulator(device="cuda")
    scene_metrics = {}
    per_frame = []
    scored = last_report = 0
    torch.cuda.synchronize()
    torch.cuda.reset_peak_memory_stats()
    started = time.perf_counter()
    print(f"Starting old UNet, DDIM{args.steps}, eta=0, {expected} frames, batch={args.batch_size}", flush=True)
    with torch.no_grad():
        for dense in loader:
            prior = make_prior_training_batch({k: v.cuda() if torch.is_tensor(v) else v for k, v in dense.items()})
            names = frame_names_by_sample(prior, batch_size=prior["target"].shape[0], window_size=1)
            noise = deterministic_prior_noise_like(prior["target"], names, seed=config.sampling.seed)
            prediction = sampler.sample(model, prior, initial_noise=noise, steps=args.steps, eta=0)
            fields = [prediction, prior["target"], prior["building"], prior["vehicle"], prior["sampling_mask"]]
            metrics.update(*fields)
            for index, name in enumerate(names):
                scene = name[0].split("/")[0]
                scene_metrics.setdefault(scene, MetricAccumulator(device="cuda"))
                scene_metrics[scene].update(*(field[index:index+1] for field in fields))
                error = (prediction[index].double() - prior["target"][index].double()).square().mean()
                per_frame.append({"frame": name[0], "mse": error.item()})
            scored += prediction.shape[0]
            if scored - last_report >= 100 or scored == expected:
                print(f"Old UNet sampled {scored}/{expected} frames, elapsed={time.perf_counter()-started:.1f}s", flush=True)
                last_report = scored
    if scored != expected:
        raise RuntimeError(f"scored {scored}, expected {expected}")
    torch.cuda.synchronize()
    write_result(output, dict(schema="tx_prior_legacy_transfer_eval_v1", smoke=bool(args.smoke_frames),
        scored_frames=scored, checkpoint=str(Path(args.checkpoint).resolve()),
        condition_encoding="old_native_raw_traffic_div255_not_binary_vehicle",
        metrics=metrics.compute(), raw=metrics.raw(),
        scene_metrics={scene: accumulator.compute() for scene, accumulator in scene_metrics.items()},
        per_frame=per_frame, resolved_config=config.to_dict(), steps=args.steps, eta=0.0,
        seed=config.sampling.seed, noise="tx-prior-ddim-noise-v1",
        precision="fp32", device=torch.cuda.get_device_name(),
        cuda_visible_devices=os.environ.get("CUDA_VISIBLE_DEVICES"),
        elapsed_seconds=time.perf_counter()-started, peak_memory_bytes=torch.cuda.max_memory_allocated(),
        source_head=subprocess.check_output(["git", "rev-parse", "HEAD"], text=True).strip(),
        source_dirty=subprocess.check_output(["git", "status", "--porcelain"], text=True)))
    print(f"Saved old-model evaluation: {output}", flush=True)


if __name__ == "__main__":
    main()
