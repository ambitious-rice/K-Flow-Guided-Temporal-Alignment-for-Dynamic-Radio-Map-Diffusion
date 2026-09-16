"""Paired old x0 DiT observation ablation on current validation frames."""

from __future__ import annotations

import argparse
from collections import Counter
from pathlib import Path
import subprocess
import time

import torch
from torch.utils.data import DataLoader

from rmdm.diffusion import DDIMSampler
from rmdm.evaluation.fixed_sparse_protocol import (
    apply_fixed_sparse_observations, deterministic_frame_noise_like, frame_names_by_sample,
)
from rmdm.evaluation.metrics import MetricAccumulator
from rmdm_hvdit_v4_joint.config import ExperimentConfig, _from_mapping
from rmdm_hvdit_v4_joint.evaluation.evaluator import manifest_video_ids
from rmdm_hvdit_v4_x0.model import build_t1_system
from rmdm_hvdit_v4_x0_continue import ARCHITECTURE_ID
from .adapter import build_scene_prior_system
from .config import load_config
from .data import deterministic_prior_noise_like, make_prior_training_batch
from .diagnose import DiagnosticFrames
from .evaluate import load_model_checkpoint, write_result
from .runner import _dataset, resolve_task_root


def old_config(payload, current):
    if (payload.get("schema") != "rmdm_hvdit_v4_x0_continue_checkpoint_v1"
            or payload.get("architecture_id") != ARCHITECTURE_ID):
        raise ValueError("expected historical x0 continuation checkpoint")
    config = _from_mapping(ExperimentConfig, payload["resolved_config"])
    if config.diffusion.prediction_type != "sample" or config.model.use_explicit_tx_condition:
        raise ValueError("expected old no-Tx x0 model")
    config.data.root = current.data.root
    config.data.split_file = current.data.split_file
    config.evaluation.subset_manifest = current.evaluation.subset_manifest
    config.validate()
    return config


def zero_observations(batch):
    """Zero all RSS inputs before encoding, including the HWM branch."""
    return make_prior_training_batch(batch)


@torch.no_grad()
def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--config", default="experimental/tx_prior/remote.yaml")
    parser.add_argument("--checkpoint", required=True)
    parser.add_argument("--current-checkpoint", default="")
    parser.add_argument("--frames", type=int, default=30)
    parser.add_argument("--batch-size", type=int, default=4)
    parser.add_argument("--precision", choices=("fp32", "bf16"), default="fp32")
    parser.add_argument("--noise-protocol", choices=("shared-prior", "historical-p1"), default="shared-prior")
    parser.add_argument("--output", required=True)
    args = parser.parse_args()
    config = load_config(args.config, smoke=False)
    output = Path(args.output).resolve()
    allowed = resolve_task_root(Path.cwd(), config.pipeline.output_root) / "train/diagnostics"
    if not output.is_relative_to(allowed) or output.exists():
        raise ValueError("output must be new and under train/diagnostics")
    if args.batch_size < 1 or not 2 <= args.frames <= 3000 or torch.cuda.device_count() != 1:
        raise ValueError("require one visible GPU, positive batch, 2..3000 frames")
    payload = torch.load(args.checkpoint, map_location="cpu", weights_only=False)
    legacy_config = old_config(payload, config)
    legacy = build_t1_system(legacy_config)
    legacy.load_state_dict(payload["model"], strict=True)
    metadata = dict(path=str(Path(args.checkpoint).resolve()), step=payload["global_step"],
                    original_resolved_config=payload["resolved_config"])
    del payload
    legacy = legacy.cuda().eval()
    current = None
    current_metadata = None
    if args.current_checkpoint:
        current = build_scene_prior_system(config)
        current_metadata = load_model_checkpoint(Path(args.current_checkpoint), current)
        current = current.cuda().eval()
    ids = manifest_video_ids(config.evaluation.subset_manifest, "stage_a")
    dataset = DiagnosticFrames(_dataset(config, split="val", fixed_starts=tuple(range(100)),
                                        video_ids=ids), args.frames, include_legacy=False)
    loader = DataLoader(dataset, batch_size=args.batch_size, shuffle=False, num_workers=0)
    labels = ["old_zero", "old_p1", "old_p2", "old_p3"] + (["current_zero"] if current else [])
    accumulators = {label: MetricAccumulator(device="cuda") for label in labels}
    scenes = Counter()
    prediction_delta = {label: 0.0 for label in labels if label != "old_zero"}
    pixel_count = 0
    scored = 0
    started = time.perf_counter()
    samplers = dict(old=DDIMSampler(legacy_config.diffusion), current=DDIMSampler(config.diffusion))
    print(f"Old DiT paired observation test: {args.frames} frames, DDIM20, {args.precision}", flush=True)
    for dense in loader:
        dense = {k: v.cuda() if torch.is_tensor(v) else v for k, v in dense.items()}
        prior = zero_observations(dense)
        names = frame_names_by_sample(prior, batch_size=prior["target"].shape[0], window_size=1)
        scenes.update(name[0].split("/")[0] for name in names)
        noise = (deterministic_frame_noise_like(prior["target"], names, rate=1, seed=config.sampling.seed)
                 if args.noise_protocol == "historical-p1" else
                 deterministic_prior_noise_like(prior["target"], names, seed=config.sampling.seed))
        predictions = {}
        for label in labels:
            batch = (apply_fixed_sparse_observations(dense, rate=int(label[-1]), split="val")
                     if label.startswith("old_p") else prior)
            model = current if label == "current_zero" else legacy
            sampler = samplers["current" if label == "current_zero" else "old"]
            with torch.autocast("cuda", dtype=torch.bfloat16, enabled=args.precision == "bf16"):
                prediction = sampler.sample(model, batch, initial_noise=noise, steps=20, eta=0)
            predictions[label] = prediction
            accumulators[label].update(prediction, batch["target"], batch["building"],
                                       batch["vehicle"], batch["sampling_mask"])
            if label != "old_zero":
                prediction_delta[label] += (prediction.double()-predictions["old_zero"].double()).square().sum().item()
        pixel_count += prior["target"].numel()
        scored += len(names)
        print(f"Paired old DiT scored {scored}/{args.frames}", flush=True)
    if scored != args.frames:
        raise RuntimeError("frame count mismatch")
    write_result(output, dict(schema="tx_prior_legacy_dit_observation_diagnosis_v1",
        checkpoint=metadata, current_checkpoint=current_metadata, scored_frames=scored,
        scene_coverage=dict(scenes), steps=20, eta=0, precision=args.precision,
        device=torch.cuda.get_device_name(), noise=args.noise_protocol,
        results={k: dict(metrics=v.compute(), raw=v.raw()) for k, v in accumulators.items()},
        prediction_mse_delta_vs_old_zero={k: v/pixel_count for k, v in prediction_delta.items()},
        zero_ablation="RSS_and_mask_zero_before_all_encoding_HWM_recomputed_not_raw_only_ablation",
        limits="zero_observation_inputs_are_out_of_training_distribution_not_equivalent_to_retraining",
        elapsed_seconds=time.perf_counter()-started,
        source_head=subprocess.check_output(["git", "rev-parse", "HEAD"], text=True).strip()))
    print(f"Saved {output}", flush=True)


if __name__ == "__main__":
    main()
