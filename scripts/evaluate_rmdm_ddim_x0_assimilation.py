#!/usr/bin/env python3
"""Evaluate RMDM-SF with the same late-step x0 assimilation used for V4 W16.

The default first96 domain matches the frames scored by W16 Stage-A. This is
an evaluation-only diagnostic and does not modify the formal RMDM sampler or
checkpoint.
"""

from __future__ import annotations

import argparse
from pathlib import Path
from typing import Any, Iterable

import torch
from accelerate import Accelerator, DataLoaderConfiguration
from torch.utils.data import DataLoader

from evaluate_hvdit_v4_w16_ddim_x0_guidance import (
    CachedDDIMSampler,
    _method_name,
    _observed_metrics,
    _observed_update,
    _parse_csv_floats,
    _parse_csv_ints,
    _require_visible_gpus,
    _starts,
)
from rmdm.data import SamplingPolicy, WindowDataset
from rmdm.diffusion import deterministic_noise_like
from rmdm.evaluation.metrics import MetricAccumulator
from rmdm.evaluation.fixed_sparse_protocol import (
    MASK_MANIFEST_SEED,
    add_fixed_observation_noise,
    apply_fixed_sparse_observations,
    deterministic_frame_noise_like,
    frame_names_by_sample,
)
from rmdm_hvdit_v4_x0_w16_ratebalanced.inverse_sampling import add_observation_noise
from rmdm_hvdit_v4_joint.config import ExperimentConfig, load_config
from rmdm_hvdit_v4_joint.evaluation.evaluator import manifest_video_ids
from rmdm_hvdit_v4_joint.evaluation.legacy_rmdm import LegacyRMDMT1ProtocolAdapter
from rmdm_hvdit_v4_joint.training.engine import write_json_atomic
from train_sparse_dynamic_rmdm import build_model_config
from utils import build_unet_from_config


def _load_rmdm(checkpoint: Path) -> tuple[torch.nn.Module, dict[str, Any]]:
    payload = torch.load(checkpoint, map_location="cpu", weights_only=False, mmap=True)
    if (
        payload.get("schema") != "rmdm_sf_sparse_checkpoint_v1"
        or "model" not in payload
        or "args" not in payload
    ):
        raise ValueError("checkpoint is not an RMDM-SF sparse artifact")
    train_args = argparse.Namespace(**payload["args"])
    model = build_unet_from_config(build_model_config(train_args))
    model.load_state_dict(payload["model"], strict=True)
    metadata = {
        "schema": payload.get("schema"),
        "epoch": int(payload.get("epoch", -1)),
        "global_step": int(payload.get("global_step", -1)),
    }
    del payload
    without_tx = bool(getattr(train_args, "without_tx", False))
    metadata["without_tx"] = without_tx
    return LegacyRMDMT1ProtocolAdapter(model, without_tx=without_tx), metadata


def _evaluate(
    accelerator: Accelerator,
    model: torch.nn.Module,
    config: ExperimentConfig,
    *,
    split: str,
    subset_stage: str,
    manifest_path: Path,
    rates: Iterable[float],
    strengths: list[float],
    steps: int,
    guided_steps: int,
    max_update: float,
    max_videos: int,
    frames_per_video: int,
    batch_size: int,
    log_interval: int,
    fixed_paper_protocol: bool,
    progress_output: Path | None = None,
    observation_noise_std: float = 0.0,
    include_noise_aware: bool = False,
) -> dict[str, Any]:
    if frames_per_video not in {96, 100}:
        raise ValueError("frames_per_video must be 96 or 100")
    video_ids = manifest_video_ids(manifest_path, subset_stage)
    if max_videos > 0:
        video_ids = video_ids[:max_videos]
    fixed_starts = tuple(range(frames_per_video))
    dataset = WindowDataset(
        root=config.data.root,
        split=split,
        split_file=config.data.split_file,
        window_size=1,
        seed=config.sampling.seed,
        cache_size=config.data.cache_size,
        tx_heatmap_sigma_px=config.data.tx_heatmap_sigma_px,
        fixed_starts=fixed_starts,
        video_ids=video_ids,
    )
    loader = DataLoader(
        dataset,
        batch_size=batch_size,
        shuffle=False,
        num_workers=min(config.data.workers, 2),
        pin_memory=True,
        persistent_workers=config.data.workers > 0,
        drop_last=False,
    )
    loader = accelerator.prepare_data_loader(loader)
    core = accelerator.unwrap_model(model)
    core.eval()
    core.requires_grad_(False)
    sampler = CachedDDIMSampler(config)
    method_specs: list[tuple[str, float | None, float]] = [("baseline", None, 0.0)]
    method_specs.extend(
        (_method_name(value, "rms"), value, 0.0) for value in strengths
    )
    if include_noise_aware and observation_noise_std > 0:
        method_specs.extend(
            (
                f"guided_noise_aware_s{value:.6g}".replace(".", "p"),
                value,
                observation_noise_std**2,
            )
            for value in strengths
    )
    results: dict[str, dict[str, Any]] = {name: {} for name, _, _ in method_specs}
    results["noisy_input_observed_points"] = {}

    for rate in rates:
        policy = SamplingPolicy(config.sampling, split="val")
        policy.set_epoch(0)
        accumulators = {
            name: MetricAccumulator(device=accelerator.device)
            for name, _, _ in method_specs
        }
        observed_sums = {
            name: torch.zeros(4, dtype=torch.float64, device=accelerator.device)
            for name, _, _ in method_specs
        }
        noisy_input_sums = torch.zeros(4, dtype=torch.float64, device=accelerator.device)
        scored_frames_local = 0

        for batch_index, dense_batch in enumerate(loader):
            if fixed_paper_protocol:
                sparse_batch = apply_fixed_sparse_observations(
                    dense_batch,
                    rate=float(rate),
                    split=split,
                )
                sparse_batch = add_fixed_observation_noise(
                    sparse_batch,
                    standard_deviation=observation_noise_std,
                    rate=float(rate),
                    seed=config.sampling.seed,
                )
            else:
                sparse_batch = policy(dense_batch, fixed_rate=float(rate))
                sparse_batch = add_observation_noise(
                    sparse_batch,
                    standard_deviation=observation_noise_std,
                    rate=float(rate),
                    seed=config.sampling.seed,
                )
            target = sparse_batch["target"]
            _observed_update(
                noisy_input_sums,
                sparse_batch["observed_rss"],
                target,
                sparse_batch["sampling_mask"],
            )
            starts = _starts(sparse_batch)
            if fixed_paper_protocol:
                initial_noise = deterministic_frame_noise_like(
                    target,
                    frame_names_by_sample(
                        sparse_batch,
                        batch_size=target.shape[0],
                        window_size=target.shape[1],
                    ),
                    rate=float(rate),
                    seed=config.sampling.seed,
                )
            else:
                initial_noise = deterministic_noise_like(
                    target,
                    video_ids=[str(value) for value in sparse_batch["video_id"]],
                    starts=starts,
                    rate=float(rate),
                    seed=config.sampling.seed,
                )
            with torch.no_grad(), accelerator.autocast():
                condition_cache = core.encode_conditions(sparse_batch)

            predictions: dict[str, torch.Tensor] = {
                "baseline": sampler.baseline(
                    core,
                    condition_cache,
                    initial_noise,
                    steps=steps,
                    accelerator=accelerator,
                )
            }
            for name, strength, noise_variance in method_specs[1:]:
                assert strength is not None
                predictions[name] = sampler.guided(
                    core,
                    condition_cache,
                    sparse_batch,
                    initial_noise,
                    steps=steps,
                    guided_steps=guided_steps,
                    strength=strength,
                    max_update=max_update,
                    gradient_normalization="rms",
                    observation_noise_variance=noise_variance,
                    accelerator=accelerator,
                )

            for name, prediction in predictions.items():
                accumulators[name].update(
                    prediction,
                    target,
                    sparse_batch["building"],
                    sparse_batch["vehicle"],
                    sparse_batch["sampling_mask"],
                )
                _observed_update(
                    observed_sums[name],
                    prediction,
                    target,
                    sparse_batch["sampling_mask"],
                )
            scored_frames_local += int(target.shape[0])

            if batch_index % log_interval == 0:
                partial_frames = int(
                    accelerator.reduce(
                        torch.tensor(
                            scored_frames_local,
                            device=accelerator.device,
                            dtype=torch.int64,
                        ),
                        reduction="sum",
                    ).item()
                )
                partial_methods: dict[str, Any] = {}
                for name, strength, noise_variance in method_specs:
                    partial_accumulator = MetricAccumulator(device="cpu")
                    partial_accumulator.sums = accelerator.reduce(
                        accumulators[name].sums.clone(), reduction="sum"
                    ).cpu()
                    partial_observed = accelerator.reduce(
                        observed_sums[name].clone(), reduction="sum"
                    )
                    observed_metrics, observed_raw = _observed_metrics(partial_observed)
                    partial_methods[name] = {
                        "guidance_strength": strength,
                        "observation_noise_variance_subtracted": noise_variance,
                        "metrics": partial_accumulator.compute(),
                        "raw": partial_accumulator.raw(),
                        "observed_points": {
                            "metrics": observed_metrics,
                            "raw": observed_raw,
                        },
                    }
                partial_input = accelerator.reduce(noisy_input_sums.clone(), reduction="sum")
                input_metrics, input_raw = _observed_metrics(partial_input)
                if accelerator.is_main_process:
                    if progress_output is not None:
                        write_json_atomic(
                            progress_output,
                            {
                                "schema": "rmdm_sf_ddim_x0_assimilation_progress_v1",
                                "status": "in_progress",
                                "current_rate": f"{float(rate):g}",
                                "completed_batches": batch_index + 1,
                                "total_batches": len(loader),
                                "scored_frames_so_far": partial_frames,
                                "noisy_input_observed_points": {
                                    "metrics": input_metrics,
                                    "raw": input_raw,
                                },
                                "methods": partial_methods,
                            },
                        )
                    print(
                        f"[rmdm-x0-assimilation] p={float(rate):g} "
                        f"batch={batch_index}/{len(loader)} methods={','.join(predictions)}",
                        flush=True,
                    )

        scored_tensor = torch.tensor(
            scored_frames_local, device=accelerator.device, dtype=torch.int64
        )
        scored_frames = int(accelerator.reduce(scored_tensor, reduction="sum").item())
        expected_frames = len(video_ids) * frames_per_video
        if scored_frames != expected_frames:
            raise RuntimeError(f"scored {scored_frames} frames, expected {expected_frames}")

        for name, _, _ in method_specs:
            accumulators[name].sums = accelerator.reduce(
                accumulators[name].sums, reduction="sum"
            )
            observed_sums[name] = accelerator.reduce(
                observed_sums[name], reduction="sum"
            )
            observed_metrics, observed_raw = _observed_metrics(observed_sums[name])
            results[name][f"{float(rate):g}"] = {
                "metrics": accumulators[name].compute(),
                "raw": accumulators[name].raw(),
                "observed_points": {
                    "metrics": observed_metrics,
                    "raw": observed_raw,
                },
                "scored_frames": scored_frames,
            }
        noisy_input_sums = accelerator.reduce(noisy_input_sums, reduction="sum")
        input_metrics, input_raw = _observed_metrics(noisy_input_sums)
        results["noisy_input_observed_points"][f"{float(rate):g}"] = {
            "metrics": input_metrics,
            "raw": input_raw,
        }

    rate_names = [f"{float(rate):g}" for rate in rates]
    methods: dict[str, Any] = {}
    for name, strength, noise_variance in method_specs:
        methods[name] = {
            "guidance_strength": strength,
            "observation_noise_variance_subtracted": noise_variance,
            "rates": results[name],
            "macro_full_image_nmse_selected_rates": sum(
                results[name][rate]["metrics"]["full_image"]["nmse"]
                for rate in rate_names
            )
            / len(rate_names),
            "macro_unobserved_free_space_nmse_selected_rates": sum(
                results[name][rate]["metrics"]["unobserved_free_space"]["nmse"]
                for rate in rate_names
            )
            / len(rate_names),
            "macro_observed_point_mse_selected_rates": sum(
                results[name][rate]["observed_points"]["metrics"]["mse"]
                for rate in rate_names
            )
            / len(rate_names),
        }

    return {
        "schema": "rmdm_sf_ddim_x0_observation_assimilation_v1",
        "evaluated_model": "rmdm_sf",
        "variant": "t1",
        "split": split,
        "subset_stage": subset_stage,
        "manifest": str(manifest_path),
        "video_ids": video_ids,
        "video_count": len(video_ids),
        "window_count": len(dataset),
        "scored_frames_per_video": frames_per_video,
        "ddim_steps": steps,
        "guided_last_steps": guided_steps,
        "rates": [float(rate) for rate in rates],
        "batch_size_per_gpu": batch_size,
        "fixed_paper_protocol": fixed_paper_protocol,
        "mask_manifest_seed": MASK_MANIFEST_SEED if fixed_paper_protocol else None,
        "guidance": {
            "loss": "visible-point mean squared error between unclipped x0_pred and observed_rss",
            "gradient_target": "current x_t",
            "update_target": "scheduler-produced x_(t-1)",
            "normalization": "per-frame residual RMS divided by full-latent gradient RMS",
            "max_abs_update": max_update,
            "uses_unobserved_target": False,
            "observation_noise_std": observation_noise_std,
            "observation_noise_clipped": False,
        },
        "methods": methods,
        "noisy_input_observed_points": results["noisy_input_observed_points"],
    }


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--config", required=True)
    parser.add_argument("--checkpoint", required=True)
    parser.add_argument("--output", required=True)
    parser.add_argument("--split", choices=("val", "test"), default="val")
    parser.add_argument(
        "--subset-stage",
        choices=("stage_a", "stage_b_extra", "all"),
        default="stage_a",
    )
    parser.add_argument(
        "--manifest",
        help="evaluation manifest; defaults to the config validation subset manifest",
    )
    parser.add_argument("--rates", default="1,2,3")
    parser.add_argument("--strengths", default="0.5")
    parser.add_argument("--ddim-steps", type=int, default=20)
    parser.add_argument("--guided-steps", type=int, default=7)
    parser.add_argument("--max-update", type=float, default=0.25)
    parser.add_argument("--observation-noise-std", type=float, default=0.0)
    parser.add_argument("--include-noise-aware", action="store_true")
    parser.add_argument("--max-videos", type=int, default=0)
    parser.add_argument("--frames-per-video", type=int, default=96)
    parser.add_argument("--batch-size", type=int, default=32)
    parser.add_argument("--log-interval", type=int, default=10)
    parser.add_argument("--expected-visible-gpus", default="0,1,2,3")
    parser.add_argument(
        "--baseline-only",
        action="store_true",
        help="Evaluate unguided DDIM only; skip observation-gradient assimilation.",
    )
    parser.add_argument(
        "--fixed-paper-protocol",
        action="store_true",
        help="Use absolute-frame masks and frame-keyed DDIM noise.",
    )
    args = parser.parse_args()

    expected_gpus = _parse_csv_ints(args.expected_visible_gpus)
    _require_visible_gpus(expected_gpus)
    rates = _parse_csv_floats(args.rates)
    strengths = [] if args.baseline_only else _parse_csv_floats(args.strengths)
    if any(value <= 0 for value in strengths):
        raise ValueError("guidance strengths must be positive")

    config_path = Path(args.config).expanduser().resolve()
    checkpoint_path = Path(args.checkpoint).expanduser().resolve()
    config = load_config(config_path)
    manifest_path = (
        Path(args.manifest).expanduser().resolve()
        if args.manifest
        else Path(config.evaluation.subset_manifest).expanduser().resolve()
    )
    accelerator = Accelerator(
        mixed_precision="bf16",
        dataloader_config=DataLoaderConfiguration(even_batches=False),
    )
    if accelerator.num_processes != len(expected_gpus):
        raise RuntimeError(
            f"expected {len(expected_gpus)} accelerator processes, "
            f"got {accelerator.num_processes}"
        )

    model, checkpoint_metadata = _load_rmdm(checkpoint_path)
    model.requires_grad_(False)
    model = accelerator.prepare(model)
    result = _evaluate(
        accelerator,
        model,
        config,
        split=args.split,
        subset_stage=args.subset_stage,
        manifest_path=manifest_path,
        rates=rates,
        strengths=strengths,
        steps=args.ddim_steps,
        guided_steps=args.guided_steps,
        max_update=args.max_update,
        max_videos=args.max_videos,
        frames_per_video=args.frames_per_video,
        batch_size=args.batch_size,
        log_interval=args.log_interval,
        fixed_paper_protocol=args.fixed_paper_protocol,
        progress_output=Path(f"{args.output}.progress.json").expanduser().resolve(),
        observation_noise_std=args.observation_noise_std,
        include_noise_aware=args.include_noise_aware,
    )
    result["checkpoint"] = {
        "path": str(checkpoint_path),
        **checkpoint_metadata,
    }
    result["config"] = {"path": str(config_path)}
    result["visible_physical_gpus"] = expected_gpus

    accelerator.wait_for_everyone()
    if accelerator.is_main_process:
        write_json_atomic(args.output, result)
        print(
            {
                "output": str(Path(args.output).expanduser().resolve()),
                "scores": {
                    name: values["macro_full_image_nmse_selected_rates"]
                    for name, values in result["methods"].items()
                },
            },
            flush=True,
        )


if __name__ == "__main__":
    main()
