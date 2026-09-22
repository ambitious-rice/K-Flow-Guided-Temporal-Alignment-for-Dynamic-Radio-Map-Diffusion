"""Deterministic validation manifest filtering and DDIM evaluation."""

from __future__ import annotations

import json
import time
from pathlib import Path

import torch
from torch.utils.data import DataLoader

from rmdm.data import SamplingPolicy, WindowDataset
from rmdm.diffusion import DDIMSampler, deterministic_noise_like
from rmdm_hvdit_v4_joint.training.engine import write_json_atomic

from .checkpoint import load
from .config import ExperimentConfig
from .model import build_model
from .noise import add_fixed_measurement_noise, apply_variance_conditioning


def validation_video_ids(
    manifest_path: str | Path,
    *,
    included_scenes: list[str],
    excluded_scenes: list[str],
    stage: str = "stage_a",
) -> list[str]:
    with Path(manifest_path).expanduser().resolve().open("r", encoding="utf-8") as handle:
        manifest = json.load(handle)
    videos = manifest.get(stage, {}).get("videos", [])
    included = set(included_scenes)
    excluded = set(excluded_scenes)
    if included & excluded:
        raise ValueError("included and excluded validation scenes overlap")
    selected = [
        str(item["video_id"])
        for item in videos
        if str(item.get("scene_id", str(item["video_id"]).split("/", 1)[0])) in included
        and str(item.get("scene_id", str(item["video_id"]).split("/", 1)[0])) not in excluded
    ]
    if not selected:
        raise ValueError("validation scene filter selected no videos")
    found_scenes = {video_id.split("/", 1)[0] for video_id in selected}
    if found_scenes != included:
        raise ValueError(f"validation manifest is missing scenes: {sorted(included - found_scenes)}")
    return selected


def _to_device(batch: dict, device: torch.device) -> dict:
    return {
        key: value.to(device, non_blocking=True) if torch.is_tensor(value) else value
        for key, value in batch.items()
    }


@torch.no_grad()
def run_validation(
    config: ExperimentConfig,
    *,
    checkpoint_path: str | Path,
    repository_root: str | Path,
    output_path: str | Path,
    evaluation_split: str = "val",
    device: str = "cuda",
    ddim_steps: int | None = None,
    max_batches: int = 0,
) -> dict:
    """Evaluate the fixed periodic-val or final-partial-test protocol."""

    root = Path(repository_root).expanduser().resolve()
    if evaluation_split not in {"val", "test"}:
        raise ValueError("evaluation_split must be val or test")
    protocol = config.validation if evaluation_split == "val" else config.final_test
    manifest = Path(protocol.subset_manifest)
    if not manifest.is_absolute():
        manifest = root / manifest
    split_file = Path(config.data.split_file)
    if not split_file.is_absolute():
        split_file = root / split_file
    video_ids = validation_video_ids(
        manifest,
        included_scenes=protocol.included_scenes,
        excluded_scenes=protocol.excluded_scenes,
    )
    dataset = WindowDataset(
        root=config.data.root,
        split=evaluation_split,
        split_file=str(split_file),
        window_size=config.data.window_size,
        seed=config.sampling.seed,
        cache_size=config.data.cache_size,
        include_tx=False,
        fixed_starts=protocol.frame_starts,
        video_ids=video_ids,
    )
    loader = DataLoader(
        dataset,
        batch_size=protocol.batch_size,
        shuffle=False,
        num_workers=config.data.workers,
        pin_memory=True,
        persistent_workers=config.data.workers > 0,
    )
    torch_device = torch.device(device)
    model = build_model(config).to(torch_device)
    payload = load(checkpoint_path, model, expected_phase=config.runtime.phase)
    model.eval()
    sampling = SamplingPolicy(config.sampling, split=evaluation_split)
    sampler = DDIMSampler(config.diffusion)
    results = []
    started = time.monotonic()
    for rate in protocol.rates:
        for sigma in protocol.noise_standard_deviations:
            mse_sum = mae_sum = psnr_sum = 0.0
            count = 0
            for batch_index, dense in enumerate(loader):
                if max_batches and batch_index >= max_batches:
                    break
                dense = _to_device(dense, torch_device)
                sparse = sampling(dense, fixed_rate=float(rate))
                sparse = add_fixed_measurement_noise(
                    sparse, float(sigma), seed=config.measurement_noise.seed
                )
                sparse = apply_variance_conditioning(
                    sparse, enabled=config.model.known_measurement_variance
                )
                starts = sparse["start"].detach().cpu().tolist()
                initial = deterministic_noise_like(
                    sparse["target"], video_ids=list(sparse["video_id"]), starts=starts,
                    rate=float(rate), seed=config.train.seed,
                )
                generated = sampler.sample(
                    model, sparse, initial_noise=initial, steps=ddim_steps or protocol.ddim_steps
                )
                difference = (generated.float() - sparse["target"].float()).flatten(1)
                mse = difference.square().mean(1)
                mae = difference.abs().mean(1)
                psnr = 10.0 * torch.log10(1.0 / mse.clamp_min(1.0e-12))
                mse_sum += float(mse.sum())
                mae_sum += float(mae.sum())
                psnr_sum += float(psnr.sum())
                count += int(mse.numel())
                if batch_index == 0 or (batch_index + 1) % 20 == 0:
                    print(json.dumps({"sampling_rate": float(rate), "measurement_sigma": float(sigma),
                                      "batches_completed": batch_index + 1, "samples_completed": count,
                                      "elapsed_seconds": time.monotonic() - started}), flush=True)
            if count == 0:
                raise RuntimeError("validation processed no samples")
            results.append({
                "sampling_rate": float(rate),
                "measurement_sigma": float(sigma),
                "samples": count,
                "mse": mse_sum / count,
                "mae": mae_sum / count,
                "psnr": psnr_sum / count,
            })
            print(json.dumps({**results[-1], "elapsed_seconds": time.monotonic() - started}), flush=True)
    summary = {
        "schema": f"noise_temporal_rmdm_{config.runtime.phase}_evaluation_v1",
        "phase": config.runtime.phase,
        "window_size": config.data.window_size,
        "evaluation_split": evaluation_split,
        "selection_role": "checkpoint_selection" if evaluation_split == "val" else "final_report_only",
        "checkpoint": str(Path(checkpoint_path).expanduser().resolve()),
        "checkpoint_step": int(payload["global_step"]),
        "ddim_steps": int(ddim_steps or protocol.ddim_steps),
        "tx_input": False,
        "source_loss": False,
        "manifest": str(manifest),
        "frame_starts": protocol.frame_starts,
        "batch_size": protocol.batch_size,
        "elapsed_seconds": time.monotonic() - started,
        "results": results,
    }
    write_json_atomic(Path(output_path).expanduser().resolve(), summary)
    return summary
