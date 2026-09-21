"""Deterministic validation manifest filtering and DDIM evaluation."""

from __future__ import annotations

import json
from pathlib import Path

import torch
from torch.utils.data import DataLoader

from rmdm.data import SamplingPolicy, WindowDataset
from rmdm.diffusion import DDIMSampler, deterministic_noise_like
from rmdm_hvdit_v4_joint.training.engine import write_json_atomic

from .checkpoint import load
from .config import ExperimentConfig
from .model import build_model
from .noise import add_fixed_measurement_noise


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
    device: str = "cuda",
    ddim_steps: int | None = None,
    max_batches: int = 0,
) -> dict:
    """Evaluate fixed rates/noise levels with deterministic masks and DDIM noise."""

    root = Path(repository_root).expanduser().resolve()
    manifest = Path(config.validation.subset_manifest)
    if not manifest.is_absolute():
        manifest = root / manifest
    split_file = Path(config.data.split_file)
    if not split_file.is_absolute():
        split_file = root / split_file
    video_ids = validation_video_ids(
        manifest,
        included_scenes=config.validation.included_scenes,
        excluded_scenes=config.validation.excluded_scenes,
    )
    dataset = WindowDataset(
        root=config.data.root,
        split="val",
        split_file=str(split_file),
        window_size=1,
        seed=config.sampling.seed,
        cache_size=config.data.cache_size,
        include_tx=False,
        fixed_starts=config.validation.frame_starts,
        video_ids=video_ids,
    )
    loader = DataLoader(
        dataset,
        batch_size=config.validation.batch_size,
        shuffle=False,
        num_workers=config.data.workers,
        pin_memory=True,
        persistent_workers=config.data.workers > 0,
    )
    torch_device = torch.device(device)
    model = build_model(config).to(torch_device)
    payload = load(checkpoint_path, model)
    model.eval()
    sampling = SamplingPolicy(config.sampling, split="validation")
    sampler = DDIMSampler(config.diffusion)
    results = []
    for rate in config.validation.rates:
        for sigma in config.validation.noise_standard_deviations:
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
                starts = sparse["start"].detach().cpu().tolist()
                initial = deterministic_noise_like(
                    sparse["target"], video_ids=list(sparse["video_id"]), starts=starts,
                    rate=float(rate), seed=config.train.seed,
                )
                generated = sampler.sample(
                    model, sparse, initial_noise=initial, steps=ddim_steps or config.diffusion.ddim_steps
                )
                difference = (generated.float() - sparse["target"].float()).flatten(1)
                mse = difference.square().mean(1)
                mae = difference.abs().mean(1)
                psnr = 10.0 * torch.log10(1.0 / mse.clamp_min(1.0e-12))
                mse_sum += float(mse.sum())
                mae_sum += float(mae.sum())
                psnr_sum += float(psnr.sum())
                count += int(mse.numel())
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
    summary = {
        "schema": "noise_temporal_rmdm_t1_validation_v1",
        "checkpoint": str(Path(checkpoint_path).expanduser().resolve()),
        "checkpoint_step": int(payload["global_step"]),
        "ddim_steps": int(ddim_steps or config.diffusion.ddim_steps),
        "tx_input": False,
        "source_loss": False,
        "results": results,
    }
    write_json_atomic(Path(output_path).expanduser().resolve(), summary)
    return summary
