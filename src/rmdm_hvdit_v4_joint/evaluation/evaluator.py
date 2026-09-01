"""Distributed, deterministic Stage-A/Stage-B evaluation for T1 and W16."""

from __future__ import annotations

import json
from pathlib import Path
from typing import Any, Iterable

import torch
from torch import nn
from torch.utils.data import DataLoader

from rmdm.data import WindowDataset
from rmdm.diffusion import DDIMSampler
from rmdm.evaluation.fixed_sparse_protocol import (
    MASK_MANIFEST_SEED,
    MASK_SAMPLER_VERSION,
    apply_fixed_sparse_observations,
    deterministic_frame_noise_like,
    frame_names_by_sample,
)
from rmdm.evaluation.metrics import MetricAccumulator
from rmdm_hvdit_v4_joint.config import ExperimentConfig

from .stitching import scoring_windows, validate_scoring_domain


def manifest_video_ids(path: str | Path, stage: str) -> list[str]:
    with Path(path).expanduser().resolve().open("r", encoding="utf-8") as handle:
        manifest = json.load(handle)
    if stage == "all":
        videos = manifest.get("videos", [])
    elif stage in {"stage_a", "stage_b_extra"}:
        videos = manifest.get(stage, {}).get("videos", [])
    else:
        raise ValueError("stage must be stage_a, stage_b_extra or all")
    if not videos:
        raise ValueError(f"No videos in manifest stage {stage!r}")
    return [str(item["video_id"]) for item in videos]


def _starts(batch: dict[str, Any]) -> list[int]:
    values = batch["start"]
    values = values.detach().cpu().tolist() if torch.is_tensor(values) else values
    return [int(value) for value in values]


def _prepare_exact_evaluation_loader(accelerator: Any, loader: DataLoader) -> Any:
    """Shard evaluation without padding or duplicating protocol samples.

    Training deliberately uses even batches, but fixed evaluation domains must
    contain every ``(video_id, start)`` pair exactly once across all ranks.
    Accelerate otherwise repeats leading batches when the batch count is not
    divisible by the world size (for example, 300 T1 batches on eight ranks).
    """

    original_even_batches = bool(accelerator.even_batches)
    try:
        accelerator.even_batches = False
        return accelerator.prepare_data_loader(loader)
    finally:
        accelerator.even_batches = original_even_batches


class _RawObservationAblation(nn.Module):
    def __init__(self, model: nn.Module) -> None:
        super().__init__()
        self.model = model

    def encode_conditions(self, sparse_batch: dict[str, Any]) -> dict[str, torch.Tensor]:
        cache = self.model.encode_conditions(sparse_batch)
        return self.model.ablate_raw_observations(cache)

    def denoise(
        self,
        noisy_target: torch.Tensor,
        diffusion_step: torch.Tensor,
        condition_cache: dict[str, torch.Tensor],
    ) -> torch.Tensor:
        return self.model.denoise(noisy_target, diffusion_step, condition_cache)


@torch.no_grad()
def evaluate_stage_a(
    accelerator: Any,
    model: nn.Module,
    config: ExperimentConfig,
    *,
    variant: str,
    subset_stage: str = "stage_a",
    rates: Iterable[float] | None = None,
    ablate_raw_observations: bool = False,
    full100: bool = False,
    split: str = "val",
    manifest_path: str | Path | None = None,
    log_interval: int = 50,
) -> dict[str, Any]:
    if variant not in {"t1", "w16"}:
        raise ValueError("variant must be t1 or w16")
    if variant == "t1" and full100:
        raise ValueError("full100 is a W16-only evaluation domain")
    if split not in {"val", "test"}:
        raise ValueError("split must be val or test")
    selected_manifest = manifest_path or config.evaluation.subset_manifest
    video_ids = manifest_video_ids(selected_manifest, subset_stage)
    if variant == "t1":
        window_size = 1
        fixed_starts = tuple(range(config.data.frames_per_video))
        score_by_start = {start: slice(0, 1) for start in fixed_starts}
        expected_frames_per_video = config.data.frames_per_video
    else:
        windows = scoring_windows(full100=full100)
        validate_scoring_domain(windows, expected_frames=100 if full100 else 96)
        window_size = 16
        fixed_starts = tuple(window.start for window in windows)
        score_by_start = {window.start: slice(window.local_start, window.local_stop) for window in windows}
        expected_frames_per_video = 100 if full100 else 96
    dataset = WindowDataset(
        root=config.data.root,
        split=split,
        split_file=config.data.split_file,
        window_size=window_size,
        seed=config.sampling.seed,
        cache_size=config.data.cache_size,
        tx_heatmap_sigma_px=config.data.tx_heatmap_sigma_px,
        fixed_starts=fixed_starts,
        video_ids=video_ids,
    )
    loader = DataLoader(
        dataset,
        batch_size=(
            config.evaluation.t1_evaluation_batch_size
            if variant == "t1"
            else config.evaluation.w16_evaluation_batch_size
        ),
        shuffle=False,
        num_workers=min(config.data.workers, 2),
        pin_memory=True,
        persistent_workers=config.data.workers > 0,
        drop_last=False,
    )
    loader = _prepare_exact_evaluation_loader(accelerator, loader)
    core = accelerator.unwrap_model(model)
    was_training = core.training
    core.eval()
    sampling_model: nn.Module = _RawObservationAblation(core) if ablate_raw_observations else core
    sampler = DDIMSampler(config.diffusion)
    results: dict[str, Any] = {}
    for rate in rates or config.evaluation.rates:
        accumulator = MetricAccumulator(device=accelerator.device)
        scored_frames_local = 0
        for batch_index, dense_batch in enumerate(loader):
            sparse_batch = apply_fixed_sparse_observations(
                dense_batch,
                rate=float(rate),
                split=split,
            )
            target = sparse_batch["target"]
            starts = _starts(sparse_batch)
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
            with accelerator.autocast():
                prediction = sampler.sample(
                    sampling_model,
                    sparse_batch,
                    initial_noise=initial_noise,
                    steps=config.evaluation.ddim_steps,
                )
            for item_index, start in enumerate(starts):
                score_slice = score_by_start[start]
                selection = (slice(item_index, item_index + 1), score_slice)
                accumulator.update(
                    prediction[selection],
                    target[selection],
                    sparse_batch["building"][selection],
                    sparse_batch["vehicle"][selection],
                    sparse_batch["sampling_mask"][selection],
                )
                scored_frames_local += score_slice.stop - score_slice.start
            if accelerator.is_main_process and batch_index % log_interval == 0:
                print(
                    f"[hvdit-eval] variant={variant} stage={subset_stage} p={float(rate):g} "
                    f"window={batch_index}/{len(loader)} ablated={ablate_raw_observations}",
                    flush=True,
                )
        accumulator.sums = accelerator.reduce(accumulator.sums, reduction="sum")
        scored_tensor = torch.tensor(scored_frames_local, device=accelerator.device, dtype=torch.int64)
        scored_frames = int(accelerator.reduce(scored_tensor, reduction="sum").item())
        expected_frames = len(video_ids) * expected_frames_per_video
        if scored_frames != expected_frames:
            raise RuntimeError(f"Scored {scored_frames} frames, expected {expected_frames}")
        results[f"{float(rate):g}"] = {
            "metrics": accumulator.compute(),
            "raw": accumulator.raw(),
            "scored_frames": scored_frames,
        }
    primary = [f"{float(rate):g}" for rate in config.evaluation.rates]
    macro = sum(results[rate]["metrics"]["full_image"]["nmse"] for rate in primary) / len(primary)
    if was_training:
        core.train()
    return {
        "schema": "rmdm_hvdit_v4_joint_evaluation_v1",
        "variant": variant,
        "subset_stage": subset_stage,
        "split": split,
        "manifest": str(Path(selected_manifest).expanduser().resolve()),
        "video_count": len(video_ids),
        "window_count": len(dataset),
        "scored_frames_per_video": expected_frames_per_video,
        "ddim_steps": config.evaluation.ddim_steps,
        "sparse_mask_protocol": {
            "sampler_version": MASK_SAMPLER_VERSION,
            "manifest_seed": MASK_MANIFEST_SEED,
            "keyed_by_physical_frame": True,
        },
        "ddim_noise_protocol": "rmdm-paper-ddim-frame-noise-v1",
        "raw_observations_ablated": bool(ablate_raw_observations),
        "full100": bool(full100),
        "rates": results,
        "macro_full_image_nmse_p1_p2_p3": macro,
    }


def combine_results(*evaluations: dict[str, Any]) -> dict[str, Any]:
    if not evaluations:
        raise ValueError("At least one evaluation is required")
    rate_names = set(evaluations[0]["rates"])
    if any(set(item["rates"]) != rate_names for item in evaluations[1:]):
        raise ValueError("Evaluation rate sets do not match")
    combined: dict[str, Any] = {}
    for rate in sorted(rate_names, key=float):
        accumulator = MetricAccumulator()
        for evaluation in evaluations:
            accumulator.add_raw(evaluation["rates"][rate]["raw"])
        combined[rate] = {"metrics": accumulator.compute(), "raw": accumulator.raw()}
    primary = [rate for rate in ("1", "2", "3") if rate in combined]
    return {
        "schema": "rmdm_hvdit_v4_joint_combined_evaluation_v1",
        "subset_stage": "+".join(str(item["subset_stage"]) for item in evaluations),
        "video_count": sum(int(item["video_count"]) for item in evaluations),
        "rates": combined,
        "macro_full_image_nmse_p1_p2_p3": sum(
            combined[rate]["metrics"]["full_image"]["nmse"] for rate in primary
        )
        / len(primary),
    }
