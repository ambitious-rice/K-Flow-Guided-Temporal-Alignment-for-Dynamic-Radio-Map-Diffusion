"""Distributed fixed-manifest W16 evaluation."""

from __future__ import annotations

import json
from pathlib import Path
from typing import Any, Iterable

import torch
from torch.utils.data import DataLoader

from rmdm.config import ExperimentConfig
from rmdm.data import SamplingPolicy, WindowDataset
from rmdm.diffusion import DDIMSampler, deterministic_noise_like

from .metrics import DOMAIN_NAMES, STAT_NAMES, MetricAccumulator


def _manifest_video_ids(path: str | Path, stage: str) -> list[str]:
    with Path(path).expanduser().resolve().open("r", encoding="utf-8") as handle:
        manifest = json.load(handle)
    if stage not in {"stage_a", "stage_b_extra"}:
        raise ValueError("stage must be stage_a or stage_b_extra")
    return [str(item["video_id"]) for item in manifest[stage]["videos"]]


def _starts(batch: dict[str, Any]) -> list[int]:
    values = batch["start"]
    return [int(value) for value in (values.detach().cpu().tolist() if torch.is_tensor(values) else values)]


@torch.no_grad()
def evaluate_rates(
    accelerator: Any,
    model: torch.nn.Module,
    config: ExperimentConfig,
    *,
    subset_stage: str,
    rates: Iterable[float],
    ddim_steps: int,
    log_interval: int = 50,
) -> dict[str, Any]:
    video_ids = _manifest_video_ids(config.evaluation.subset_manifest, subset_stage)
    dataset = WindowDataset(
        root=config.data.root,
        split="val",
        split_file=config.data.split_file,
        window_size=config.data.window_size,
        seed=config.sampling.seed,
        cache_size=config.data.cache_size,
        tx_heatmap_sigma_px=config.data.tx_heatmap_sigma_px,
        fixed_starts=config.evaluation.starts,
        video_ids=video_ids,
    )
    model_core = accelerator.unwrap_model(model)
    was_training = model_core.training
    model_core.eval()
    sampler = DDIMSampler(config.diffusion)
    results: dict[str, Any] = {}
    for rate in rates:
        loader = DataLoader(
            dataset,
            batch_size=1,
            shuffle=False,
            num_workers=min(config.data.workers, 2),
            pin_memory=True,
            persistent_workers=config.data.workers > 0,
            drop_last=False,
        )
        loader = accelerator.prepare_data_loader(loader)
        policy = SamplingPolicy(config.sampling, split="val")
        policy.set_epoch(0)
        accumulator = MetricAccumulator(device=accelerator.device)
        for batch_index, dense_batch in enumerate(loader):
            sparse_batch = policy(dense_batch, fixed_rate=float(rate))
            target = sparse_batch["target"]
            initial_noise = deterministic_noise_like(
                target,
                video_ids=[str(value) for value in sparse_batch["video_id"]],
                starts=_starts(sparse_batch),
                rate=float(rate),
                seed=config.sampling.seed,
            )
            with accelerator.autocast():
                prediction = sampler.sample(
                    model_core,
                    sparse_batch,
                    initial_noise=initial_noise,
                    steps=ddim_steps,
                )
            accumulator.update(
                prediction,
                target,
                sparse_batch["building"],
                sparse_batch["vehicle"],
                sparse_batch["sampling_mask"],
            )
            if accelerator.is_main_process and batch_index % log_interval == 0:
                print(
                    f"[joint-eval] stage={subset_stage} p={float(rate):g}% "
                    f"window={batch_index}/{len(loader)}",
                    flush=True,
                )
        accumulator.sums = accelerator.reduce(accumulator.sums, reduction="sum")
        results[f"{float(rate):g}"] = {
            "metrics": accumulator.compute(),
            "raw": accumulator.raw(),
        }
    if was_training:
        model_core.train()
    primary_rates = [str(float(rate)).rstrip("0").rstrip(".") for rate in config.evaluation.stage_a_rates]
    available = [rate for rate in primary_rates if rate in results]
    macro = None
    if available:
        macro = sum(results[rate]["metrics"]["full_image"]["nmse"] for rate in available) / len(available)
    return {
        "subset_stage": subset_stage,
        "video_count": len(video_ids),
        "window_count": len(dataset),
        "ddim_steps": int(ddim_steps),
        "rates": results,
        "macro_full_image_nmse_p1_p2_p3": macro,
    }


def combine_evaluation_results(*evaluations: dict[str, Any]) -> dict[str, Any]:
    if not evaluations:
        raise ValueError("At least one evaluation is required")
    rate_names = set(evaluations[0]["rates"])
    if any(set(item["rates"]) != rate_names for item in evaluations[1:]):
        raise ValueError("Evaluation rate sets do not match")
    combined_rates: dict[str, Any] = {}
    for rate in sorted(rate_names, key=float):
        accumulator = MetricAccumulator()
        for evaluation in evaluations:
            accumulator.add_raw(evaluation["rates"][rate]["raw"])
        combined_rates[rate] = {"metrics": accumulator.compute(), "raw": accumulator.raw()}
    primary = [rate for rate in ("1", "2", "3") if rate in combined_rates]
    macro = (
        sum(combined_rates[rate]["metrics"]["full_image"]["nmse"] for rate in primary) / len(primary)
        if primary
        else None
    )
    return {
        "subset_stage": "+".join(str(item["subset_stage"]) for item in evaluations),
        "video_count": sum(int(item["video_count"]) for item in evaluations),
        "window_count": sum(int(item["window_count"]) for item in evaluations),
        "ddim_steps": evaluations[0]["ddim_steps"],
        "rates": combined_rates,
        "macro_full_image_nmse_p1_p2_p3": macro,
    }

