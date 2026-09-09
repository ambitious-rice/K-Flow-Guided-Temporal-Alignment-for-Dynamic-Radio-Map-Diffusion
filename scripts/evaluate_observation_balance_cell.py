#!/usr/bin/env python3
"""Evaluate one fixed observation-balance test cell."""

from __future__ import annotations

import os

# CUDA determinism must be configured before importing torch.
os.environ.setdefault("CUBLAS_WORKSPACE_CONFIG", ":4096:8")

import argparse
import json
from pathlib import Path
import random
import sys
from typing import Any

REPOSITORY_ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(REPOSITORY_ROOT))
sys.path.insert(0, str(REPOSITORY_ROOT / "src"))

import numpy as np
import torch
from accelerate import Accelerator, DataLoaderConfiguration
from torch.utils.data import DataLoader

from rmdm.data import WindowDataset
from rmdm.evaluation.metrics import MetricAccumulator
from rmdm.evaluation.fixed_sparse_protocol import (
    DDIM_NOISE_VERSION,
    MASK_SAMPLER_VERSION,
    OBSERVATION_NOISE_VERSION,
    add_fixed_observation_noise,
    apply_fixed_sparse_observations,
    deterministic_frame_noise_like,
    frame_names_by_sample,
)
from rmdm.evaluation.observation_balance_suite import (
    CELL_SCHEMA,
    RESULT_SCHEMA,
    read_json,
    write_json_atomic,
)
from rmdm_hvdit_v4_joint.config import load_config
from rmdm_hvdit_v4_joint.model import build_t1_system, build_w16_system
from rmdm_noise_estimation.assimilation import NoiseAwareDDIMSampler


def _configure_determinism(seed: int) -> None:
    random.seed(seed)
    np.random.seed(seed % (2**32))
    torch.manual_seed(seed)
    if torch.cuda.is_available():
        torch.cuda.manual_seed_all(seed)
    torch.backends.cudnn.benchmark = False
    torch.backends.cudnn.deterministic = True
    torch.backends.cuda.matmul.allow_tf32 = False
    torch.backends.cudnn.allow_tf32 = False
    torch.use_deterministic_algorithms(True)


def _observed_sums(
    prediction: torch.Tensor,
    target: torch.Tensor,
    mask: torch.Tensor,
) -> torch.Tensor:
    selected = mask > 0.5
    error = prediction.double() - target.double()
    selected64 = selected.double()
    return torch.stack(
        (
            (error.square() * selected64).sum(),
            (error.abs() * selected64).sum(),
            (target.double().square() * selected64).sum(),
            selected64.sum(),
        )
    )


def _observed_metrics(raw: torch.Tensor) -> tuple[dict[str, float], dict[str, float]]:
    squared, absolute, energy, count = [float(value) for value in raw.detach().cpu()]
    if count <= 0:
        raise RuntimeError("observed-point metric domain is empty")
    values = {
        "sum_squared_error": squared,
        "sum_absolute_error": absolute,
        "sum_target_energy": energy,
        "pixel_count": count,
    }
    return {
        "mse": squared / count,
        "nmse": squared / max(energy, 1.0e-12),
        "mae": absolute / count,
    }, values


def _load_model(variant: str, config: Any, checkpoint_path: Path) -> tuple[torch.nn.Module, dict[str, Any]]:
    payload = torch.load(checkpoint_path, map_location="cpu", weights_only=False, mmap=True)
    model = build_t1_system(config) if variant == "w1" else build_w16_system(config)
    model.load_state_dict(payload["model"], strict=True)
    metadata = {
        "schema": payload.get("schema"),
        "architecture_id": payload.get("architecture_id"),
        "global_step": int(payload.get("global_step", -1)),
        "epoch": int(payload.get("completed_epoch", payload.get("epoch", -1))),
    }
    return model, metadata


def evaluate(
    accelerator: Accelerator,
    model: torch.nn.Module,
    config: Any,
    cell: dict[str, Any],
) -> dict[str, Any]:
    variant = str(cell["variant"])
    if not set(cell["methods"]) <= {"no_da", "known_noise_da"}:
        raise ValueError("cell methods must be no_da or known_noise_da")
    window_size = 1 if variant == "w1" else 16
    if cell["starts"] == "all":
        starts = list(range(config.data.frames_per_video)) if variant == "w1" else [0, 16, 32, 48, 64, 80]
    else:
        starts = [int(value) for value in cell["starts"]]
    dataset = WindowDataset(
        root=config.data.root,
        split=cell["split"],
        split_file=config.data.split_file,
        window_size=window_size,
        seed=int(cell["seeds"]["dataset"]),
        cache_size=config.data.cache_size,
        tx_heatmap_sigma_px=config.data.tx_heatmap_sigma_px,
        fixed_starts=tuple(starts),
        video_ids=list(cell["video_ids"]),
    )
    batch_size = (
        config.evaluation.t1_evaluation_batch_size
        if variant == "w1"
        else config.evaluation.w16_evaluation_batch_size
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
    core.eval().requires_grad_(False)
    sampler = NoiseAwareDDIMSampler(config.diffusion)
    accumulators = {
        method: MetricAccumulator(device=accelerator.device)
        for method in cell["methods"]
    }
    observed = {
        method: torch.zeros(4, dtype=torch.float64, device=accelerator.device)
        for method in cell["methods"]
    }
    scored_local = 0
    for dense_batch in loader:
        sparse = apply_fixed_sparse_observations(
            dense_batch,
            rate=float(cell["rate"]),
            split=cell["split"],
            manifest_seed=int(cell["seeds"]["mask"]),
        )
        sparse = add_fixed_observation_noise(
            sparse,
            standard_deviation=float(cell["noise_std"]),
            rate=float(cell["rate"]),
            seed=int(cell["seeds"]["observation_noise"]),
        )
        target = sparse["target"]
        initial_noise = deterministic_frame_noise_like(
            target,
            frame_names_by_sample(
                sparse,
                batch_size=target.shape[0],
                window_size=target.shape[1],
            ),
            rate=float(cell["rate"]),
            seed=int(cell["seeds"]["ddim_noise"]),
        )
        with torch.no_grad(), accelerator.autocast():
            cache = core.encode_conditions(sparse)
        predictions: dict[str, torch.Tensor] = {}
        if "no_da" in cell["methods"]:
            predictions["no_da"] = sampler.baseline(
                core,
                cache,
                initial_noise,
                steps=int(cell["ddim_steps"]),
                accelerator=accelerator,
            )
        if "known_noise_da" in cell["methods"]:
            guidance = cell["guidance"]
            predictions["known_noise_da"] = sampler.guided(
                core,
                cache,
                sparse,
                initial_noise,
                steps=int(cell["ddim_steps"]),
                guided_steps=min(int(guidance["guided_steps"]), int(cell["ddim_steps"])),
                strength=float(guidance["strength"]),
                max_update=float(guidance["max_update"]),
                noise_variance=float(cell["noise_std"]) ** 2,
                accelerator=accelerator,
            )
        for method, prediction in predictions.items():
            accumulators[method].update(
                prediction,
                target,
                sparse["building"],
                sparse["vehicle"],
                sparse["sampling_mask"],
            )
            observed[method] += _observed_sums(
                prediction, target, sparse["sampling_mask"]
            )
        scored_local += int(target.shape[0] * target.shape[1])

    methods: dict[str, Any] = {}
    for method in cell["methods"]:
        accumulators[method].sums = accelerator.reduce(accumulators[method].sums, reduction="sum")
        observed[method] = accelerator.reduce(observed[method], reduction="sum")
        point_metrics, point_raw = _observed_metrics(observed[method])
        methods[method] = {
            "metrics": accumulators[method].compute(),
            "raw": accumulators[method].raw(),
            "observed_points": {"metrics": point_metrics, "raw": point_raw},
        }
    scored = int(
        accelerator.reduce(
            torch.tensor(scored_local, dtype=torch.int64, device=accelerator.device),
            reduction="sum",
        ).item()
    )
    expected = len(cell["video_ids"]) * len(starts) * window_size
    if scored != expected:
        raise RuntimeError(f"scored {scored} frames, expected {expected}")
    return {
        "schema": RESULT_SCHEMA,
        "cell": cell,
        "scored_frames": scored,
        "methods": methods,
        "random_generators": {
            "mask": MASK_SAMPLER_VERSION,
            "ddim_noise": DDIM_NOISE_VERSION,
            "observation_noise": OBSERVATION_NOISE_VERSION,
            "frame_keyed": True,
        },
    }


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--cell", required=True)
    parser.add_argument("--config")
    parser.add_argument("--checkpoint")
    parser.add_argument("--output")
    parser.add_argument("--expected-world-size", type=int)
    args = parser.parse_args()

    cell = read_json(args.cell)
    if cell.get("schema") != CELL_SCHEMA:
        raise ValueError("cell file has the wrong schema")
    _configure_determinism(int(cell["seeds"]["global"]))
    accelerator = Accelerator(
        mixed_precision="bf16",
        dataloader_config=DataLoaderConfiguration(even_batches=False),
    )
    expected_world_size = int(cell["world_size"])
    if args.expected_world_size is not None and args.expected_world_size != expected_world_size:
        parser.error("--expected-world-size differs from cell world size")
    if accelerator.num_processes != expected_world_size:
        raise RuntimeError(
            f"expected world size {expected_world_size}, got {accelerator.num_processes}"
        )
    config_path = Path(args.config or cell["config"]).expanduser().resolve()
    checkpoint_path = Path(args.checkpoint or cell["checkpoint"]).expanduser().resolve()
    config = load_config(config_path)
    model, checkpoint_metadata = _load_model(cell["variant"], config, checkpoint_path)
    model = accelerator.prepare(model)
    result = evaluate(accelerator, model, config, cell)
    if accelerator.is_main_process:
        result["checkpoint"] = checkpoint_metadata
        result["artifacts"] = {
            "config": str(config_path),
            "checkpoint": str(checkpoint_path),
        }
        output = Path(args.output or cell["result"]).expanduser().resolve()
        write_json_atomic(output, result)
        print(json.dumps({"cell_id": cell["cell_id"], "output": str(output)}))


if __name__ == "__main__":
    main()
