"""Ablate T16 temporal boundaries on identical clips and DDIM noise."""

from __future__ import annotations

import argparse
import json
from pathlib import Path

import torch
from torch import nn

from experimental.noise_temporal_rmdm.checkpoint import load
from experimental.noise_temporal_rmdm.config import load_config
from experimental.noise_temporal_rmdm.model import build_model
from experimental.noise_temporal_rmdm.validation import run_validation


STAGES = ("calibration", "condition", "denoised")


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--config", default="experimental/noise_temporal_rmdm/t16_local_4gpu.yaml")
    parser.add_argument("--checkpoint", required=True)
    parser.add_argument("--output", required=True)
    parser.add_argument("--variants", nargs="+", default=["all", "none", "no_condition", "no_denoised", "no_calibration"])
    parser.add_argument("--max-batches", type=int, default=5)
    parser.add_argument("--batch-size", type=int, default=2)
    parser.add_argument("--rates", nargs="+", type=float, default=[1.0, 3.0])
    parser.add_argument("--sigmas", nargs="+", type=float, default=[0.0, 0.09])
    args = parser.parse_args()

    config = load_config(args.config)
    config.data.window_size = 16
    config.data.workers = 4
    config.validation.batch_size = args.batch_size
    config.validation.rates = args.rates
    config.validation.noise_standard_deviations = args.sigmas
    model = build_model(config)
    phase = config.runtime.phase
    payload = load(args.checkpoint, model, expected_phase=phase)
    originals = dict(model.temporal_hook.stages) if phase == "t16" else {}
    output = Path(args.output)
    output.mkdir(parents=True, exist_ok=True)
    scores = {}
    for variant in args.variants:
        if phase == "t16":
            active = set(STAGES) if variant == "all" else set()
            if variant.startswith("no_"):
                active = set(STAGES) - {variant[3:]}
            if variant.startswith("only_"):
                active = {variant[5:]}
            for stage in STAGES:
                model.temporal_hook.stages[stage] = originals[stage] if stage in active else nn.Identity()
        result = run_validation(config, checkpoint_path=args.checkpoint,
            repository_root=Path(__file__).resolve().parents[2],
            output_path=output/f"{variant}.json", model=model,
            checkpoint_step=int(payload["global_step"]),
            evaluated_model=f"{phase}_{variant}", max_batches=args.max_batches)
        scores[variant] = sum(row["mse"] for row in result["results"]) / len(result["results"])
        print(json.dumps({"variant": variant, "mse": scores[variant]}), flush=True)
    (output/"scores.json").write_text(json.dumps(scores, indent=2))


if __name__ == "__main__":
    main()
