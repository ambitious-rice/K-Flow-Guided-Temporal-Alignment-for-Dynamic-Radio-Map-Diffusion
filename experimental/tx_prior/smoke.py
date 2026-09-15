"""Safe tx_prior configuration/model preflight; training uses train.py."""

from __future__ import annotations

import argparse
import json

from .adapter import build_scene_prior_system
from .config import load_config


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--config", default="experimental/tx_prior/smoke.yaml")
    parser.add_argument("--build-model", action="store_true")
    args = parser.parse_args()
    config = load_config(args.config, smoke=True)
    summary = {
        "mode": "preflight_only",
        "gpus": config.pipeline.allowed_physical_gpus,
        "per_gpu_batch_size": config.t1_train.per_gpu_batch_size,
        "gradient_accumulation_steps": config.t1_train.gradient_accumulation_steps,
        "effective_global_batch_size": config.t1_train.effective_global_batch_size,
        "max_steps": config.t1_train.max_steps,
        "output_root": config.pipeline.output_root,
    }
    if args.build_model:
        model = build_scene_prior_system(config)
        summary["parameters"] = sum(parameter.numel() for parameter in model.parameters())
    print(json.dumps(summary, indent=2))


if __name__ == "__main__":
    main()
