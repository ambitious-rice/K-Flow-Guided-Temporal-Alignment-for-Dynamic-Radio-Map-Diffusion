"""Evaluate an x0 continuation checkpoint with a selectable Stage-A protocol."""

from __future__ import annotations

import argparse
import json
from pathlib import Path

import torch
from accelerate import Accelerator, DataLoaderConfiguration

from rmdm_hvdit_v4_joint.evaluation.evaluator import evaluate_stage_a
from rmdm_hvdit_v4_joint.training.engine import write_json_atomic
from rmdm_hvdit_v4_x0.model import build_t1_system
from rmdm_hvdit_v4_x0_continue import ARCHITECTURE_ID
from rmdm_hvdit_v4_x0_continue.config import load_config


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--config", required=True)
    parser.add_argument("--checkpoint", required=True)
    parser.add_argument("--output", required=True)
    parser.add_argument("--rates", default="")
    parser.add_argument("--manifest", default="")
    parser.add_argument("--split", choices=("val", "test"), default="val")
    parser.add_argument("--subset-stage", default="stage_a")
    parser.add_argument("--full100", action="store_true")
    parser.add_argument("--num-processes", type=int, default=0)
    args = parser.parse_args()

    config = load_config(args.config)
    accelerator = Accelerator(
        mixed_precision=config.t1_train.mixed_precision,
        dataloader_config=DataLoaderConfiguration(even_batches=False),
    )
    if args.num_processes and accelerator.num_processes != args.num_processes:
        raise RuntimeError(f"expected {args.num_processes} processes, got {accelerator.num_processes}")
    payload = torch.load(Path(args.checkpoint).expanduser().resolve(), map_location="cpu", weights_only=False)
    if payload.get("architecture_id") != ARCHITECTURE_ID or "model" not in payload:
        raise ValueError("checkpoint is not compatible with the x0 continuation model")
    model = build_t1_system(config)
    model.load_state_dict(payload["model"], strict=True)
    model = accelerator.prepare(model)
    rates = [float(value) for value in args.rates.split(",") if value.strip()] or None
    result = evaluate_stage_a(
        accelerator,
        model,
        config,
        variant="t1",
        subset_stage=args.subset_stage,
        split=args.split,
        manifest_path=args.manifest or None,
        rates=rates,
        full100=args.full100,
    )
    if accelerator.is_main_process:
        result["checkpoint"] = str(Path(args.checkpoint).expanduser().resolve())
        write_json_atomic(args.output, result)
        print(json.dumps({"output": str(Path(args.output).resolve()), "score": result["macro_full_image_nmse_p1_p2_p3"]}), flush=True)


if __name__ == "__main__":
    main()
