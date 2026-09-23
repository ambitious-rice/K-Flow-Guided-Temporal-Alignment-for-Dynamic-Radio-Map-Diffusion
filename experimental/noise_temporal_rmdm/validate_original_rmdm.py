"""Evaluate a Tx-free original single-frame RMDM with the T1 noise protocol."""

from __future__ import annotations

import argparse
import json

import torch

from rmdm_hvdit_v4_joint.evaluation.legacy_rmdm import LegacyRMDMT1ProtocolAdapter
from train_sparse_dynamic_rmdm import build_model_config
from utils import build_unet_from_config

from .config import load_config
from .validation import run_validation


def _load_model(checkpoint_path: str) -> tuple[torch.nn.Module, int]:
    payload = torch.load(checkpoint_path, map_location="cpu", weights_only=False, mmap=True)
    if (
        payload.get("schema") != "rmdm_sf_sparse_checkpoint_v1"
        or "model" not in payload
        or "args" not in payload
    ):
        raise ValueError("checkpoint is not an RMDM-SF sparse artifact")
    train_args = argparse.Namespace(**payload["args"])
    if not bool(getattr(train_args, "without_tx", False)):
        raise ValueError("comparison requires a checkpoint trained without Tx input")
    model = build_unet_from_config(build_model_config(train_args))
    model.load_state_dict(payload["model"], strict=True)
    step = int(payload.get("global_step", -1))
    if step < 0:
        raise ValueError("checkpoint has no valid global_step")
    return LegacyRMDMT1ProtocolAdapter(model, without_tx=True), step


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--config", required=True)
    parser.add_argument("--checkpoint", required=True)
    parser.add_argument("--output", required=True)
    parser.add_argument("--repository-root", default=".")
    parser.add_argument("--split", choices=("val", "test"), default="val")
    parser.add_argument("--device", default="cuda")
    parser.add_argument("--ddim-steps", type=int, default=0)
    parser.add_argument("--max-batches", type=int, default=0)
    parser.add_argument("--data-root")
    parser.add_argument("--manifest")
    parser.add_argument("--rates", type=float, nargs="+")
    parser.add_argument("--sigmas", type=float, nargs="+")
    parser.add_argument("--frame-starts", type=int, nargs="+")
    parser.add_argument("--batch-size", type=int)
    parser.add_argument("--workers", type=int)
    args = parser.parse_args()

    config = load_config(args.config)
    protocol = config.validation if args.split == "val" else config.final_test
    for argument, field in (
        ("manifest", "subset_manifest"),
        ("rates", "rates"),
        ("sigmas", "noise_standard_deviations"),
        ("frame_starts", "frame_starts"),
        ("batch_size", "batch_size"),
    ):
        value = getattr(args, argument)
        if value is not None:
            setattr(protocol, field, value)
    if args.data_root is not None:
        config.data.root = args.data_root
    if args.workers is not None:
        config.data.workers = args.workers
    config.validate()

    model, checkpoint_step = _load_model(args.checkpoint)
    summary = run_validation(
        config,
        checkpoint_path=args.checkpoint,
        repository_root=args.repository_root,
        output_path=args.output,
        evaluation_split=args.split,
        device=args.device,
        ddim_steps=args.ddim_steps or None,
        max_batches=args.max_batches,
        model=model,
        checkpoint_step=checkpoint_step,
        evaluated_model="original_rmdm_clean_baseline",
    )
    print(json.dumps(summary, indent=2))


if __name__ == "__main__":
    main()
