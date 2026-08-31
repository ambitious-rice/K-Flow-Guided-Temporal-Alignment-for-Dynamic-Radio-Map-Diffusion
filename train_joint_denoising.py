#!/usr/bin/env python3
"""Train the W16 Stage1-conditioned joint RMDM."""

from __future__ import annotations

import argparse

from rmdm.config import load_config
from rmdm.training import run_training


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--config", default="configs/joint_denoising/w16_pilot.yaml")
    parser.add_argument("--output_dir", default="")
    parser.add_argument("--resume_from", default="")
    parser.add_argument("--max_steps", type=int, default=None)
    parser.add_argument("--workers", type=int, default=None)
    return parser.parse_args()


def main() -> None:
    args = parse_args()
    config = load_config(args.config)
    if args.output_dir:
        config.train.output_dir = args.output_dir
    if args.resume_from:
        config.train.resume_from = args.resume_from
    if args.max_steps is not None:
        config.train.max_steps = args.max_steps
    if args.workers is not None:
        config.data.workers = args.workers
    config.validate()
    run_training(config)


if __name__ == "__main__":
    main()
