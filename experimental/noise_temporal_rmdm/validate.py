"""CLI for deterministic T1 DDIM validation."""

from __future__ import annotations

import argparse
import json

from .config import load_config
from .validation import run_validation


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
    args = parser.parse_args()
    config = load_config(args.config)
    summary = run_validation(
        config,
        checkpoint_path=args.checkpoint,
        repository_root=args.repository_root,
        output_path=args.output,
        evaluation_split=args.split,
        device=args.device,
        ddim_steps=args.ddim_steps or None,
        max_batches=args.max_batches,
    )
    print(json.dumps(summary, indent=2))


if __name__ == "__main__":
    main()
