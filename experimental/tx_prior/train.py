"""Explicit CLI for real tx_prior training."""

from __future__ import annotations

import argparse
from pathlib import Path

from .config import load_config
from .runner import run


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--config", required=True)
    parser.add_argument("--repository-root", default=".")
    parser.add_argument("--resume-from", default="")
    parser.add_argument("--smoke", action="store_true")
    parser.add_argument("--smoke-data-limit", type=int, default=0)
    args = parser.parse_args()
    config_path = Path(args.config).expanduser().resolve()
    config = load_config(config_path, smoke=args.smoke)
    if args.smoke and args.smoke_data_limit <= 0:
        parser.error("smoke config requires --smoke-data-limit")
    if not args.smoke and args.smoke_data_limit:
        parser.error("formal training rejects --smoke-data-limit")
    run(config, config_path=config_path,
        repository_root=Path(args.repository_root).expanduser().resolve(),
        resume_from=args.resume_from, smoke=args.smoke, smoke_limit=args.smoke_data_limit)


if __name__ == "__main__":
    main()
