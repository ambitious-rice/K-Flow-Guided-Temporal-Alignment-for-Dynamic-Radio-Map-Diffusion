"""CLI for local noise-aware T1 training."""

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
    config = load_config(args.config, smoke=args.smoke)
    run(
        config,
        config_path=Path(args.config),
        repository_root=Path(args.repository_root),
        smoke=args.smoke,
        smoke_data_limit=args.smoke_data_limit,
        resume_from=args.resume_from,
    )


if __name__ == "__main__":
    main()
