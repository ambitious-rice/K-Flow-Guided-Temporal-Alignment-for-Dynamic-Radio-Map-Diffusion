"""CLI for noise-aware T1 or T16 training."""

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
    parser.add_argument("--initialize-from-t1", default="")
    args = parser.parse_args()
    config = load_config(args.config)
    run(
        config,
        config_path=Path(args.config),
        repository_root=Path(args.repository_root),
        resume_from=args.resume_from,
        initialize_from_t1=args.initialize_from_t1,
    )


if __name__ == "__main__":
    main()
