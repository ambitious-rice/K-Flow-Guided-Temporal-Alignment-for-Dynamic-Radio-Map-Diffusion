"""Build the contiguous training cache used by the noise-aware T1 runner."""

from __future__ import annotations

import argparse
from pathlib import Path

from .config import load_config
from .packed_data import build_packed_cache


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--config", required=True)
    parser.add_argument("--output", required=True)
    parser.add_argument("--repository-root", default=".")
    parser.add_argument("--workers", type=int, default=32)
    args = parser.parse_args()
    config = load_config(args.config)
    repository_root = Path(args.repository_root).expanduser().resolve()
    split_file = Path(config.data.split_file)
    if not split_file.is_absolute():
        split_file = repository_root / split_file
    build_packed_cache(
        data_root=config.data.root,
        split_file=split_file,
        output=args.output,
        frames_per_video=config.data.frames_per_video,
        workers=args.workers,
    )


if __name__ == "__main__":
    main()
