"""Build the contiguous training cache used by the noise-aware T1 runner."""

from __future__ import annotations

import argparse
from pathlib import Path

from .config import load_config
from .packed_data import build_packed_cache, verify_packed_cache


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--config", required=True)
    parser.add_argument("--output", required=True)
    parser.add_argument("--repository-root", default=".")
    parser.add_argument("--data-root", default="", help="Override the config dataset root for this machine")
    parser.add_argument("--workers", type=int, default=32)
    parser.add_argument("--verify-videos", type=int, default=64)
    args = parser.parse_args()
    config = load_config(args.config)
    repository_root = Path(args.repository_root).expanduser().resolve()
    split_file = Path(config.data.split_file)
    if not split_file.is_absolute():
        split_file = repository_root / split_file
    data_root = args.data_root or config.data.root
    build_packed_cache(
        data_root=data_root,
        split_file=split_file,
        output=args.output,
        frames_per_video=config.data.frames_per_video,
        workers=args.workers,
    )
    result = verify_packed_cache(
        args.output, data_root=data_root, split_file=split_file,
        sample_videos=args.verify_videos,
    )
    print(result, flush=True)


if __name__ == "__main__":
    main()
