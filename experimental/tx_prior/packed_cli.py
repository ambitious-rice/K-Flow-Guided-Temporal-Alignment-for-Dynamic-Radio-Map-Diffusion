"""Build or inspect the disposable tx_prior packed cache."""

from __future__ import annotations

import argparse
import json
import time
from pathlib import Path

from .packed import DEFAULT_ROOT, build_cache, inspect_cache, remove_cache, verify_cache


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("action", choices=("build", "inspect", "verify", "remove"))
    parser.add_argument("--root", type=Path, default=DEFAULT_ROOT)
    parser.add_argument("--source-root", required=False)
    parser.add_argument("--split-file", required=False)
    parser.add_argument("--confirm", action="store_true")
    args = parser.parse_args()
    if args.action == "build":
        if not args.source_root or not args.split_file:
            parser.error("build requires --source-root and --split-file")
        started = time.monotonic()

        def progress(done: int, total: int) -> None:
            if done == 1 or done % 100 == 0 or done == total:
                elapsed = time.monotonic() - started
                print(
                    f"[tx-prior-pack] done={done}/{total} elapsed_seconds={elapsed:.1f}",
                    flush=True,
                )

        result = build_cache(
            root=args.root,
            source_root=args.source_root,
            split_file=args.split_file,
            progress=progress,
        )
    elif args.action == "inspect":
        result = inspect_cache(args.root)
    elif args.action == "verify":
        if not args.source_root:
            parser.error("verify requires --source-root")
        if not args.split_file:
            parser.error("verify requires --split-file")
        result = verify_cache(
            args.root, source_root=args.source_root, split_file=args.split_file
        )
    else:
        remove_cache(args.root, confirm=args.confirm)
        result = {"state": "removed", "root": str(args.root)}
    # The manifest contains all 10,500 record mappings. Keep routine build and
    # inspect output concise; the complete mapping remains in manifest.json.
    printable = (
        {key: value for key, value in result.items() if key != "records"}
        if isinstance(result, dict)
        else result
    )
    print(json.dumps(printable, indent=2))


if __name__ == "__main__":
    main()
