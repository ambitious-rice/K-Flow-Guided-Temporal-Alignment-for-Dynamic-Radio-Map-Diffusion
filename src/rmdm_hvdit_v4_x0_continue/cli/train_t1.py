"""Continue the validated V4-W1 x0 model from step 10k toward step 50k."""

from __future__ import annotations

import argparse
from pathlib import Path

from rmdm_hvdit_v4_x0_continue.config import load_config
from rmdm_hvdit_v4_x0_continue.runner_t1 import run_continuation


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--config", required=True)
    parser.add_argument("--repository-root", default=".")
    parser.add_argument(
        "--source-checkpoint",
        default="runs/rmdm_hvdit_v4_x0/t1_pilot_10k/checkpoints/last.pth",
    )
    parser.add_argument("--output-dir", default="")
    parser.add_argument("--resume-from", default="")
    parser.add_argument("--finetune-from", default="")
    parser.add_argument("--observation-alignment-weight", type=float, default=0.0)
    parser.add_argument("--from-scratch", action="store_true")
    args = parser.parse_args()
    config_path = Path(args.config).expanduser().resolve()
    root = Path(args.repository_root).expanduser().resolve()
    run_continuation(
        load_config(config_path),
        config_path=config_path,
        repository_root=root,
        source_checkpoint=(
            None if args.from_scratch else Path(args.source_checkpoint).expanduser().resolve()
        ),
        output_dir=args.output_dir or None,
        resume_from=args.resume_from or None,
        finetune_from=args.finetune_from or None,
        observation_alignment_weight=args.observation_alignment_weight,
        from_scratch=args.from_scratch,
    )


if __name__ == "__main__":
    main()
