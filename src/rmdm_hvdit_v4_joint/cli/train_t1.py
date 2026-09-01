"""Train and gate the explicit T1 model."""

from __future__ import annotations

import argparse
import json

from rmdm_hvdit_v4_joint.training.runner_t1 import run_t1_training
from rmdm_hvdit_v4_joint.training.execution import parse_physical_gpus, parse_wall_clock

from .common import config_argument, load_arguments


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    config_argument(parser)
    parser.add_argument("--output-dir", default="")
    parser.add_argument("--resume-from", default="")
    parser.add_argument("--execution-gpus", default="")
    parser.add_argument("--gradient-accumulation-steps", type=int, default=0)
    parser.add_argument("--stop-at", default="")
    parser.add_argument("--pause-before-validation-seconds", type=int, default=600)
    args = parser.parse_args()
    config, config_path, root = load_arguments(args)
    result = run_t1_training(
        config,
        config_path=config_path,
        repository_root=root,
        output_dir=args.output_dir or None,
        resume_from=args.resume_from or None,
        execution_gpus=parse_physical_gpus(args.execution_gpus) if args.execution_gpus else None,
        gradient_accumulation_steps=args.gradient_accumulation_steps or None,
        stop_at=parse_wall_clock(args.stop_at),
        pause_before_validation_seconds=args.pause_before_validation_seconds,
    )
    print(json.dumps(result, ensure_ascii=False, sort_keys=True), flush=True)
    if result.get("status") != "paused" and not result["passed"]:
        raise SystemExit(2)


if __name__ == "__main__":
    main()
