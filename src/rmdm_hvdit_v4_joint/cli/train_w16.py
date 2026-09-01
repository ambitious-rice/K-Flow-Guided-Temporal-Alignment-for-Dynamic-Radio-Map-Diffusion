"""Train W16 from the strict inflation artifact."""

from __future__ import annotations

import argparse
import json

from rmdm_hvdit_v4_joint.training.runner_w16 import run_w16_training
from rmdm_hvdit_v4_joint.training.execution import parse_physical_gpus

from .common import config_argument, load_arguments


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    config_argument(parser)
    parser.add_argument("--initialization", required=True)
    parser.add_argument("--output-dir", default="")
    parser.add_argument("--per-gpu-batch-size", type=int, default=0)
    parser.add_argument("--resume-from", default="")
    parser.add_argument("--execution-gpus", default="")
    args = parser.parse_args()
    config, config_path, root = load_arguments(args)
    result = run_w16_training(
        config,
        config_path=config_path,
        repository_root=root,
        initialization_path=args.initialization,
        output_dir=args.output_dir or None,
        per_gpu_batch_size=args.per_gpu_batch_size or None,
        resume_from=args.resume_from or None,
        execution_gpus=parse_physical_gpus(args.execution_gpus) if args.execution_gpus else None,
    )
    print(json.dumps(result, ensure_ascii=False, sort_keys=True), flush=True)


if __name__ == "__main__":
    main()
