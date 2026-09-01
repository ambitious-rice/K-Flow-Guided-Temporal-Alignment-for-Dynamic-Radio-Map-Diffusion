"""Train the isolated W1 x0-prediction pilot to 10k and run Stage-A once."""

from __future__ import annotations

import argparse

from rmdm_hvdit_v4_x0.training.runner_t1 import run_t1_pilot

from .common import config_argument, load_arguments


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    config_argument(parser)
    parser.add_argument("--output-dir", default="")
    parser.add_argument("--resume-from", default="")
    args = parser.parse_args()
    config, config_path, root = load_arguments(args)
    run_t1_pilot(
        config,
        config_path=config_path,
        repository_root=root,
        output_dir=args.output_dir or None,
        resume_from=args.resume_from or None,
    )


if __name__ == "__main__":
    main()
